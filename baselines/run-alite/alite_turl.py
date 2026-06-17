from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import time,json


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import trange
from tqdm.autonotebook import tqdm

from data_loader.hybrid_data_loaders import *
from data_loader.header_data_loaders import *
from data_loader.CT_Wiki_data_loaders import *
from data_loader.RE_data_loaders import *
from data_loader.EL_data_loaders import *
from model.configuration import TableConfig

from model.model import HybridTableMaskedLM, HybridTableCER, TableHeaderRanking, HybridTableCT,HybridTableEL,HybridTableRE,BertRE
from model.transformers.configuration_bert import BertConfig
from model.transformers.tokenization_bert import BertTokenizer
from model.transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from utils.util import *
from baselines.row_population.metric import average_precision,ndcg_at_k
from baselines.cell_filling.cell_filling import *
from model import metric

from scipy.spatial.distance import cosine, euclidean, jaccard
import nltk, string, re
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)
import glob
from collections import Counter
from random import sample
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import codecs
import gc

gc.collect()

data_dir = 'data/'
config_name = "configs/table-base-config_v2.json"
device = torch.device('cpu')
# device = torch.device('cuda')
# load entity vocab from entity_vocab.txt
entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2)
entity_wikid2id = {entity_vocab[x]['wiki_id']:x for x in entity_vocab}

config_class, model_class, _ = (TableConfig, HybridTableMaskedLM, BertTokenizer)
config = config_class.from_pretrained(config_name)
config.output_attentions = True

checkpoint = "checkpoint/"
model = model_class(config, is_simple=True)
# checkpoint = torch.load(os.path.join(checkpoint, 'pytorch_model.bin'))
checkpoint = torch.load(os.path.join(checkpoint, 'pytorch_model.bin'), map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# pdb.set_trace()

# with open(os.path.join(data_dir,"CF_test_data.json"), 'r') as f:
#     dev_data = json.load(f)
dataset = WikiHybridTableDataset(data_dir,entity_vocab,max_cell=100, max_input_tok=350, max_input_ent=150, src="dev", max_length = [50, 10, 10], force_new=False, tokenizer = None, mode=0)
# CF = cell_filling(data_dir)

# pdb.set_trace()

# This is an example of converting an arbitrary table to input
# Here we show an example for cell filling task
# The input entites are entities in the subject column, we append [ENT_MASK] and use its representation to match with the candidate entities
def CF_build_input1(pgEnt, pgTitle, secTitle, caption, headers, core_entities, core_entities_text, entity_cand, config):
#     print(pgEnt) 
#     print(pgTitle)
#     print(secTitle)
#     print(caption)
#     print(headers)
#     print(core_entities)
#     print(core_entities_text)
#     print(entity_cand)
    tokenized_pgTitle = config.tokenizer.encode(pgTitle, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_meta = tokenized_pgTitle+\
                    config.tokenizer.encode(secTitle, max_length=config.max_title_length, add_special_tokens=False)
    if caption != secTitle:
        tokenized_meta += config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_headers = [config.tokenizer.encode(header, max_length=config.max_header_length, add_special_tokens=False) for header in headers]
    input_tok = []
    input_tok_pos = []
    input_tok_type = []
    tokenized_meta_length = len(tokenized_meta)
    input_tok += tokenized_meta
    input_tok_pos += list(range(tokenized_meta_length))
    input_tok_type += [0]*tokenized_meta_length
    header_span = []
    for tokenized_header in tokenized_headers:
        tokenized_header_length = len(tokenized_header)
        header_span.append([len(input_tok), len(input_tok)+tokenized_header_length])
        input_tok += tokenized_header
        input_tok_pos += list(range(tokenized_header_length))
        input_tok_type += [1]*tokenized_header_length
    
    input_ent = [config.entity_wikid2id[pgEnt] if pgEnt!=-1 else 0]
    input_ent_text = [tokenized_pgTitle[:config.max_cell_length]]
    input_ent_type = [2]
    
    # core entities in the subject column
    input_ent += [config.entity_wikid2id[entity] for entity in core_entities]
    input_ent_text += [config.tokenizer.encode(entity_text, max_length=config.max_cell_length, add_special_tokens=False) if len(entity_text)!=0 else [] for entity_text in core_entities_text]
    input_ent_type += [3]*len(core_entities)

    # append [ent_mask]
    input_ent += [config.entity_wikid2id['[ENT_MASK]']]*len(core_entities)
    input_ent_text += [[]]*len(core_entities)
    input_ent_type += [4]*len(core_entities)

    input_ent_cell_length = [len(x) if len(x)!=0 else 1 for x in input_ent_text]
    max_cell_length = max(input_ent_cell_length)
    input_ent_text_padded = np.zeros([len(input_ent_text), max_cell_length], dtype=int)
    for i,x in enumerate(input_ent_text):
        input_ent_text_padded[i, :len(x)] = x
    assert len(input_ent) == 1+2*len(core_entities)

    input_tok_mask = np.ones([1, len(input_tok), len(input_tok)+len(input_ent)], dtype=int)
    for header_i in header_span:
        input_tok_mask[0, header_i[0]:header_i[1], len(input_tok)+1+len(core_entities):] = 0
    input_tok_mask[0, :, len(input_tok)+1+len(core_entities):] = 0

    # pdb.set_trace()

    # build the mask for entities
    input_ent_mask = np.ones([1, len(input_ent), len(input_tok)+len(input_ent)], dtype=int)
    for header_i in header_span[1:]:
        input_ent_mask[0, 1:1+len(core_entities), header_i[0]:header_span[1][1]] = 0
        input_ent_mask[0, 1:1+len(core_entities), len(input_tok)+1+len(core_entities):] = np.eye(len(core_entities), dtype=int)
    input_ent_mask[0, 1+len(core_entities):, header_span[0][0]:header_span[0][1]] = 0
    input_ent_mask[0, 1+len(core_entities):, len(input_tok)+1:len(input_tok)+1+len(core_entities)] = np.eye(len(core_entities), dtype=int)
    input_ent_mask[0, 1+len(core_entities):, len(input_tok)+1+len(core_entities):] = np.eye(len(core_entities), dtype=int)

    input_tok_mask = torch.LongTensor(input_tok_mask)
    input_ent_mask = torch.LongTensor(input_ent_mask)

    input_tok = torch.LongTensor([input_tok])
    input_tok_type = torch.LongTensor([input_tok_type])
    input_tok_pos = torch.LongTensor([input_tok_pos])
    
    input_ent = torch.LongTensor([input_ent])
    input_ent_text = torch.LongTensor([input_ent_text_padded])
    input_ent_cell_length = torch.LongTensor([input_ent_cell_length])
    input_ent_type = torch.LongTensor([input_ent_type])

    input_ent_mask_type = torch.zeros_like(input_ent)
    input_ent_mask_type[:,1+len(core_entities):] = config.entity_wikid2id['[ENT_MASK]']
    
    candidate_entity_set = [config.entity_wikid2id[entity] for entity in entity_cand]
    candidate_entity_set = torch.LongTensor([candidate_entity_set])
    
    # pdb.set_trace()
    return input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
            input_ent, input_ent_text, input_ent_cell_length, input_ent_type, input_ent_mask_type, input_ent_mask, candidate_entity_set


results = []

import pandas as pd
import pdb

def read_tables(folder):
    tables = {}
    table_to_cluster = {}
    for folder_in in glob.glob(folder):
        for file in glob.glob(folder_in+'/*'):
            if 'json' in file:
                continue
            table_name = os.path.basename(file)
            # print(table_name)
            table_to_cluster[table_name] = os.path.basename(folder_in)
            table_content = pd.read_csv(file, encoding='latin-1',warn_bad_lines=True, error_bad_lines=False)
            # Drop nan columns
            table_content = table_content.dropna(axis=1, how='all')
            tables[table_name] = table_content
            # if "stockport_contracts_7.csv" in table_name:
            #     pdb.set_trace()
    return tables, table_to_cluster

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    # text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    punc_with_ = [s for s in string.punctuation if s != '_']
    text = re.sub('[%s]' % punc_with_, '', text)
    text = re.sub('\n', '', text)
    # Removes words that contain numbers:
    # text = re.sub('\w*\d\w*', '', text)
    return text


def text_preprocessing(text):
    """
    Cleaning and parsing the text.

    """
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(tokenized_text)
    return combined_text

text_to_entity = {}
for e in entity_vocab:
    wiki_title = text_preprocessing(entity_vocab[e]['wiki_title'])
    wiki_id = entity_vocab[e]['wiki_id']
    text_to_entity[wiki_title] = wiki_id

gc.collect()
torch.cuda.empty_cache()

# torch.cuda
# torch.cuda.memory_summary(device=None, abbreviated=False)

def read_table_representations(folder):
    all_files = glob.glob(folder + "/*.json")
    table_representations = {}
    for filename in all_files:
        temp = json.load(open(filename))
        table_representations = {**table_representations, **temp}
    return table_representations


def get_columns_represenation_data_lake_increment(tables, 
                                                  text_to_entity,
                                                  embedding_type,
                                                  folder,
                                                  sampling_size = 20, 
                                                  sampling_method = 'random', 
                                                  save_only_centroid = True, 
                                                  batches = 10, 
                                                  epsilon = 0.01):
    times = []
    table_representations = {}
    save_format = folder + 'table_representations_{}_{}.json'
    i = 0
    for table_id in tqdm(tables):
        if "json" in table_id:
            continue
        # print(table_id)
        table_representations[table_id] = {}
        table = tables[table_id]
        for col in table.columns:
            start_time = time.time()
            col_as_table = pd.DataFrame(table[col])
            pgEnt = '[PAD]'
            pgTitle = ''
            secTitle = ''
            caption = ''
            headers = list(col_as_table.columns)
            core_entities = [] # This will be the subject column entities
            core_entities_text = []
            all_entity_cand = []
            if len(col_as_table) < sampling_size:
                sampling_size_update = len(col_as_table) - 1
            else:
                sampling_size_update = sampling_size
            iters = 0
#             print(i)
            while len(col_as_table) > sampling_size_update:
                iters += 1
#                 if i == 69:
#                     print(iters)
                table_subset = col_as_table.head(sampling_size)
                col_as_table = col_as_table.iloc[sampling_size:, :]
                for index, row in table_subset.iterrows():
                    for columnIndex, value in row.items():
                        entity = text_preprocessing(str(value).replace(' ', '_'))
                        if entity in text_to_entity:
                            core_entities.append(text_to_entity[entity])
                            core_entities_text.append(entity) 
                            all_entity_cand.append(text_to_entity[entity])
                        else:
                            sub_entities = entity.split('_')
                            if sub_entities != None:
                                for sub_entity in sub_entities:
                                    if sub_entity in text_to_entity:
                                        core_entities.append(text_to_entity[sub_entity])
                                        core_entities_text.append(sub_entity) 
                                        all_entity_cand.append(text_to_entity[sub_entity])
                all_entity_cand = list(set(all_entity_cand))
                input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
                        input_ent, input_ent_text, input_ent_text_length, input_ent_type, input_ent_mask_type, input_ent_mask, \
                        candidate_entity_set = CF_build_input1(pgEnt, pgTitle, secTitle, caption, headers, core_entities, core_entities_text, all_entity_cand, dataset)
                
                
                input_tok = input_tok.to(device)
                input_tok_type = input_tok_type.to(device)
                input_tok_pos = input_tok_pos.to(device)
                input_tok_mask = input_tok_mask.to(device)
                input_ent_text = input_ent_text.to(device)
                input_ent_text_length = input_ent_text_length.to(device)
                input_ent = input_ent.to(device)
                input_ent_type = input_ent_type.to(device)
                input_ent_mask_type = input_ent_mask_type.to(device)
                input_ent_mask = input_ent_mask.to(device)
                candidate_entity_set = candidate_entity_set.to(device)
                with torch.no_grad():
                    tok_outputs, ent_outputs = model(input_tok, input_tok_type, input_tok_pos, input_tok_mask,
                                    input_ent_text, input_ent_text_length, input_ent_mask_type,
                                    input_ent, input_ent_type, input_ent_mask, candidate_entity_set)
                
                # pdb.set_trace()
                if save_only_centroid:
#                     table_mean_rep = {}
                    if embedding_type == 'entities_only':
                        new_mean = torch.mean(ent_outputs[1][0], 0, dtype=torch.float)
                    elif embedding_type == 'column_names_only':
                        new_mean = torch.mean(tok_outputs[1][0], 0, dtype=torch.float)
                    elif embedding_type == 'entities_and_column_name':
                        new_mean = torch.mean(tok_outputs[1][0], 0, dtype=torch.float).add(torch.mean(ent_outputs[1][0], 0, dtype=torch.float))/2
                    
                    if col not in table_representations[table_id]:
#                         table_mean_rep['entities_only'] = new_mean
                        table_representations[table_id][col] = new_mean
                    else:
                        current_mean = table_representations[table_id][col]
                        new_mean_add = current_mean.add(new_mean)/2
                        change = 1-cosine(current_mean, new_mean_add)
#                         print(change)
                        if epsilon < change or iters > 10:
#                             print(cosine(current_mean, new_mean_add))
                            break
                        else:
#                             table_mean_rep['entities_only'] = new_mean_add
                            table_representations[table_id][col] = new_mean_add
                    if tok_outputs in locals():
                        del tok_outputs, ent_outputs,input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
                            input_ent, input_ent_text, input_ent_text_length, input_ent_type, input_ent_mask_type, input_ent_mask, \
                            candidate_entity_set
            if col in table_representations[table_id]:
                final_rep = table_representations[table_id][col]
                table_representations[table_id][col] = {}
                table_representations[table_id][col][embedding_type] = final_rep.cpu().numpy().tolist()
                # table_representations[table_id][col]['entities_only'] = final_rep.cpu().numpy().tolist()
                # table_representations[table_id][col]['column_names_only'] = final_rep.cpu().numpy().tolist()
                # table_representations[table_id][col]['entities_and_column_name'] = final_rep.cpu().numpy().tolist()
                times.append(time.time() - start_time)
        i+=1
        # if i % 10 == 0:
        if i % 5 == 0:
#             np.savez('mydata.npz', **table_representations)
#             print(table_representations[table_id][col])
            # print(f"Saved intermediate representations for {i} tables")
            # print(save_format.format(i-batches, i))
            json.dump(table_representations, open(save_format.format(i-batches, i),'w'))
#             break
            del table_representations
            if tok_outputs in locals():
                del tok_outputs, ent_outputs,input_tok, input_tok_type, input_tok_pos, input_tok_mask,\
                    input_ent, input_ent_text, input_ent_text_length, input_ent_type, input_ent_mask_type, input_ent_mask, \
                    candidate_entity_set
            gc.collect()
            torch.cuda.empty_cache()
            table_representations = {}        
    # print(f"Saved final representations for all tables")
    # print(save_format.format(i-batches, i))
    json.dump(table_representations, open(save_format.format(i-batches, i),'w'))
    table_representations = read_table_representations(folder)
    return table_representations, times

# path_tables = 'Real Benchmark/' #folder containing folders of tables (integration sets)
# path_tables = 'SCHINT-Benchmark/' #folder containing folders of tables (integration sets)
path_tables = 'SINT-Benchmark/' #folder containing folders of tables (integration sets)
tables, table_to_cluster = read_tables(path_tables + '*')
# pdb.set_trace()
# embedding type
embedding_type = 'entities_and_column_name' #'entities_only', 'column_names_only', 
# pdb.set_trace()
# save_to = 'RealBenchmark_input/'
# save_to = 'SCHINT-Benchmark_input/'
# overall_folder = 'SCHINT-Benchmark_all/'
# overall_folder = 'SCHINT-Benchmark_with_numbers/'
overall_folder = 'SINT-Benchmark_with_numbers/'
# overall_folder = 'Real Benchmark_with_numbers/'
save_to = f'{overall_folder}/{embedding_type}/'

if not os.path.exists(save_to):
    os.makedirs(save_to)

table_representations, times = get_columns_represenation_data_lake_increment(tables, text_to_entity, embedding_type, save_to, 30, 'random', True, 5, 0.05)
table_representations = read_table_representations(save_to)

# pdb.set_trace()

table_representations_by_seed = {}
for table in table_representations:
    try:
        seed_table = table_to_cluster[table]
        # Load column mappings
        # column_mappings = sienna.load(f'{overall_folder}/{seed_table}_mappings.json')
        column_mappings = json.load(open(f'{path_tables}{seed_table}/{seed_table}_mappings.json'))
        if seed_table not in table_representations_by_seed:
            table_representations_by_seed[seed_table] = {}
        table_representations_by_seed[seed_table].setdefault(table, {})
        # table_representations_by_seed[seed_table][table].setdefault(embedding_type, table_representations[table])
        # Save the columns as their GT header
        # table_representations_by_seed[seed_table][table] = table_representations[table]
        # pdb.set_trace()
        table_representations_by_seed[seed_table][table] = {column_mappings[table][column]: table_representations[table][column] for column in table_representations[table]}
    except Exception as e:
        pdb.set_trace()
        
for seed in table_representations_by_seed:
    if f'table_representations_'+ seed + '.json' in os.listdir(overall_folder):
        # Load the file
        # existing_table_respresentations = sienna.load(f'{overall_folder}/table_representations_'+ seed + '.json')
        existing_table_respresentations = json.load(open(f'{overall_folder}/table_representations_'+ seed + '.json'))
        # For each column of the table add the new embedding type
        for table in existing_table_respresentations:
            for column in existing_table_respresentations[table]:
                existing_table_respresentations[table][column][embedding_type] = table_representations_by_seed[seed][table][column][embedding_type]
        json.dump(existing_table_respresentations, open(f'{overall_folder}/table_representations_'+ seed + '.json','w'))
    else:
        json.dump(table_representations_by_seed[seed], open(f'{overall_folder}/table_representations_'+ seed + '.json','w'))