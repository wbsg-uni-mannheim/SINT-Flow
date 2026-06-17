import torch
import pdb
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json, os, tqdm
import glob
import numpy as np
import sienna
from numpy.linalg import norm
# import sys
# sys.path.append("../alite-main/codes/align_utilities/")
# from align_phase import getColumnType, get_glove_embeddings, get_fasttext_embeddings, get_sentence_bert_embeddings, get_sentence_bert_embeddings_serialize, get_bert_embeddings, get_bert_embeddings_serialize
import re
import string
import random
import nltk
# import utilities as utl

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

## ALITE CODE ##
def CosineSimilarity(array1, array2):
    return np.dot(array1,array2)/(norm(array1)*norm(array2))

## ALITE CODE: fit for decoder LLMs ##
# A function to load the pretrained model and use it to encode tables.
def encode_finetuned(column_list, model, tokenizer):
    # Tokenize input sentences and convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # inputs = tokenizer(column_list, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
    inputs = tokenizer(" ".join(column_list), padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
    # pdb.set_trace()
    # Get the model outputs
    outputs = model(**inputs)
    embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # Generate embeddings for input sentences
    # with torch.no_grad():
    #     embeddings = model(input_ids, attention_mask)
    # print("embeddings tensor:", embeddings.shape)
    # Convert embeddings to numpy array and print
    # embeddings = embeddings.cpu().numpy()
    embeddings = embeddings.cpu().detach().numpy()
    # print("average embeddings len:", len(np.mean(embeddings, axis = 0)))
    # sys.exit()
    return embeddings

## ALITE CODE ##
def EmbedTable(serialized_rows, model, tokenizer, column_name, sample_size = 20, sim_threshold = 0.05):
    total_rows = len(serialized_rows)
    used_rows = 0
    #serialized_rows = set(serialized_rows) #using set of rows so that we can quickly sample without replacement
    sample1_list = column_name + random.sample(serialized_rows, min(sample_size, len(serialized_rows)))
    # Sample embeddings
    sample1_embeddings = encode_finetuned(sample1_list, model, tokenizer)
    sample1_average_embeddings = np.mean(sample1_embeddings, axis=0)
    serialized_rows = list(set(serialized_rows) - set(sample1_list))
    while(len(serialized_rows) > 0):
        sample2_list = column_name + random.sample(serialized_rows, min(sample_size, len(serialized_rows)))
        sample2_embeddings = encode_finetuned(sample2_list, model, tokenizer)
        sample2_average_embeddings = np.mean(sample2_embeddings, axis = 0)
        serialized_rows = list(set(serialized_rows) - set(sample2_list))
        cosine = CosineSimilarity(sample1_average_embeddings, sample2_average_embeddings)
        sample1_average_embeddings = (sample1_average_embeddings + sample2_average_embeddings) / 2
        # print("Length of average embedding:", len(sample1_average_embeddings))
        # print("Current cosine similarity:", cosine)
        if cosine >= (1 - sim_threshold):
            break
    used_rows = total_rows - len(serialized_rows)               
    # print("Total rows:", total_rows)
    # print("Used rows for serialization:", total_rows - len(serialized_rows))
    return sample1_average_embeddings, total_rows, used_rows

# Embedding model name:
model_name = 'Qwen/Qwen3-Embedding-8B'
max_length = 4096

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir="../../hf_cache")
model = AutoModel.from_pretrained(model_name, cache_dir="../../hf_cache")

method = "qwen"
embedding_method = "entities_and_column_name"

# input_folder_name = r"../alite-main/codes/Align Benchmark/"
# input_folder_name = r"../alite-main/codes/Real Benchmark/"
# input_folder_name = r"../alite-main/codes/align_utilities/SCHINT-Benchmark/"
input_folder_name = r"../alite-main/codes/align_utilities/SINT-Benchmark/"
# input_folder_name = r"../alite-main/codes/align_utilities/Real Benchmark/"

foldernames = os.listdir(input_folder_name)
# Create output folder if not exists
output_folder_path = method+"/"+input_folder_name.split("/")[-2]+"/"
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for foldername in tqdm.tqdm(foldernames, total=len(foldernames)):
    if ".csv" in foldername:
        continue

    if f"table_representations_{foldername}.json" in os.listdir(output_folder_path):
        # Load the file
        results_embeddings = sienna.load(f"{output_folder_path}table_representations_{foldername}.json")
        if embedding_method in results_embeddings[list(results_embeddings.keys())[0]]:
            continue
    else:
        results_embeddings = {}
    
    for tablename in os.listdir(input_folder_name+foldername):
        if ".csv" not in tablename:
            continue
        results_embeddings.setdefault(tablename, {}) # initialize the dictionary for the table if it doesn't exist
        # Load the column mappings
        column_mappings = sienna.load(f'{input_folder_name}{foldername}/{foldername}_mappings.json')
        try:
            df = pd.read_csv(input_folder_name+foldername+"/"+ tablename, encoding='latin1', on_bad_lines='skip')
        except Exception as e:
            print(input_folder_name+foldername+"/"+ tablename)
            print("table not found")
            print(e)
            break

        # Embed each column in the table and get the representative embedding for each column
        for column in df.columns:
            # column_data = list(set(df[column].dropna().astype(str).tolist()))
            column_data = list(set(df[column].map(str)))
            column_name = []
            if embedding_method == "entities_and_column_name":
                column_name = ["Column name: " + column]
            all_text = ' '.join(column_data)
            # .join(map(str.lower, original_list))
            all_text = re.sub(r'\([^)]*\)', '', all_text)
            column_data = list(set(re.sub(r"[^a-z0-9]+", " ", all_text.lower()).split()))
            if len(column_data) == 0:
                continue
            embeddings, total_rows, used_rows = EmbedTable(column_data, model, tokenizer, column_name)
            results_embeddings[tablename].setdefault(column_mappings[tablename][column], {}) # initialize the dictionary for the column if it doesn't exist
            results_embeddings[tablename][column_mappings[tablename][column]][embedding_method] = embeddings.tolist()
    # pdb.set_trace()
    sienna.save(results_embeddings, f"{output_folder_path}table_representations_{foldername}.json")
