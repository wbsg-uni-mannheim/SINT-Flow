
from unsloth import FastLanguageModel
import pandas as pd
import sienna
from model import llm_predict
from evaluation import SchemaIntegrationEvaluation
import os
from dotenv import dotenv_values
from langchain.chat_models import init_chat_model
import pdb
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import parse_model_response
from dataset_utils import load_ground_truth, table_serialization
import tqdm



if __name__ == "__main__":
    # Initialize LLM
    # model_name = "gpt-5.2-2025-12-11"
    reasoning = "None"
    model_name = "Qwen/Qwen3.6-27B"
    model_name_output = "Qwen" if "qwen" in model_name.lower() else "GPT-5.2"
    # model_type = "openai"
    model_type = "hf"
    
    if model_type == "hf":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16"
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../../hf_cache/")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", cache_dir="../../hf_cache/", quantization_config=quant_config)

        if "qwen" in model_name.lower():
            # Update chat format
            with open("qwen_chat_template.jinja", "r") as f:
                tokenizer.chat_template = f.read()
    else:
        config = dotenv_values("../key.env")
        os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
        OPENAI_API_KEY = config["OPENAI_API_KEY"]
        
        if reasoning != "None":
            model = init_chat_model(openai_api_key=OPENAI_API_KEY, temperature=1, model=model_name, reasoning = {"effort": reasoning, "summary": "auto"})
        else:
            model = init_chat_model(openai_api_key=OPENAI_API_KEY, temperature=1, model=model_name)

    # benchmark = "SINT-Benchmark"
    benchmark = "Real Benchmark"
    

    # SI-LLM prompts from SI-LLM repository: https://github.com/PierreWoL/SILLM/
    table_attribute_renaming = """The following is a column, its belonging table and its most specific entity type. \
Your task is to infer attribute name for the given column in the given table considering both column’s header, \
context and table’s most specific type. An entity type is typically defined as a word or a short phrase representing \
the overarching category or domain shared by the entities described in the table rows.   

An attribute name should be a concise and semantically meaningful label that reflects not only the type of data\
in the column but also its specific role in the table’s context. When multiple columns share similar value types,\
attribute names should distinguish their roles by leveraging contextual cues from the table context \
and its most specific entity type.

Provide attribute name for the given column.
 
Please ONLY output the attribute name for the given column, with no additional text.
"""

    attribute_merging = """You are given a list of attribute names inferred from table columns, which may be semantically redundant, paraphrased, or differently phrased.
Your task is to group those names into unified conceptual attribute groups, and assign a clear, canonical attribute name to each group. Please output python dictionary format, where: \
the key is the canonical attribute name of the group, the value is the list of original attribute names grouped under that attribute.
Be careful to distinguish different roles (e.g., supplier vs. consumer) even if they share similar semantics (like company names).

Provide ONLY output unified conceptual attribute names in the JSON format unified_conceptual_attribute: {list of attributes belonging to this group}, with no additional text. \
You must ensure that **every** attribute name from the input is included in the output!"""

    missing_attributes_refinement = """You are given a list of attribute names (i.e., unified conceptual groups),\
and a separate list of attribute names that were not yet assigned to any group.

Your task is to assign each missing attribute to one of the **existing canonical attribute groups**. \
If an attribute does not match any existing group, create a new canonical group for it.

Output in JSON format, where:
– Each key is a canonical attribute name (either chosen from the provided list or newly created),
– Each value is a list of one or more of the provided missing attribute names grouped under that key.

Only include groups for the given missing attributes — do not repeat any previously grouped attributes or their canonical names.

Provide ONLY A VALID JSON format as output, with no additional text.
"""
    
    table_most_specific_entity_types = [
        "Transaction",
        "Expense",
        "Transaction",
        "Client Information",
        "Organization Information",
        "Event",
        "Applicant",
        "Expense",
        "Job",
        "School Report",
        "Contract"
    ]
    
    for fi, folder_name in tqdm.tqdm(enumerate(os.listdir(f"../data/selected-tables/{benchmark}/")), total=len(os.listdir(f"../data/selected-tables/{benchmark}/"))):
        res = {}
        gt_file_names, gt_file, gt_mappings, gt_final_integrated_schema = load_ground_truth(folder = folder_name, benchmark = benchmark)
        if benchmark != "SINT-Benchmark":
            table_most_specific_entity_type = table_most_specific_entity_types[fi]
        
        for table_name in gt_file_names:
            res[table_name] = {}
            if benchmark == "SINT-Benchmark":
                if folder_name == 'airline-data':
                    table_most_specific_entity_type = "Flight" if "Flight" in gt_file["schema_by_entity"][table_name] else "Airport"
                elif folder_name == 'car-listings':
                    table_most_specific_entity_type = "Car"
                elif folder_name == 'games':
                    table_most_specific_entity_type = "Games"
                elif folder_name == 'park_events' or folder_name == 'goby_selected':
                    table_most_specific_entity_type = "Event"
                elif folder_name == 'wikidbs_horror-film-character':
                    table_most_specific_entity_type = "Movie Character"
                elif folder_name == 'wikidbs_monuments':
                    table_most_specific_entity_type = "Monument"
                elif folder_name == 'wikidbs_paintings-collection':
                    table_most_specific_entity_type = "Painting"
                elif folder_name == 'wikidbs_research-articles':
                    table_most_specific_entity_type = "Article"
                elif folder_name == 'volcanic-eruptions':
                    table_most_specific_entity_type = "Eruption" if "Eruption" in gt_file["schema_by_entity"][table_name] else "Volcano"
                else:
                    raise ValueError(f"Unknown folder name: {folder_name}")

            df = pd.read_csv(f"../data/selected-tables/{benchmark}/{folder_name}/{table_name}")
            for column in df.columns:
                messages_list = messages_list = [
                    {"role": "system", "content": table_attribute_renaming},
                    {"role": "user", "content": f"Header: {column}\nColumn Values: {df[column].tolist()[:5]}\nTable most specific entity type: {table_most_specific_entity_type}\n{table_serialization(df, nr_rows=5)}"},
                ]
                response = llm_predict(messages_list, tokenizer, model, max_seq_length=2048, temperature=0.001, model_name=model_name)
                res[table_name][column] = response

        messages_list = [
            {"role": "system", "content": attribute_merging},
            {"role": "user", "content": f"Attribute names: {[res[tab][col] for tab in res for col in res[tab]]}"},
        ]

        merging_response = llm_predict(messages_list, tokenizer, model, max_seq_length=2048, temperature=0.001, model_name=model_name)
        # Save as for each group: a dict with table name as key and list of columns
        attribute_groups = []
        for group in parse_model_response(merging_response):
            attributes_in_group = {}
            for attribute in parse_model_response(merging_response)[group]:
                for tab in res:
                    for col, val in res[tab].items():
                        if val == attribute:
                            if tab not in attributes_in_group:
                                attributes_in_group[tab] = []
                            if attribute not in attributes_in_group[tab]:
                                attributes_in_group[tab].append(col)
            attribute_groups.append(attributes_in_group)

        # Catch missing attributes that were not included in any group
        all_grouped_attributes = [attribute for group in parse_model_response(merging_response) for attribute in parse_model_response(merging_response)[group]]
        missing_attributes = [res[tab][col] for tab in res for col in res[tab] if res[tab][col] not in all_grouped_attributes]

        for col in missing_attributes:
            messages_list = [
                {"role": "system", "content": missing_attributes_refinement},
                {"role": "user", "content": f"Existing attribute groups: {parse_model_response(merging_response)}\nMissing attributes to be assigned: {missing_attributes}"},
            ]
            refinement_response = llm_predict(messages_list, tokenizer, model, max_seq_length=2048, temperature=0.001, model_name=model_name)
            
            # Update attribute groups with the refinement response
            for group in parse_model_response(refinement_response):
                if group not in parse_model_response(merging_response):
                    parse_model_response(merging_response)[group] = []
                for attribute in parse_model_response(refinement_response)[group]:
                    if attribute not in parse_model_response(merging_response)[group]:
                        parse_model_response(merging_response)[group].append(attribute)

        sienna.save({"column_renaming": res, "merging_response": parse_model_response(merging_response), "attribute_groups": attribute_groups}, f"si-llm_results_per_use_case/logs_{folder_name}_{benchmark}_{model_name_output}-SI-LLM.json")


    json_results = {}
    # Take the schema matching first runs and evaluate them as ALITE does
    # for file_name in os.listdir("evaluation_json/"):
    for file_name in os.listdir("si-llm_results_per_use_case/"):
        if benchmark in file_name:
            log_file = sienna.load(f"si-llm_results_per_use_case/{file_name}")
            use_case = file_name.split("logs_")[1].split(f"_{benchmark}_{model_name_output}-SI-LLM")[0]
            print(use_case)
            # load ground truth
            gt_mappings = sienna.load(f"../data/gt/{benchmark}/{use_case}/{use_case}_gt_mappings_with_values.json")

            # GT combinations
            true_combinations = set()
            for group in gt_mappings["grouped_attributes"].values():
                for tabattr in group:
                    for table, attribute in tabattr.items():
                        # Inner loop
                        for other_tabattr in group:
                            for other_table, other_attribute in other_tabattr.items():
                                true_combinations.add(tuple(sorted([f"{table}.{attribute}", f"{other_table}.{other_attribute}"])))

            # Predicted combinations
            combinations = []
            for group in log_file["attribute_groups"]:
                for table, attributes in group.items():
                    for attribute in attributes:
                        # Inner loop
                        for other_table, other_attributes in group.items():
                            for other_attribute in other_attributes:
                                if tuple(sorted([f"{table}.{attribute}", f"{other_table}.{other_attribute}"])) not in combinations:
                                    combinations.append(tuple(sorted([f"{table}.{attribute}", f"{other_table}.{other_attribute}"])))

            
            tps = 0
            fps = 0
            fns = 0
            for combination in combinations:
                if combination in true_combinations:
                    tps += 1
                else:
                    fps += 1
            for combination in true_combinations:
                if combination not in combinations:
                    fns += 1

            json_results[use_case] = {
                "true_combinations": len(true_combinations),
                "predicted_combinations": len(combinations),
                "true_positives": tps,
                "false_positives": fps,
                "false_negatives": fns,
                "precision": tps/(tps+fps) if (tps+fps) > 0 else 0,
                "recall": tps/(tps+fns) if (tps+fns) > 0 else 0,
                "f1_score": 2*tps/(2*tps + fps + fns) if (2*tps + fps + fns) > 0 else 0
            }
                # except Exception as e:
                #     print(f"Error processing {use_case}: {e}")
                    
    # Calculate average metrics across use cases
    f1s = [json_results[folder]["f1_score"] for folder in json_results]
    p = [json_results[folder]["precision"] for folder in json_results]
    r = [json_results[folder]["recall"] for folder in json_results]
    average_f1 = sum(f1s) / len(f1s) if f1s else 0
    average_precision = sum(p) / len(p) if p else 0
    average_recall = sum(r) / len(r) if r else 0

    json_results["average_results"] = {
        "average_f1": average_f1,
        "average_precision": average_precision,
        "average_recall": average_recall
    }
    sienna.save(json_results, f"overall_baseline_results/results_{benchmark}-{model_name_output}-si-llm.json")
