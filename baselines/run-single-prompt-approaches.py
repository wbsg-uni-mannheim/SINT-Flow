import sys
sys.append("..")
from model import llm_predict
from evaluation_utils import evaluate_instruction_prompt, evaluate_cot_prompt
from evaluation import SchemaIntegrationEvaluation
from utils import CACHE_DIR, parse_json
import os
from dotenv import dotenv_values
from langchain.chat_models import init_chat_model
# import warnings
# Ignore user warnings globally or specifically for bitsandbytes
# warnings.filterwarnings("ignore", module="bitsandbytes")
# warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataset_utils import table_serialization, load_ground_truth
import pandas as pd
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import sienna
import time
import tqdm


# Initialize LLM
# model_name = "gpt-5.2-2025-12-11"
model_name = "Qwen/Qwen3.6-27B"
reasoning = "None"
# model_type = "openai"
model_type = "hf" # For models loaded through HuggingFace
# prompt = "_cot"
prompt = "_instruction"


if model_type == "hf":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16"
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", cache_dir=CACHE_DIR, quantization_config=quant_config)
    if "qwen" in model_name.lower():
        # Update chat format to fix issues with tool calls
        with open("qwen_chat_template.jinja", "r") as f:
            tokenizer.chat_template = f.read()
else:
    config = dotenv_values("../key.env")
    os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
    OPENAI_API_KEY = config["OPENAI_API_KEY"]
    
    if reasoning != "None":
        model = init_chat_model(openai_api_key=OPENAI_API_KEY, temperature=1, model=model_name, reasoning = {"effort": reasoning, "summary": "auto"})
    else:
        model = init_chat_model(openai_api_key=OPENAI_API_KEY, temperature=0, model=model_name)
    tokenizer = None

if prompt == "_instruction":
    system_message = """You are a data engineer tasked to design an integrated schema that will be used to represent all data from a set of source tables. The integrated schema should fulfill the following requirements:\n Completeness: All attributes of the source tables should be covered\n Correctness: All data should be represented semantically correct\n Minimality: The integrated schema should be minimal in respect to the number of tables and attributes (redundancy-free)\n Understandability: The schema should be easy to understand.\n\n You split the schema integration process into three steps: 1. Table splitting: This step involves splitting each source table that describes multiple entity types (e.g. single table describing films and actors) into separate tables per entity type (e.g. one table for films and all their attributes and one table for actors and all their attributes. 2. Group (Cluster) the tables resulting from step 1 that describe the same entity type. 3. Merge the attributes which correspond together and output for each group/entity type the overall merged attributes. 4. Determine primary/foreign keys between table types (if there exist more than one table type) and add tables if necessary to model the relationship between table types.

Your input will be a set of tables. There are four phases to pass through to generate the final integrated schema. The phases are as follows:

1. Table Splitting Phase: For each input table, your task is to identify the entity types present in the table and split the table if needed based on the types found. This phase has the following instructions: For each table: 1. Identify the entity types described by the table. 2. Choose one main entity type that the table primarily describes. 3. Identify any other entity types that relate to the main entity. 4. For the main entity and each related entity, list the attributes (columns) that belong to it. 5. Split the table into multiple tables ONLY if there is low overlap between attribute groups. 6. Do not split the table if there are less than three attributes in any resulting table. The output of this phase should be written in the following JSON format: {\"overall_table_name\": {\"table_name\": {\"attributes\": [\"names\"]}}}, where overall_table_name is the name of the input table, table_name is the name of the entity type and attributes is a list of the attributes that belong to it. Always append a number that matches the number of the input table to the table names of the splits found.

2. Schema Matching Phase: Your task is to perform schema matching i.e. find matching columns between the tables created in the previous phase. Please find correspondences between the columns of all tables. Correspondences between columns mean that the column match semantically. For the columns to semantically match, they should also refer to the same semantic type. Not all columns must have correspondences. The output of this phase should be written in the following JSON format: {\"table_name\": {\"other_table_name\": {\"attribute_name\": \"other_attribute_name\"}}}, where table_name is the name of the table, other_table_name is the name of the other table that has a matching attribute, attribute_name is the name of the attribute in table_name and other_attribute_name is the name of the matching attribute in other_table_name. Null should not be a correspondence!

3. Table Grouping Phase:  Your task is to group (cluster) the tables that describe the same entity type, so that all tables that describe the same or semantically similar entity types are put into the same group. Name each group according to the entity type of the grouped tables. If a hierarchy can be observed between the found groups, combine the tables that are connected through this hierarchy and name the overall group after the highest entity type in the hierarchy. Please group (cluster) the set of tables by entity type, so that all tables that describe the same or similar entity types are put into the same group. The output of these phase should be the names of the groups together with the names of the tables in each group strictly using the following JSON format {group_name: [names of tables in the group]} with no additional commentary. Be brief in the naming of the groups and use general terms without parenthesis. Make sure that each table is part of exactly one group.

4. Attribute Clustering Phase: For each entity group found in the last phase, group the attributes in these entity groups based on the correspondences found in the schema matching phase to form attribute clusters. Your task is to assign to each attribute cluster a representative attribute from the group. Prefer choosing the attribute that can include all other attributes in the groups e.g. between year and date, date includes year so date should be chosen. The output of this phase should be written in the following JSON format: {\"entity_type\": {\"representative_attribute\": {\"table_name\":\"attribute\"}}}, where entity_type is the name of the entity type, representative_attribute is the name of the attribute that represents the cluster and its content is the attributes that belong to the cluster that are derived from the correspondences found in the previous phase.

5. Final Integrated Schema Generation Phase: The output of the previous phase contains the intermediate integrated schemata for each detected entity type. Examine the entity schemata. This phase has two tasks: create identifiers for the entity schemata IF needed (e.g. when more than one entity schema is given) and remove redundant attributes. At this phase, do NOT add any new entity schemata! Add identifiers for each entity schema that can be used as foreign keys. If there are already existing identifiers do not add new ones. If there is only one entity schema there is no need to add non-existing identifiers. Remove redundant attributes e.g. if the same attribute shows in two different entity schemas and it is not used as a foreign key, decide on which table it should be left in. Do not change the naming of the entities! Return the final schemata of all entities strictly as JSON in the format {entity_type: {attributes: [list of attributes], foreign_keys: {attribute_name: {other_entity_type: other_attribute_name}}}}.

Return the overall result of the four phases as a JSON object where the keys are the phase names and the values are the outputs of each phase. The keys that should be included are: Table Splitting Phase, Schema Matching Phase, Table Grouping Phase, Attribute Clustering Phase, Final Integrated Schema Generation Phase. Do not forget any key!
The output should be strictly in JSON format and should not contain any additional text or explanations. Be careful with the JSON format, it should be correct JSON!
"""

elif prompt == "_cot":
    system_message = """You are a data engineer tasked to design an integrated schema that will be used to represent all data from a set of source tables. The integrated schema should fulfill the following requirements:\n Completeness: All attributes of the source tables should be covered\n Correctness: All data should be represented semantically correct\n Minimality: The integrated schema should be minimal in respect to the number of tables and attributes (redundancy-free)\n Understandability: The schema should be easy to understand.\n\n You split the schema integration process into three steps: 1. Table splitting: This step involves splitting each source table that describes multiple entity types (e.g. single table describing films and actors) into separate tables per entity type (e.g. one table for films and all their attributes and one table for actors and all their attributes. 2. Group (Cluster) the tables resulting from step 1 that describe the same entity type. 3. Merge the attributes which correspond together and output for each group/entity type the overall merged attributes. 4. Determine primary/foreign keys between table types (if there exist more than one table type) and add tables if necessary to model the relationship between table types.

Your input will be a set of tables. There are four phases to pass through to generate the final integrated schema. The phases are as follows:

1. Table Splitting Phase: For each input table, your task is to identify the entity types present in the table and split the table if needed based on the types found. This phase has the following instructions: For each table: 1. Identify the entity types described by the table. 2. Choose one main entity type that the table primarily describes. 3. Identify any other entity types that relate to the main entity. 4. For the main entity and each related entity, list the attributes (columns) that belong to it. 5. Split the table into multiple tables ONLY if there is low overlap between attribute groups. 6. Do not split the table if there are less than three attributes in any resulting table. The output of this phase should be written in the following JSON format: {\"overall_table_name\": {\"table_name\": {\"attributes\": [\"names\"]}}}, where overall_table_name is the name of the input table, table_name is the name of the entity type and attributes is a list of the attributes that belong to it. Always append a number that matches the number of the input table to the table names of the splits found.

2. Schema Matching Phase: Your task is to perform schema matching i.e. find matching columns between the tables created in the previous phase. Please find correspondences between the columns of all tables. Correspondences between columns mean that the column match semantically. For the columns to semantically match, they should also refer to the same semantic type. Not all columns must have correspondences. The output of this phase should be written in the following JSON format: \"table_name\": {\"other_table_name\": {\"attribute_name\": \"other_attribute_name\"}}, where table_name is the name of the table, other_table_name is the name of the other table that has a matching attribute, attribute_name is the name of the attribute in table_name and other_attribute_name is the name of the matching attribute in other_table_name. Do not include null correspondences in the output! Keys and values cannot be null.

3. Table Grouping Phase:  Your task is to group (cluster) the tables that describe the same entity type, so that all tables that describe the same or semantically similar entity types are put into the same group. Name each group according to the entity type of the grouped tables. If a hierarchy can be observed between the found groups, combine the tables that are connected through this hierarchy and name the overall group after the highest entity type in the hierarchy. Please group (cluster) the set of tables by entity type, so that all tables that describe the same or similar entity types are put into the same group. The output of these phase should be the names of the groups together with the names of the tables in each group strictly using the following JSON format {group_name: [names of tables in the group]} with no additional commentary. Be brief in the naming of the groups and use general terms without parenthesis. Make sure that each table is part of exactly one group.

4. Attribute Clustering Phase: For each entity group found in the last phase, group the attributes in these entity groups based on the correspondences found in the schema matching phase to form attribute clusters. Your task is to assign to each attribute cluster a representative attribute from the group. Prefer choosing the attribute that can include all other attributes in the groups e.g. between year and date, date includes year so date should be chosen. The output of this phase should be written in the following JSON format: {\"entity_type\": {\"representative_attribute\": {\"table_name\":\"attribute\"}}}, where entity_type is the name of the entity type, representative_attribute is the name of the attribute that represents the cluster and its content is the attributes that belong to the cluster that are derived from the correspondences found in the previous phase.

5. Final Integrated Schema Generation Phase: The output of the previous phase contains the intermediate integrated schemata for each detected entity type. Examine the entity schemata. This phase has two tasks: create identifiers for the entity schemata IF needed (e.g. when more than one entity schema is given) and remove redundant attributes. At this phase, do NOT add any new entity schemata! Add identifiers for each entity schema that can be used as foreign keys. If there are already existing identifiers do not add new ones. If there is only one entity schema there is no need to add non-existing identifiers. Remove redundant attributes e.g. if the same attribute shows in two different entity schemas and it is not used as a foreign key, decide on which table it should be left in. Do not change the naming of the entities! Return the final schemata of all entities strictly as JSON in the format {entity_type: {attributes: [list of attributes], foreign_keys: {attribute_name: {other_entity_type: other_attribute_name}}}}.

6. Return with the key input_column_mapping the mapping of input columns to the final integrated schema attributes. The output should be in the following JSON format: {\"input_table_name\": {\"input_column_name\": {\"final_entity_name\": \"final_attribute_name\"}}}, where input_table_name is the name of the input table, input_column_name is the name of the column in the input table and final_attribute_name is the name of the attribute in the final integrated schema that corresponds to it.

7. Return with the key reasoning the reasoning behind the outputs with the structure: {\"phase_name\": \"reasoning_text\"}.

Think step-by-step and follow the instructions for each phase.
Return the overall result of the four phases as a JSON object where the keys are the phase names. Check that the JSON is valid.
"""

elif prompt == "_cot":
    system_message = """You are a data engineer tasked to design an integrated schema that will be used to represent all data from a set of source tables. The integrated schema should fulfill the following requirements:\n Completeness: All attributes of the source tables should be covered\n Correctness: All data should be represented semantically correct\n Minimality: The integrated schema should be minimal in respect to the number of tables and attributes (redundancy-free)\n Understandability: The schema should be easy to understand.\n\n

Your task is to detect the different entity types present in the given source tables, and design an integrated schema that represents all the data from the source tables for each entity type. 

Think step-by-step and plan your approach.
Return your answer in the following JSON format: {"integrated_schemas": {"entity_type": {"integrated_schema": [list of attributes]}}, "input_column_mappings": {"source_table_name": {"source_column_name": {"entity_type": "integrated_schema_attribute"}}}, "reasoning": "explanation of the reasoning process"}. Do not add any additinal text.
"""


folders = os.listdir("../data/selected-tables/SINT-Benchmark/")
for folder in tqdm.tqdm(folders, total=len(folders)):
    if os.path.exists(f"single_prompt_approaches_per_use_case/{'gpt-5.2' if model_type == 'openai' else 'qwen'}/{folder}_output{prompt}.json"):
        continue
    all_runs = {}
    for ri in tqdm.tqdm(range(3), total=3, leave=False):
        start_time = time.perf_counter()
        input_tables_string = "Input tables:\n"
        for i, file in enumerate(os.listdir(f"../data/selected-tables/SINT-Benchmark/{folder}")):
            table_df = pd.read_csv(f"../data/selected-tables/SINT-Benchmark/{folder}/{file}")
            input_tables_string += f"Table {i + 1}:\n{table_serialization(table_df)}\n\n"
        
        if model_type != "hf":
            messages_list = [
                SystemMessage(content=system_message),
                HumanMessage(content=input_tables_string)
            ]
        else:
            messages_list = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_tables_string}
            ]

        if model_type == "hf":
            response = llm_predict(messages_list, tokenizer, model, max_seq_length=20000, temperature=0.001, model_name=model_name)
        else:
            response = model.invoke(messages_list)

        # if len(parse_json(response)) == 0:
        #     print(f"Run {ri} for folder {folder} returned invalid JSON. Cancelling further runs for this folder.")
        #     print(f"Response: {parse_json(response).keys()}")
        #     break
        #     # Bad JSON output

        end_time = time.perf_counter()
        all_runs[ri] = {}
        if model_type != "hf":
             all_runs[ri]["response"] = parse_json(response.content)
        else:
            all_runs[ri]["response"] = parse_json(response)
        all_runs[ri]["response_unparsed"] = response
        all_runs[ri]["time"] = end_time - start_time
    
    if len(all_runs) > 0:
        sienna.save(all_runs, f"single_prompt_approaches_per_use_case/{'gpt-5.2' if model_type == 'openai' else 'qwen'}/{folder}_output{prompt}.json")


if prompt == "_instruction":
    # Process instruction prompt
    for folder in folders:
        runs = sienna.load(f"single_prompt_approaches_per_use_case/{'gpt-5.2' if model_type == 'openai' else 'qwen'}/{folder}_output{prompt}.json")
        processed_runs = evaluate_instruction_prompt(runs, folder, benchmark="SINT-Benchmark")
        # Evaluate the results of all runs
        sch_integration_eval = SchemaIntegrationEvaluation(f"data/selected-tables/SINT-Benchmark/", folder=folder)
        sch_integration_eval.other_parameters={}
        sch_integration_eval.init_manually(benchmark="SINT-Benchmark", folder=folder, sequence_of_phases=["detect_tables_phase", "schema_matching_phase", "grouping_phase", "schema_integration_phase", "final_integration_phase"], predictions=processed_runs, num_runs=3)
        sch_integration_eval.init_gt_properties()
        sch_integration_eval.run_evaluation()

        # Save the evaluation results
        eval_results_content = {
            "run_full_results": sch_integration_eval.eval_results,
            "average_metrics": sch_integration_eval.average_metrics,
            "std_dev_metrics": sch_integration_eval.std_dev_metrics,
        }
        
        sienna.save(processed_runs, f"single_prompt_approaches_per_use_case/{'gpt-5.2' if model_type == 'openai' else 'qwen'}/{folder}_processed{prompt}.json")
        sienna.save(eval_results_content, f"single_prompt_approaches_per_use_case/{'gpt-5.2' if model_type == 'openai' else 'qwen'}/{folder}_evaluation{prompt}.json")

if prompt == "_cot":
    # Process COT prompt results
    for folder in folders:
        if not os.path.exists(f"single_prompt_approaches_per_use_case/{'gpt-5.2' if model_type == 'openai' else 'qwen'}/{folder}_output{prompt}.json"):
            print(f"Output file for folder {folder} does not exist. Skipping.")
            continue
        if os.path.exists(f"single_prompt_approaches_per_use_case/{'gpt-5.2' if model_type == 'openai' else 'qwen'}/{folder}_evaluation{prompt}.json"):
            print(f"Evaluation file for folder {folder} already exists. Skipping.")
            continue
        runs = sienna.load(f"single_prompt_approaches_per_use_case/{'gpt-5.2' if model_type == 'openai' else 'qwen'}/{folder}_output{prompt}.json")
        all_eval_results = evaluate_cot_prompt(runs, folder, benchmark="SINT-Benchmark")
        sienna.save(all_eval_results, f"single_prompt_approaches_per_use_case/{'gpt-5.2' if model_type == 'openai' else 'qwen'}/{folder}_evaluation{prompt}.json")

# Read all evaluation results and average them across all folders
evaluation_results = {}
std_dev_evaluation_results = {}
for folder in folders:
    eval_results = sienna.load(f"single_prompt_approaches_per_use_case/{'gpt-5.2' if model_type == 'openai' else 'qwen'}/{folder}_evaluation{prompt}.json")
    for phase in eval_results["average_metrics"]:
        if phase not in evaluation_results:
            evaluation_results[phase] = {}
        for metric, value in eval_results["average_metrics"][phase].items():
            if metric not in evaluation_results[phase]:
                evaluation_results[phase][metric] = []
            evaluation_results[phase][metric].append(value)
            if phase not in std_dev_evaluation_results:
                std_dev_evaluation_results[phase] = {}
            if metric not in std_dev_evaluation_results[phase]:
                std_dev_evaluation_results[phase][metric] = []
            std_dev_evaluation_results[phase][metric].append(eval_results["std_dev_metrics"][phase][metric])


# Average the metrics across all folders and standard deviation
average_evaluation_results = {}
avg_std_dev_evaluation_results = {}

for phase, metrics in evaluation_results.items():
    average_evaluation_results[phase] = {}
    for metric, values in metrics.items():
        average_evaluation_results[phase][metric] = sum(values) / len(values)
        if phase not in avg_std_dev_evaluation_results:
            avg_std_dev_evaluation_results[phase] = {}
        if metric not in avg_std_dev_evaluation_results[phase]:
            avg_std_dev_evaluation_results[phase][metric] = 0
        avg_std_dev_evaluation_results[phase][metric] = sum(std_dev_evaluation_results[phase][metric]) / len(std_dev_evaluation_results[phase][metric])

sienna.save({"average_metrics": average_evaluation_results, "std_dev_metrics": std_dev_evaluation_results, "avg_std_dev_metrics": avg_std_dev_evaluation_results}, f"single_prompt_approaches_per_use_case/{'gpt-5.2' if model_type == 'openai' else 'qwen'}/overall_evaluation{prompt}.json")
