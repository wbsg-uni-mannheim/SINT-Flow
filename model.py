import random
import time
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from utils import langchain_message_to_dict, parse_json, write_txt_file, parse_model_response
from dataset_utils import table_serialization
from dataset_utils import load_ground_truth
import pandas as pd
import sienna
import copy
from langchain.tools import tool
from langchain.chat_models import BaseChatModel
from self_consistency import SelfConsistency
import pdb
import tqdm
import torch

@tool
def check_subset_schemata(schemata_1_name: str, schemata_1_attributes: list[str], schemata_2_name: str, schemata_2_attributes: list[str]) -> str:
    """
    Check if a table schemata is a subset of another table schemata.

    Args:
        schemata_1_name: The name of the first schemata to check.
        schemata_1_attributes: The attributes of the first schemata to check.
        schemata_2_name: The name of the second schemata to check.
        schemata_2_attributes: The attributes of the second schemata to check.
    """
    if set(schemata_1_attributes).issubset(set(schemata_2_attributes)):
        return f"Schemata ({schemata_1_name}) is a subset of schemata ({schemata_2_name})."
    else:
        return f"Schemata ({schemata_1_name}) is not a subset of schemata ({schemata_2_name})."
    
@tool
def check_number_of_attributes_in_table_split(table_split_name: str, table_split_attributes: list[str]) -> str:
    """
    Check if a table split has less than 3 attributes.

    Args:
        table_split_name: The name of the table split to check.
        table_split_attributes: The attributes of the table split to check.
    """
    if len(table_split_attributes)<3:
        return f"Table split ({table_split_name}) has less than 3 attributes."
    else:
        return f"Table split ({table_split_name}) has 3 or more attributes."

@tool
def check_if_ids_exist(table_schema_name: str, table_schema_attributes: list[str], id: bool) -> str:
    """
    Check if a table schema has an identifier in its attributes.

    Args:
        table_schema_name: The name of the table schema to check.
        table_schema_attributes: The attributes of the table schema to check.
        id: Whether an identifier exists or not.
    """
    if id:
        return f"Table schema ({table_schema_name}) has an identifier in its attributes."
    else:
        return f"Table schema ({table_schema_name}) does not have an identifier in its attributes."

@tool
def check_if_entities_are_added(new_table_schemas:dict, old_table_schemas:dict) -> str:
    """
    Check if new table schemata have more attributes than old table schemata.

    Args:
        new_table_schemas: The new table schemata to check.
        old_table_schemas: The old table schemata to check.
    """
    added_entities = set(new_table_schemas.keys()) - set(old_table_schemas.keys())
    
    return f"Some entities were added: {added_entities}. This should not happen."


FUNCTION_NAME_TO_FUNCTION = {
    "check_number_of_attributes_in_table_split": check_number_of_attributes_in_table_split,
    "check_subset_schemata": check_subset_schemata,
    "check_if_ids_exist": check_if_ids_exist,
    "check_if_entities_are_added": check_if_entities_are_added
}

def llm_predict(messages, tokenizer, model, max_seq_length, temperature, enable_thinking=False, model_name=None, tools=[]):
    if "qwen" in model_name.lower():
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tools=tools,
            return_tensors="pt",
            enable_thinking=enable_thinking
        ).to(model.device)
    else:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    
    if "llama" in model_name.lower() or "jellyfish" in model_name.lower():
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_seq_length,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    else:
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_seq_length,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7, 
            # temperature=temperature,
            top_p=0.80,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.0
        )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

class SchemaIntegrationWorkflow:
    def __init__(self, folder_name: str, table_path: str, model: BaseChatModel, prompt_messages: dict, sequence_of_phases: list[str], tools_available: list[str], table_format="markdown", table_rows=10, gt_part="schema_by_entity", model_type="openai", benchmark="SINT-Benchmark", model_name=None, tokenizer=None, self_consistency=False):
        self.folder_name = folder_name
        self.table_path = table_path
        self.model = model
        self.model_type = model_type
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.self_consistency = self_consistency
        self.sequence_of_phases = sequence_of_phases
        self.table_format = table_format
        self.table_rows = table_rows # Number of rows to show for the input tables
        self.results = {}
        self.prompt_messages = prompt_messages
        self.gt_part = gt_part
        # Load ground truth
        self.file_names, self.gt_file, self.gt_mappings, self.gt_final_integrated_schema = load_ground_truth(folder_name, benchmark=benchmark)
        self.file_names_to_index = {file_name: i+1 for i, file_name in enumerate(self.file_names)}
        self.index_to_file_names = {i+1: file_name for i, file_name in enumerate(self.file_names)}
        # Phase information
        self.table_detection_phase_name = "detect_tables_phase"
        self.table_detection_phase_name = self.table_detection_phase_name if self.table_detection_phase_name in sequence_of_phases else None

        if "grouping_phase" in sequence_of_phases:
            self.grouping_phase_name = "grouping_phase"
        else:
            self.grouping_phase_name = None
        # Record time of each workflow phase
        self.phase_times = {phase: None for phase in sequence_of_phases}
        if self.table_detection_phase_name:
            self.splitting_after_integration = self.sequence_of_phases.index(self.table_detection_phase_name) > self.sequence_of_phases.index("schema_integration_phase")
        else:
            self.splitting_after_integration = True
        # Mappings and important results
        self.results_parameters = {
            "detected_tables": {},
            "detected_tables_grouped_and_mapped": {},
            "column_correspondences": {},
            "removed_correspondences": {},
            "integrated_attributes": {},
            "integrated_schemas": {},
            "groups": {},
            "table_splits_to_group": {},
            "attribute_groups": {},
            "attribute_groups_with_no_removals": {},
            "primary_keys": {},
            "foreign_keys": {},
            "relationships": {}
        }
    
    def tables_to_string(self, group_of_tables):
        tables_str = ""
        for table_name, table_splits in group_of_tables.items():
            for table_split in table_splits:
                df = pd.read_csv(f"{self.table_path}/{table_name}")
                table = table_splits[table_split]
                tables_str += f"'{table['table_name']}':\n{table_serialization(df[table['attributes']], nr_rows=3)}\n\n"
        return tables_str

    def get_column_values(self, table_name, column_name):
        if "_table_" in table_name:
            table_split, table_index = table_name.split("_table_")
        elif "table_" in table_name:
            table_index = table_name.split("table_")[-1]
            table_split = table_name 
        df = pd.read_csv(f"{self.table_path}/{self.index_to_file_names[int(table_index)]}")
        if column_name is None:
            column_name = df.columns.to_list()
        return f"{table_name}:\n{table_serialization(df[column_name], nr_rows=5)}"
    

    def add_message(self, message_type, content):
        if self.model_type != "hf":
            if message_type == "system":
                return SystemMessage(content=content)
            elif message_type == "user":
                return HumanMessage(content=content)
            elif message_type == "assistant":
                return AIMessage(content=content)
        else:
            return {"role": message_type, "content": content}

    def invoke_model(self, messages_list):
        if self.model_type != "hf":
            response = self.model.invoke(messages_list)
            return response
        else:
            response_content = llm_predict(messages_list, self.tokenizer, self.model, max_seq_length=2048, temperature=0.001, model_name=self.model_name)
            return response_content

    # Format prompts
    def get_messages(self, phase, **kwargs):
        # Retrieve prompt messages for the specified phase
        prompt_messages = self.prompt_messages[phase]
        # Start message list with system message
        message_list = []
        message_list.append(self.add_message("system", prompt_messages["system"]))
        
        if phase == "detect_tables_phase" or phase == "detect_tables_phase_with_correspondences":          
            # Serialize input table
            df_serialized = table_serialization(kwargs["df"], nr_rows=self.table_rows, format=self.table_format)            
            # Add input table to messages
            message_list.append(self.add_message("user", prompt_messages["human"].replace("[TABLE_INPUT]", df_serialized).replace("[CORRESPONDENCES]\n\n\n", kwargs["correspondences"]).replace("[DEMONSTRATION]\n\n\n", "")))
            return message_list
        
        elif phase == "grouping_phase" or phase == "grouping_phase_with_correspondences":
            # Add input tables split
            message_list.append(self.add_message("user", prompt_messages["human"].replace("[TABLES]", kwargs["detected_tables"]).replace("[CORRESPONDENCES]\n\n", kwargs["correspondences"]).replace("[DEMONSTRATION]\n\n\n", "")))
            return message_list
        
        elif phase == "schema_matching_phase":
            # Add two tables to match
            message_list.append(self.add_message("user", prompt_messages["human"].replace("[TABLES]", kwargs["detected_tables"]).replace("[DEMONSTRATION]\n\n\n", "")))
            return message_list
    
        elif phase == "schema_integration_phase" or phase == "schema_integration_phase_with_correspondences":
            # Add tables to integrate
            message_list.append(self.add_message("user", prompt_messages["human"].replace("[TABLES]", kwargs["tables_to_integrate"]).replace("[ATTRIBUTE_GROUPS]", kwargs["attribute_groups"]).replace("[DEMONSTRATION]\n\n\n", "")))
            return message_list
        
        elif phase == "final_integration_phase" or phase == "final_integration_phase_with_tables":
            # Add integrated schemata
            message_list.append(self.add_message("user", prompt_messages["human"].replace("[SCHEMATA]", kwargs["schemata"]).replace("[DEMONSTRATION]\n\n\n", "").replace("[RELATIONSHIPS]", kwargs["rel_str"]).replace("[PRIMARY_KEYS]", kwargs["primary_keys"]).replace("[TABLE_INPUT]", kwargs["integrated_tables"])))
            return message_list
        
    def run_model_with_tools(self, messages_list: list, tool_list: list, tool_choice=None):
        full_responses = []
        if self.model_type == "hf":
            # Initialize model with tools
            response_content = llm_predict(messages_list, self.tokenizer, self.model, max_seq_length=2048, temperature=0.001, model_name=self.model_name, tools=tool_list)
            full_responses.append(langchain_message_to_dict(response_content))
            
            while "<tool_call>" in response_content:
                # Parse response to call the tool
                tool_calls = [parse_json(res.replace("<tool_call>","").strip()) for res in response_content.split("</tool_call>")]
                tool_calls = [tool_call for tool_call in tool_calls if tool_call]
                messages_list.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": tool_call} for tool_call in tool_calls]})
                # If the model has called a tool, execute the tool
                for tool_call in tool_calls:
                    tool_result = FUNCTION_NAME_TO_FUNCTION[tool_call["name"]](**tool_call["arguments"])
                    messages_list.append(self.add_message("tool", tool_result))
                # Pass the tool results to the model to get the final response
                response_content = llm_predict(messages_list, self.tokenizer, self.model, max_seq_length=2048, temperature=0.001, model_name=self.model_name, tools=tool_list)
                full_responses.append(langchain_message_to_dict(response_content))

            return full_responses, parse_model_response(response_content), messages_list
        
        else:
            # Initialize model with tools
            # model_with_tools = self.model.bind_tools(tool_list)
            model_with_tools = self.model.bind_tools(tool_list, tool_choice=tool_choice)
            # Initial model response
            response = model_with_tools.invoke(messages_list)
            full_responses.append(langchain_message_to_dict(response))

            while response.tool_calls:
                messages_list.append(response)
                # If the model has called a tool, execute the tool
                for tool_call in response.tool_calls:
                    tool_result = FUNCTION_NAME_TO_FUNCTION[tool_call["name"]].invoke(tool_call)
                    # Add tool result to messages
                    messages_list.append(tool_result)
                # Pass the tool results to the model to get the final response
                response = model_with_tools.invoke(messages_list)
                full_responses.append(langchain_message_to_dict(response))
        
            return full_responses, parse_model_response(response), messages_list

    def get_integrated_df(self, entity):
        if entity == "all":
            merged_schemas = list(set(list(self.results_parameters["integrated_attributes"][entity].values())))
        else:
            merged_schemas = self.results_parameters["integrated_schemas"][entity]
        
        final_grouped_tables = self.results_parameters["detected_tables_grouped_and_mapped"][entity]
        # For each attribute in the merged schema, record in which original tables it appears
        merged_schemas_to_tables = {attribute: [] for attribute in merged_schemas}
        # Create a dataframe using the generated integrated schema
        table_records = {attribute: [] for attribute in merged_schemas}
        for table_name, table_splits in final_grouped_tables.items():
            for table_split in table_splits.values():
                df = pd.read_csv(self.table_path+table_name)
                integrated_schema_to_columns = {v: k for k, v in table_split["column_mappings_to_integrated_schema"].items()}
                # Add for each attribute in the integrated schema 5 values from the tables
                for attribute in merged_schemas:
                    if attribute in integrated_schema_to_columns:
                        column_name = integrated_schema_to_columns[attribute]
                        for value in df[column_name].to_list()[:5]:
                            table_records[attribute].append(value)
                        merged_schemas_to_tables[attribute].append(table_name)
                    else:
                        for value in range(5):
                            table_records[attribute].append(None)
        self.results_parameters["merged_schemas_to_tables"] = merged_schemas_to_tables
        return table_records
    
    def split_correspondences(self):
        # Add schema matching results if schema matching phase has been run
        if "schema_matching_phase" in self.sequence_of_phases and self.sequence_of_phases.index("schema_matching_phase") < self.sequence_of_phases.index(self.table_detection_phase_name):
            print("Re-arranging correspondences according to table splits.")
            # Split correspondences according to table splits detected
            table_split_correspondences = {}
            for ti, (table_name, table_splits) in enumerate(self.results_parameters["detected_tables"].items()):
                for table_split in table_splits.values():
                    if f"table_{ti+1}" in self.results_parameters["column_correspondences"]:
                        for (target_tab, correspondences) in self.results_parameters["column_correspondences"][f"table_{ti+1}"].items():
                            original_table_idx = int(target_tab.split("_")[-1])-1
                            original_table_name = list(self.results_parameters["detected_tables"].keys())[original_table_idx]

                            for source_attr, target_attr in correspondences.items():
                                if source_attr in table_split["attributes"]:
                                    target_tab_split = None
                                    # Find target table split based on the name of the table split
                                    for other_splits in self.results_parameters["detected_tables"][original_table_name].values():
                                        if target_attr in other_splits["attributes"]:
                                            target_tab_split = other_splits["table_name"]
                                            break
                                    table_split_correspondences.setdefault(table_split["table_name"], {}).setdefault(target_tab_split, {})
                                    table_split_correspondences[table_split["table_name"]][target_tab_split][source_attr] = target_attr
            # Replace the correspondences in the schema matching phase with the table split correspondences
            self.results_parameters["original_column_correspondences"] = copy.deepcopy(self.results_parameters["column_correspondences"])
            self.results_parameters["column_correspondences"] = table_split_correspondences

            # Attribute groups should also be updated according to the detected tables
            for attribute_group_type in ["attribute_groups", "attribute_groups_with_no_removals"]:
                updated_attribute_groups = []
                for attribute_group in self.results_parameters[attribute_group_type]["all"]:
                    updated_attribute_group = {}
                    for table_name, column_names in attribute_group.items():
                        for column_name in column_names:
                            # print(table_name, column_name)
                            # table_name should be replaced with the name of the table split where the column_name appears
                            original_table_idx = int(table_name.split("_")[-1])-1
                            original_table_name = list(self.results_parameters["detected_tables"].keys())[original_table_idx]
                            # print("Original table name:", original_table_name)
                            for table_split in self.results_parameters["detected_tables"][original_table_name].values():
                                if column_name in table_split["attributes"]:
                                    # print("Found table split where the column appears:", table_split["table_name"])
                                    updated_attribute_group.setdefault(table_split["table_name"], []).append(column_name)
                                    # updated_attribute_group.append({table_split["table_name"]: column_name})
                    updated_attribute_groups.append(updated_attribute_group)
                self.results_parameters["original_" + attribute_group_type] = {}
                self.results_parameters["original_" + attribute_group_type]["all"] = copy.deepcopy(self.results_parameters[attribute_group_type]["all"])
                self.results_parameters[attribute_group_type]["all"] = updated_attribute_groups
 
    def has_connections(self, attribute, attribute_group, in_group_attribute, correspondences):
        # Check if the attribute has any matches within the found group
        # nr_connections = 0
        for other_table_name, other_column_names in attribute_group.items():
            for other_column_name in other_column_names:
                if {other_table_name: other_column_name} != in_group_attribute and {other_table_name: other_column_name} != attribute:
                    for table_name, column_name in attribute.items():
                        # If the correspondence exists
                        if other_table_name in correspondences and table_name in correspondences[other_table_name] and other_column_name in correspondences[other_table_name][table_name]:
                            if correspondences[other_table_name][table_name][other_column_name]==column_name:
                                # If this corresponde is not in the removed correspondences
                                if not (self.results_parameters["removed_correspondences"] != {} and other_table_name in self.results_parameters["removed_correspondences"] and table_name in self.results_parameters["removed_correspondences"][other_table_name] and other_column_name in self.results_parameters["removed_correspondences"][other_table_name][table_name] and self.results_parameters["removed_correspondences"][other_table_name][table_name][other_column_name] == column_name):
                                    return True
                                    # nr_connections += 1

                        # Check the other way around as well
                        if table_name in correspondences and other_table_name in correspondences[table_name] and column_name in correspondences[table_name][other_table_name]:
                            if correspondences[table_name][other_table_name][column_name]==other_column_name:
                                # If this correspondence is not in the removed correspondences
                                if not (self.results_parameters["removed_correspondences"] != {} and table_name in self.results_parameters["removed_correspondences"] and other_table_name in self.results_parameters["removed_correspondences"][table_name] and column_name in self.results_parameters["removed_correspondences"][table_name][other_table_name] and self.results_parameters["removed_correspondences"][table_name][other_table_name][column_name] == other_column_name):
                                    return True
                                    # nr_connections += 1
        return False
        # return nr_connections >= 2
    
    def detect_schema_matching_false_positives(self, **kwargs):
        messages_list = []
        messages_list.append(self.add_message("system", "You are an assistant for schema integration. The given groups are composed of attributes from different tables that were found to be matches during a schema matching phase. A link between the two groups has been found and your task is to decide based on the attributes and their column values if the link is correct or a false positive."))

        if "found_target_group" in kwargs and "found_source_group" in kwargs:
            messages_list.append(self.add_message("user", f"Here is group 1: {kwargs['found_source_group']}.\nHere are the table columns that are included in this group:\n\n"+"\n\n".join([self.get_column_values(table, column) for table, column in kwargs["found_source_group"].items()])))
            messages_list.append(self.add_message("user", f"Here is group 2: {kwargs['found_target_group']}.\nHere are the table columns that are included in this group:\n\n"+"\n\n".join([self.get_column_values(table, column) for table, column in kwargs["found_target_group"].items()])))
        
        elif "found_source_group" in kwargs:
            messages_list.append(self.add_message("user", f"Here is group 1: {kwargs['found_source_group']}.\nHere are the table columns that are included in this group:\n\n"+"\n\n".join([self.get_column_values(table, column) for table, column in kwargs["found_source_group"].items()])))
            messages_list.append(self.add_message("user", "Here is group 2: {"+f"{kwargs['target_table_name']}: {[kwargs['target_attribute']]}"+"}"+f".\nHere are the table columns that are included in this group:\n\n"+"\n\n".join([self.get_column_values(table, column) for table, column in {kwargs['target_table_name']: [kwargs['target_attribute']]}.items()])))
        
        elif "found_target_group" in kwargs:
            messages_list.append(self.add_message("user", f"Here is group 1: {kwargs['found_target_group']}.\nHere are the table columns that are included in this group:\n\n"+"\n\n".join([self.get_column_values(table, column) for table, column in kwargs["found_target_group"].items()])))
            messages_list.append(self.add_message("user", "Here is group 2: {"+f"{kwargs['table_name']}: {[kwargs['source_attribute']]}"+"}"+f".\nHere are the table columns that are included in this group:\n\n"+"\n\n".join([self.get_column_values(table, column) for table, column in {kwargs['table_name']: [kwargs['source_attribute']]}.items()])))
        
        elif "found_source_group" not in kwargs and "found_target_group" not in kwargs:
            # In this step show all table to have some more context
            messages_list.append(self.add_message("user", f"The column {kwargs['source_attribute']} from table {kwargs['table_name']} is in the first group:\n\n"+"\n\n".join([self.get_column_values(table, column) for table, column in {kwargs['table_name']: None}.items()])))
            messages_list.append(self.add_message("user", f"The column {kwargs['target_attribute']} from table {kwargs['target_table_name']} is in the second group:\n\n"+"\n\n".join([self.get_column_values(table, column) for table, column in {kwargs['target_table_name']: None}.items()])))

        # pdb.set_trace()
        messages_list.append(self.add_message("user", "Your task is schema matching, i.e. finding matches between columns of different tables. Based on the attributes in each group and their column values, do all the columns refer to the same semantic concept and therefore be merged into one column (do not focus on value formats or value overlaps rather only the semantic concept)? Please answer with only yes or no.")) # and an explanation and the semantic concepts of all columns.
        responses = []
        # Run 3 times for self-consistency or once if not
        num_runs = 3 if self.self_consistency else 1
        for _ in range(num_runs):
            response = self.invoke_model(messages_list)
            if self.model_type != "hf":
                response = response.content
            if "yes" in response.lower():
                responses.append(True)
            else:
                responses.append(False)
        # Majority voting
        if responses.count(True) > responses.count(False):
            print(f"Based on LLM review, the groups where the attributes {kwargs['table_name']}.{kwargs['source_attribute']} and {kwargs['target_table_name']}.{kwargs['target_attribute']} will be merged (true positive).")
            return True
        else:
            self.results_parameters["removed_correspondences"].setdefault(kwargs["table_name"], {}).setdefault(kwargs["target_table_name"], {})[kwargs["source_attribute"]] = kwargs["target_attribute"]
            print(f"Based on LLM review, the groups where the attributes {kwargs['table_name']}.{kwargs['source_attribute']} and {kwargs['target_table_name']}.{kwargs['target_attribute']} are found will not be merged (schema matching false positive).")
            return False
        
    def create_attribute_groups(self, detected_groups, grouped_tables, entity_group):
        if "grouping_phase" not in self.sequence_of_phases or self.sequence_of_phases.index("schema_matching_phase") < self.sequence_of_phases.index("grouping_phase") or entity_group=="all":#("schema_integration_phase" in self.results and self.results["schema_integration_phase"]["merged_schemas"]):
            overall_correspondences = self.results_parameters["column_correspondences"]
        else:
            # Select the correspondences from a specific entity group
            overall_correspondences = self.results_parameters["column_correspondences"][entity_group]

        # Cluster/group attributes based on correspondences
        # Create two types of groups: 1) the actual groups that get reviewed 2) the groups without review (will be used only for analysis)
        for attribute_group_type in ["attribute_groups", "attribute_groups_with_no_removals"]:
            reviewed_correspondences = {}
            nr_of_reviews = 0
            attribute_groups = []
            for table_name, table_correspondences in overall_correspondences.items():
                if table_name in detected_groups[entity_group]:
                    for target_table_name, attribute_correspondences in table_correspondences.items():
                        if target_table_name in detected_groups[entity_group]:
                            for source_attribute, target_attribute in attribute_correspondences.items():
                                skip_correspondence = False
                                # Check if this correspondence has been marked as a false positive, if yes, skip it
                                if self.results_parameters["removed_correspondences"] != {} and table_name in self.results_parameters["removed_correspondences"] and target_table_name in self.results_parameters["removed_correspondences"][table_name] and source_attribute in self.results_parameters["removed_correspondences"][table_name][target_table_name] and self.results_parameters["removed_correspondences"][table_name][target_table_name][source_attribute] == target_attribute:
                                    # Skip this correspondence as it has been marked as a false positive
                                    print(f"Skipping correspondence between {table_name}.{source_attribute} and {target_table_name}.{target_attribute} as it has been marked as a false positive based on LLM review.")
                                    skip_correspondence = True
                                # Check also the opposite direction, and skip if it has been marked as a false positive
                                elif self.results_parameters["removed_correspondences"] != {} and target_table_name in self.results_parameters["removed_correspondences"] and table_name in self.results_parameters["removed_correspondences"][target_table_name] and target_attribute in self.results_parameters["removed_correspondences"][target_table_name][table_name] and self.results_parameters["removed_correspondences"][target_table_name][table_name][target_attribute] == source_attribute:
                                    # Skip this correspondence as it has been marked as a false positive
                                    print(f"Skipping correspondence between {table_name}.{source_attribute} and {target_table_name}.{target_attribute} as it has been marked as a false positive based on LLM review.")
                                    skip_correspondence = True
                                
                                if skip_correspondence:
                                    continue
                                
                                # If source_attribute not added to any group yet
                                if not any(table_name in group and source_attribute in group[table_name] for group in attribute_groups):
                                    # Check if target_attribute is already in any group
                                    if any(target_table_name in group and target_attribute in group[target_table_name] for group in attribute_groups):
                                        # If target attribute is already in a group, add source attribute to that group
                                        found_group = None
                                        for group in attribute_groups:
                                            if target_table_name in group and target_attribute in group[target_table_name]:
                                                found_group = group
                                                break

                                        # If its for analysis add without checking
                                        if attribute_group_type == "attribute_groups_with_no_removals":
                                            found_group.setdefault(table_name, []).append(source_attribute)
                                            continue
                                        
                                        # Before adding: Check if this could be a false positive
                                        # First check: Does the source attribute match any of the other columns in the group
                                        connected = self.has_connections({table_name: source_attribute}, found_group, {target_table_name: target_attribute}, overall_correspondences)
                                        if connected:
                                            # If yes, this means that it has 2 connections to this group and it can be added to the group
                                            found_group.setdefault(table_name, []).append(source_attribute)
                                        else:
                                            # If it doesn't have any connections other than 1, ask an LLM if this attribute is okay to be added
                                            reviewed_correspondences.setdefault(table_name, {}).setdefault(target_table_name, {})[source_attribute] = target_attribute
                                            nr_of_reviews += 1
                                            correct_match = self.detect_schema_matching_false_positives(table_name=table_name, source_attribute=source_attribute, target_table_name=target_table_name, target_attribute=target_attribute, found_target_group=found_group)
                                            if correct_match: #and not skip_correspondence
                                                found_group.setdefault(table_name, []).append(source_attribute)
                                    else:
                                        # For analysis, add without checking
                                        if attribute_group_type == "attribute_groups_with_no_removals":
                                            attribute_groups.append({table_name: [source_attribute], target_table_name: [target_attribute]})
                                            continue
                                        # Both source and target attributes not assigned to any group yet, create new group
                                        # But first check that this is not a false positive by asking an LLM based on the attributes and their column values
                                        reviewed_correspondences.setdefault(table_name, {}).setdefault(target_table_name, {})[source_attribute] = target_attribute
                                        nr_of_reviews += 1
                                        correct_match = self.detect_schema_matching_false_positives(table_name=table_name, source_attribute=source_attribute, target_table_name=target_table_name, target_attribute=target_attribute)
                                        if correct_match: #and not skip_correspondence
                                            attribute_groups.append({table_name: [source_attribute], target_table_name: [target_attribute]})
                                else:
                                    if any(target_table_name in group and target_attribute in group[target_table_name] for group in attribute_groups):
                                        # Otherwise, if both source and target attributes are in different groups, check if the groups can be merged or if this correspondence is a FP
                                        # Find the groups they are in
                                        found_target_group = None
                                        for group in attribute_groups:
                                            # if target_table_name in group and group[target_table_name] == target_attribute:
                                            if target_table_name in group and target_attribute in group[target_table_name]:
                                                found_target_group = group
                                                break
                                        found_source_group = None
                                        for group in attribute_groups:
                                            # if table_name in group and group[table_name] == source_attribute:
                                            if table_name in group and source_attribute in group[table_name]:
                                                found_source_group = group
                                                break

                                        # If the groups are different:
                                        if not attribute_groups.index(found_source_group) == attribute_groups.index(found_target_group):
                                            # Merge groups without checking
                                            if attribute_group_type == "attribute_groups_with_no_removals":
                                                # Merge the two groups
                                                merged_group = copy.deepcopy(found_source_group)
                                                for inner_tab in found_target_group:
                                                    if inner_tab in merged_group:
                                                        merged_group[inner_tab].extend(found_target_group[inner_tab])
                                                    else:
                                                        merged_group[inner_tab] = found_target_group[inner_tab]
                                                # And remove the previous separate groups
                                                attribute_groups.remove(found_target_group)
                                                attribute_groups.remove(found_source_group)
                                                # And add the merged group
                                                attribute_groups.append(merged_group)
                                                continue
                                            # Ask an LLM if the connection is correct
                                            reviewed_correspondences.setdefault(table_name, {}).setdefault(target_table_name, {})[source_attribute] = target_attribute
                                            nr_of_reviews += 1
                                            correct_match = self.detect_schema_matching_false_positives(table_name=table_name, source_attribute=source_attribute, target_table_name=target_table_name, target_attribute=target_attribute, found_source_group=found_source_group, found_target_group=found_target_group)
                                            if correct_match: # and not skip_correspondence:
                                                # Merge groups
                                                # print("Merging groups based on LLM response.")
                                                # Merge the two groups
                                                merged_group = copy.deepcopy(found_source_group)
                                                for inner_tab in found_target_group:
                                                    if inner_tab in merged_group:
                                                        merged_group[inner_tab].extend(found_target_group[inner_tab])
                                                    else:
                                                        merged_group[inner_tab] = found_target_group[inner_tab]
                                                # And remove the previous separate groups
                                                attribute_groups.remove(found_target_group)
                                                attribute_groups.remove(found_source_group)
                                                # And add the merged group
                                                attribute_groups.append(merged_group)

                                    else:
                                        # Source attribute is already in a group, add target attribute to that group
                                        found_group = None
                                        for group in attribute_groups:
                                            if table_name in group and source_attribute in group[table_name]:
                                                found_group = group
                                                break

                                        # For analysis add directly without checking
                                        if attribute_group_type == "attribute_groups_with_no_removals":
                                            found_group.setdefault(target_table_name, []).append(target_attribute)
                                            continue
                                        
                                        # Check if the target attribute has any connections to the other attributes in the found group:
                                        connected = self.has_connections({target_table_name: target_attribute}, found_group, {table_name: source_attribute}, overall_correspondences)
                                        if connected:
                                            # If yes, it means that the target attribute has at least 2 connections to this group so it can be added to the group
                                            found_group.setdefault(target_table_name, []).append(target_attribute)
                                        else:
                                            # If not, ask an LLM if the correspondence is a FP
                                            reviewed_correspondences.setdefault(table_name, {}).setdefault(target_table_name, {})[source_attribute] = target_attribute
                                            nr_of_reviews += 1
                                            correct_match = self.detect_schema_matching_false_positives(table_name=table_name, source_attribute=source_attribute, target_table_name=target_table_name, target_attribute=target_attribute, found_source_group=found_group)
                                            if correct_match: # and not skip_correspondence:
                                                found_group.setdefault(target_table_name, []).append(target_attribute)

            # Add attributes that were not matched to any other attribute as their own group
            for table_name, table_splits in grouped_tables[entity_group].items():
                for table_split in table_splits.values():
                    if table_split["table_name"] in detected_groups[entity_group]:
                        for attribute in table_split["attributes"]:
                            if not any(table_split["table_name"] in group and attribute in group[table_split["table_name"]] for group in attribute_groups):
                                print(f"Adding unmatched attribute {table_split['table_name']}.{attribute} as its own group.")
                                attribute_groups.append({table_split["table_name"]: [attribute]})

            if entity_group != "all" and attribute_group_type == "attribute_groups":
                self.results["schema_matching_phase"]["reviewed_correspondences"][entity_group] = reviewed_correspondences
                self.results["schema_matching_phase"]["nr_of_reviews"][entity_group] = nr_of_reviews
            elif attribute_group_type == "attribute_groups":
                self.results["schema_matching_phase"]["reviewed_correspondences"] = reviewed_correspondences
                self.results["schema_matching_phase"]["nr_of_reviews"] = nr_of_reviews
            
            attribute_groups_str = "\n".join([f"Group_{gi+1}: "+", ".join(list(set([str(attribute) for attributes in group.values() for attribute in attributes]))) for gi, group in enumerate(attribute_groups)])
            if attribute_groups_str.strip() == "":
                # In the case that all attributes share the same naming or there is only one attribute per all groups
                self.results_parameters[attribute_group_type][entity_group] = [{grouped_tables[entity_group][table][split]["table_name"]: attribute} for table in grouped_tables[entity_group] for split in grouped_tables[entity_group][table] for attribute in grouped_tables[entity_group][table][split]["attributes"]]
            else:
                self.results_parameters[attribute_group_type][entity_group] = attribute_groups
    
    def update_mappings_after_table_detection(self):
        # Update the mappings according to the detected tables in the normalization phase    
        all_tables_split = {entity: {} for entity in self.results_parameters["detected_tables"]["full_table"]}
        for overall_table_name, tables in self.results_parameters["detected_tables_grouped_and_mapped"]["all"].items():
            # Update this table and split it into these detected splits according to the column mappings
            for table_name, table_info in tables.items():
                for attribute, mapped_attribute in table_info["column_mappings_to_integrated_schema"].items():
                    for table_split_name, table_split_info in self.results_parameters["detected_tables"]["full_table"].items():
                        if mapped_attribute in table_split_info["attributes"]:
                            all_tables_split[table_split_name].setdefault(overall_table_name, {}).setdefault(f"{table_split_name}_{table_name}", {"attributes": [], "column_mappings_to_integrated_schema": {}})
                            all_tables_split[table_split_name][overall_table_name][f"{table_split_name}_{table_name}"]["attributes"].append(attribute)
                            all_tables_split[table_split_name][overall_table_name][f"{table_split_name}_{table_name}"]["table_name"] = f"{table_split_name}_{table_name}"
                            all_tables_split[table_split_name][overall_table_name][f"{table_split_name}_{table_name}"]["column_mappings_to_integrated_schema"][attribute] = mapped_attribute

        self.results_parameters["original_detected_tables_grouped_and_mapped"] = copy.deepcopy(self.results_parameters["detected_tables_grouped_and_mapped"])
        self.results_parameters["detected_tables_grouped_and_mapped"] = all_tables_split
        self.results_parameters["integrated_schemas"] = {entity: list(set(self.results_parameters["detected_tables"]["full_table"][entity]["attributes"])) for entity in self.results_parameters["detected_tables"]["full_table"]}
    
    def run_detect_tables_phase(self, **kwargs):
        # Detect tables phase: Detect tables in each input table
        phase_name = "detect_tables_phase"
        self.results[phase_name] = {"full_model_response":{}, "messages_list": {}, "missing_loop": {}}
        df_dict = {table_name: pd.read_csv(f"{self.table_path}/{table_name}") for table_name in self.file_names} if not self.splitting_after_integration else {f"full_table": pd.DataFrame(self.get_integrated_df("all")).sample(frac=1)}

        start = time.perf_counter()
        for ti, (table_name, df) in enumerate(df_dict.items()):
            correct_response = False
            maximum_iterations = 3
            iteration = 0
            messages_list = self.get_messages(phase=phase_name, df=df, correspondences="", fds="")

            while not correct_response and iteration < maximum_iterations:
                # Run table detection
                tools = [check_number_of_attributes_in_table_split]
                full_responses, detected_tables, messages_list = self.run_model_with_tools(messages_list, tools)
                # Clear non-existing attributes from detected tables
                detected_tables = {table_split: {"attributes": [attr for attr in table_split_info["attributes"] if attr in df.columns]} for table_split, table_split_info in detected_tables.items()}
                # Check that all original attributes appear in at least one detected table
                correct_response = all(any(attr in table_split["attributes"] for table_split in detected_tables.values()) for attr in df.columns)
                if not correct_response:
                    # Ask the model to add the missing attribute(s) to a split
                    missing_attributes = [attr for attr in df.columns if not any(attr in table_split["attributes"] for table_split in detected_tables.values())]
                    messages_list.append(self.add_message("assistant", full_responses[-1]["content"]))
                    messages_list.append(self.add_message("user", f"The detected tables are missing the following attributes from the original table: {missing_attributes}. Please add these attributes to one of the detected tables. Here are the currently detected tables:\n{detected_tables}"))
                    missing_response = self.invoke_model(messages_list)
                    # Save corrected response
                    detected_tables = parse_model_response(missing_response)
                    full_responses.append(langchain_message_to_dict(missing_response))
                    # Check again if all original attributes appear in at least one detected table
                    correct_response = all(any(attr in table_split["attributes"] for table_split in detected_tables.values()) for attr in df.columns)
                # print(f"Correct response: {correct_response}\n")
                iteration += 1
            
            # Record response
            self.results[phase_name]["full_model_response"][table_name] = full_responses
            self.results[phase_name]["messages_list"][table_name] = langchain_message_to_dict(messages_list)
            # Save how many times the missing attributes loop was called
            self.results[phase_name]["missing_loop"][table_name] = iteration-1
            # Save the detected tables
            self.results_parameters["detected_tables"][table_name] = detected_tables
            # Record relationships betwen the detected table types
            if len(detected_tables) > 1:
                self.results_parameters["relationships"][table_name] = list(detected_tables.keys())
            
            # Log Results for debugging
            print(f"Table: {table_name}\n\n{table_serialization(df)}\n")
            print(f"Tables detected: {detected_tables}\n\n")

        if self.splitting_after_integration:
            # Check if one entity has only one original table
            merged_schemas_to_tables = self.results_parameters["merged_schemas_to_tables"]
            single_table_entities = []
            for entity, tables in detected_tables.items():
                all_tables_in_this_entity = set([table for attribute in merged_schemas_to_tables if attribute in tables["attributes"] for table in merged_schemas_to_tables[attribute]])
                if len(all_tables_in_this_entity) == 1:
                    single_table_entities.append(entity)
            
            # If there are single-table entities, ask the model to assign them to the other entities
            if len(single_table_entities) > 0:
                other_entities_str = "\n".join([f"{entity}: {tables['attributes']}" for entity, tables in detected_tables.items() if entity not in single_table_entities])
                entities_str = "\n".join([f"{entity}: {tables['attributes']}" for entity, tables in detected_tables.items() if entity in single_table_entities])
                messages_list = []
                messages_list.append(self.add_message("system", "You are an assistant for schema integration. You are given an integrated table, the different entity schemas detected from the table. Re-assign the given schema to one of the other table schemas and return the re-assignment witht the JSON format {\"table_name\": {\"attributes\": [\"names\"]}}."))
                messages_list.append(self.add_message("user", f"The integrated table is as follows:\n{table_serialization(df_dict['full_table'])}\n\nThe detected entity schemas are as follows:\n{other_entities_str}\n\nThe following entities have only one original table assigned to them: {entities_str}. Please re-assign these entities to one of the other entities and add the attributes to the assigned entity."))
                
                response = self.invoke_model(messages_list)
                reassignment = parse_model_response(response)
                self.results_parameters["detected_tables"]["full_table"] = merged_schemas_to_tables
                self.results_parameters["relationships"]["full_table"] = [entity for entity in self.results_parameters["relationships"]["full_table"] if entity in reassignment]

        end = time.perf_counter()
        # Record time taken for table detection
        self.phase_times[phase_name] = end - start

    def post_table_splitting_processing(self):
        # Add table names for each split detected to be traced back
        for table_name, tables in self.results_parameters["detected_tables"].items():
            for table_split in tables:
                table_index = 1 if table_name == "full_table" else self.file_names_to_index[table_name]
                self.results_parameters["detected_tables"][table_name][table_split]["table_name"] = f"{table_split}_table_{table_index}"
        
        # If the tables have been split after the merging of the attributes
        if self.splitting_after_integration:
            self.update_mappings_after_table_detection()
 
    def run_grouping_phase(self, **kwargs):
        grouping_phase_name = "grouping_phase"
        self.results[grouping_phase_name] = {"reassigned_tables": 0, "missed_tables_reassigned": 0}
        # Group tables by detected entity types
        all_types_detected = [table.lower().capitalize() for tables in self.results_parameters["detected_tables"].values() for table in tables]
        all_types_equal = len(set(all_types_detected))==1 and len(all_types_detected) == len(self.results_parameters["detected_tables"])

        start_time = time.perf_counter()

        # Re-arrange correspondences if they have been detected before splitting the tables
        self.split_correspondences()

        # Skip if only one table type is detected (i.e. all entity types are the same)
        if all_types_equal:
            groups = {table_split: [entities[ent]["table_name"] for entities in self.results_parameters["detected_tables"].values() for ent in entities] for table_split in set(all_types_detected)}
            self.results_parameters["detected_tables_grouped_and_mapped"] = {group: self.results_parameters["detected_tables"] for group in groups}
            self.results_parameters["groups"] = groups
            end_time = time.perf_counter()
            self.phase_times[grouping_phase_name] = end_time - start_time
            return
        
        # Prepare correspondences string if schema matching phase has been run
        corr_str = ""
        if "schema_matching_phase" in self.results and self.model_type != "hf":
            corr_str += "Correspondences between the detected tables:\n"
            for correspondence in self.results_parameters["column_correspondences"]:
                corr_str += f"{correspondence}: {self.results_parameters['column_correspondences'][correspondence]}\n"


        # Show first 3 rows of each detected table
        input_str = self.tables_to_string(group_of_tables=self.results_parameters["detected_tables"])

        print(input_str)

        # Run grouping phase
        messages_list = self.get_messages(phase=self.grouping_phase_name+"_with_correspondences", detected_tables=input_str, correspondences=corr_str)
        response = self.invoke_model(messages_list)
        groups = parse_model_response(response)

        # Remove tables that do not exist from the response
        for group, grouped_tables in groups.items():
            groups[group] = [table for table in grouped_tables if table in [previous_tables[previous_table]["table_name"] for previous_tables in self.results_parameters["detected_tables"].values() for previous_table in previous_tables]]
            if len(groups[group]) == 0:
                del groups[group]
        
        print(f"Groups found: {groups}\n\n")

        # If there is a group with only one table, this table should be assigned to one of the other groups as it is not a case for integration
        for group, grouped_tables in groups.items():
            if len(grouped_tables) == 1:
                table_to_reassign = grouped_tables[0]
                # Other groups that the table can be reassigned to: groups with more than 1 table
                other_groups = {g: t for g, t in groups.items() if g != group and len(t) > 1}
                reassignment_messages_list = copy.deepcopy(messages_list)
                reassignment_messages_list.append(self.add_message("assistant", response.content if self.model_type != "hf" else response))
                reassignment_messages_list.append(self.add_message("user", f"The group {group} contains only one table: {table_to_reassign}. Please reassign this table to one of the other groups. Here are the other groups: {other_groups}. Choose response from the other groups! Respond only with the new group name."))
                reassignment_response = self.invoke_model(reassignment_messages_list)
                reassignment_group = reassignment_response.content if self.model_type != "hf" else reassignment_response
                # Check that the reassignment group is valid
                if reassignment_group not in other_groups:
                    print("Error: The reassignment group returned by the model is not valid. Rerunning grouping phase.")
                    print(f"Groups found: {groups}\n\n")
                    self.run_grouping_phase()
                else:
                    self.results[self.grouping_phase_name]["reassigned_tables"] += 1
                    groups[reassignment_group].append(table_to_reassign)
        
        # Remove the groups that have only one table (after the reassignment)
        groups = {group: grouped_tables for group, grouped_tables in groups.items() if len(grouped_tables) > 1}
        print(f"New groups: {groups}\n\n")

        # Assign tables that are not assigned to any group:
        assigned_tables = [table for grouped_tables in groups.values() for table in grouped_tables]
        unassigned_tables = [previous_tables[previous_table]["table_name"] for previous_tables in self.results_parameters["detected_tables"].values() for previous_table in previous_tables if previous_tables[previous_table]["table_name"] not in assigned_tables]
        for unassigned_table in unassigned_tables:
            reassignment_messages_list = copy.deepcopy(messages_list)
            reassignment_messages_list.append(self.add_message("assistant", response.content if self.model_type != "hf" else response))
            reassignment_messages_list.append(self.add_message("user", f"The table {unassigned_table} is not assigned to any group. Please assign this table to one of the groups. Here are the groups: {groups}. Respond only with the group name."))
            reassignment_response = self.invoke_model(reassignment_messages_list)
            reassignment_group = reassignment_response.content if self.model_type != "hf" else reassignment_response
            # Check that the reassignment group is valid
            if reassignment_group not in groups:
                print("Error: The reassignment group returned by the model is not valid. Rerunning grouping phase.")
                print(f"Groups found: {groups}\n\n")
                self.run_grouping_phase()
            else:
                self.results[self.grouping_phase_name]["missed_tables_reassigned"] += 1
                groups[reassignment_group].append(unassigned_table)

        try:
            # Check that the groups are valid (i.e. that they contain only detected tables from the previous phase)
            assert all(grouped_table in [previous_tables[previous_table]["table_name"] for previous_tables in self.results_parameters["detected_tables"].values() for previous_table in previous_tables] for grouped_tables in groups.values() for grouped_table in grouped_tables)
            # Check that all tables from the previous phase are assigned to a group
            assert all(previous_tables[previous_table]["table_name"] in [grouped_table for grouped_tables in groups.values() for grouped_table in grouped_tables] for previous_tables in self.results_parameters["detected_tables"].values() for previous_table in previous_tables)
            # Check that one table is not assigned to multiple groups
            assert all(grouped_table not in [grouped_table2 for group2, grouped_tables2 in groups.items() if group2 != group for grouped_table2 in grouped_tables2] for group, grouped_tables in groups.items() for grouped_table in grouped_tables)
        except AssertionError:
            print("Error: The groups returned by the model are not valid. Rerunning grouping phase.")
            # print(f"Groups found: {groups}\n\n")
            self.run_grouping_phase()
        
        # Record results
        self.results[grouping_phase_name]["full_model_response"] = langchain_message_to_dict(response)
        self.results[grouping_phase_name]["messages_list"] = langchain_message_to_dict(messages_list)
        self.results_parameters["groups"] = groups
        end_time = time.perf_counter()
        self.phase_times[grouping_phase_name] = end_time - start_time

    def post_grouping_phase_processing(self):
        # Add table group info to detected tables
        grouped_detected_tables = {} # Group detected tables by their assigned group
        table_splits_to_group = {}
        
        for group, table_splits in self.results_parameters["groups"].items():
            grouped_detected_tables[group] = {}
            for table in table_splits:
                # Add group info
                table_split, table_index = table.split("_table_")
                table_name = self.index_to_file_names[int(table_index)]
                self.results_parameters["detected_tables"][table_name][table_split]["table_group"] = group
                # Group detected tables by their assigned group
                grouped_detected_tables[group].setdefault(table_name, {})
                grouped_detected_tables[group][table_name][table_split] = self.results_parameters["detected_tables"][table_name][table_split]
                # Table split to group mapping
                table_splits_to_group.setdefault(table_name, {})
                table_splits_to_group[table_name][table_split] = group
            # TODO: !!!
            # # If two splits from the same original table are assigned to the same group, merge them into one split with all the attributes of the two splits
            # for table_name, table_splits in grouped_detected_tables[group].items():
            #     if len(table_splits) > 1:
            #         merged_attributes = []
            #         for table_split in table_splits.values():
            #             merged_attributes.extend(table_split["attributes"])
            #         merged_table_split_name = f"{table_name}_merged"
            #         grouped_detected_tables[group][table_name][merged_table_split_name] = {"attributes": list(set(merged_attributes)), "table_name": merged_table_split_name, "column_mappings_to_integrated_schema": {}, "table_group": group}
            #         del grouped_detected_tables[group][table_name]
            #         # Update table split to group mapping
            #         for table_split in table_splits:
            #             table_splits_to_group[table_name][table_split] = group

        self.results_parameters["detected_tables_grouped_and_mapped"] = grouped_detected_tables
        self.results_parameters["table_splits_to_group"] = table_splits_to_group

        # Update relationships based on the groups
        if "relationships" in self.results[self.table_detection_phase_name]:
            relationships_updated = copy.deepcopy(self.results[self.table_detection_phase_name]["relationships"])
            for table_name, table_splits in self.results[self.table_detection_phase_name]["relationships"].items():
                for table_split in table_splits:
                    relationships_updated[table_name].remove(table_split)
                    relationships_updated[table_name].append(table_splits_to_group[table_name][table_split])
                if len(set(relationships_updated[table_name])) == 1:
                    del relationships_updated[table_name]
            self.results_parameters["relationships"] = relationships_updated

        # Update attribute groups if schema matching is in the sequence of phases and split them into the detected groups
        if "schema_matching_phase" in self.sequence_of_phases:
            for attribute_group_type in ["attribute_groups", "attribute_groups_with_no_removals"]:
                for entity_group in self.results_parameters["groups"]:
                    entity_attribute_groups = []
                    for attribute_group in self.results_parameters[attribute_group_type]["all"]:
                        entity_attribute_group = {}
                        for table in attribute_group:
                            if table in self.results_parameters["groups"][entity_group]:
                                entity_attribute_group[table] = attribute_group[table]
                        if len(entity_attribute_group) > 0:
                            entity_attribute_groups.append(entity_attribute_group)
                    
                    self.results_parameters[attribute_group_type][entity_group] = entity_attribute_groups

    def find_column_correspondences(self, group_of_tables, group=None, batch_nr=1):
        phase_name = "schema_matching_phase"
        # Group of tables: Choose group to run schema matching on, can be either all table splits or a specific group
        column_correspondences = {}
        responses = []
        messages_list = []
        # for table_name, table_splits in tqdm.tqdm(group_of_tables[:1].items(), total=len(group_of_tables[:1])):
        for table_name, table_splits in tqdm.tqdm(group_of_tables.items(), total=len(group_of_tables)):
            # For each table in the group, run schema matching against all other tables
            for table_split in table_splits.values():
                # Inner loop: batch the other tables
                batched_table_splits = [[other_table_name, other_table_split] for other_table_name, other_table_splits in group_of_tables.items() for other_table_split in other_table_splits.values() if other_table_split["table_name"] != table_split["table_name"]]
                batched_table_splits = [batched_table_splits[i:i + batch_nr] for i in range(0, len(batched_table_splits), batch_nr)]

                for batch in batched_table_splits:
                    # Delete from the batch the tables that have already been compared to this table split in the previous batches
                    cleaned_batch = [[other_table_name, other_table_split] for other_table_name, other_table_split in batch if not (other_table_split["table_name"] in column_correspondences and table_split["table_name"] in column_correspondences[other_table_split["table_name"]]) and other_table_name!=table_name]
                    cleaned_batch_names = [other_table_split["table_name"] for other_table_name, other_table_split in batch if not (other_table_split["table_name"] in column_correspondences and table_split["table_name"] in column_correspondences[other_table_split["table_name"]]) and other_table_name!=table_name]
                    
                    if len(cleaned_batch) == 0:
                        continue
                    # Prepare prompt
                    # Source table
                    df_1 = pd.read_csv(f"{self.table_path}/{table_name}")
                    table_1_attributes = table_split["attributes"]
                    table_1_str = table_serialization(df_1[table_1_attributes], nr_rows=5, format=self.table_format)
                    # Target tables
                    other_table_attributes = {}
                    other_table_str = ""
                    for other_table_name, other_table_split in cleaned_batch:
                        df_2 = pd.read_csv(f"{self.table_path}/{other_table_name}")
                        other_table_attributes[other_table_split["table_name"]] = other_table_split["attributes"]
                        other_table_str += f"{other_table_split['table_name']}:\n{table_serialization(df_2[other_table_attributes[other_table_split['table_name']]], nr_rows=5, format=self.table_format)}\n\n"

                    mess = self.get_messages(phase="schema_matching_phase", detected_tables=f"Source table:\n{table_1_str}\n\nTarget tables:\n{other_table_str}")
                    # LLM call
                    response = self.invoke_model(mess)
                    # Parse response
                    correspondences = parse_model_response(response)
                    # Check that 1. all target tables exist and 2. the attributes in the correspondences exist in the respective tables
                    valid_correspondences = {}
                    for target_table in correspondences:
                        if target_table in cleaned_batch_names:
                            for source_attr, target in correspondences[target_table].items():
                                if source_attr in table_1_attributes and target in other_table_attributes[target_table]:
                                    valid_correspondences.setdefault(target_table, {})[source_attr] = target

                    if valid_correspondences: # If there exists at least one valid correspondence between the tables, record it
                        for other_tab in valid_correspondences:
                            for source_attr, target in valid_correspondences[other_tab].items():
                                column_correspondences.setdefault(f"{table_split['table_name']}", {})
                                column_correspondences[f"{table_split['table_name']}"].setdefault(other_tab, {})[source_attr] = target

                    responses.append(langchain_message_to_dict(response))
                    messages_list.append(mess)
        
        if group:
            self.results[phase_name]["full_model_responses"][group] = responses
            self.results[phase_name]["messages_list"][group] = langchain_message_to_dict(messages_list)
            self.results_parameters["column_correspondences"][group] = column_correspondences
        else:
            self.results[phase_name]["full_model_responses"] = responses
            self.results[phase_name]["messages_list"] = langchain_message_to_dict(messages_list)
            self.results_parameters["column_correspondences"] = column_correspondences

    def run_schema_matching_phase(self, **kwargs):
        phase_name = "schema_matching_phase"
        self.results[phase_name] = {"full_model_responses": {}, "messages_list": {}}

        start_time = time.perf_counter()
        # Schema matching is running after grouping
        if "grouping_phase" in self.results:
            # Schema matching on each group separately
            for group, group_of_tables in self.results_parameters["detected_tables_grouped_and_mapped"].items():
                self.find_column_correspondences(group_of_tables=group_of_tables, group=group, batch_nr=kwargs["schema_matching_batch_size"])
        # Schema matching is run first on all tables
        elif self.sequence_of_phases.index("schema_matching_phase") == 0:
            group_of_tables = {table_name: {f"table_{file_index}": { "attributes": pd.read_csv(f"{self.table_path}{table_name}").columns.tolist(), "table_name": f"table_{file_index}" }} for table_name, file_index in self.file_names_to_index.items()}
            self.find_column_correspondences(group_of_tables=group_of_tables, batch_nr=kwargs["schema_matching_batch_size"])
        else:
            # Schema matching on the detected split tables
            self.find_column_correspondences(group_of_tables=self.results_parameters["detected_tables"], batch_nr=kwargs["schema_matching_batch_size"])
        end_time = time.perf_counter()
        self.phase_times[phase_name] = end_time - start_time

    def post_schema_matching_phase_processing(self):
        # Create the attribute groups after schema matching IF grouping phase is done, or grouping phase does not exist
        if "grouping_phase" in self.results:
            grouped_tables = self.results_parameters["detected_tables_grouped_and_mapped"]
            detected_groups = self.results_parameters["groups"]
        elif "grouping_phase" not in self.sequence_of_phases or ("schema_matching_phase" in self.sequence_of_phases and self.sequence_of_phases.index("schema_matching_phase") < self.sequence_of_phases.index(self.table_detection_phase_name)):
            # If no grouping is run beforehand:
            grouped_tables = {"all": {table_name: {f"table_{ti+1}": { "attributes": pd.read_csv(f"{self.table_path}{table_name}").columns.tolist(), "table_name": f"table_{ti+1}" }} for ti, table_name in enumerate(self.gt_file[self.gt_part])}}
            detected_groups = { "all": [f"table_{ti+1}" for ti, table in enumerate(self.file_names)] }
            self.results_parameters["detected_tables_grouped_and_mapped"] = grouped_tables
            self.results_parameters["groups"] = detected_groups
        else:
            # If grouping will be run after schema matching and the splitting of tables has been run
            grouped_tables = {"all": self.results_parameters["detected_tables"]}
            detected_groups = {"all": [table_info["table_name"] for table in self.results_parameters["detected_tables"] for table_info in self.results_parameters["detected_tables"][table].values()]}

        # Group matching attributes by correspondences
        for entity_group in detected_groups:
            self.create_attribute_groups(detected_groups, grouped_tables, entity_group)

    def run_schema_integration_phase(self, **kwargs):
        grouped_tables = self.results_parameters["detected_tables_grouped_and_mapped"]
        detected_groups = self.results_parameters["groups"]

        # Schema integration phase starts
        phase_name = "schema_integration_phase"
        self.results[phase_name] = {"full_model_response":{}, "messages_list": {}}
        start_time = time.perf_counter()

        if "schema_matching_phase" in self.sequence_of_phases:
            # Group matching attributes by correspondences
            for entity_group in detected_groups:
                attribute_groups = self.results_parameters["attribute_groups"][entity_group]
                # Prepare group and attribute strings
                group_to_index = {f"Group_{gi+1}": group for gi, group in enumerate(attribute_groups)}
                attribute_groups_str = "\n".join([f"Group_{gi+1}: "+", ".join(list(set([str(attribute) for attributes in group.values() for attribute in attributes]))) for gi, group in enumerate(attribute_groups)])

                print(attribute_groups)
                print(attribute_groups_str)

                if attribute_groups_str.strip() == "":
                    # In the case that all attributes share the same naming or there is only one attribute per all groups
                    print("Empty attribute groups! Only one table in this entity group or all tables have the same attributes.")
                    self.results[phase_name]["full_model_response"][entity_group] = {}
                    self.results_parameters["integrated_attributes"][entity_group] = {f"Group_{ai+1}": attribute for ai, attribute in enumerate(list(set([attribute for table in grouped_tables[entity_group] for split in grouped_tables[entity_group][table] for attribute in grouped_tables[entity_group][table][split]["attributes"]])))}
                else:
                    if kwargs["merging_batch_size"] == "all":
                        # Prepare group and attribute strings
                        group_str = self.tables_to_string(group_of_tables=grouped_tables[entity_group])
                        print(group_str)
                        # Retrieve integrated schema from model
                        messages_list = self.get_messages(phase=phase_name+"_with_correspondences", tables_to_integrate=group_str, attribute_groups=attribute_groups_str)
                        # LLM call
                        response = self.invoke_model(messages_list)
                        integration_response = parse_model_response(response)
                    else:
                        # One by one merging
                        integration_response = {}
                        for gi, group in enumerate(attribute_groups):
                            group_str = "\n\n".join([self.get_column_values(table, columns) for table, columns in group.items()])
                            unique_attributes = set([str(attribute) for attributes in group.values() for attribute in attributes])
                            attribute_group_str = f"\n\nGroup_{gi+1}: "+", ".join(list(unique_attributes))+"\n\n"
                            if len(unique_attributes) == 1:
                                # If there is only one table in the group, add all its attributes to the integrated schema without asking the model
                                response = None
                                integration_response[f"Group_{gi+1}"] = list(unique_attributes)[0]
                                continue
                            # Retrieve integrated schema from model
                            messages_list = self.get_messages(phase=phase_name+"_with_correspondences", tables_to_integrate=group_str, attribute_groups=attribute_group_str)
                            # LLM call
                            response = self.invoke_model(messages_list)
                            parsed_response = parse_model_response(response)
                            if f"Group_{gi+1}" in parsed_response:
                                integration_response[f"Group_{gi+1}"] = parsed_response[f"Group_{gi+1}"]
                            else:
                                # Error choose randomly
                                integration_response[f"Group_{gi+1}"] = random.choice(list(unique_attributes))
                    try:
                        # Check that all attribute groups are included in the output
                        assert all(group in integration_response for group in group_to_index)
                    except AssertionError as e:
                        print(f"Error: The number of attribute groups ({len(attribute_groups)}) does not match the number of attributes in the integrated schema ({len(integration_response)}). Rerunning schema integration phase.")
                        # pdb.set_trace()
                        self.run_schema_integration_phase(**kwargs)

                    if response:
                        self.results[phase_name]["full_model_response"][entity_group] = langchain_message_to_dict(response)
                    self.results[phase_name]["messages_list"][entity_group] = langchain_message_to_dict(messages_list)
                    self.results_parameters["integrated_attributes"][entity_group] = integration_response
                
        else:
            # Schema integration without explicit schema matching
            for entity_group in grouped_tables:
                group_str = self.tables_to_string(group_of_tables=grouped_tables[entity_group])
                print(group_str)
                messages_list = self.get_messages(phase=phase_name, tables_to_integrate=group_str, attribute_groups="")
                response = self.invoke_model(messages_list)
                integration_response = parse_model_response(response)
                
                # Check all attribute have been added to the mappings
                try:
                    assert("mappings_of_old_attributes" in integration_response)
                    assert all(attribute in integration_response["mappings_of_old_attributes"][grouped_tables[entity_group][table][split]["table_name"]] for table in grouped_tables[entity_group] for split in grouped_tables[entity_group][table] for attribute in grouped_tables[entity_group][table][split]["attributes"])
                except Exception as e:
                    print(e)
                    print(f"Error: Not all attributes have been mapped in the integrated schema for entity group {entity_group}. Rerunning phase.")
                    # pdb.set_trace()
                    self.run_schema_integration_phase(**kwargs)
                
                self.results[phase_name]["full_model_response"][entity_group] = langchain_message_to_dict(response)
                self.results[phase_name]["messages_list"][entity_group] = langchain_message_to_dict(messages_list)
                self.results_parameters["integrated_attributes"][entity_group] = integration_response
      
        end_time = time.perf_counter()
        self.phase_times[phase_name] = end_time - start_time

    def post_schema_integration_phase_processing(self):
        if "schema_matching_phase" in self.sequence_of_phases:
            # for attribute_group_type in ["attribute_groups", "attribute_groups_with_no_removals"]:
            for entity_group in self.results_parameters["detected_tables_grouped_and_mapped"]:
                group_to_index = {f"Group_{gi+1}": group for gi, group in enumerate(self.results_parameters["attribute_groups"][entity_group])}
                attribute_groups_str = "\n".join([f"Group_{gi+1}: "+", ".join(list(set([str(attribute) for attributes in group.values() for attribute in attributes]))) for gi, group in enumerate(self.results_parameters["attribute_groups"][entity_group])])

                if attribute_groups_str.strip() == "":
                    # Add column mappings of the attributes: since only one table, all attributes are mapped to themselves
                    for table in self.results_parameters["detected_tables_grouped_and_mapped"][entity_group]: 
                        for split in self.results_parameters["detected_tables_grouped_and_mapped"][entity_group][table]: 
                            for attribute in self.results_parameters["detected_tables_grouped_and_mapped"][entity_group][table][split]["attributes"]:
                                self.results_parameters["detected_tables_grouped_and_mapped"][entity_group][table][split].setdefault("column_mappings_to_integrated_schema", {})
                                self.results_parameters["detected_tables_grouped_and_mapped"][entity_group][table][split]["column_mappings_to_integrated_schema"][attribute] = attribute
                else:
                    # Add to each table split the mapping of the original attributes to the attributes in the integrated schema
                    for group_name, representative_attribute in self.results_parameters["integrated_attributes"][entity_group].items():
                        for table_split_name, table_attributes in group_to_index[group_name].items():
                            # Find the original table name
                            if "_table_" in table_split_name:
                                table_split, table_index = table_split_name.split("_table_")
                            elif "table_" in table_split_name:
                                table_index = table_split_name.split("table_")[-1]
                                table_split = table_split_name
                            
                            table_name = self.index_to_file_names[int(table_index)]
                            for table_attribute in table_attributes:
                                # If there are no mappings for this table split yet, initialize empty dict
                                self.results_parameters["detected_tables_grouped_and_mapped"][entity_group][table_name][table_split].setdefault("column_mappings_to_integrated_schema", {})
                                # Add the mapping from the original attribute to the representative attribute in the integrated schema
                                self.results_parameters["detected_tables_grouped_and_mapped"][entity_group][table_name][table_split]["column_mappings_to_integrated_schema"][table_attribute] = representative_attribute
        
                # Save the integrated schemas for each entity
                self.results_parameters["integrated_schemas"][entity_group] = list(set(self.results_parameters["integrated_attributes"][entity_group].values()))

            # If there are more than 1 table schemas, check based on correspondences if there are inter-schema attributes that are linked to a different integrated attribute and normalize them
            if not self.splitting_after_integration and len(self.results_parameters["integrated_schemas"])>1: # and "schema_matching_phase" in self.sequence_of_phases:
                self.prepare_final_integration_input()
        
        else:
            # Add these mappings to the detected_tables_grouped_and_mapped result of the workflow to use it in the next phase
            for entity in self.results_parameters["detected_tables_grouped_and_mapped"]:
                for table_name in self.results_parameters["detected_tables_grouped_and_mapped"][entity]:
                    for table_split in self.results_parameters["detected_tables_grouped_and_mapped"][entity][table_name].values():
                        # Add the mappings to integrated schema for each attribute in the table split
                        # Delete non-existing attributes
                        relevant_mappings = self.results_parameters["integrated_attributes"][entity][table_split["table_name"]]
                        table_split["column_mappings_to_integrated_schema"] = {attr: mapping for attr, mapping in relevant_mappings.items() if attr in table_split["attributes"]}
            
            # Add the integrated schema to the results parameters for each entity group
            integrated_schemas = {}
            for entity in self.results_parameters["integrated_attributes"]:
                integrated_schemas[entity] = []
                for table_mappings in self.results_parameters["integrated_attributes"][entity].values():
                    for integrated_attribute in table_mappings.values():
                        if integrated_attribute not in integrated_schemas[entity]:
                            integrated_schemas[entity].append(integrated_attribute)
            self.results_parameters["integrated_schemas"] = integrated_schemas

        # Create the integrated tables
        self.results_parameters["integrated_tables"] = {entity: self.get_integrated_df(entity) for entity in self.results_parameters["integrated_schemas"]}

    def prepare_final_integration_input(self):
        # Check across groups if there are corresponding attributes in different final entity groups that have different surface forms
        attribute_groups = self.results_parameters["attribute_groups"]["all"]
        final_grouped_tables = self.results_parameters["detected_tables_grouped_and_mapped"]
        table_split_to_table_split = {table_splits[table_split]["table_name"]: table_splits[table_split] for tables in final_grouped_tables.values() for table_splits in tables.values() for table_split in table_splits}
        
        # Clean the attribute groups to have only the remaining relevant attributes for the integration = only the attributes that show up in multiple schemas
        cleaned_attribute_groups = []
        cleaned_original_attribute_groups = []
        for group in attribute_groups:
            cleaned_attributes = {}
            cleaned_original_attributes = {}
            for tabsplit, attributes in group.items():
                for attribute in attributes:
                    if any(table_split_to_table_split[tabsplit]["column_mappings_to_integrated_schema"][attribute] in schemas for schemas in self.results_parameters["integrated_schemas"].values()):
                        cleaned_attributes.setdefault(tabsplit, []).append(table_split_to_table_split[tabsplit]["column_mappings_to_integrated_schema"][attribute])
                        cleaned_original_attributes.setdefault(tabsplit, []).append(attribute)     
            if len(cleaned_attributes) > 0:
                cleaned_attribute_groups.append(cleaned_attributes)
                cleaned_original_attribute_groups.append(cleaned_original_attributes)

        already_unique_attributes = set()
        non_unique_groups = []
        for i, grouped_attributes in enumerate(cleaned_attribute_groups):
            unique_attributes = set([attr for attrs in grouped_attributes.values() for attr in attrs])
            if len(unique_attributes) > 1:
                non_unique_groups.append(i)
            else:
                already_unique_attributes.update(unique_attributes)

        for group_index in non_unique_groups:
            unique_attributes = set([attr for attrs in cleaned_attribute_groups[group_index].values() for attr in attrs])
            cleaned_unique_attributes = unique_attributes - already_unique_attributes
            if len(cleaned_unique_attributes) > 1:
                # Ask LLM to choose between the remaining unique attributes based on the values in the tables
                mess = [
                    self.add_message("system", "You are an assistant for schema integration. The columns given to you have been matched previously, but their final naming differs. They should have the same naming. Choose between them the naming that best represents the values of all columns."),
                    self.add_message("user", "Here are the namings grouped: "+", ".join(cleaned_unique_attributes)+"\n\n Here are the columns grouped under them: "+"\n\n".join([self.get_column_values(table, column) for table, column in cleaned_original_attribute_groups[group_index].items()])+"\n\nChoose one of the namings and answer only with it."),
                ]
                res = self.invoke_model(mess)
                chosen_attribute = res
                if self.model_type != "hf":
                    chosen_attribute = res.content
            elif len(cleaned_unique_attributes) == 1:
                chosen_attribute = cleaned_unique_attributes.pop()
            else:
                # A new name should be picked to avoid confusion in the schema
                mess = [
                    self.add_message("system", "You are an assistant for schema integration. The columns given to you have been matched previously, but their final naming differs. They should have the same naming. Choose between them the naming that best represents the values of all columns."),
                    self.add_message("user", "Here are the namings of the other attributes: "+", ".join(already_unique_attributes)+"\n\n Here are the columns grouped under them: "+"\n\n".join([self.get_column_values(table, column) for table, column in cleaned_original_attribute_groups[group_index].items()])+"\n\nChoose a naming for these columns that is not in the attributes given to you. Reply only with the chosen naming."),
                ]
                res = self.invoke_model(mess)
                chosen_attribute = res
                if self.model_type != "hf":
                    chosen_attribute = res.content
                print(f"New attribute name chosen! {chosen_attribute}")

            # Update the mappings to the chosen attribute
            for tabsplit, attrs in cleaned_attribute_groups[group_index].items():
                for attr in table_split_to_table_split[tabsplit]["column_mappings_to_integrated_schema"]:
                    if table_split_to_table_split[tabsplit]["column_mappings_to_integrated_schema"][attr] in attrs:
                        table_split_to_table_split[tabsplit]["column_mappings_to_integrated_schema"][attr] = chosen_attribute
            # Add this as unique attribute for the next iterations
            already_unique_attributes.add(chosen_attribute)

        # Update the integrated schema with the newly selected attributes:
        updated_integrated_schema = {}
        for entity, tables in final_grouped_tables.items():
            updated_integrated_schema[entity] = []
            for table_name, table_splits in tables.items():
                for table_split in table_splits.values():
                    # Add attributes to integrated schema
                    for attribute in table_split["column_mappings_to_integrated_schema"].values():
                        if attribute not in updated_integrated_schema[entity]:
                            updated_integrated_schema[entity].append(attribute)
        self.results_parameters["integrated_schemas"] = updated_integrated_schema

    
    def run_final_integration_phase(self, phase_iteration=0, **kwargs):
        phase_name = "final_integration_phase"
        integrated_schemas = self.results_parameters["integrated_schemas"]
        integrated_table_records = self.results_parameters["integrated_tables"]
        integrated_tables = {entity: pd.DataFrame(table_rows) for entity, table_rows in integrated_table_records.items()}
    
        # Final phase in schema integration starts
        start_time = time.perf_counter()

        if len(integrated_schemas)==1:
            # If there is only one table schema, it can be considered directly as the final schema
            final_integrated_schema = {}
            for entity in integrated_schemas:
                final_integrated_schema[entity] = {"attributes": integrated_schemas[entity], "foreign_keys": {}}
            
            self.results[phase_name] = {
                "full_model_response": {},
                "final_integrated_schema": final_integrated_schema,
            }
            self.results_parameters["final_integrated_schema"] = final_integrated_schema
            end_time = time.perf_counter()
            self.phase_times[phase_name] = end_time - start_time
            return

        # Prepare string input with the integrated schemas from the previous phase
        integrated_schemas_str = "\n".join([f"{group}: "+", ".join(integrated_schemas[group]) for group in integrated_schemas])
        integrated_tables_str = "\n\n".join([f"{group}:\n\n"+table_serialization(integrated_tables[group], nr_rows=20) for group in integrated_tables])
        print(integrated_tables_str)
        
        # Generate final integrated schema
        relationships = self.results_parameters["relationships"]
        relationships_str = "\n".join(list(set([" and ".join(rels)+ " schemas are linked." for rels in list(relationships.values())])))

        messages_list = self.get_messages(phase=phase_name+"_with_tables", schemata=integrated_schemas_str, rel_str=relationships_str, primary_keys="", integrated_tables=integrated_tables_str)
        tools = [check_subset_schemata, check_if_ids_exist, check_if_entities_are_added]
        full_responses, final_integrated_schema, messages_list = self.run_model_with_tools(messages_list, tools)
        # full_responses = self.invoke_model(messages_list)
        # final_integrated_schema = parse_model_response(full_responses)
        correct_response = all(any(attribute in final_integrated_schema[entity]["attributes"] for entity in final_integrated_schema) for group in integrated_schemas for attribute in integrated_schemas[group])
        iteration = 0

        # Remove added entities:
        if not all(entity in integrated_schemas for entity in final_integrated_schema):
            print("Removing automatically added entities")
            final_integrated_schema = {entity: final_integrated_schema[entity] for entity in final_integrated_schema if entity in integrated_schemas}

        while not correct_response:
            # Attributes can be moved around/deleted from certain entities but not removed entirely
            removed_attributes = [attribute for group in integrated_schemas for attribute in integrated_schemas[group] if not any(attribute in final_integrated_schema[entity]["attributes"] for entity in final_integrated_schema)]
            adding_attributes_messages_list = copy.deepcopy(messages_list)
            adding_attributes_messages_list.append(self.add_message("assistant", f"The following attributes were removed in the final integrated schema, but they should be included at least in one of the schemas since attributes should not be removed entirely. Here are the removed attributes: {', '.join(removed_attributes)}. Please add them to the final integrated schema in the correct table and respond with the updated schema."))
            full_updated_responses = self.invoke_model(adding_attributes_messages_list)
            updated_final_integrated_schema = parse_model_response(full_updated_responses)
            correct_response = all(any(attribute in updated_final_integrated_schema[entity]["attributes"] for entity in updated_final_integrated_schema) for group in integrated_schemas for attribute in integrated_schemas[group])
            iteration += 1
            # final_integrated_schema = updated_final_integrated_schema
            if iteration >= 5:  # Prevent infinite loop
                break
        if iteration>0:
            final_integrated_schema = updated_final_integrated_schema

        correct_response = all(any(attribute in final_integrated_schema[entity]["attributes"] for entity in final_integrated_schema) for group in integrated_schemas for attribute in integrated_schemas[group]) and all(entity in integrated_schemas for entity in final_integrated_schema)
        iteration = 0

        # If an entity is added:
        while not correct_response:
            # Ask LLM to delete added entity
            added_entities = [entity for entity in final_integrated_schema if entity not in integrated_schemas]
            remove_added_entities_messages_list = copy.deepcopy(messages_list)
            remove_added_entities_messages_list.append(self.add_message("assistant", f"The following entities were added in the final integrated schema, but they should not be included since they were not present in any of the integrated schemas: {', '.join(added_entities)}. Please remove them from the final integrated schema and respond with the updated schema. Don't forget to keep all the attributes from the integrated schemas! The added entities' attributes should be returned to their correct entity."))
            full_updated_responses = self.invoke_model(remove_added_entities_messages_list)
            updated_final_integrated_schema = parse_model_response(full_updated_responses)
            correct_response = all(any(attribute in updated_final_integrated_schema[entity]["attributes"] for entity in updated_final_integrated_schema) for group in integrated_schemas for attribute in integrated_schemas[group]) and all(entity in integrated_schemas for entity in updated_final_integrated_schema)
            iteration += 1
            # final_integrated_schema = updated_final_integrated_schema
            if iteration >= 5:  # Prevent infinite loop
                break
        if iteration>0:
            final_integrated_schema = updated_final_integrated_schema
                
        try:
            # Check that there are no extra entities added, entities can be missing but not added
            assert all(entity in integrated_schemas for entity in final_integrated_schema)
            # Check that none of the previous attributes have been removed, they can be moved around the entities but not removed entirely
            assert all(any(attribute in final_integrated_schema[entity]["attributes"] for entity in final_integrated_schema) for group in integrated_schemas for attribute in integrated_schemas[group])
        except AssertionError:
            print("The final integrated schema contains entities not present in the integrated schemas or some attributes were removed. Re-running phase.")
            # pdb.set_trace()
            if phase_iteration < 5 and "schema_matching_phase" in self.sequence_of_phases:  # Prevent infinite loop
                self.run_final_integration_phase(phase_iteration=phase_iteration+1)

        end_time = time.perf_counter()
        self.phase_times[phase_name] = end_time - start_time

        # Record response
        self.results[phase_name] = {
            "full_model_response": full_responses,
            "messages_list": langchain_message_to_dict(messages_list),
            "final_integrated_schema": final_integrated_schema,
            "missed_attributes_loop": iteration
        }
        self.results_parameters["final_integrated_schema"] = final_integrated_schema

    def post_final_integration_phase_processing(self, **kwargs):
        # Record attributes that were added that do not belong in any of the original tables: these should be the identitfiers
        new_added_attributes_per_entity = {entity: [attribute for attribute in self.results_parameters["final_integrated_schema"][entity]["attributes"] if attribute not in self.results_parameters["integrated_schemas"][entity] and not any(attribute in self.results_parameters["integrated_schemas"][other_entity] for other_entity in self.results_parameters["integrated_schemas"] if other_entity != entity)] for entity in self.results_parameters["final_integrated_schema"]}
        # Update the column mappings of every table based on the final schema
        final_integrated_schema_mappings = copy.deepcopy(self.results_parameters["detected_tables_grouped_and_mapped"])
        removed_attributes_per_table = {}
        
        for entity_name, entity_schema in self.results_parameters["final_integrated_schema"].items():
            # Check the additions and deletions in this group compared to the previous phase, table by table
            for table_name, table_mappings in self.results_parameters["detected_tables_grouped_and_mapped"][entity_name].items():
                for table_split, table_info in table_mappings.items():
                    # Previous schema of this table for this entity
                    previous_phase_schema = set(list(table_info["column_mappings_to_integrated_schema"].values()))
                    # Final schema for this entity
                    final_phase_schema = entity_schema["attributes"]
                    # Compare them and find which attributes were removed
                    removed_attributes = previous_phase_schema - set(final_phase_schema)
                    added_attributes = set(final_phase_schema) - previous_phase_schema

                    df = pd.read_csv(f"{self.table_path}{table_name}")
                    for column_name, mapped_attribute in table_info["column_mappings_to_integrated_schema"].items():
                        # If the mapped attribute is not in the final schema for this entity it should be removed
                        if mapped_attribute in removed_attributes:
                            # Remove this attribute from both the attributes list and the mappings
                            del final_integrated_schema_mappings[entity_name][table_name][table_split]["column_mappings_to_integrated_schema"][column_name]
                            final_integrated_schema_mappings[entity_name][table_name][table_split]["attributes"].remove(column_name)
                            found_in_other_entity = False
                            # Are these removed attributes present in other entities? 
                            for other_entity in self.results_parameters["detected_tables_grouped_and_mapped"]:
                                if other_entity != entity_name and table_name in self.results_parameters["detected_tables_grouped_and_mapped"][other_entity]:
                                    if any(column_name in other_table_info["column_mappings_to_integrated_schema"] and other_table_info["column_mappings_to_integrated_schema"][column_name] in removed_attributes for other_table_info in self.results_parameters["detected_tables_grouped_and_mapped"][other_entity][table_name].values()):
                                        for other_table_info in self.results_parameters["detected_tables_grouped_and_mapped"][other_entity][table_name].values():
                                            if column_name in other_table_info["column_mappings_to_integrated_schema"].values() and other_table_info["column_mappings_to_integrated_schema"][column_name] in removed_attributes:
                                                found_in_other_entity = True
                            # If not, record them to the correct entity
                            if not found_in_other_entity:
                                removed_attributes_per_table.setdefault(table_name, {}).setdefault(table_split, {}).setdefault("mapping", {})
                                removed_attributes_per_table[table_name][table_split]["mapping"][column_name] = mapped_attribute
                                removed_attributes_per_table[table_name][table_split]["table_name"] = table_info["table_name"]
                    
                    for column_name in df.columns:
                        for other_entity in self.results_parameters["detected_tables_grouped_and_mapped"]:
                            if other_entity != entity_name and table_name in self.results_parameters["detected_tables_grouped_and_mapped"][other_entity]:
                                if any(column_name in other_table_info["column_mappings_to_integrated_schema"] and other_table_info["column_mappings_to_integrated_schema"][column_name] in added_attributes for other_table_info in self.results_parameters["detected_tables_grouped_and_mapped"][other_entity][table_name].values()):
                                    for other_table_info in self.results_parameters["detected_tables_grouped_and_mapped"][other_entity][table_name].values():
                                        if column_name in other_table_info["column_mappings_to_integrated_schema"] and other_table_info["column_mappings_to_integrated_schema"][column_name] in added_attributes:
                                            found_mapping = {column_name: other_table_info["column_mappings_to_integrated_schema"][column_name]}
                                    # Add this attribute to the current entity's mappings and attributes
                                    if found_mapping[column_name] not in final_integrated_schema_mappings[entity_name][table_name][table_split]["column_mappings_to_integrated_schema"].values():
                                        final_integrated_schema_mappings[entity_name][table_name][table_split]["column_mappings_to_integrated_schema"][column_name] = found_mapping[column_name]
                                        final_integrated_schema_mappings[entity_name][table_name][table_split]["attributes"].append(column_name)
                    # Add the new attributes (identifiers) to this table split
                    final_integrated_schema_mappings[entity_name][table_name][table_split]["new_attributes"] = list(new_added_attributes_per_entity[entity_name])
        
        # Add the removed attributes
        for entity_name, entity_schema in self.results_parameters["final_integrated_schema"].items():
            for table_name, table_splits in removed_attributes_per_table.items():
                for table_split, table_info in table_splits.items():
                    for column_name, mapped_attribute in table_info["mapping"].items():            
                        if mapped_attribute in entity_schema["attributes"]:
                            print(f"Attribute '{column_name}' mapped to '{mapped_attribute}' is now added to entity '{entity_name}'.")
                            final_integrated_schema_mappings[entity_name].setdefault(table_name, {})
                            # If the split does not exist in this entity:
                            if table_split not in final_integrated_schema_mappings[entity_name][table_name]:
                                final_integrated_schema_mappings[entity_name][table_name][table_split] = {
                                    "table_name": f"({entity_name})" + table_info["table_name"],
                                    "column_mappings_to_integrated_schema": {},
                                    "attributes": [],
                                    "table_group": entity_name
                                }
                            
                            final_integrated_schema_mappings[entity_name][table_name][table_split]["column_mappings_to_integrated_schema"][column_name] = mapped_attribute
                            final_integrated_schema_mappings[entity_name][table_name][table_split]["attributes"].append(column_name)
                            # Add the new attributes to this table split
                            final_integrated_schema_mappings[entity_name][table_name][table_split]["new_attributes"] = list(new_added_attributes_per_entity[entity_name])
        # Save the final mappings and the new attributes
        self.results_parameters["final_integrated_schema_mappings"] = final_integrated_schema_mappings
        self.results_parameters["new_added_attributes_per_entity"] = new_added_attributes_per_entity

class SchemaIntegrationRuns:
    def __init__(self, folder_name: str, sequence_of_phases: list[str], table_path: str, model: BaseChatModel, reasoning: str, prompt_name: str, self_consistency: bool, table_format="markdown", table_rows=10, num_runs=1, fds=False, model_type="openai", model_name=None, tokenizer=None, schema_matching_batch_size=1, merging_batch_size="all", benchmark="SINT-Benchmark"):
        self.folder_name = folder_name
        self.table_path = table_path
        self.benchmark = benchmark
        self.model = model
        self.prompt_messages = sienna.load("prompts/prompts.json")[prompt_name] if model_type != "hf" else sienna.load("prompts/prompts_hf.json")[prompt_name]
        self.table_format = table_format
        self.table_rows = table_rows # Number of rows to show for the input tables
        self.sequence_of_phases = sequence_of_phases
        self.schema_matching_batch_size = schema_matching_batch_size
        self.merging_batch_size = merging_batch_size
        self.gt_part = "schema_by_entity" #if "detect_tables_phase" in self.sequence_of_phases else "schema_by_normalization"
        self.reasoning = reasoning if reasoning else "None"
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.num_runs = num_runs
        self.fds = fds
        self.self_consistency = self_consistency
        self.results = {}
        self.initialize_logfile()

    def get_function(self, phase_name, run_nr):
        workflow = self.workflows[run_nr]
        if phase_name == "detect_tables_phase":
            return {"function": workflow.run_detect_tables_phase, "postprocessing_function": workflow.post_table_splitting_processing, "variable": "detected_tables"}
        elif phase_name == "schema_matching_phase":
            return {"function": workflow.run_schema_matching_phase, "postprocessing_function": workflow.post_schema_matching_phase_processing, "variable": "column_correspondences"}
        elif phase_name == "grouping_phase":
            return {"function": workflow.run_grouping_phase, "postprocessing_function": workflow.post_grouping_phase_processing, "variable": "groups"}
        elif phase_name == "schema_integration_phase":
            return {"function": workflow.run_schema_integration_phase, "postprocessing_function": workflow.post_schema_integration_phase_processing, "variable": "integrated_attributes"}
        elif phase_name == "final_integration_phase":
            return {"function": workflow.run_final_integration_phase, "postprocessing_function": workflow.post_final_integration_phase_processing, "variable": "final_integrated_schema"}
        else:
            raise ValueError(f"Phase '{phase_name}' does not exist in the pipeline.")

    def initialize_workflow(self):
        # For each overall number of runs: initialize a separate workflow
        self.workflows = {}
        for run in range(self.num_runs):
            self.workflows[run] = SchemaIntegrationWorkflow(folder_name=self.folder_name, table_path=self.table_path, model=self.model, prompt_messages=self.prompt_messages, table_format=self.table_format, table_rows=self.table_rows, gt_part=self.gt_part, sequence_of_phases=self.sequence_of_phases, tools_available=self.tools_available, model_type=self.model_type, model_name=self.model_name, tokenizer=self.tokenizer, benchmark=self.benchmark, self_consistency=self.self_consistency)

        # Initialize self-consistency
        if self.self_consistency:
            self.self_consistency_run = SelfConsistency(num_runs=self.num_runs, file_names=self.workflows[0].file_names, benchmark=self.benchmark, folder_name=self.folder_name)
            self.self_consistency_function_mapping = {
                "detect_tables_phase": self.self_consistency_run.run_table_splitting_self_consistency,
                "schema_matching_phase": self.self_consistency_run.run_schema_matching_self_consistency,
                "grouping_phase": self.self_consistency_run.run_grouping_self_consistency,
                "schema_integration_phase": self.self_consistency_run.run_schema_integration_self_consistency,
                "final_integration_phase": self.self_consistency_run.run_final_integration_self_consistency,
            }

    def initialize_logfile(self):
        file_prefix = f"llm-workflow-{self.benchmark}-{self.folder_name}-"
        timestr = time.strftime("%Y%m%d-%H%M%S")
        logfile_name = f"logs/{file_prefix}{self.model_name.split('/')[-1]}-{self.prompt_name}-{timestr}"
        print(f"Log file name: {logfile_name}")
        
        self.tools_available = ["check_number_of_attributes_in_table_split", "check_subset_schemata"]
        self.record_run_info = {
            "benchmark": self.benchmark,
            "folder_name": self.folder_name,
            "model_name": self.model_name,
            "reasoning": self.reasoning,
            "prompt_name": self.prompt_name,
            "sequence_of_phases": self.sequence_of_phases,
            "prompts":{
                "prompt_name": self.prompt_name,
                "prompts": self.prompt_messages
            },
            "demonstration": 0,
            "schema_matching_batch_size": self.schema_matching_batch_size,
            "merging_batch_size": self.merging_batch_size,
            "logfile_name": logfile_name,
            "num_runs": self.num_runs,
            "other_parameters": {
                "tools_available": self.tools_available,
                "self-consistency": self.self_consistency,
            }
        }
        self.log_file_name = logfile_name

    def run_workflow(self):
        # Initialize workflow
        self.initialize_workflow()
        
        for phase_name in self.sequence_of_phases:
            for run in range(self.num_runs):
                print(f"Running phase, run {run+1}: {phase_name}")
                phase_function = self.get_function(phase_name=phase_name, run_nr=run)["function"]
                # Call function
                phase_function(run_feedback_loop=True, fds=self.fds, schema_matching_batch_size=self.schema_matching_batch_size, merging_batch_size=self.merging_batch_size)

            if self.self_consistency:
                print(f"Running self-consistency for phase: {phase_name}")
                # At the end of each phase run self-consistency for this phase and get the result to go in the next phase
                self_consistency_input = {run: self.workflows[run].results_parameters[self.get_function(phase_name=phase_name, run_nr=run)["variable"]] for run in range(self.num_runs)}
                self.self_consistency_run.results[phase_name] = self_consistency_input
                # Call function
                self.self_consistency_function_mapping[phase_name](phase_name=phase_name, schema_integration_results=self.workflows[0].results_parameters["integrated_schemas"], integrated_attributes=self.workflows[0].results_parameters["integrated_attributes"], detected_tables_grouped_and_mapped=self.workflows[0].results_parameters["detected_tables_grouped_and_mapped"])
                self_consistency_results = self.self_consistency_run.self_consistency_results[phase_name]
                # Update the results of the workflow with the self-consistency results for this phase to use it in the next phase and run postprocessing to update the workflows                
                for run in range(self.num_runs):
                    # Replace with new result
                    phase_variable = self.get_function(phase_name=phase_name, run_nr=run)["variable"]
                    self.workflows[run].results_parameters[phase_variable] = self_consistency_results
                    if run == 0:
                        # Run the postprocessing function only once
                        phase_postprocessing_function = self.get_function(phase_name=phase_name, run_nr=0)["postprocessing_function"]
                        if phase_postprocessing_function:
                            phase_postprocessing_function()
                    else:
                        # For the other runs, copy the output of the first postprocessing function
                        if phase_postprocessing_function:
                            self.workflows[run].results_parameters = copy.deepcopy(self.workflows[0].results_parameters)
            else:
                # For regular (non-self-constency) runs, the post-processing function is run separately for each run
                for run in range(self.num_runs):
                    phase_postprocessing_function = self.get_function(phase_name=phase_name, run_nr=run)["postprocessing_function"]
                    if phase_postprocessing_function:
                        phase_postprocessing_function()
        
        # Record all run results
        for run in range(self.num_runs):
            self.workflows[run].results["parameters"] = self.workflows[run].results_parameters
            self.workflows[run].results["time_info"] = self.workflows[run].phase_times
            self.results[run] = self.workflows[run].results
        # Save results of run
        self.save_results()

    def save_results(self):
        log_file_content = {
            "run_info": self.record_run_info,
            "runs": self.results,
            "time_info": {run: sum(self.results[run]["time_info"].values()) for run in self.results}
        }
        if self.self_consistency:
            log_file_content["self_consistency_results"] = self.self_consistency_run.self_consistency_results
        sienna.save(log_file_content, f"{self.log_file_name}.json")