import numpy as np
# import pdb
from dataset_utils import load_ground_truth
import os
import copy
import pandas as pd

def calculate_f1_p_r(report, labels):
    for label in report:
        if report[label]['TP'] == 0:
            precision = 0
            recall = 0
        else:
            precision = report[label]['TP'] / (report[label]['TP'] + report[label]['FP'])
            recall = report[label]['TP'] / (report[label]['TP'] + report[label]['FN'])

        if np.isnan(precision) or precision == 0:
            f1 = 0
        elif np.isnan(recall) or precision == 0:
            f1 = 0
        else:
            f1 = 2*precision*recall / (precision + recall)

        report[label]['p'] =  precision
        report[label]['r'] =  recall
        report[label]['f1'] = f1
    
    all_fn = 0
    all_tp = 0
    all_fp = 0

    for r in report:
        # if r != num_classes-1:
        all_fn += report[r]['FN']
        all_tp += report[r]['TP']
        all_fp += report[r]['FP']
        
    class_f1s = [ report[class_]['f1'] for class_ in report]
    class_p = [ 0 if np.isnan(report[class_]['p']) else report[class_]['p'] for class_ in report]
    class_r = [ 0 if np.isnan(report[class_]['r']) else report[class_]['r'] for class_ in report]
    # macro_f1 = sum(class_f1s[:-1]) / (num_classes-1)
    macro_f1 = sum(class_f1s) / len(labels)
    
    p =  sum(class_p) / len(labels)
    r =  sum(class_r) / len(labels)
    micro_f1 = all_tp / ( all_tp + (1/2 * (all_fp + all_fn) )) 
    
    per_class_eval = {}
    errors_per_class = {}
    for index, t in enumerate(labels):
        per_class_eval[t] = {"Precision":class_p[index], "Recall": class_r[index], "F1": class_f1s[index]}
        errors_per_class[t] = report[t]["FN"] + report[t]["FP"]
    
    return {
        "per_class_eval": per_class_eval,
        "errors_per_class": errors_per_class
    }

def calculate_minimality_and_completeness(predicted_integrated_schema, gt_integrated_schema, type="entity"):
    try:
        total_attributes_gt = sum([len(attrs["attributes"]) for attrs in gt_integrated_schema.values()])
        total_entities_gt = len(gt_integrated_schema)
    except Exception as e:
        total_attributes_gt = sum([sum([len(tab_splits["attributes"]) for tab_splits in gt_integrated_schema[table].values()]) for table in gt_integrated_schema])
        total_entities_gt = sum([len(gt_integrated_schema[table]) for table in gt_integrated_schema])
        type = "table"

    if type != "entity":
        # Table are sent
        total_correct_attributes = 0
        total_predicted_attributes = sum([sum([len(tab_splits["attributes"]) for tab_splits in predicted_integrated_schema[table].values()]) for table in predicted_integrated_schema])
        total_correct_entities = 0
        total_predicted_entities = sum([len(predicted_integrated_schema[table]) for table in predicted_integrated_schema])
        for table in predicted_integrated_schema:
            res = calculate_minimality_and_completeness(predicted_integrated_schema[table], gt_integrated_schema[table], type="entity")
            total_correct_attributes += res["total_correct_attributes"]
            total_correct_entities += res["total_correct_entities"]
    else:
        # Record number of correctly predicted attributes and entities and number of predicted attributes and entities
        total_correct_attributes = 0
        total_predicted_attributes = sum([len(attrs["attributes"]) for attrs in predicted_integrated_schema.values()])
        total_correct_entities = 0
        total_predicted_entities = len(predicted_integrated_schema)

        for entity in predicted_integrated_schema:
            if entity in gt_integrated_schema:
                total_correct_entities += 1
            for attr in predicted_integrated_schema[entity]["attributes"]:
                if entity in gt_integrated_schema and attr in gt_integrated_schema[entity]["attributes"]:
                    total_correct_attributes += 1

    attribute_completeness = total_correct_attributes / total_attributes_gt if total_attributes_gt > 0 else 0
    attribute_minimality = 1 - ( (total_predicted_attributes-total_correct_attributes) / total_attributes_gt) if total_attributes_gt > 0 else 0

    entity_completeness = total_correct_entities / total_entities_gt if total_entities_gt > 0 else 0
    entity_minimality = 1 - ( (total_predicted_entities-total_correct_entities) / total_entities_gt) if total_entities_gt > 0 else 0

    return {
        "entity_completeness": entity_completeness,
        "entity_minimality": entity_minimality,
        "attribute_completeness": attribute_completeness,
        "attribute_minimality": attribute_minimality,
        "total_correct_attributes": total_correct_attributes,
        "total_predicted_attributes": total_predicted_attributes,
        "total_correct_entities": total_correct_entities,
        "total_predicted_entities": total_predicted_entities,
    }


def evaluate_schemas(predictions, actual, entity_labels, attribute_labels, join_tables=None):
    matrix_entities = {label: {'TP': 0, 'FP': 0, 'FN': 0} for label in entity_labels}
    matrix_attributes = {label: {'TP': 0, 'FP': 0, 'FN': 0} for label in attribute_labels}
    fps_list = []
    fns_list = []

    overall_tps_entities = 0
    overall_fps_entities = 0
    overall_fns_entities = 0

    overall_tps_attributes = 0
    overall_fps_attributes = 0
    overall_fns_attributes = 0
    
    # errors_per_table = {}

    # TODO: Fix foreign keys evaluation!
    success_fk_detections = 0
    # overall_fks = len([fk for entity in actual for fk in actual[entity]["foreign_keys"]])
    wrong_fk_detections = 0 # FPs
    missed_fk_detections = 0 # FNs

    # Counting TPs, FPs, FNs for attributes, entities and foreign keys
    for (table_name, predicted) in predictions.items():
        for pred in predicted:
            if pred in actual[table_name]:
                # Correct entity prediction
                matrix_entities[pred]['TP'] += 1
                overall_tps_entities += 1

                # Check predicted attributes for this entity
                for pred_attr in predicted[pred]["attributes"]:
                    if pred_attr not in actual[table_name][pred]["attributes"]:
                        # Incorrect attribute prediction: FP
                        overall_fps_attributes += 1
                        if f"{pred}.{pred_attr}" in matrix_attributes:
                            matrix_attributes[f"{pred}.{pred_attr}"]['FP'] += 1
                        # if table_name not in errors_per_table:
                        #     errors_per_table[table_name] = []
                        # errors_per_table[table_name].append(f"Wrong attribute detection for entity '{pred}': predicted attribute '{pred_attr}'")
                        fps_list.append(f"Wrong attribute detection for entity '{pred}': predicted attribute '{pred_attr}'")
                
                for act_attr in actual[table_name][pred]["attributes"]:
                    if act_attr not in predicted[pred]["attributes"]:
                        # Missed attribute prediction: FN
                        overall_fns_attributes += 1
                        if f"{pred}.{act_attr}" in matrix_attributes:
                            matrix_attributes[f"{pred}.{act_attr}"]['FN'] += 1
                        # if table_name not in errors_per_table:
                        #     errors_per_table[table_name] = []
                        # errors_per_table[table_name].append(f"Missed attribute '{act_attr}' for entity '{pred}'")
                        fns_list.append(f"Missed attribute '{act_attr}' for entity '{pred}'")

            else:
                # Incorrect prediction: FP
                overall_fps_entities += 1
                if pred in entity_labels:
                    matrix_entities[pred]['FP'] += 1 
                # if table_name not in errors_per_table:
                #     errors_per_table[table_name] = []
                # errors_per_table[table_name].append(f"Wrong entity type detection: predicted '{pred}'")
                fps_list.append(f"Wrong entity type detection: predicted '{pred}'")

                # Since entity is incorrect: all predicted attributes are also incorrect: FP
                for pred_attr in predicted[pred]["attributes"]:
                    overall_fps_attributes += 1
                    if any([pred_attr in actual[table_name][act]["attributes"] for act in actual[table_name]]):
                        found_entity = [act for act in actual[table_name] if pred_attr in actual[table_name][act]["attributes"]][0]
                        matrix_attributes[f"{found_entity}.{pred_attr}"]['FP'] += 1
                    # if table_name not in errors_per_table:
                    #     errors_per_table[table_name] = []
                    # errors_per_table[table_name].append(f"Wrong attribute detection for entity '{pred}': predicted attribute '{pred_attr}'")
                    fps_list.append(f"Wrong attribute detection for entity '{pred}': predicted attribute '{pred_attr}'")
            
            # If foreign keys should be predicted: Evaluate FKs
            if "foreign_keys" in predicted[pred]:
                if pred in actual[table_name]:
                    for fk, value in predicted[pred]["foreign_keys"].items():
                        if fk in actual[table_name][pred]["foreign_keys"].keys():
                            if value == actual[table_name][pred]["foreign_keys"][fk]:
                                success_fk_detections += 1  # TP
                            else:
                                wrong_fk_detections += 1  # FP
                        else:
                            wrong_fk_detections += 1
                    for fk in actual[table_name][pred]["foreign_keys"]:
                        if fk not in predicted[pred]["foreign_keys"]:
                            missed_fk_detections += 1  # FN
                else:
                    wrong_fk_detections += len(predicted[pred]["foreign_keys"])

        for act in actual[table_name]:
            if act not in predicted:
                # Missed prediction: FN
                overall_fns_entities += 1
                matrix_entities[act]['FN'] += 1
                # if table_name not in errors_per_table:
                #     errors_per_table[table_name] = []
                # errors_per_table[table_name].append(f"Missed entity type '{act}'")
                fns_list.append(f"Missed entity type '{act}'")

                # Since entity is missed: all its attributes are also missed: FN
                for act_attr in actual[table_name][act]["attributes"]:
                    overall_fns_attributes += 1
                    if f"{act}.{act_attr}" in matrix_attributes:
                        matrix_attributes[f"{act}.{act_attr}"]['FN'] += 1
                    # if table_name not in errors_per_table:
                    #     errors_per_table[table_name] = []
                    # errors_per_table[table_name].append(f"Missed attribute '{act_attr}' for entity '{act}'")
                    fns_list.append(f"Missed attribute '{act_attr}' for entity '{act}'")


                # Evaluate FKs: since entity is missed, all FKs are also missed
                if "foreign_keys" in actual[table_name][act]:
                    for fk in actual[table_name][act]["foreign_keys"]:
                        missed_fk_detections += 1

            else:
                for act_attr in actual[table_name][act]["attributes"]:
                    if act_attr in predicted[act]["attributes"]:
                        overall_tps_attributes += 1
                        matrix_attributes[f"{act}.{act_attr}"]['TP'] += 1

                    # Count duplicates as FPs
                    if sum([ 1 for p in predicted[act]["attributes"] if p==act_attr]) > 1:
                        print(act_attr)
                        overall_fps_attributes += sum([ 1 for p in predicted[act]["attributes"] if p==act_attr])-1
    
    # Check the evaluation results are correct
    assert(overall_tps_entities + overall_fns_entities == sum([ len(actual[table_name]) for table_name in actual]))
    assert(overall_tps_attributes + overall_fns_attributes == sum([ len(actual[table_name][entity]["attributes"]) for table_name in actual for entity in actual[table_name]]))
    assert(overall_tps_entities + overall_fps_entities == sum([ len(predictions[table_name]) for table_name in predictions]))
    assert(overall_tps_attributes + overall_fps_attributes == sum([ len(predictions[table_name][entity]["attributes"]) for table_name in predictions for entity in predictions[table_name]]))
    
    eval_results_entity = calculate_f1_p_r(matrix_entities, entity_labels)
    eval_results_attributes = calculate_f1_p_r(matrix_attributes, attribute_labels)

    completeness_minimality_results = calculate_minimality_and_completeness(predictions, actual)

    # Calculate overall entities evaluation metrics
    eval_results_entity["evaluation"] = {"tps": overall_tps_entities, "fps": overall_fps_entities, "fns": overall_fns_entities, "overall_recall": overall_tps_entities/(overall_tps_entities + overall_fns_entities), "overall_precision": overall_tps_entities/(overall_tps_entities + overall_fps_entities)}
    eval_results_entity["evaluation"]["overall_f1"] = 2*eval_results_entity["evaluation"]["overall_precision"]*eval_results_entity["evaluation"]["overall_recall"] / (eval_results_entity["evaluation"]["overall_precision"] + eval_results_entity["evaluation"]["overall_recall"]) if (eval_results_entity["evaluation"]["overall_precision"] + eval_results_entity["evaluation"]["overall_recall"])>0 else 0
    eval_results_entity["evaluation"]["overall_completeness"] = completeness_minimality_results["entity_completeness"]
    eval_results_entity["evaluation"]["overall_minimality"] = completeness_minimality_results["entity_minimality"]

    # Calculate overall attributes evaluation metrics
    eval_results_attributes["evaluation"] = {"tps": overall_tps_attributes, "fps": overall_fps_attributes, "fns": overall_fns_attributes, "overall_recall": overall_tps_attributes/(overall_tps_attributes + overall_fns_attributes), "overall_precision": overall_tps_attributes/(overall_tps_attributes + overall_fps_attributes)}
    eval_results_attributes["evaluation"]["overall_f1"] = 2*eval_results_attributes["evaluation"]["overall_precision"]*eval_results_attributes["evaluation"]["overall_recall"] / (eval_results_attributes["evaluation"]["overall_precision"] + eval_results_attributes["evaluation"]["overall_recall"]) if (eval_results_attributes["evaluation"]["overall_precision"] + eval_results_attributes["evaluation"]["overall_recall"])>0 else 0
    eval_results_attributes["evaluation"]["overall_completeness"] = completeness_minimality_results["attribute_completeness"]
    eval_results_attributes["evaluation"]["overall_minimality"] = completeness_minimality_results["attribute_minimality"]

    return {"eval_results_entity": eval_results_entity, "eval_results_attributes": eval_results_attributes, "errors_list": {"fps": fps_list, "fns": fns_list}}

def evaluate_instruction_prompt(runs, folder, benchmark="SINT-Benchmark"):
    processed_runs = {}
    file_names, gt_file, gt_mappings, gt_final_integrated_schema = load_ground_truth(folder, benchmark=benchmark)
    # Files to index and index to file names dictionary
    file_names_to_index = {file_name: i+1 for i, file_name in enumerate(file_names)}
    index_to_file_names = {i+1: file_name for i, file_name in enumerate(file_names)}
    
    splitting_phase = "Table Splitting Phase"
    schema_matching_phase = "Schema Matching Phase"
    grouping_phase = "Table Grouping Phase"
    attribute_clustering_phase = "Attribute Clustering Phase"
    final_integration_phase = "Final Integrated Schema Generation Phase"

    for run in runs:
        processed_runs[run] = {}

        # Detect tables
        file_names_to_ids = {file: f"Table {i+1}" if f"Table {i+1}" in runs[run]["response"][splitting_phase] else f"Table{i+1}" for i, file in enumerate(os.listdir(f"data/selected-tables/{benchmark}/{folder}"))}
        detected_tables = {file: runs[run]["response"][splitting_phase][file_names_to_ids[file]] for file in os.listdir(f"data/selected-tables/{benchmark}/{folder}")}
        table_splits_to_tables = {key: f"{key}_table_{file_names_to_index[table_name]}" for i, (table_name, table) in enumerate(detected_tables.items()) for key in table.keys()}
        # Add names to detected tables
        for i, (table_name, table) in enumerate(detected_tables.items()):
            for key in table.keys():
                table[key]["table_name"] = f"{key}_table_{file_names_to_index[table_name]}"

        # Schema Matching
        schema_matching_input = {}
        for table_split, table_corrs in runs[run]["response"][schema_matching_phase].items():
            for other_table_split, corrs in table_corrs.items():
                table_index = int(table_splits_to_tables[table_split].split("_table_")[-1])
                if other_table_split not in table_splits_to_tables:
                    continue
                other_table_index = int(table_splits_to_tables[other_table_split].split("_table_")[-1])

                if table_index < other_table_index:
                    # If the index of this table is lower than the other table, add as is
                    if f"table_{table_index}" not in schema_matching_input:
                        schema_matching_input[f"table_{table_index}"] = {}
                    if f"table_{other_table_index}" not in schema_matching_input[f"table_{table_index}"]:
                        schema_matching_input[f"table_{table_index}"][f"table_{other_table_index}"] = corrs
                    else:
                        schema_matching_input[f"table_{table_index}"][f"table_{other_table_index}"].update(corrs)
                else:
                    # otherwise invert the correspondences
                    if f"table_{other_table_index}" not in schema_matching_input:
                        schema_matching_input[f"table_{other_table_index}"] = {}
                    if f"table_{table_index}" not in schema_matching_input[f"table_{other_table_index}"]:
                        schema_matching_input[f"table_{other_table_index}"][f"table_{table_index}"] = {v: k for k, v in corrs.items()}
                    else:
                        schema_matching_input[f"table_{other_table_index}"][f"table_{table_index}"].update({v: k for k, v in corrs.items()})


        # Group tables
        grouping_input = {group_name: [table_splits_to_tables[tab] for tab in group_tables if tab in table_splits_to_tables] for group_name, group_tables in runs[run]["response"][grouping_phase].items()}
        table_splits_to_groups = {tab: group_name for group_name, group_tables in grouping_input.items() for tab in group_tables}
        # Update detected tables with group names
        for table in detected_tables.values():
            for key in table.keys():
                table[key]["table_group"] = table_splits_to_groups[table[key]["table_name"]]

        # Schema Integration
        schema_integration_input = {group_name: [{table_splits_to_tables[tab]: [repr_attr] for tab in tabs if tab in table_splits_to_tables} for repr_attr, tabs in attribute_groups.items()] for group_name, attribute_groups in runs[run]["response"][attribute_clustering_phase].items() }
        predicted_integrated_schemas = {group_name: [repr_attr for repr_attr in attribute_groups.keys()] for group_name, attribute_groups in runs[run]["response"][attribute_clustering_phase].items() }
        detected_tables_grouped_and_mapped = {group_name: {} for group_name in grouping_input.keys()}

        for table_name, table_splits in detected_tables.items():
            for table_split_name, table_split in table_splits.items():
                table_split_info = {
                    "attributes": table_split["attributes"],
                    "table_name": table_split["table_name"],
                    "table_group": table_split["table_group"],
                    "column_mappings_to_integrated_schema": {}
                }
                # Add the mappings
                for attribute in table_split["attributes"]:
                    for repr_attr, tabs in runs[run]["response"][attribute_clustering_phase][table_split["table_group"]].items():
                    # for repr_attr, tabs in runs[run]["response"][attribute_clustering_phase]["output"][table_split["table_group"]].items():
                        if table_split_name in tabs and tabs[table_split_name]==attribute:
                            table_split_info["column_mappings_to_integrated_schema"][attribute] = repr_attr
                
                # Add to the detected_tables_grouped_and_mapped
                if table_split["table_group"] not in detected_tables_grouped_and_mapped:
                    detected_tables_grouped_and_mapped[table_split["table_group"]] = {}
                if table_name not in detected_tables_grouped_and_mapped[table_split["table_group"]]:
                    detected_tables_grouped_and_mapped[table_split["table_group"]][table_name] = {}
                detected_tables_grouped_and_mapped[table_split["table_group"]][table_name][table_split_name] = table_split_info
        
        # Final Integration Phase
        new_added_attributes_per_entity = {entity: [attribute for attribute in runs[run]["response"][final_integration_phase][entity]["attributes"] if attribute not in predicted_integrated_schemas[entity] and not any(attribute in predicted_integrated_schemas[other_entity] for other_entity in predicted_integrated_schemas if other_entity != entity)] for entity in runs[run]["response"][final_integration_phase] if entity!="reasoning" and entity!="input_column_mapping"}
        
        # remove faulty keys from final integrated schema
        for entity in list(runs[run]["response"][final_integration_phase].keys()):
            if entity=="reasoning" or entity=="input_column_mapping":
                del runs[run]["response"][final_integration_phase][entity]
        
        final_integrated_schema_mappings = copy.deepcopy(detected_tables_grouped_and_mapped)
        # Update mappings to match the final integrated schema
        for group_name, group_tables in final_integrated_schema_mappings.items():
            for table_name, table_splits in group_tables.items():
                for table_split_name, table_split in table_splits.items():
                    # Update the column mappings to match the final integrated schema
                    updated_mappings = {}
                    for attribute, repr_attr in table_split["column_mappings_to_integrated_schema"].items():
                        if repr_attr not in runs[run]["response"][final_integration_phase][group_name]["attributes"]:
                            # Remove attribute from list and ignore the mapping
                            table_split["attributes"].remove(attribute)
                            continue
                        
                        updated_mappings[attribute] = repr_attr
                    
                    final_integrated_schema_mappings[group_name][table_name][table_split_name]["column_mappings_to_integrated_schema"] = updated_mappings
                    # Any new attributes added compared to the previous phase?
                    final_integrated_schema_mappings[group_name][table_name][table_split_name]["new_attributes"] = new_added_attributes_per_entity[table_split["table_group"]]

        processed_runs[run]["parameters"] = {
            "detected_tables": detected_tables,
            "detected_tables_grouped_and_mapped": detected_tables_grouped_and_mapped,
            "column_correspondences": schema_matching_input,
            "groups": grouping_input,
            "attribute_groups": schema_integration_input,
            "integrated_schemas": predicted_integrated_schemas,
            "final_integrated_schema": runs[run]["response"][final_integration_phase],
            "final_integrated_schema_mappings": final_integrated_schema_mappings,
            "removed_correspondences": {}
        }
    
    return processed_runs


def evaluate_cot_prompt(runs, folder, benchmark="SINT-Benchmark"):
    names_to_ids = {file: f"Table {i+1}" for i, file in enumerate(os.listdir(f"data/selected-tables/{benchmark}/{folder}"))}
    file_names, gt_file, gt_mappings, gt_final_integrated_schema = load_ground_truth(folder, benchmark=benchmark)
    all_eval_results = {run: {} for run in runs}
    std_dev_evaluation_results = {}
    
    for run in runs:
        if not any("entity_type" in col for col in runs[run]["response"]["input_column_mappings"]["Table 1"].values()):
            # Format input
            simple_prompt_formatted_results = {}
            for tab, tab_mappings in runs[run]["response"]["input_column_mappings"].items():
                simple_prompt_formatted_results[tab] = {}
                for col, col_mapping in tab_mappings.items():
                    for ent, attr in col_mapping.items():
                        simple_prompt_formatted_results[tab][col] = {"entity_type": ent, "integrated_schema_attribute": attr}
            runs[run]["response"]["input_column_mappings"] = simple_prompt_formatted_results
            
        integrated_attributes_values = {}
        file_names_to_ids = {f"Table {i+1}": file for i, file in enumerate(os.listdir(f"data/selected-tables/{benchmark}/{folder}"))}
        
        # Loop over the input mappings and selct the three first column values to add to the mapped attributes
        for table_name, table_mapping in runs[run]["response"]["input_column_mappings"].items():
            df = pd.read_csv(f"../data/selected-tables/SINT-Benchmark/{folder}/{file_names_to_ids[table_name]}")

            for col, col_mapping in table_mapping.items():
                integrated_attr = col_mapping["integrated_schema_attribute"]
                # print(f"Processing column {col} mapped to {col_mapping}")
                if integrated_attr not in integrated_attributes_values:
                    integrated_attributes_values[integrated_attr] = []

                # Add the first three values of each column to the corresponding integrated attribute
                try:
                    for value in df[col].dropna().tolist()[:3]:
                        integrated_attributes_values[integrated_attr].append(value)
                except KeyError:
                    print(f"Column {col} not found in table {table_name}. Skipping.")
                    print(df.columns)
                    continue

        # First map the integrated attributes that are the exact same as the gt attributes
        mapped_attributes = {}
        for attr, values in integrated_attributes_values.items():
            if any(attr in gt_final_integrated_schema["integrated_schema_by_entity"][entity]["attributes"] for entity in gt_final_integrated_schema["integrated_schema_by_entity"]):
                mapped_attributes[attr] = attr

        attribute_overlaps = {}
        for pred_attr in integrated_attributes_values:
            if pred_attr in mapped_attributes:
                continue
                            
            for entity, entity_info in gt_mappings["integrated_schema_by_entity"].items():
                entity_attributes_with_values = entity_info["attributes_with_values"]
                for gt_attr in entity_attributes_with_values:
                    # Count overlap of values
                    overlap = len(set(entity_attributes_with_values[gt_attr]).intersection(set(integrated_attributes_values[pred_attr])))
                    if pred_attr not in attribute_overlaps:
                        attribute_overlaps[pred_attr] = {}
                    attribute_overlaps[pred_attr][gt_attr] = overlap

        # For the rest based on value overlap with gt attributes: maximum overlap of values between the integrated attribute and the gt attributes
        highest_attribute_overlaps = {}
        for pred_attr in attribute_overlaps:
            maximum_attr = max(attribute_overlaps[pred_attr], key=attribute_overlaps[pred_attr].get)
            maximum_value = max(attribute_overlaps[pred_attr].values())
            if maximum_value != 0:
                highest_attribute_overlaps[pred_attr] = {"maximum_attr": maximum_attr, "maximum_value": maximum_value}
            else:
                highest_attribute_overlaps[pred_attr] = {"maximum_attr": None, "maximum_value": 0}

        
        for pred_attr in highest_attribute_overlaps:
            if highest_attribute_overlaps[pred_attr]["maximum_attr"] is not None and highest_attribute_overlaps[pred_attr]["maximum_attr"] not in mapped_attributes.values():
                other_occurences = [high["maximum_value"] for high in highest_attribute_overlaps.values() if high["maximum_attr"] == highest_attribute_overlaps[pred_attr]["maximum_attr"] and high != highest_attribute_overlaps[pred_attr]]
                if len(other_occurences):
                    if highest_attribute_overlaps[pred_attr]["maximum_value"] > max(other_occurences):
                        mapped_attributes[pred_attr] = highest_attribute_overlaps[pred_attr]["maximum_attr"]
                    else:
                        mapped_attributes[pred_attr] = None
                else:
                    mapped_attributes[pred_attr] = highest_attribute_overlaps[pred_attr]["maximum_attr"]
            else:
                mapped_attributes[pred_attr] = None

        
        # Map entities based on mapped attributes: if the majority of the mapped attributes of an entity type are mapped to a gt entity type, then map the entity type to that gt entity type
        entity_overlaps = {}
        for pred_entity, entity_info in runs[run]["response"]["integrated_schemas"].items():
            pred_entity_mapped_gt_attributes = [mapped_attributes[attr] for attr in entity_info if attr in mapped_attributes and mapped_attributes[attr] is not None]
            entity_overlaps[pred_entity] = {}
            for entity in gt_final_integrated_schema["integrated_schema_by_entity"]:
                gt_entity_attributes = gt_final_integrated_schema["integrated_schema_by_entity"][entity]["attributes"]
                overlap_count = len(set(pred_entity_mapped_gt_attributes).intersection(set(gt_entity_attributes)))
                entity_overlaps[pred_entity][entity] = overlap_count


        entity_mappings = {}
        for pred_entity, overlaps in entity_overlaps.items():
            max_overlap_entity = max(overlaps, key=overlaps.get)
            highest_occurence = max([entity_overlaps[e][max_overlap_entity] for e in entity_overlaps if e!=pred_entity and max_overlap_entity in entity_overlaps[e]]) if len(entity_overlaps) > 1 else 0
            if overlaps[max_overlap_entity] > highest_occurence and max_overlap_entity not in entity_mappings.values():
                entity_mappings[pred_entity] = max_overlap_entity
            else:
                entity_mappings[pred_entity] = f"None_{pred_entity}"

        file_names_to_index = {file_name: i+1 for i, file_name in enumerate(file_names)}

        final_mappings = {}
        for entity, entity_info in gt_mappings["integrated_schema_by_entity"].items():
            if entity not in final_mappings:
                final_mappings[entity] = {}
            for table_name, cols_maps in gt_mappings["integrated_schema_by_entity"][entity]["column_mappings_by_table"].items():
                for col, mapped_attr in cols_maps.items():
                    if mapped_attr in gt_final_integrated_schema["integrated_schema_by_entity"][entity]["attributes"]:
                        if table_name not in final_mappings[entity]:
                            final_mappings[entity][table_name] = {}
                        final_mappings[entity][table_name][col] = mapped_attr
                    else:
                        # Find where the mapped attribute is in the final integrated schema
                        found_ent = None
                        for ent, ent_info in gt_final_integrated_schema["integrated_schema_by_entity"].items():
                            if mapped_attr in ent_info["attributes"]:
                                found_ent = ent
                        
                        if found_ent is not None:
                            # Move the mapping there
                            if found_ent not in final_mappings:
                                final_mappings[found_ent] = {}
                            if table_name not in final_mappings[found_ent]:
                                final_mappings[found_ent][table_name] = {}
                            final_mappings[found_ent][table_name][col] = mapped_attr

        added_attributes_per_entity = {entity: [attr for attr in gt_final_integrated_schema["integrated_schema_by_entity"][entity]["attributes"] if attr not in gt_mappings["integrated_schema_by_entity"][entity]["attributes"]] for entity in gt_final_integrated_schema["integrated_schema_by_entity"]}

        entity_wise_mappings = {}
        for table_name, table_mapping in runs[run]["response"]["input_column_mappings"].items():
            for col, col_mapping in table_mapping.items():
                pred_ent = entity_mappings[col_mapping["entity_type"]]
                gt_table_name = file_names_to_ids[table_name]
                if pred_ent not in entity_wise_mappings:
                    entity_wise_mappings[pred_ent] = {}
                if gt_table_name not in entity_wise_mappings[pred_ent]:
                    entity_wise_mappings[pred_ent][gt_table_name] = {}
                entity_wise_mappings[pred_ent][gt_table_name][col] = mapped_attributes[col_mapping["integrated_schema_attribute"]]

        overall_tps = 0
        overall_fps = 0
        overall_fns = 0

        for group in entity_wise_mappings:
            if group not in final_mappings:
                overall_fps += len([col for table_name, table_mapping in entity_wise_mappings[group].items() for col in table_mapping])
            else:
                for table_name, cols_maps in entity_wise_mappings[group].items():
                    if table_name not in final_mappings[group]:
                        overall_fps += len(cols_maps)
                        continue

                    for original_attr, mapped_attr in entity_wise_mappings[group][table_name].items():
                        if original_attr not in final_mappings[group][table_name]:
                            overall_fps += 1
                        else:
                            if final_mappings[group][table_name][original_attr] == mapped_attr:
                                overall_tps += 1
                            else:
                                overall_fps += 1
                                overall_fns += 1

        for group in final_mappings:
            if group not in entity_mappings.values():
                overall_fns += len([col for table_name, table_mapping in final_mappings[group].items() for col in table_mapping])
            else:
                for table_name, cols_maps in final_mappings[group].items():
                    if table_name not in entity_wise_mappings[group]:
                        overall_fns += len(cols_maps)
                        continue

                    if len(cols_maps) > len(entity_wise_mappings[group][table_name]):
                        overall_fns += len(cols_maps) - len(entity_wise_mappings[group][table_name])

            
        eval_results = {}

        add = 0
        if not overall_tps + overall_fns == sum([len([col for table_name, cols_maps in final_mappings[group].items() for col in cols_maps]) for group in final_mappings]):
            add = sum([ len([attribute for cols_maps in gt_mappings["integrated_schema_by_entity"][group]["column_mappings_by_table"].values() for attribute in cols_maps]) for group in gt_mappings["integrated_schema_by_entity"]]) - (overall_tps + overall_fns)
            print(f"Adding {add} to overall_fns to match the total number of attributes in the GT mappings")

        add = max(0, add)

        if not overall_tps + overall_fps == len([col for table_name, table_mapping in runs[run]["response"]["input_column_mappings"].items() for col, col_mapping in table_mapping.items()]):
            print("Not all predictions are counted!!")

        added_ids = [(entity_group, attribute) for entity_group in gt_final_integrated_schema["integrated_schema_by_entity"] for attribute in gt_final_integrated_schema["integrated_schema_by_entity"][entity_group]["attributes"] if "id" in attribute.lower() and attribute not in gt_mappings["integrated_schema_by_entity"][entity_group]["attributes"]]
        model_added_ids = []
        # Check if model has added ids
        for entity in runs[run]["response"]["integrated_schemas"]:
            for attribute in runs[run]["response"]["integrated_schemas"][entity]:
                if "id" in attribute.lower() and attribute not in mapped_attributes:
                    print(f"Model has added id attribute {attribute} for entity {entity}.")
                    if (entity_mappings[entity], f"{entity_mappings[entity]}_id") not in model_added_ids:
                        model_added_ids.append((entity_mappings[entity], f"{entity_mappings[entity]}_id"))

        for entity, added_id in added_ids:
            if (entity, added_id) not in model_added_ids:
                print(f"Model has not added id attribute {added_id} for entity {entity}.")
                overall_fns += len(final_mappings[entity])
            else:
                overall_tps += len(final_mappings[entity])
        for entity, model_added_id in model_added_ids:
            if entity not in gt_final_integrated_schema["integrated_schema_by_entity"]:
                if entity not in entity_mappings:
                    overall_fps += 1
                    continue
                print(f"Model has added id attribute {model_added_id} for entity {entity} but it is not in the GT.")
                overall_fps += len(entity_wise_mappings[entity])

        eval_results["column_mappings"] = {"tps": overall_tps, "fps": overall_fps, "fns": overall_fns+add, "overall_recall": overall_tps/(overall_tps + overall_fns+add), "overall_precision": overall_tps/(overall_tps + overall_fps)}
        eval_results["column_mappings"]["overall_f1"] = 2*eval_results["column_mappings"]["overall_precision"]*eval_results["column_mappings"]["overall_recall"] / (eval_results["column_mappings"]["overall_precision"] + eval_results["column_mappings"]["overall_recall"]) if (eval_results["column_mappings"]["overall_precision"] + eval_results["column_mappings"]["overall_recall"])>0 else 0
        
        integrated_schema_mapped ={entity_mappings[entity]: {"attributes": [mapped_attributes[attr] if attr in mapped_attributes else None for attr in runs[run]["response"]["integrated_schemas"][entity]]} for entity in runs[run]["response"]["integrated_schemas"]}
        integrated_attribute_labels = [f"{ent}.{attr}" for ent, attrs in gt_final_integrated_schema["integrated_schema_by_entity"].items() for attr in attrs["attributes"]]
        integrated_entity_labels = list(gt_final_integrated_schema["integrated_schema_by_entity"].keys())
        schema_result = evaluate_schemas({"tab": integrated_schema_mapped}, {"tab": gt_final_integrated_schema["integrated_schema_by_entity"]}, integrated_entity_labels, integrated_attribute_labels)

        eval_results["entity_results"] = schema_result["eval_results_entity"]["evaluation"]
        eval_results["attribute_results"] = schema_result["eval_results_attributes"]["evaluation"]
        all_eval_results[run] = eval_results
        
    # Calculate standard deviation for each metric across all runs
    for aspect in ["column_mappings", "entity_results", "attribute_results"]:
        if aspect not in std_dev_evaluation_results:
            std_dev_evaluation_results[aspect] = {}
        for metric in ["overall_recall", "overall_precision", "overall_f1", "overall_minimality", "overall_completeness"]:
            metric_values = [all_eval_results[run][aspect][metric] for run in all_eval_results if metric in all_eval_results[run][aspect]]
            if len(metric_values) > 0:
                std_dev_evaluation_results[aspect][metric] = np.std(metric_values)

    
    # Calculate average metrics and standard deviation across all runs
    average_metrics = {}

    for aspect in ["column_mappings", "entity_results", "attribute_results"]:
        average_metrics[aspect] = {}
        for metric in ["overall_recall", "overall_precision", "overall_f1", "overall_minimality", "overall_completeness"]:
            metric_values = [all_eval_results[run][aspect][metric] for run in all_eval_results if metric in all_eval_results[run][aspect]]
            if len(metric_values) > 0:
                average_metrics[aspect][metric] = sum(metric_values) / len(metric_values)

    all_eval_results["average_metrics"] = average_metrics
    all_eval_results["std_dev_metrics"] = std_dev_evaluation_results

    return all_eval_results