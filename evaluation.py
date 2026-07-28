import csv
import sienna
from dataset_utils import load_ground_truth
# from utils import write_txt_file
import pandas as pd
import collections
import numpy as np
import os
import copy
import random

import numpy as np
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
    
    evaluation = {
        "Micro-F1": micro_f1,
        "Macro-F1": macro_f1,
        "Precision": p,
        "Recall": r
    }
    
    return {
        "evaluation": evaluation,
        "per_class_eval": per_class_eval,
        "errors_per_class": errors_per_class
    }

class SchemaIntegrationEvaluation:
    def __init__(self, tables_path, log_file_name):
        self.tables_path = tables_path
        self.log_file_name = log_file_name
        self.init_predictions()
        # Load ground truth
        self.init_gt_properties()
        self.eval_results = {}
        self.overall_metrics = {}
        self.analysis_results = {}
        self.overall_mappings = None

    def init_predictions(self):
        file = sienna.load(f"{self.log_file_name}.json")
        self.run_info = file["run_info"]
        self.benchmark = file["run_info"]["benchmark"] if "benchmark" in file["run_info"] else "SINT-Benchmark"
        self.folder = file["run_info"]["folder_name"]
        self.sequence_of_phases = file["run_info"]["sequence_of_phases"]
        self.other_parameters = file["run_info"]["other_parameters"]
        self.table_splitting_phase_name = "detect_tables_phase" # if "detect_tables_phase" in self.sequence_of_phases else "normalization_phase"
        self.self_consistency = self.other_parameters["self-consistency"]
        self.num_runs = file["run_info"]["num_runs"] if not self.self_consistency else 1
        self.all_predictions = file["runs"]
        self.predictions = file["runs"] if not self.self_consistency else {"0": file["runs"]["0"]}

    def init_gt_properties(self):
        # initialize Ground truth
        self.file_names, gt_file, gt_mappings, gt_final_integrated_schema = load_ground_truth(self.folder, benchmark=self.benchmark)
        gt_part = "schema_by_entity"
        gt_integration_part = "integrated_schema_by_entity"
        try:
            self.splitting_after_integration = self.sequence_of_phases.index(self.table_splitting_phase_name) > self.sequence_of_phases.index("schema_integration_phase")
        except Exception:
            self.splitting_after_integration = True
        # print(gt_integration_part)
        self.gt_file = gt_file[gt_part]
        self.gt_mappings = gt_mappings[gt_integration_part] if gt_integration_part in gt_mappings else gt_mappings
        self.overall_gt_mappings = gt_mappings["column_mappings"]
        self.gt_final_integrated_schema = gt_final_integrated_schema[gt_integration_part]
        
        # Schema matching ground truth
        self.schema_matching_correspondences = gt_mappings["schema_matching"]

        # Files to index and index to file names dictionary
        self.file_names_to_index = {file_name: i+1 for i, file_name in enumerate(self.file_names)}
        self.index_to_file_names = {i+1: file_name for i, file_name in enumerate(self.file_names)}

        if len(self.sequence_of_phases) == 1 and self.sequence_of_phases[0] == "schema_matching_phase":
            # Special case, for only schema matching evaluation
            self.gt_mappings = gt_mappings
            return

        if self.splitting_after_integration:
            # Merge all attributes, column mappings and schema matching
            # print(self.folder)
            # print(self.gt_mappings.keys())
            merged_attributes = list(set([attr for ent, attrs in self.gt_mappings.items() for attr in attrs["attributes"]]))
            # Merge attribute values
            merged_attribute_values = {}
            for _, attrs in self.gt_mappings.items():
                for attr, values in attrs["attributes_with_values"].items():
                    merged_attribute_values.setdefault(attr, []).extend(values)
            # Merge column mappings
            merged_column_mappings = gt_mappings["column_mappings"]
            grouped_attributes = gt_mappings["grouped_attributes"]
            self.gt_mappings = {
                "all": {
                    "attributes": merged_attributes,
                    "attributes_with_values": merged_attribute_values,
                    "column_mappings_by_table": merged_column_mappings,
                    "grouped_attributes": grouped_attributes
                }
            }
        
        # Table splitting ground truth
        self.entity_labels = list(set([entity for table_splits in self.gt_file.values() for entity in table_splits]))
        self.attribute_labels = set([f"{entity}.{attribute}" for table in self.gt_file.values() for entity in table for attribute in table[entity]["attributes"]])
        
        # Extra tables
        self.join_tables = [entity for entity in self.gt_final_integrated_schema if entity not in self.gt_mappings]
        # Integrated schema
        self.integrated_attribute_labels = [f"{ent}.{attr}" for ent, attrs in self.gt_final_integrated_schema.items() for attr in attrs["attributes"]]
        self.integrated_entity_labels = list(self.gt_final_integrated_schema.keys())

    def evaluate_schemas(self, predictions, actual, entity_labels, attribute_labels, join_tables=None):
        report_entities = {label: {'TP': 0, 'FP': 0, 'FN': 0} for label in entity_labels}
        report_attributes = {label: {'TP': 0, 'FP': 0, 'FN': 0} for label in attribute_labels}
        fps_list = []
        fns_list = []

        overall_tps_entities = 0
        overall_fps_entities = 0
        overall_fns_entities = 0

        overall_tps_attributes = 0
        overall_fps_attributes = 0
        overall_fns_attributes = 0
        
        errors_per_table = {}

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
                    report_entities[pred]['TP'] += 1
                    overall_tps_entities += 1

                    # Check predicted attributes for this entity
                    for pred_attr in predicted[pred]["attributes"]:
                        if pred_attr not in actual[table_name][pred]["attributes"]:
                            # Incorrect attribute prediction: FP
                            overall_fps_attributes += 1
                            if f"{pred}.{pred_attr}" in report_attributes:
                                report_attributes[f"{pred}.{pred_attr}"]['FP'] += 1
                            if table_name not in errors_per_table:
                                errors_per_table[table_name] = []
                            errors_per_table[table_name].append(f"Wrong attribute detection for entity '{pred}': predicted attribute '{pred_attr}'")
                            fps_list.append(f"Wrong attribute detection for entity '{pred}': predicted attribute '{pred_attr}'")
                    
                    for act_attr in actual[table_name][pred]["attributes"]:
                        if act_attr not in predicted[pred]["attributes"]:
                            # Missed attribute prediction: FN
                            overall_fns_attributes += 1
                            if f"{pred}.{act_attr}" in report_attributes:
                                report_attributes[f"{pred}.{act_attr}"]['FN'] += 1
                            if table_name not in errors_per_table:
                                errors_per_table[table_name] = []
                            errors_per_table[table_name].append(f"Missed attribute '{act_attr}' for entity '{pred}'")
                            fns_list.append(f"Missed attribute '{act_attr}' for entity '{pred}'")

                else:
                    # Incorrect prediction: FP
                    overall_fps_entities += 1
                    if pred in entity_labels:
                        report_entities[pred]['FP'] += 1 
                    if table_name not in errors_per_table:
                        errors_per_table[table_name] = []
                    errors_per_table[table_name].append(f"Wrong entity type detection: predicted '{pred}'")
                    fps_list.append(f"Wrong entity type detection: predicted '{pred}'")

                    # Since entity is incorrect: all predicted attributes are also incorrect: FP
                    for pred_attr in predicted[pred]["attributes"]:
                        overall_fps_attributes += 1
                        if any([pred_attr in actual[table_name][act]["attributes"] for act in actual[table_name]]):
                            found_entity = [act for act in actual[table_name] if pred_attr in actual[table_name][act]["attributes"]][0]
                            report_attributes[f"{found_entity}.{pred_attr}"]['FP'] += 1
                        if table_name not in errors_per_table:
                            errors_per_table[table_name] = []
                        errors_per_table[table_name].append(f"Wrong attribute detection for entity '{pred}': predicted attribute '{pred_attr}'")
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
                    report_entities[act]['FN'] += 1
                    if table_name not in errors_per_table:
                        errors_per_table[table_name] = []
                    errors_per_table[table_name].append(f"Missed entity type '{act}'")
                    fns_list.append(f"Missed entity type '{act}'")

                    # Since entity is missed: all its attributes are also missed: FN
                    for act_attr in actual[table_name][act]["attributes"]:
                        overall_fns_attributes += 1
                        if f"{act}.{act_attr}" in report_attributes:
                            report_attributes[f"{act}.{act_attr}"]['FN'] += 1
                        if table_name not in errors_per_table:
                            errors_per_table[table_name] = []
                        errors_per_table[table_name].append(f"Missed attribute '{act_attr}' for entity '{act}'")
                        fns_list.append(f"Missed attribute '{act_attr}' for entity '{act}'")


                    # Evaluate FKs: since entity is missed, all FKs are also missed
                    if "foreign_keys" in actual[table_name][act]:
                        for fk in actual[table_name][act]["foreign_keys"]:
                            missed_fk_detections += 1

                else:
                    for act_attr in actual[table_name][act]["attributes"]:
                        if act_attr in predicted[act]["attributes"]:
                            overall_tps_attributes += 1
                            report_attributes[f"{act}.{act_attr}"]['TP'] += 1

                        # Count duplicates as FPs
                        if sum([ 1 for p in predicted[act]["attributes"] if p==act_attr]) > 1:
                            print(act_attr)
                            overall_fps_attributes += sum([ 1 for p in predicted[act]["attributes"] if p==act_attr])-1
        
        # Check the evaluation results are correct
        assert(overall_tps_entities + overall_fns_entities == sum([ len(actual[table_name]) for table_name in actual]))
        assert(overall_tps_attributes + overall_fns_attributes == sum([ len(actual[table_name][entity]["attributes"]) for table_name in actual for entity in actual[table_name]]))
        assert(overall_tps_entities + overall_fps_entities == sum([ len(predictions[table_name]) for table_name in predictions]))
        assert(overall_tps_attributes + overall_fps_attributes == sum([ len(predictions[table_name][entity]["attributes"]) for table_name in predictions for entity in predictions[table_name]]))
        
        eval_results_entity = calculate_f1_p_r(report_entities, entity_labels)
        eval_results_attributes = calculate_f1_p_r(report_attributes, attribute_labels)

        eval_results_entity["eval_stats"] = {"tps": overall_tps_entities, "fps": overall_fps_entities, "fns": overall_fns_entities, "overall_recall": overall_tps_entities/(overall_tps_entities + overall_fns_entities), "overall_precision": overall_tps_entities/(overall_tps_entities + overall_fps_entities)}
        eval_results_attributes["eval_stats"] = {"tps": overall_tps_attributes, "fps": overall_fps_attributes, "fns": overall_fns_attributes, "overall_recall": overall_tps_attributes/(overall_tps_attributes + overall_fns_attributes), "overall_precision": overall_tps_attributes/(overall_tps_attributes + overall_fps_attributes)}
        eval_results_entity["eval_stats"]["overall_f1"] = 2*eval_results_entity["eval_stats"]["overall_precision"]*eval_results_entity["eval_stats"]["overall_recall"] / (eval_results_entity["eval_stats"]["overall_precision"] + eval_results_entity["eval_stats"]["overall_recall"]) if (eval_results_entity["eval_stats"]["overall_precision"] + eval_results_entity["eval_stats"]["overall_recall"])>0 else 0
        eval_results_attributes["eval_stats"]["overall_f1"] = 2*eval_results_attributes["eval_stats"]["overall_precision"]*eval_results_attributes["eval_stats"]["overall_recall"] / (eval_results_attributes["eval_stats"]["overall_precision"] + eval_results_attributes["eval_stats"]["overall_recall"]) if (eval_results_attributes["eval_stats"]["overall_precision"] + eval_results_attributes["eval_stats"]["overall_recall"])>0 else 0

        return {"eval_results_entity": eval_results_entity, "eval_results_attributes": eval_results_attributes, "errors_list": {"fps": fps_list, "fns": fns_list}}
    
    def evaluate_detect_tables_phase(self, run_nr):
        # Tables to evaluate
        predicted_table_splits = self.predictions[run_nr]["parameters"]["detected_tables"]

        if self.splitting_after_integration:
            predicted_table_splits = {}
            for ent, tables in self.predictions[run_nr]["parameters"]["detected_tables_grouped_and_mapped"].items():
                for table, splits in tables.items():
                    predicted_table_splits.setdefault(table, {})
                    for split in splits:
                        predicted_table_splits[table][ent] = splits[split]
                        predicted_table_splits[table][ent]["table_name"] = split

        # Calculate overlap of attributes for the found entities
        detected_tables_mapped = {}
        overall_mappings = {}
        for (table_name, table_splits) in predicted_table_splits.items():
            matched_entities = {}
            # First check if any entity name matches with GT and directly map
            for table_split in table_splits:
                for gt_entity in self.gt_file[table_name]:
                    if table_split == gt_entity or table_split.lower() == gt_entity.lower():
                        matched_entities[table_split] = gt_entity
                        # Replace old table name with gt entity name
                        detected_tables_mapped.setdefault(table_name, {})
                        detected_tables_mapped[table_name][gt_entity] = predicted_table_splits[table_name][table_split]
                        break

            # For other non-mapped entities, find overlap of attributes and based on the highest overlaps assign GT entity
            entity_overlaps = {pred_entity: {gt_entity: 0 for gt_entity in self.gt_file[table_name]} for pred_entity in table_splits if pred_entity not in matched_entities}   
            for table_split in table_splits:
                if table_split not in matched_entities:
                    predicted_attributes = table_splits[table_split]["attributes"]
                    # Calculate overlap
                    for entity in self.gt_file[table_name]:
                        gt_attributes = self.gt_file[table_name][entity]["attributes"]
                        overlap = len(set(predicted_attributes).intersection(set(gt_attributes)))
                        entity_overlaps[table_split][entity] = overlap
            
            # For each predicted entity, find the gt entity with the highest overlap            
            max_entity_overlap = {}
            for entity_overlap in entity_overlaps:
                if entity_overlap not in matched_entities:
                    if max(entity_overlaps[entity_overlap].values()) != 0:
                        best_entity = max(entity_overlaps[entity_overlap], key=entity_overlaps[entity_overlap].get)
                        max_entity_overlap[entity_overlap] = {"max_entity": best_entity, "overlap": entity_overlaps[entity_overlap][best_entity]}
                    else:
                        max_entity_overlap[entity_overlap] = {"max_entity": None, "overlap": 0}
            
            for entity in max_entity_overlap:
                if max_entity_overlap[entity]["max_entity"] is not None:
                    best_entity = max_entity_overlap[entity]["max_entity"]
                    # If this entity has already been matched, skip
                    if best_entity not in matched_entities.values():
                        # If this entity is not highest overlap for another table split
                        other_occurences = [e for e in max_entity_overlap if max_entity_overlap[e]["max_entity"] == best_entity and e != entity]
                        if max_entity_overlap[entity]["overlap"] > max([max_entity_overlap[e]["overlap"] for e in other_occurences], default=0):
                            matched_entities[entity] = best_entity
                            # Replace old table name with gt entity name
                            detected_tables_mapped.setdefault(table_name, {})
                            detected_tables_mapped[table_name][best_entity] = predicted_table_splits[table_name][entity]

            # Extra table splits that is not in GT are marked with None
            for not_matched in (set(table_splits.keys()) - set(matched_entities.keys())):
                matched_entities[not_matched] = f"None_{not_matched}"
                detected_tables_mapped.setdefault(table_name, {})
                detected_tables_mapped[table_name][f"None_{not_matched}"] = predicted_table_splits[table_name][not_matched]
            # Record the mappings
            overall_mappings[table_name] = matched_entities

        if self.splitting_after_integration:
            # Update entity mappings
            # For each split gather the mappings and count the occurences of each mapping
            mappings_by_splits = {}
            for table_mappings in overall_mappings.values():
                for split, mapping in table_mappings.items():
                    if split not in mappings_by_splits:
                        mappings_by_splits[split] = {}
                    mappings_by_splits[split].setdefault(mapping, 0)
                    mappings_by_splits[split][mapping] += 1
            self.mapped_groups = {}
            for split in mappings_by_splits:
                # Remove None mappings if there are other mappings
                if len([mapping for mapping in mappings_by_splits[split] if "None" not in mapping]) > 0:
                    # Choose a mapping for this split between the other mappings
                    highest_mapping = max({mapping: mappings_by_splits[split][mapping] for mapping in mappings_by_splits[split] if "None" not in mapping}, key={mapping: mappings_by_splits[split][mapping] for mapping in mappings_by_splits[split] if "None" not in mapping}.get)
                    self.mapped_groups[split] = highest_mapping
                else:
                    # Choose the highest mapping for this split
                    highest_mapping = max(mappings_by_splits[split], key=mappings_by_splits[split].get)
                    self.mapped_groups[split] = highest_mapping
            # Update table namings
            for table_name, table_mappings in overall_mappings.items():
                for split, mapping in table_mappings.items():
                    if split in self.mapped_groups and mapping != self.mapped_groups[split]:
                        # print(f"Table {table_name} has split {split} mapped to {mapping} but majority is {self.mapped_groups[split]}")
                        # Update the mappings of the tables to the new mapping
                        detected_tables_mapped[table_name][self.mapped_groups[split]] = copy.deepcopy(detected_tables_mapped[table_name][mapping])
                        del detected_tables_mapped[table_name][mapping]
            # self.mapped_groups = {ent: mapped for ent_mapped in overall_mappings.values() for ent, mapped in ent_mapped.items()}
            # Update attribute mappings to by entity
            self.predicted_mapped_attributes = {entity: {pred_attr: self.predicted_mapped_attributes["all"][pred_attr] for pred_attr in self.predictions[run_nr]["parameters"]["detected_tables"]["full_table"][entity]["attributes"]} for entity in self.predictions[run_nr]["parameters"]["detected_tables"]["full_table"]}

        # Evaluate mapped entities
        self.eval_results[run_nr]["detect_tables_phase"] = self.evaluate_schemas(detected_tables_mapped, self.gt_file, self.entity_labels, self.attribute_labels)
        self.overall_metrics.setdefault("detect_tables_phase_entity", [])
        self.overall_metrics["detect_tables_phase_entity"].append(self.eval_results[run_nr]["detect_tables_phase"]["eval_results_entity"]["eval_stats"])
        self.overall_metrics.setdefault("detect_tables_phase_attributes", [])
        self.overall_metrics["detect_tables_phase_attributes"].append(self.eval_results[run_nr]["detect_tables_phase"]["eval_results_attributes"]["eval_stats"])
        self.analysis_results.setdefault(run_nr, {})
        self.analysis_results[run_nr]["detect_tables_phase"] = self.eval_results[run_nr]["detect_tables_phase"]["errors_list"]

        # Save mappings to GT
        self.overall_mappings = overall_mappings
        self.detected_tables_mapped = detected_tables_mapped

    def evaluate_grouping_phase(self, run_nr):
        # Evaluate grouping of tables, each table has an assigned group
        groups_mapped = {group: [] for group in self.predictions[run_nr]["parameters"]["groups"]}
        # Using old mappings, map entities again to gt labels
        # for (table_name, table_splits) in self.predictions[run_nr]["parameters"]["detected_tables_grouped"].items():
        for (table_name, table_splits) in self.predictions[run_nr]["parameters"]["detected_tables"].items():
            for (table_split, table) in table_splits.items():
                groups_mapped[table["table_group"]].append(self.overall_mappings[table_name][table_split])

        # For each group, find the entity that mostly shows up in the list
        representative_entities = {}
        for group in groups_mapped:
            if len(groups_mapped[group]) == 0:
                continue
            most_common_entity = collections.Counter(groups_mapped[group]).most_common(1)[0][0]
            representative_entities[group] = { "most_common_entity": most_common_entity, "count": groups_mapped[group].count(most_common_entity) }

        # Refine dictionary based on duplicate most common entities
        refined_representative_entities = {}
        for group in representative_entities:
            # Check if most common entity appears in the other groups
            if representative_entities[group]["most_common_entity"] in [representative_entities[g]["most_common_entity"] for g in representative_entities if g != group]:
                # If the number of appearances is higher, add as mapping
                number_of_groups = [representative_entities[g]["count"] for g in representative_entities if g != group and representative_entities[g]["most_common_entity"] == representative_entities[group]["most_common_entity"] ]
                current_count = representative_entities[group]["count"]
                if current_count >= max(number_of_groups) and representative_entities[group]["most_common_entity"] not in refined_representative_entities.values():
                    refined_representative_entities[group] = representative_entities[group]["most_common_entity"]
                else:
                    refined_representative_entities[group] = f"None_{representative_entities[group]['most_common_entity']}"
            # If not keep as is
            else:
                refined_representative_entities[group] = representative_entities[group]["most_common_entity"]

        # Switch predicted group names to representative entity names
        detected_tables_grouped_mapped = {}
        for (table_name, table_splits) in self.predictions[run_nr]["parameters"]["detected_tables"].items():
            detected_tables_grouped_mapped[table_name] = {}
            for (table_split, table) in table_splits.items():
                detected_tables_grouped_mapped[table_name][self.overall_mappings[table_name][table_split]] = {
                    "attributes": table["attributes"],
                    "table_group": refined_representative_entities[table["table_group"]], # replace with gt label
                    "table_name": table["table_name"].replace(table_split, self.overall_mappings[table_name][table_split])
                }

        # Evaluate grouping
        overall_tps = 0
        overall_fps = 0
        overall_fns = 0
        errors_list = []
        preds = []
        gts = []

        for ti, (table_name, table_splits) in enumerate(detected_tables_grouped_mapped.items()):
            gt_groups = [f"{table_split['table_group']}.{table_split_name}_table_{ti+1}" for (table_split_name, table_split) in self.gt_file[table_name].items()]
            predicted_groups = [f"{table_split['table_group']}.{table_split['table_name']}" for table_split in table_splits.values()]
            preds.append(predicted_groups)
            gts.append(gt_groups)
            # The predictions are correct if both group and entity name are in GT
            for predicted_group in predicted_groups:
                if predicted_group in gt_groups:
                    overall_tps += 1
                else:
                    overall_fps += 1
                    errors_list.append(f"Wrong group detection for table {table_name}: predicted group '{predicted_group}' not in GT groups {gt_groups}")
            
            for gt_group in gt_groups:
                if gt_group not in predicted_groups:
                    overall_fns += 1
                    errors_list.append(f"Missed group '{gt_group}' for table {table_name}")

        assert(overall_tps + overall_fns == sum([len(self.gt_file[table]) for table in self.gt_file]))
        assert(overall_tps + overall_fps == sum([len(detected_tables_grouped_mapped[table]) for table in detected_tables_grouped_mapped]))
        
        eval_results = {"tps": overall_tps, "fps": overall_fps, "fns": overall_fns, "overall_recall": overall_tps/(overall_tps + overall_fns), "overall_precision": overall_tps/(overall_tps + overall_fps)}
        eval_results["overall_f1"] = 2*eval_results["overall_precision"]*eval_results["overall_recall"] / (eval_results["overall_precision"] + eval_results["overall_recall"]) if (eval_results["overall_precision"] + eval_results["overall_recall"])>0 else 0
        # TODO: Save class-wise evaluation
        # Save evaluation results
        self.eval_results[run_nr][ "grouping_phase"] = eval_results
        # Save for average calculation
        self.overall_metrics.setdefault("grouping_phase", [])
        self.overall_metrics["grouping_phase"].append(eval_results)
        # Save mapped detected tables and group mappings for next phase evaluation
        self.detected_tables_grouped_mapped = detected_tables_grouped_mapped
        self.mapped_groups = refined_representative_entities

        self.analysis_results.setdefault(run_nr, {})
        self.analysis_results[run_nr]["grouping_phase"] = errors_list
        self.analysis_results[run_nr]["grouping_phase_preds"] = {"predicted_groups": preds, "gt_groups": gts}

    def evaluate_schema_matching_phase(self, run_nr, remove_correspondences=False): #
        gt_column_correspondences = {f"table_{self.file_names_to_index[table_name]}": {f"table_{self.file_names_to_index[other_table_name]}": table_corrs[other_table_name] for other_table_name in table_corrs} for ti, (table_name, table_corrs) in enumerate(self.schema_matching_correspondences.items())}

        # If schema matching is run after splitting tables and before grouping, merge original tables
        if self.splitting_after_integration or ("schema_matching_phase" in self.sequence_of_phases and self.sequence_of_phases.index("schema_matching_phase")==0):
            column_correspondences = self.predictions[run_nr]["parameters"]["original_column_correspondences"] if "original_column_correspondences" in self.predictions[run_nr]["parameters"] else self.predictions[run_nr]["parameters"]["column_correspondences"]
        
            if remove_correspondences:
                removed_correspondences = self.predictions[run_nr]["parameters"]["removed_correspondences"]
                schema_matching_predictions = {table: {other_table: {attr: attr_corr for attr, attr_corr in table_corrs[other_table].items() if not(table in removed_correspondences and other_table in removed_correspondences[table] and attr in removed_correspondences[table][other_table] and removed_correspondences[table][other_table][attr]==attr_corr) and not(other_table in removed_correspondences and table in removed_correspondences[other_table] and attr in removed_correspondences[other_table][table] and removed_correspondences[other_table][table][attr]==attr_corr) } for other_table in table_corrs} for table, table_corrs in column_correspondences.items() }
                column_correspondences_updated = copy.deepcopy(schema_matching_predictions)
                column_correspondences = copy.deepcopy(schema_matching_predictions)
            else:
                column_correspondences_updated = copy.deepcopy(column_correspondences)
            
            # Catch any problems
            for table, table_corrs in column_correspondences.items():
                for other_table, corrs in table_corrs.items():
                    for attr, mapped_attr in corrs.items():
                        if int(other_table.split('_')[-1]) < int(table.split('_')[-1]):
                            # print(int(other_table.split('_')[-1]), int(table.split('_')[-1]))
                            # Switch and delete to avoid duplicates
                            column_correspondences_updated.setdefault(other_table, {})
                            column_correspondences_updated[other_table].setdefault(table, {})
                            column_correspondences_updated[other_table][table][mapped_attr] = attr
                            del column_correspondences_updated[table][other_table][attr]
            column_correspondences = column_correspondences_updated
        else:
            if self.sequence_of_phases.index("schema_matching_phase") > self.sequence_of_phases.index("grouping_phase"):
                # Flatten the correspondences when they are grouped
                schema_matching_predictions = {table: table_corrs[table] for entity, table_corrs in self.predictions[run_nr]["parameters"]["column_correspondences"].items() for table in table_corrs}
            else:
                schema_matching_predictions = self.predictions[run_nr]["parameters"]["column_correspondences"]

            if remove_correspondences:
                removed_correspondences = self.predictions[run_nr]["parameters"]["removed_correspondences"]
                schema_matching_predictions = {table: {other_table: {attr: attr_corr for attr, attr_corr in table_corrs[other_table].items() if not(table in removed_correspondences and other_table in removed_correspondences[table] and attr in removed_correspondences[table][other_table] and removed_correspondences[table][other_table][attr]==attr_corr) and not(other_table in removed_correspondences and table in removed_correspondences[other_table] and attr in removed_correspondences[other_table][table] and removed_correspondences[other_table][table][attr]==attr_corr) } for other_table in table_corrs} for table, table_corrs in schema_matching_predictions.items() }

            # If schema matching after splitting of tables: merge them first based on the table names
            column_correspondences = {}
            for tab_name in self.file_names_to_index.values():
                full_tab_name = f"table_{tab_name}"
                column_correspondences[full_tab_name] = {}
                for table_split_name, table_corrs in schema_matching_predictions.items():
                    if table_split_name.endswith(full_tab_name):
                        for other_split_name, corrs in table_corrs.items():
                            # If both from same table, skip
                            if other_split_name.endswith(full_tab_name):
                                # print("Both from same table, skipping")
                                # print(table_split_name, other_split_name)
                                continue
                            if int(other_split_name.split('_')[-1]) < int(tab_name):
                                full_other_split_name = f"table_{other_split_name.split('_')[-1]}"
                                column_correspondences[full_other_split_name].setdefault(full_tab_name, {})
                                for attr, mapped_attr in corrs.items():
                                    if mapped_attr not in column_correspondences[full_other_split_name][full_tab_name]:
                                        column_correspondences[full_other_split_name][full_tab_name][mapped_attr] = attr
                            else:
                                full_other_split_name = f"table_{other_split_name.split('_')[-1]}"
                                column_correspondences[full_tab_name].setdefault(full_other_split_name, {})
                                column_correspondences[full_tab_name][full_other_split_name].update(corrs)
            # Same for removed correspondences
            if remove_correspondences:
                removed_correspondences_updated = {}
                for tab_name in self.file_names_to_index.values():
                    full_tab_name = f"table_{tab_name}"
                    removed_correspondences_updated[full_tab_name] = {}
                    for table_split_name, table_corrs in removed_correspondences.items():
                        if table_split_name.endswith(full_tab_name):
                            for other_split_name, corrs in table_corrs.items():
                                if other_split_name.endswith(full_tab_name):
                                    continue
                                if int(other_split_name.split('_')[-1]) < int(tab_name):
                                    full_other_split_name = f"table_{other_split_name.split('_')[-1]}"
                                    removed_correspondences_updated.setdefault(full_other_split_name, {})
                                    removed_correspondences_updated[full_other_split_name].setdefault(full_tab_name, {})
                                    for attr, mapped_attr in corrs.items():
                                        if mapped_attr not in removed_correspondences_updated[full_other_split_name][full_tab_name]:
                                            removed_correspondences_updated[full_other_split_name][full_tab_name][mapped_attr] = attr
                                else:
                                    full_other_split_name = f"table_{other_split_name.split('_')[-1]}"
                                    removed_correspondences_updated.setdefault(full_tab_name, {})
                                    removed_correspondences_updated[full_tab_name].setdefault(full_other_split_name, {})
                                    removed_correspondences_updated[full_tab_name][full_other_split_name].update(corrs)
                removed_correspondences = removed_correspondences_updated
        
        if remove_correspondences:
            total_changes = sum([1 for table, table_corrs in removed_correspondences.items() for other_table, corrs in table_corrs.items() for attr, attr_corr in corrs.items()])
            correct_changes = 0
            incorrect_changes = 0
            # Check which of the correspondences are correct and incorrect changes in the removed correspondences
            for table, table_corrs in removed_correspondences.items():
                for other_table, corrs in table_corrs.items():
                    for attr, attr_corr in corrs.items():
                        if table in gt_column_correspondences and other_table in gt_column_correspondences[table] and attr in gt_column_correspondences[table][other_table] and gt_column_correspondences[table][other_table][attr] == attr_corr:
                            # A correct correspondence was removed
                            incorrect_changes += 1
                        else:
                            # An incorrect correspondence was removed
                            correct_changes += 1
            self.eval_results[run_nr]["removed_correspondences_evaluation"] = {"total_changes": total_changes, "correct_changes": correct_changes, "incorrect_changes": incorrect_changes, "percentage of correct changes": correct_changes/total_changes if total_changes>0 else 0, "percentage of incorrect changes": incorrect_changes/total_changes if total_changes>0 else 0}
        
        overall_tps = 0
        overall_fps = 0
        overall_fns = 0

        # Record FNs and FPs for later analysis of errors
        fps_list = []
        fns_list = []

        for table_split_name, table_split_correspondences in column_correspondences.items():
            for other_table_split_name, predicted_corrs in table_split_correspondences.items():
                for source_attr, target_attr in predicted_corrs.items():
                    if source_attr not in gt_column_correspondences[table_split_name][other_table_split_name]:
                        # Wrong correspondence: FP
                        overall_fps += 1
                        fps_list.append(f"Wrong correspondence: {table_split_name}.{source_attr} -> {other_table_split_name}.{target_attr} (not in GT)")
                    elif gt_column_correspondences[table_split_name][other_table_split_name][source_attr] == target_attr:
                        # Correct correspondence
                        overall_tps += 1
                    else:
                        # Incorrect correspondence: FP
                        overall_fps += 1
                        fns_list.append(f"Missed correct correspondence: {table_split_name}.{source_attr} -> {other_table_split_name}.{gt_column_correspondences[table_split_name][other_table_split_name][source_attr]} (predicted {table_split_name}.{source_attr} -> {other_table_split_name}.{target_attr})")

        for overall_table_name in gt_column_correspondences:
            for other_overall_table_name in gt_column_correspondences[overall_table_name]:
                for source_attr, target_attr in gt_column_correspondences[overall_table_name][other_overall_table_name].items():
                    if overall_table_name not in column_correspondences:
                        # Missed correspondence: FN
                        overall_fns += 1
                        fns_list.append(f"Missed correct correspondence: {overall_table_name}.{source_attr} -> {other_overall_table_name}.{target_attr} (not predicted)")
                        continue
                    if other_overall_table_name not in column_correspondences[overall_table_name]:
                        # Missed correspondence: FN
                        overall_fns += 1
                        fns_list.append(f"Missed correct correspondence: {overall_table_name}.{source_attr} -> {other_overall_table_name}.{gt_column_correspondences[overall_table_name][other_overall_table_name][source_attr]} (not predicted)")
                    else:
                        if source_attr not in column_correspondences[overall_table_name][other_overall_table_name]:
                            # Missed correspondence: FN
                            overall_fns += 1
                            fns_list.append(f"Missed correct correspondence: {overall_table_name}.{source_attr} -> {other_overall_table_name}.{gt_column_correspondences[overall_table_name][other_overall_table_name][source_attr]} (not predicted)")
                        if source_attr in column_correspondences[overall_table_name][other_overall_table_name] and column_correspondences[overall_table_name][other_overall_table_name][source_attr] != target_attr:
                            # Missed the correct correspondence of this attribute: FN
                            overall_fns += 1
                            fns_list.append(f"Missed correct correspondence: {overall_table_name}.{source_attr} -> {other_overall_table_name}.{gt_column_correspondences[overall_table_name][other_overall_table_name][source_attr]} (predicted {overall_table_name}.{source_attr} -> {other_overall_table_name}.{column_correspondences[overall_table_name][other_overall_table_name].get(source_attr, 'None')})")
        
        assert(overall_tps+overall_fns == sum([ len(corrs) for table_corrs in self.schema_matching_correspondences.values() for corrs in table_corrs.values()]))
        assert(overall_tps+overall_fps == sum([ len(predicted_corrs) for table_split_correspondences in column_correspondences.values() for predicted_corrs in table_split_correspondences.values()]))

        
        eval_results = {"tps": overall_tps, "fps": overall_fps, "fns": overall_fns, "overall_recall": overall_tps/(overall_tps + overall_fns) if (overall_tps + overall_fns) > 0 else 0, "overall_precision": overall_tps/(overall_tps + overall_fps) if (overall_tps + overall_fps) > 0 else 0}
        if eval_results["overall_precision"] + eval_results["overall_recall"] > 0:
            eval_results["overall_f1"] = 2*eval_results["overall_precision"]*eval_results["overall_recall"] / (eval_results["overall_precision"] + eval_results["overall_recall"]) if (eval_results["overall_precision"] + eval_results["overall_recall"])>0 else 0
        else:
            eval_results["overall_f1"] = 0
        # Save evaluation results
        phase_name = "schema_matching_phase_removed" if remove_correspondences else "schema_matching_phase"
        self.eval_results[run_nr][phase_name] = eval_results
        # Save for average calculation
        self.overall_metrics.setdefault(phase_name, [])
        self.overall_metrics[phase_name].append(eval_results)
        self.analysis_results.setdefault(run_nr, {})
        self.analysis_results[run_nr][phase_name] = {"fps": fps_list, "fns": fns_list}

    def evaluate_schema_integration_phase(self, run_nr):
        predicted_column_mappings = {}
        predicted_mapped_attributes = {}
        eval_results = {}
        # If table is split after schema integration, all tables are grouped together
        self.mapped_groups = {"all": "all"} if self.splitting_after_integration else self.mapped_groups

        for integration_type in ["integrated_schemas", "integrated_attributes_no_removals"]:
            # If the original attribute groups before schema matching loop were not recorded: skip
            if "attribute_groups_with_no_removals" not in self.predictions[run_nr]["parameters"] and integration_type == "integrated_attributes_no_removals":
                continue

            # Map the integrated attributes to GT attributes for evaluation
            predicted_mapped_attributes[integration_type] = {}
            # Based on the above mappings, map all columns to the GT attributes for evaluation
            predicted_column_mappings[integration_type] = {}
            # Retrieve the attribute groups
            attribute_groups = self.predictions[run_nr]["parameters"]["attribute_groups_with_no_removals"] if integration_type == "integrated_attributes_no_removals" else self.predictions[run_nr]["parameters"]["attribute_groups"]
            
            # Retrieve the integrated schemata for all entity types
            if self.splitting_after_integration:
                if integration_type == "integrated_attributes_no_removals":
                    random.seed(0)
                    predicted_integrated_schemas = {entity: [random.choice(list(set([str(attribute) for attributes in group.values() for attribute in attributes]))) for gi, group in enumerate(attribute_groups)] for entity, attribute_groups in attribute_groups.items()}
                else:
                    predicted_integrated_schemas = {entity: list(self.predictions[run_nr]["parameters"]["integrated_attributes"][entity].values()) for entity in self.predictions[run_nr]["parameters"]["integrated_attributes"]}
            else:
                if integration_type == "integrated_attributes_no_removals":
                    random.seed(0)
                    predicted_integrated_schemas = {entity: [random.choice(list(set([str(attribute) for attributes in group.values() for attribute in attributes]))) for gi, group in enumerate(attribute_groups)] for entity, attribute_groups in attribute_groups.items() if entity!="all"}
                else:
                    predicted_integrated_schemas = self.predictions[run_nr]["parameters"]["integrated_schemas"]

            for entity_group in predicted_integrated_schemas:
                if self.mapped_groups[entity_group] not in self.gt_mappings:
                    # If the entity group was not previously mapped to a GT entity type, all attributes are wrong = FPs
                    predicted_column_mappings[integration_type][self.mapped_groups[entity_group]] = self.predictions[run_nr]["parameters"]["detected_tables_grouped_and_mapped"][entity_group]
                    predicted_mapped_attributes[integration_type][entity_group] = {pred_attr: f"None_{pred_attr}" for pred_attr in predicted_integrated_schemas[entity_group]}
                    continue

                group_attributes_with_values = {}
                mapped_attributes = {}
                attribute_overlaps = {}

                # Map equal integrated attributes to GT attributes directly, otherwise retrieve the column values for all the others
                for i, pred_attr in enumerate(predicted_integrated_schemas[entity_group]):
                    # Predicted attribute is in GT, directly map to it
                    if pred_attr in self.gt_mappings[self.mapped_groups[entity_group]]["attributes"]:
                        mapped_attributes[pred_attr] = pred_attr
                    else:
                        attribute_overlaps[pred_attr] = {}
                        # Collect 3 values from all columns that are linked to this predicted attribute
                        if "original_detected_tables_grouped_and_mapped" in self.predictions[run_nr]["parameters"] and integration_type != "integrated_attributes_no_removals":
                            connected_columns = [f"{table_name}|||{table_split}|||{col}" for table_name, table_splits in self.predictions[run_nr]["parameters"]["original_detected_tables_grouped_and_mapped"][entity_group].items() for table_split, table_info in table_splits.items() if pred_attr in table_info["column_mappings_to_integrated_schema"].values() for col in table_info["column_mappings_to_integrated_schema"].keys() if table_info["column_mappings_to_integrated_schema"][col] == pred_attr]
                        elif integration_type == "integrated_attributes_no_removals":
                            connected_columns = [f"{self.index_to_file_names[int(table_name.split('_table_')[-1])]}|||{table_name}|||{attribute}" if "_table_" in table_name else f"{self.index_to_file_names[int(table_name.split('table_')[-1])]}|||{table_name}|||{attribute}" for table_name, table_attributes in attribute_groups[entity_group][i].items() for attribute in table_attributes]
                        else:
                            connected_columns = [f"{table_name}|||{table_split}|||{col}" for table_name, table_splits in self.predictions[run_nr]["parameters"]["detected_tables_grouped_and_mapped"][entity_group].items() for table_split, table_info in table_splits.items() if pred_attr in table_info["column_mappings_to_integrated_schema"].values() for col in table_info["column_mappings_to_integrated_schema"].keys() if table_info["column_mappings_to_integrated_schema"][col] == pred_attr]
                        for table_name_split_col in connected_columns:
                            table_name, table_split, col = table_name_split_col.split("|||")
                            df = pd.read_csv(f"{self.tables_path}/{self.folder}/{table_name}")
                            for value in df[col].dropna().tolist()[:3]:
                                group_attributes_with_values.setdefault(pred_attr, []).append(value)
                    
                # For each attribute there are 3 values from each table, count the overlap with GT attributes
                entity_attributes_with_values = self.gt_mappings[self.mapped_groups[entity_group]]["attributes_with_values"]

                for pred_attr in group_attributes_with_values:
                    for gt_attr in entity_attributes_with_values:
                        # Count overlap of values
                        overlap = len(set(entity_attributes_with_values[gt_attr]).intersection(set(group_attributes_with_values[pred_attr])))
                        attribute_overlaps[pred_attr][gt_attr] = overlap

                highest_attribute_overlaps = {}
                for pred_attr in attribute_overlaps:
                    maximum_attr = max(attribute_overlaps[pred_attr], key=attribute_overlaps[pred_attr].get)
                    maximum_value = max(attribute_overlaps[pred_attr].values())
                    if maximum_value != 0:
                        highest_attribute_overlaps[pred_attr] = {"maximum_attr": maximum_attr, "maximum_value": maximum_value}
                    else:
                        highest_attribute_overlaps[pred_attr] = {"maximum_attr": None, "maximum_value": 0}
                
                # Map predicted attributes to GT attributes based on highest overlap, duplicate highest overlap, choose highest the others are None
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

                assert len(mapped_attributes) == len(set(predicted_integrated_schemas[entity_group]))

                # if integration_type == "integrated_schemas":
                # Update predicted column mappings with mapped attributes: the right-side
                if "original_detected_tables_grouped_and_mapped" in self.predictions[run_nr]["parameters"]:
                    gt_mapped_detected_tables_and_attributes = copy.deepcopy(self.predictions[run_nr]["parameters"]["original_detected_tables_grouped_and_mapped"][entity_group])
                else:
                    gt_mapped_detected_tables_and_attributes = copy.deepcopy(self.predictions[run_nr]["parameters"]["detected_tables_grouped_and_mapped"][entity_group])

                for table_name in gt_mapped_detected_tables_and_attributes:
                    for table_split in gt_mapped_detected_tables_and_attributes[table_name]:
                        for col, intcol in gt_mapped_detected_tables_and_attributes[table_name][table_split]["column_mappings_to_integrated_schema"].items():
                            if integration_type == "integrated_attributes_no_removals":
                                # Find in which group col is in the attribute groups with no removals
                                group_index = [i for i, attribute_group in enumerate(attribute_groups[entity_group]) for table_name in attribute_group if col in attribute_group[table_name]]
                                # Find the integrated attribute that corresponds to this group index
                                integrated_attr = mapped_attributes[predicted_integrated_schemas[entity_group][group_index[0]]]
                                # Update mapping
                                gt_mapped_detected_tables_and_attributes[table_name][table_split]["column_mappings_to_integrated_schema"][col] = integrated_attr
                            else:
                                if intcol in mapped_attributes:
                                    # Map to the GT attribute
                                    gt_mapped_detected_tables_and_attributes[table_name][table_split]["column_mappings_to_integrated_schema"][col] = mapped_attributes[intcol]

                predicted_column_mappings[integration_type][self.mapped_groups[entity_group]] = gt_mapped_detected_tables_and_attributes
                predicted_mapped_attributes[integration_type][entity_group] = mapped_attributes

            # Evaluate column mappings
            overall_tps = 0
            overall_fps = 0
            overall_fns = 0
            for group in predicted_column_mappings[integration_type]:
                if group not in self.gt_mappings:
                    # Count the mappings all as FPs
                    overall_fps += len([attribute for table_splits in predicted_column_mappings[integration_type][group].values() for table_split in table_splits.values() for attribute in table_split["column_mappings_to_integrated_schema"].keys()])
                    continue
                
                for table_name, table_splits in predicted_column_mappings[integration_type][group].items():
                    # Combine column mappings if two splits from the same table have ended together
                    combined_table_info = {}

                    for table_split, table_info in table_splits.items():
                        combined_table_info.setdefault("column_mappings_to_integrated_schema", {}).update(table_info["column_mappings_to_integrated_schema"])
                        
                    if table_name in self.gt_mappings[group]["column_mappings_by_table"]:
                        for original_attr, mapped_attr in combined_table_info["column_mappings_to_integrated_schema"].items():
                            if original_attr not in self.gt_mappings[group]["column_mappings_by_table"][table_name]:
                                # wrong attribute (mapping) not in GT
                                overall_fps += 1
                            elif self.gt_mappings[group]["column_mappings_by_table"][table_name][original_attr] == mapped_attr:
                                # Correct mapping
                                overall_tps += 1
                            else:
                                # Incorrect mapping to this attribute
                                # And missed mapping to correct attribute
                                overall_fps += 1
                                overall_fns += 1
                    else:
                        # All are incorrect FPs
                        for original_attr, mapped_attr in combined_table_info["column_mappings_to_integrated_schema"].items():
                            overall_fps += 1

            for group in self.gt_mappings:
                if group not in predicted_column_mappings[integration_type]:
                    overall_fns += len([attribute for cols_maps in self.gt_mappings[group]["column_mappings_by_table"].values() for attribute in cols_maps])
                else:
                    for table_name, cols_maps in self.gt_mappings[group]["column_mappings_by_table"].items():
                        for original_attr, mapped_attr in self.gt_mappings[group]["column_mappings_by_table"][table_name].items():
                            # This attribute from the GT is not in the predicted mappings, count as FN
                            if table_name not in predicted_column_mappings[integration_type][group]:
                                overall_fns += 1
                            elif not any([original_attr in table_split["column_mappings_to_integrated_schema"] for table_split in predicted_column_mappings[integration_type][group][table_name].values()]):
                                overall_fns += 1

            combined_predicted_mappings = {}
            for group in predicted_column_mappings[integration_type]:
                combined_predicted_mappings[group] = {}
                for table_name, table_splits in predicted_column_mappings[integration_type][group].items():
                    for table_split, table_info in table_splits.items():
                        for original_attr, mapped_attr in table_info["column_mappings_to_integrated_schema"].items():
                            combined_predicted_mappings[group].setdefault(table_name, {})[original_attr] = mapped_attr

            assert(overall_tps + overall_fns == sum([ len([attribute for cols_maps in self.gt_mappings[group]["column_mappings_by_table"].values() for attribute in cols_maps]) for group in self.gt_mappings]))
            assert(overall_tps + overall_fps == len([attribute for group in combined_predicted_mappings for table in combined_predicted_mappings[group] for attribute in combined_predicted_mappings[group][table]]))

            eval_results[integration_type] = {}
            eval_results[integration_type]["schema_integration_phase_column_mappings"] = {"tps": overall_tps, "fps": overall_fps, "fns": overall_fns, "overall_recall": overall_tps/(overall_tps + overall_fns), "overall_precision": overall_tps/(overall_tps + overall_fps)}
            eval_results[integration_type]["schema_integration_phase_column_mappings"]["overall_f1"] = 2*eval_results[integration_type]["schema_integration_phase_column_mappings"]["overall_precision"]*eval_results[integration_type]["schema_integration_phase_column_mappings"]["overall_recall"] / (eval_results[integration_type]["schema_integration_phase_column_mappings"]["overall_precision"] + eval_results[integration_type]["schema_integration_phase_column_mappings"]["overall_recall"]) if (eval_results[integration_type]["schema_integration_phase_column_mappings"]["overall_precision"] + eval_results[integration_type]["schema_integration_phase_column_mappings"]["overall_recall"])>0 else 0


        # Evaluate predicted schema (not mappings)
        entity_labels_to_evaluate = ["all"] if "all" in self.mapped_groups.values() else self.entity_labels

        integrated_schema_mapped ={self.mapped_groups[entity]: {"attributes": list(set(predicted_mapped_attributes["integrated_schemas"][entity].values()))} for entity in predicted_mapped_attributes["integrated_schemas"]}
        integrated_schema_gt = {entity: {"attributes": self.gt_mappings[entity]["attributes"]} for entity in self.gt_mappings}
        integrated_attributes = [f"{entity}.{attr}" for entity in integrated_schema_gt for attr in integrated_schema_gt[entity]["attributes"]]
        eval_results["integrated_schemas"]["schema_integration_phase_schema"] = self.evaluate_schemas({"tab": integrated_schema_mapped}, {"tab": integrated_schema_gt}, entity_labels_to_evaluate, integrated_attributes)

        if "attribute_groups_with_no_removals" in self.predictions[run_nr]["parameters"]:
            integrated_schema_no_removals_mapped ={self.mapped_groups[entity]: {"attributes": list(set(predicted_mapped_attributes["integrated_attributes_no_removals"][entity].values()))} for entity in predicted_mapped_attributes["integrated_attributes_no_removals"]}
            if "schema_matching_phase" in self.sequence_of_phases:
                eval_results["integrated_attributes_no_removals"]["schema_integration_phase_schema_no_removals"] = self.evaluate_schemas({"tab": integrated_schema_no_removals_mapped}, {"tab": integrated_schema_gt}, entity_labels_to_evaluate, integrated_attributes)
        
        # Save evaluation results
        self.eval_results[run_nr]["schema_integration_phase"] = eval_results["integrated_schemas"]["schema_integration_phase_column_mappings"]
        # Save also for analysis all other schema integration results
        self.eval_results[run_nr]["schema_integration_phase_extended"] = eval_results

        # Save for average calculation
        self.overall_metrics.setdefault("schema_integration_phase", [])
        self.overall_metrics["schema_integration_phase"].append(eval_results)
        self.predicted_mapped_column_mappings = predicted_column_mappings # mapped predicted column mappings
        self.predicted_mapped_attributes = predicted_mapped_attributes["integrated_schemas"]

    def evaluate_final_integration(self, run_nr):
        final_predicted_schema_mapped = {}
        final_attributes_mapped = {}
        for entity in self.predictions[run_nr]["parameters"]["final_integrated_schema"]:
            # If entity is not mapped in previous phase, it should be a new table added by the model
            if entity not in self.mapped_groups:
                print("Entity was not in previous phase!")
                final_predicted_schema_mapped[entity] = self.predictions[run_nr]["parameters"]["final_integrated_schema"][entity]
                continue
                # self.mapped_groups[entity] = entity
                # self.predicted_mapped_attributes[entity] = {}
            
            final_attributes_mapped[self.mapped_groups[entity]] = {}
            
            # Entity is mapped, map attributes if not mapped already
            for pred_attr in self.predictions[run_nr]["parameters"]["final_integrated_schema"][entity]["attributes"]:
                if pred_attr in self.predicted_mapped_attributes[entity]:
                    # Check if it has been mapped in the previous phase
                    mapped_attr = self.predicted_mapped_attributes[entity][pred_attr]
                
                elif any(pred_attr in self.predicted_mapped_attributes[other_entity] for other_entity in self.predicted_mapped_attributes if other_entity != entity):
                    # Map to the same attribute name in the other entity
                    for other_entity in self.predicted_mapped_attributes:
                        if other_entity != entity and pred_attr in self.predicted_mapped_attributes[other_entity]:
                            mapped_attr = self.predicted_mapped_attributes[other_entity][pred_attr]
                            break
                elif "id" in pred_attr.lower():
                    # Check if the added attribute is an added identifier
                    print("Added identifier: ", pred_attr)
                    if entity.lower() in pred_attr.lower():
                        # Check if this identifier is the identifier for the entity
                        mapped_attr = f"{self.mapped_groups[entity]}_id"
                    else:
                        # Check if it is the id of another entity
                        if any(other_entity.lower() in pred_attr.lower() for other_entity in self.mapped_groups):
                            for other_entity in self.mapped_groups:
                                if other_entity.lower() in pred_attr.lower():
                                    mapped_attr = f"{self.mapped_groups[other_entity]}_id"
                                    break
                        else:
                            # The entity does not exist
                            mapped_attr = pred_attr
                else:
                    # Check if this attribute could have been moved from another entity
                    # else:
                    # Attribute is not found and not mapped previously, probably an error or extra attribute that is not needed
                    print("attribute was not in previous phase, got added", pred_attr)
                    mapped_attr = pred_attr
                
                final_attributes_mapped[self.mapped_groups[entity]][pred_attr] = mapped_attr
                final_predicted_schema_mapped.setdefault(self.mapped_groups[entity], {"attributes": [], "foreign_keys": {}})
                # Adding an identifier to the table should not be considered as an error
                # So check if this identifier exists in GT, if not, don't add to the mapped schema
                if mapped_attr == f"{self.mapped_groups[entity]}_id" and self.mapped_groups[entity] in self.gt_final_integrated_schema and mapped_attr not in self.gt_final_integrated_schema[self.mapped_groups[entity]]["attributes"]:
                    print("Added identifier does not exist in GT, not adding to mapped schema, but should not be considered as an error", mapped_attr)
                    continue
                final_predicted_schema_mapped[self.mapped_groups[entity]]["attributes"].append(mapped_attr)
            # if not(any("id" in pred_attr.lower() for pred_attr in final_predicted_schema_mapped[self.mapped_groups[entity]]["attributes"])):
            #     # add id
            #     final_predicted_schema_mapped[self.mapped_groups[entity]]["attributes"].append(f"{self.mapped_groups[entity]}_id")

        for entity in self.predictions[run_nr]["parameters"]["final_integrated_schema"]:
            if entity not in self.mapped_groups:
                print("Entity was not in previous phase! Add fks as they are")
                # final_predicted_schema_mapped[entity] = self.predictions[run_nr]["parameters"]["final_integrated_schema"][entity]
                continue
            if "foreign_keys" not in self.predictions[run_nr]["parameters"]["final_integrated_schema"][entity]:
                continue
            for fk, reference in self.predictions[run_nr]["parameters"]["final_integrated_schema"][entity]["foreign_keys"].items():
                # print(fk, reference)
                for reference_entity, reference_attr in reference.items():
                    # Map the referenced entity, by looking into the previously mapped entities
                    if reference_entity in self.mapped_groups:
                        mapped_reference_entity = self.mapped_groups[reference_entity]
                    else:
                        mapped_reference_entity = "None"

                    # Map the foreign attribute: attribute should be in this table, so the attribute should have already been mapped above
                    if fk in final_attributes_mapped[self.mapped_groups[entity]]:
                        mapped_fk = final_attributes_mapped[self.mapped_groups[entity]][fk]
                    else:
                        print("This attribute was not mapped above! error, this attribute must not exist", fk)
                        mapped_fk = fk
                    
                    # Map the referenced attribute from the foreign table, this attribute should exist in the referenced table and should already have been mapped above
                    if mapped_reference_entity in final_attributes_mapped:
                        if reference_attr in final_attributes_mapped[mapped_reference_entity]:
                            mapped_reference_attr = final_attributes_mapped[mapped_reference_entity][reference_attr]
                        else:
                            print("This referenced attribute was not mapped above! error, this attribute must not exist", reference_attr)
                            mapped_reference_attr = reference_attr
                    else:
                        print("This referenced entity does not exits, so attribute also does not exist! error", reference_entity, reference_attr)
                        mapped_reference_attr = reference_attr

                    final_predicted_schema_mapped[self.mapped_groups[entity]]["foreign_keys"][mapped_fk] = {mapped_reference_entity: mapped_reference_attr}
        # Evaluate final integrated schema
        # self.eval_results["final_integration_phase"] = self.evaluate_schemas(final_predicted_schema_mapped, self.gt_final_integrated_schema, self.entity_labels, self.attribute_labels)
        self.eval_results[run_nr]["final_integration_phase"] = self.evaluate_schemas({"tab": final_predicted_schema_mapped}, {"tab": self.gt_final_integrated_schema}, self.integrated_entity_labels, self.integrated_attribute_labels, self.join_tables)
        self.final_predicted_schema_mapped = final_predicted_schema_mapped
        self.final_attributes_mapped = final_attributes_mapped
        # Save for average calculation
        self.overall_metrics.setdefault("final_integration_entity", [])
        self.overall_metrics["final_integration_entity"].append(self.eval_results[run_nr]["final_integration_phase"]["eval_results_entity"]["eval_stats"])
        self.overall_metrics.setdefault("final_integration_attributes", [])
        self.overall_metrics["final_integration_attributes"].append(self.eval_results[run_nr]["final_integration_phase"]["eval_results_attributes"]["eval_stats"])
        self.analysis_results.setdefault(run_nr, {})
        self.analysis_results[run_nr]["final_integration_phase"] = self.eval_results[run_nr]["final_integration_phase"]["errors_list"]

        # Evaluate mappings as well
        mapped_final_integrated_schema_mappings = {}
        for entity_group in self.predictions[run_nr]["parameters"]["final_integrated_schema_mappings"]:
            mapped_final_integrated_schema_mappings[self.mapped_groups[entity_group]] = {}
            mapped_entity_group = self.mapped_groups[entity_group]
            for table_name, table_splits in self.predictions[run_nr]["parameters"]["final_integrated_schema_mappings"][entity_group].items():
                # if len(table_splits) > 1:
                # print(f"Multiple splits for table {table_name} in entity group {entity_group}. Mappings should be combined for evaluation.")
                mapped_final_integrated_schema_mappings[mapped_entity_group][table_name] = {}
                mapped_final_integrated_schema_mappings[mapped_entity_group][table_name][mapped_entity_group] = {}
                mapped_final_integrated_schema_mappings[mapped_entity_group][table_name][mapped_entity_group]["attributes"] = []
                mapped_final_integrated_schema_mappings[mapped_entity_group][table_name][mapped_entity_group]["column_mappings_to_integrated_schema"] = {}
                mapped_final_integrated_schema_mappings[mapped_entity_group][table_name][mapped_entity_group]["new_attributes"] = []
                for table_split, table_info in table_splits.items():
                    for column_name, mapped_attribute in table_info["column_mappings_to_integrated_schema"].items():
                        if column_name not in mapped_final_integrated_schema_mappings[mapped_entity_group][table_name][mapped_entity_group]["column_mappings_to_integrated_schema"]:
                            mapped_final_integrated_schema_mappings[mapped_entity_group][table_name][mapped_entity_group]["column_mappings_to_integrated_schema"][column_name] = self.final_attributes_mapped[mapped_entity_group][mapped_attribute]
                        if column_name not in mapped_final_integrated_schema_mappings[mapped_entity_group][table_name][mapped_entity_group]["attributes"]:
                            mapped_final_integrated_schema_mappings[mapped_entity_group][table_name][mapped_entity_group]["attributes"].append(column_name)
                    for attribute in table_info["new_attributes"]:
                        if self.final_attributes_mapped[mapped_entity_group][attribute] not in mapped_final_integrated_schema_mappings[mapped_entity_group][table_name][mapped_entity_group]["new_attributes"]:
                            mapped_final_integrated_schema_mappings[mapped_entity_group][table_name][mapped_entity_group]["new_attributes"].append(self.final_attributes_mapped[mapped_entity_group][attribute])

        overall_tps = 0
        overall_fps = 0
        overall_fns = 0
        for entity_group in mapped_final_integrated_schema_mappings:
            if "None" in entity_group:
                for table_name in mapped_final_integrated_schema_mappings[entity_group]:
                    for table_split in mapped_final_integrated_schema_mappings[entity_group][table_name]:
                        predicted_column_mappings = mapped_final_integrated_schema_mappings[entity_group][table_name][table_split]["column_mappings_to_integrated_schema"]
                        overall_fps += len(predicted_column_mappings)
                continue
            
            for table_name in mapped_final_integrated_schema_mappings[entity_group]:        
                for table_split in mapped_final_integrated_schema_mappings[entity_group][table_name]:
                    table_gt_column_mappings = {column_name: mapped_attribute for column_name, mapped_attribute in self.overall_gt_mappings[table_name].items() if mapped_attribute in self.gt_final_integrated_schema[entity_group]["attributes"] }
                    predicted_column_mappings = mapped_final_integrated_schema_mappings[entity_group][table_name][table_split]["column_mappings_to_integrated_schema"]
                    # Check each mapping
                    for column_name, mapped_attribute in predicted_column_mappings.items():
                        if column_name in table_gt_column_mappings and table_gt_column_mappings[column_name] == mapped_attribute:
                            # Correct mapping
                            overall_tps += 1
                        elif column_name in table_gt_column_mappings:
                            # Else incorrect mapping and a false positive for the attribute mapped to this column
                            overall_fps += 1
                            overall_fns += 1 # Also a false negative for the gt mapping that was not predicted
                        else:
                            # Column was not in gt mappings, so it's a false positive
                            overall_fps += 1
                    # Check the other way around for false negatives
                    for column_name, mapped_attribute in table_gt_column_mappings.items():
                        if column_name not in predicted_column_mappings:
                            overall_fns += 1
                
                # for attribute in mapped_final_integrated_schema_mappings[entity_group][table_name][table_split]["new_attributes"]:
                #     if attribute not in self.gt_final_integrated_schema[entity_group]["attributes"]:
                #         overall_fps += 1
                #         print("new attribute:", attribute)
                #         # print(f"False positive new attribute '{attribute}' added to entity group '{entity_group}' for table '{table_name}', but it does not exist in GT")
                #     else:
                #         overall_tps += 1
                
            # Any missed tables from the GT that should have this entity?
            for table_name in self.overall_gt_mappings:
                table_gt_column_mappings = {column_name: mapped_attribute for column_name, mapped_attribute in self.overall_gt_mappings[table_name].items() if mapped_attribute in self.gt_final_integrated_schema[entity_group]["attributes"] }
                if len(table_gt_column_mappings):
                    if table_name not in mapped_final_integrated_schema_mappings[entity_group]:
                        # All mappings for this table are false negatives
                        overall_fns += len(table_gt_column_mappings)
                        # print(f"Table '{table_name}' was missed in the predictions for entity group '{entity_group}'. All mappings for this table are false negatives.")
        # For missed entities
        for entity_group in self.gt_final_integrated_schema:
            if entity_group not in mapped_final_integrated_schema_mappings:
                # print(f"Entity group '{entity_group}' was missed in the predictions. All mappings for this group are false negatives.")
                for table_name in self.overall_gt_mappings:
                    for column_name, mapped_attribute in self.overall_gt_mappings[table_name].items():
                        if mapped_attribute in self.gt_final_integrated_schema[entity_group]["attributes"]:
                            overall_fns += 1

        # Lost ids are false negatives
        if self.splitting_after_integration:
            added_ids = [(entity_group, attribute) for entity_group in self.gt_final_integrated_schema for attribute in self.gt_final_integrated_schema[entity_group]["attributes"] if "id" in attribute.lower() and attribute not in self.gt_mappings["all"]["attributes"]]
        else:
            added_ids = [(entity_group, attribute) for entity_group in self.gt_final_integrated_schema for attribute in self.gt_final_integrated_schema[entity_group]["attributes"] if "id" in attribute.lower() and attribute not in self.gt_mappings[entity_group]["attributes"]]
        
        id_errors = [(error_label.split(".")[0], error_label.split(".")[1]) for error_label, errors in self.eval_results["0"]["final_integration_phase"]["eval_results_attributes"]["errors_per_class"].items() if "id" in error_label.lower() and errors > 0 and any(error_label.split(".")[0] in added_id[0] and error_label.split(".")[1] == added_id[1] for added_id in added_ids)]

        overall_mappings = sum([1 for entity_group in self.gt_final_integrated_schema for table_name in self.overall_gt_mappings for column_name, mapped_attribute in self.overall_gt_mappings[table_name].items() if mapped_attribute in self.gt_final_integrated_schema[entity_group]["attributes"] ])
        overall_predicted_mappings = sum([len(mapped_final_integrated_schema_mappings[entity_group][table_name][table_split]["column_mappings_to_integrated_schema"]) for entity_group in mapped_final_integrated_schema_mappings for table_name in mapped_final_integrated_schema_mappings[entity_group] for table_split in mapped_final_integrated_schema_mappings[entity_group][table_name]])
        
        added_fns = 0
        gt_ids = 0
        table_names_to_entity_groups = {}
        # correct_ids = 0
        if len(added_ids):
            for entity_group in self.gt_final_integrated_schema:
                table_names_to_entity_groups[entity_group] = []
                if any(entity_group in added_id[0] for added_id in added_ids):
                    for table_name, mappings in self.overall_gt_mappings.items():
                        # if any(mapped_attribute in self.gt_final_integrated_schema[entity_group]["attributes"] for mapped_attribute in mappings.values()):
                        if len([mapped_attribute for mapped_attribute in mappings.values() if mapped_attribute in self.gt_final_integrated_schema[entity_group]["attributes"]])>=2:
                            table_names_to_entity_groups[entity_group].append(table_name)
                            # A GT id, add for one table as many ids as there are for this entity group
                            gt_ids += len([added_id for added_id in added_ids if entity_group in added_id[0]])
                            # Check if this table exists in the predicted mappings otherwise FN for the id
                            if table_name not in mapped_final_integrated_schema_mappings.get(entity_group, {}):
                                added_fns += len([added_id for added_id in added_ids if entity_group in added_id[0]])
                                # print(f"Missed table '{table_name}' of entity group '{entity_group}' with an id attribute leads to {len([added_id for added_id in added_ids if entity_group in added_id[0]])} false negative mappings for this table.")

            
            for entity_group, tables in mapped_final_integrated_schema_mappings.items():
                # Calculate bad mappings (FNs) if an id attribute was missed for an entity group
                number_of_table_splits = sum([len(mapped_final_integrated_schema_mappings[entity_group][table_name]) for table_name in mapped_final_integrated_schema_mappings[entity_group] if table_name in table_names_to_entity_groups.get(entity_group, [])])
                ids_for_entity_group = len([added_id for added_id in added_ids if entity_group in added_id[0]])
                error_ids_for_entity_group = len([id_error for id_error in id_errors if entity_group in id_error[0]])

                if any(entity_group in id_error[0] for id_error in id_errors):
                    # Missed id attribute, it is a wrong mapping for each table in this group
                    added_fns += number_of_table_splits*error_ids_for_entity_group
                    # print(f"Missed id attribute for entity group '{entity_group}' leads to {number_of_table_splits} false negative mappings for this group.")
                
                if any(entity_group in added_id[0] for added_id in added_ids):
                    correct_ids = len([added_id for added_id in added_ids if entity_group in added_id[0] and not any(id_error for id_error in id_errors if id_error[0] == added_id[0] and id_error[1] == added_id[1])])
                    # This entity group has an added id and was found correctly add to the tps and to the overall predicted mappings
                    # print("Correct ids", correct_ids, "leads to", number_of_table_splits*correct_ids, "true positives for this group.")
                    if correct_ids:
                        overall_tps += number_of_table_splits*correct_ids
                        # correct_ids += number_of_table_splits*correct_ids
                        print( mapped_final_integrated_schema_mappings[entity_group].keys())
                        print(number_of_table_splits*correct_ids)
                        overall_predicted_mappings += number_of_table_splits*correct_ids

            # If an entity is missed also their ids can be missed
            # for entity_group in self.gt_final_integrated_schema:
            #     if entity_group not in mapped_final_integrated_schema_mappings:
            #         # If an added id exists for this entity, for each table in the gt this is a false negative
            #         if any(entity_group in added_id[0] for added_id in added_ids):
            #             if self.splitting_after_integration:
            #                 number_of_table_splits = len(self.gt_mappings["all"]["column_mappings_by_table"])
            #             else:
            #                 number_of_table_splits = len(self.gt_mappings[entity_group]["column_mappings_by_table"])
            #             added_fns += number_of_table_splits
            #             print(f"Missed entity group '{entity_group}' with an id attribute leads to {number_of_table_splits} false negative mappings for this group.")

        assert(overall_tps + overall_fns + added_fns == overall_mappings+gt_ids)
        assert(overall_tps + overall_fps == overall_predicted_mappings)
        # Save mappings results
        eval_results = {"tps": overall_tps, "fps": overall_fps, "fns": overall_fns+added_fns, "overall_recall": overall_tps/(overall_tps + overall_fns+added_fns), "overall_precision": overall_tps/(overall_tps + overall_fps)}
        eval_results["overall_f1"] = 2*eval_results["overall_precision"]*eval_results["overall_recall"] / (eval_results["overall_precision"] + eval_results["overall_recall"]) if (eval_results["overall_precision"] + eval_results["overall_recall"])>0 else 0
        # Save evaluation results
        self.eval_results[run_nr]["final_integration_mappings"] = eval_results
        # Save for average calculation
        self.overall_metrics.setdefault("final_integration_mappings", [])
        self.overall_metrics["final_integration_mappings"].append(eval_results)

    def run_evaluation(self):
        SEQUENCE_OF_STEPS_TO_EVALUATION = {
            "normalization_phase": self.evaluate_detect_tables_phase,
            "detect_tables_phase": self.evaluate_detect_tables_phase,
            "schema_matching_phase": self.evaluate_schema_matching_phase,
            "grouping_phase": self.evaluate_grouping_phase,
            "schema_integration_phase": self.evaluate_schema_integration_phase,
            "final_integration_phase": self.evaluate_final_integration,
        }
        
        for run in [str(i) for i in range(self.num_runs)]:
            self.eval_results[run] = {}
            print(f"\n\nEvaluating run number {int(run)+1}")
            for phase in self.sequence_of_phases:
                print(f"Evaluating phase: {phase}")
                SEQUENCE_OF_STEPS_TO_EVALUATION[phase](run)
                if phase == "schema_matching_phase":
                    # Also evaluate the version with removed correspondences for analysis
                    self.evaluate_schema_matching_phase(run, remove_correspondences=True)
        # Calculate average metrics
        self.average_metrics = self.calculate_average_metrics()
        self.std_dev_metrics = self.calculate_standard_deviation()
    
    def calculate_average_metrics(self):
        average_metrics = {}
        for phase in self.overall_metrics:
            average_metrics[phase] = {}
            for metric in self.overall_metrics[phase][0]:
                # Skip tps, fps, fns
                if "overall" not in metric:
                    continue
                average_metrics[phase][metric] = sum([run[metric] for run in self.overall_metrics[phase]]) / len(self.overall_metrics[phase])
        return average_metrics

    def calculate_standard_deviation(self):
        std_dev_metrics = {}
        for phase in self.overall_metrics:
            std_dev_metrics[phase] = {}
            for metric in self.overall_metrics[phase][0]:
                # Skip tps, fps, fns
                if "overall" not in metric:
                    continue
                std_dev = np.std([run[metric] for run in self.overall_metrics[phase]])
                std_dev_metrics[phase][metric] = std_dev
        return std_dev_metrics

    def save_results(self):
        # If the evaluation file does not exist
        if f"{self.log_file_name.replace('logs/', '')}_evaluation.json" not in os.listdir("evaluation_json/"):
            eval_results_content = {
                "run_full_results": self.eval_results,
                "average_metrics": self.average_metrics
            }
            sienna.save(eval_results_content, f"{self.log_file_name.replace('logs/', 'evaluation_json/')}_evaluation.json")

            # Phase names
            phases = ['detect_tables_phase_entity', 'detect_tables_phase_attributes', 'schema_matching_phase', 'schema_matching_phase_removed', 'grouping_phase', 'schema_integration_phase', 'final_integration_entity', 'final_integration_attributes', 'final_integration_mappings']
            phases_results = [ f"{self.average_metrics[phase]['overall_f1']:.4f}" if phase in self.average_metrics else "-" for phase in phases ]

            # Write in a csv file for F1
            csv_file = f"evaluation_summary/{self.folder}_evaluation_f1_summary.csv"
            file_exists = os.path.isfile(csv_file)
            row = [self.benchmark, self.run_info["model_name"], self.run_info["prompt_name"], self.run_info["reasoning"], self.run_info["demonstration"], self.run_info["num_runs"], self.run_info["other_parameters"]["self-consistency"], self.run_info["other_parameters"]["generated_knowledge"]]+phases_results+["->".join(self.run_info["sequence_of_phases"]), self.log_file_name] #, self.other_parameters
            writing_mode = "a" if file_exists else "w"
            with open(csv_file, writing_mode, newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(["benchmark", "model_name", "prompt_name", "reasoning", "demonstration", "num_runs", "self_consistency", "generated_knowledge"] + phases + ["sequence_of_phases", "log_file_name"]) #, "other_parameters"
                writer.writerow(row)

            # Write in a csv file for Recall
            phases_results = [ f"{self.average_metrics[phase]['overall_recall']:.4f}" if phase in self.average_metrics else "-" for phase in phases ]
            csv_file = f"evaluation_summary/{self.folder}_evaluation_recall_summary.csv"
            file_exists = os.path.isfile(csv_file)
            row = [self.benchmark, self.run_info["model_name"], self.run_info["prompt_name"], self.run_info["reasoning"], self.run_info["demonstration"], self.run_info["num_runs"], self.run_info["other_parameters"]["self-consistency"], self.run_info["other_parameters"]["generated_knowledge"]]+phases_results+["->".join(self.run_info["sequence_of_phases"]), self.log_file_name] #, self.other_parameters
            writing_mode = "a" if file_exists else "w"
            with open(csv_file, writing_mode, newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(["benchmark", "model_name", "prompt_name", "reasoning", "demonstration", "num_runs", "self_consistency", "generated_knowledge"] + phases + ["sequence_of_phases", "log_file_name"]) #, "other_parameters"
                writer.writerow(row)

            # Write in a csv file for Precision
            phases_results = [ f"{self.average_metrics[phase]['overall_precision']:.4f}" if phase in self.average_metrics else "-" for phase in phases ]
            csv_file = f"evaluation_summary/{self.folder}_evaluation_precision_summary.csv"
            file_exists = os.path.isfile(csv_file)
            row = [self.benchmark, self.run_info["model_name"], self.run_info["prompt_name"], self.run_info["reasoning"], self.run_info["demonstration"], self.run_info["num_runs"], self.run_info["other_parameters"]["self-consistency"], self.run_info["other_parameters"]["generated_knowledge"]]+phases_results+["->".join(self.run_info["sequence_of_phases"]), self.log_file_name] #, self.other_parameters
            writing_mode = "a" if file_exists else "w"
            with open(csv_file, writing_mode, newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(["benchmark", "model_name", "prompt_name", "reasoning", "demonstration", "num_runs", "self_consistency", "generated_knowledge"] + phases + ["sequence_of_phases", "log_file_name"]) #, "other_parameters"
                writer.writerow(row)