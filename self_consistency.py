import sienna
import pandas as pd
import random
import copy


class SelfConsistency:
    def __init__(self, num_runs: int, file_names: list, benchmark: str, folder_name: str = None):
        self.benchmark = benchmark
        self.folder_name = folder_name
        self.num_runs = num_runs
        self.minimum_occurence = int(num_runs/2)+1
        self.results = {}
        self.self_consistency_results = {}
        self.file_names = file_names

    def run_schema_matching_self_consistency(self, phase_name: str, **kwargs):
        # Correspondences will be at the same phase
        schema_matching_results = self.results[phase_name]
        # Hold the matches that show up in more than half the predictions
        schema_matching_correspondences_occurences = {}

        for run_prediction in schema_matching_results.values():
            for source_table, source_correspondences in run_prediction.items():
                for target_table, target_correspondences in source_correspondences.items():
                    for source_attr, target_attr in target_correspondences.items():
                        schema_matching_correspondences_occurences.setdefault(source_table, {}).setdefault(target_table, {}).setdefault(source_attr, {}).setdefault(target_attr, 0)
                        # Add an occurence for this source and target attribute pair
                        schema_matching_correspondences_occurences[source_table][target_table][source_attr][target_attr] += 1
        
        # Create the final consistent schema matching correspondences
        consistent_schema_matching_correspondences = {}

        for source_table, source_correspondences in schema_matching_correspondences_occurences.items():
            for target_table, target_correspondences in source_correspondences.items():
                for source_attr, target_attr_occurence in target_correspondences.items():
                    for target_attr, occurence in target_attr_occurence.items():
                        if occurence >= self.minimum_occurence:
                            # If correspondence has been generated more than half the runs add to list
                            consistent_schema_matching_correspondences.setdefault(source_table, {}).setdefault(target_table, {}).setdefault(source_attr, {})
                            consistent_schema_matching_correspondences[source_table][target_table][source_attr] = target_attr
        
        # return consistent_schema_matching_correspondences
        self.self_consistency_results[phase_name] = consistent_schema_matching_correspondences

    def run_table_splitting_self_consistency(self, phase_name: str, integrated_attributes: dict, **kwargs):
        splitting_results = self.results[phase_name]
        # For each table, count attribute co-occurences in the splits
        consistent_table_splits = {}
        # for table_name in self.file_names:
        for table_name in splitting_results[0].keys():
            attributes_co_occurrence = {}
            overlapping_attributes = {}
            for run in range(3):
                # Read table to get attribute names or if the full table = integrated attributes
                if table_name in self.file_names:
                    table_df = pd.read_csv(f"data/selected-tables/{self.benchmark}/{self.folder_name}/{table_name}")
                    attribute_names = table_df.columns.tolist()
                else:
                    attribute_names = list(integrated_attributes["all"].values())
                    # attribute_names = list(self.results_parameters["integrated_attributes"]["all"].values())
                
                for attribute in attribute_names:
                    if attribute not in attributes_co_occurrence:
                        attributes_co_occurrence[attribute] = {}
                    # print(f"Run {run}, Table {table_name}, Attribute {attribute}")
                    for split, split_info in splitting_results[run][table_name].items():
                        if attribute in split_info["attributes"]:
                            for co_attribute in split_info["attributes"]:
                                if co_attribute != attribute:
                                    if co_attribute not in attributes_co_occurrence[attribute]:
                                        attributes_co_occurrence[attribute][co_attribute] = 0
                                    attributes_co_occurrence[attribute][co_attribute] += 1
                    # Record attributes that occur in more than 1 split and how many times this happens
                    if len([split for split, split_info in splitting_results[run][table_name].items() if attribute in split_info["attributes"]]) > 1:
                        if attribute not in overlapping_attributes:
                            overlapping_attributes[attribute] = 0
                        overlapping_attributes[attribute] += 1
                
            final_overlapping_attributes = {attribute: count for attribute, count in overlapping_attributes.items() if count >= self.minimum_occurence} # more strict

            # Remove overlapping attributes from co-occurrence counts and add them at the end
            copies_of_coocurrences = {}
            for overlapping_attribute in final_overlapping_attributes:
                if overlapping_attribute in attributes_co_occurrence:
                    copies_of_coocurrences[overlapping_attribute] = copy.deepcopy(attributes_co_occurrence[overlapping_attribute])
                    del attributes_co_occurrence[overlapping_attribute]

            for overlapping_attribute in copies_of_coocurrences:
                attributes_co_occurrence[overlapping_attribute] = copies_of_coocurrences[overlapping_attribute]

            # Based on the co-occurrence counts, determine which attributes are most often split together
            attribute_groups = []
            for attribute, co_occurrences in attributes_co_occurrence.items():
                for co_attribute, count in co_occurrences.items():
                    # Does this co_attribute show up in multiple splits? Don't use it to determine the splits
                    if co_attribute in overlapping_attributes:
                        continue
                    
                    # If both attributes are not in any group yet and they co-occur above the threshold, group them together
                    if not any(attribute in group for group in attribute_groups) and not any(co_attribute in group for group in attribute_groups):
                        if count >= self.minimum_occurence:
                            attribute_groups.append([attribute, co_attribute])
                    # If the co attribute is already in a group and the current attribute co-occurs with it above the threshold, add the current attribute to that group
                    elif any(co_attribute in group for group in attribute_groups) and not any(attribute in group for group in attribute_groups):
                        for group in attribute_groups:
                            if co_attribute in group and count >= self.minimum_occurence:
                                group.append(attribute)
                                break
                    # If the current attribute is already in a group and the co attribute co-occurs with it above the threshold, add the co attribute to that group
                    elif any(attribute in group for group in attribute_groups) and not any(co_attribute in group for group in attribute_groups):
                        for group in attribute_groups:
                            if attribute in group and count >= self.minimum_occurence:
                                group.append(co_attribute)
                                break
                    else:
                        # Find the groups that the current attribute and co attribute belong to (if any)
                        attribute_group = None
                        co_attribute_group = None   
                        for group in attribute_groups:
                            if attribute in group:
                                attribute_group = group
                            if co_attribute in group:
                                co_attribute_group = group
                        
                        # if attribute_group != co_attribute_group and count >= self.minimum_occurence:
                            # print("Not same! Attribute occurs in two groups")
                        # if count >= self.minimum_occurence:
                        if count >= self.num_runs: # only if it appears in all runs, to be more strict
                            # Add current attribute to co attribute group
                            co_attribute_group.append(attribute)
            
            # Either select the highest co-occuring attribute and put it into that group or randomly pick an attribute from the co-occuring attributes and put it into that group. Or only group attributes together if they co-occur in all runs, to be more strict.
            for attribute in attribute_names:
                if not any(attribute in group for group in attribute_groups):
                    if not attributes_co_occurrence[attribute]:
                        # Missing attribute overall
                        print(f"Attribute {attribute} is removed from the original table {table_name}")
                        continue
                    print(f"Attribute {attribute} is not in any group for table {table_name}")
                    co_attributes = attributes_co_occurrence[attribute]
                    # Get the co-occuring attribute with the highest co-occurence count
                    # Select highest number of co-occurences
                    maximum_co_occurence_number = max(co_attributes.values())
                    # Select randomly one of the co-occuring attributes that have the highest co-occurence count
                    co_attributes_with_maximum_co_occurence = [co_attribute for co_attribute, count in co_attributes.items() if count == maximum_co_occurence_number]
                    selected_co_attribute = random.choice(co_attributes_with_maximum_co_occurence)
                    # Find the group of this attribute and add the current attribute
                    for group in attribute_groups:
                        if selected_co_attribute in group:
                            group.append(attribute)
                            break
            
            attribute_groups = [list(set(group)) for group in attribute_groups]
            # Naming
            # For the naming, check if any run has the exact attributes grouped and pick that as name otherwise pick the name where the most overlap exists
            group_names = {f"Group_{gi+1}": {} for gi, group in enumerate(attribute_groups)}
            group_coocurrence_for_naming = {f"Group_{gi+1}": {"name": None, "overlap": 0} for gi, group in enumerate(attribute_groups)}
            
            for run in range(3):
                for split, split_info in splitting_results[run][table_name].items():
                    # print(split)
                    split_attributes = split_info["attributes"]
                    
                    for gi, group in enumerate(attribute_groups):
                        # If naming has not been assigned to this group yet
                        if not group_names[f"Group_{gi+1}"]:
                            # Check if this split has an exact match to the group attributes
                            if set(group) == set(split_attributes):
                                # If yes, assign this split name as name for the group
                                group_names[f"Group_{gi+1}"] = {"name": split, "attributes": group}
                                break
                            else:
                                # Other wise check overlap and record highest overlap for naming
                                overlap = len(set(group).intersection(set(split_attributes)))
                                if overlap > group_coocurrence_for_naming[f"Group_{gi+1}"]["overlap"]:
                                    group_coocurrence_for_naming[f"Group_{gi+1}"] = {"name": split, "overlap": overlap, "attributes": group}

            final_named_groups = {}
            for gi, group in enumerate(attribute_groups):
                    if group_names[f"Group_{gi+1}"]:
                        final_named_groups[group_names[f"Group_{gi+1}"]["name"]] = {"attributes": group_names[f"Group_{gi+1}"]["attributes"]}
                    else:
                        # Assign the highest overlap for this group
                        final_named_groups[group_coocurrence_for_naming[f"Group_{gi+1}"]["name"]] = {"attributes": group_coocurrence_for_naming[f"Group_{gi+1}"]["attributes"]}
                
            consistent_table_splits[table_name] = final_named_groups
        self.self_consistency_results[phase_name] = consistent_table_splits
    
    def run_grouping_self_consistency(self, phase_name: str, **kwargs):
        grouping_results = self.results[phase_name]

        split_cooccurences = {}
        # For each table split, count co-occurences of other table splits in the same group
        for group_result in grouping_results.values():
            for group in group_result.values():
                for table_split in group:
                    split_cooccurences.setdefault(table_split, {})
                    # Inner loop for same group
                    for other_table_split in group:
                        if other_table_split != table_split:
                            split_cooccurences[table_split].setdefault(other_table_split, 0)
                            split_cooccurences[table_split][other_table_split] += 1

        split_groups = []
        for table_split, cooccurences in split_cooccurences.items():
            for other_table_split, count in cooccurences.items():
                # If table split and the other table split are not added to any group
                if not any(table_split in group for group in split_groups) and not any(other_table_split in group for group in split_groups):
                    if count >= self.minimum_occurence:
                        split_groups.append([table_split, other_table_split])
                elif not any(table_split in group for group in split_groups):
                    # Other table split is already in a group
                    if count >= self.minimum_occurence:
                        # Add table split to the group of the other table split
                        for group in split_groups:
                            if other_table_split in group:
                                group.append(table_split)
                                break
                elif not any(other_table_split in group for group in split_groups):
                    # Table split is already in a group
                    if count >= self.minimum_occurence:
                        # Add other table split to the group of the table split
                        for group in split_groups:
                            if table_split in group:
                                group.append(other_table_split)
                                break
                else:
                    # If both are already in groups, check if they are in the same group, if not and count is above threshold, merge the groups
                    if count >= self.minimum_occurence:
                        table_split_group = None
                        other_table_split_group = None
                        for group in split_groups:
                            if table_split in group:
                                table_split_group = group
                            if other_table_split in group:
                                other_table_split_group = group
                        if table_split_group != other_table_split_group:
                            # Merge groups
                            merged_group = list(set(table_split_group + other_table_split_group))
                            split_groups.remove(table_split_group)
                            split_groups.remove(other_table_split_group)
                            split_groups.append(merged_group)
        
        # Add as individual groups the table splits that do not have any co-occurences
        for table_split, cooccurences in split_cooccurences.items():
            if not cooccurences:
                split_groups.append([table_split])
            # If a splits co-occurences are all under the threshold, pick the highest co-occurence and put it into that group (randomly pick from the highest co-occurence)
            elif cooccurences and all(count < self.minimum_occurence for count in cooccurences.values()):
                maximum_co_occurence_number = max(cooccurences.values())
                co_splits_with_maximum_co_occurence = [co_split for co_split, count in cooccurences.items() if count == maximum_co_occurence_number]
                selected_co_split = random.choice(co_splits_with_maximum_co_occurence)
                for group in split_groups:
                    if selected_co_split in group:
                        group.append(table_split)
                        break
        
        # Choose group name based on highest co-occurence with the splits in the group
        group_names = {f"Group_{gi+1}": {} for gi, group in enumerate(split_groups)}
        group_coocurrence_for_naming = {f"Group_{gi+1}": {"name": None, "overlap": 0} for gi, group in enumerate(split_groups)}
        # group_coocurrence_for_naming = {}
        for run in range(3):
            for group_table, tables in grouping_results[run].items():
                for gi, group in enumerate(split_groups):
                    # If naming has not been assigned to this group yet
                    if not group_names[f"Group_{gi+1}"]:
                        # Check if this split has an exact match to the group attributes
                        if set(group) == set(tables):
                            # If yes, assign this split name as name for the group
                            group_names[f"Group_{gi+1}"] = {"name": group_table, "tables": group}
                            break
                        else:
                            # Other wise check overlap and record highest overlap for naming
                            overlap = len(set(group).intersection(set(tables)))
                            if overlap > group_coocurrence_for_naming[f"Group_{gi+1}"]["overlap"]:
                                group_coocurrence_for_naming[f"Group_{gi+1}"] = {"name": group_table, "overlap": overlap, "tables": group}

        final_named_groups = {}
        for gi, group in enumerate(split_groups):
                if group_names[f"Group_{gi+1}"]:
                    final_named_groups[group_names[f"Group_{gi+1}"]["name"]] = group_names[f"Group_{gi+1}"]["tables"]
                else:
                    # Assign the highest overlap for this group
                    final_named_groups[group_coocurrence_for_naming[f"Group_{gi+1}"]["name"]] = group_coocurrence_for_naming[f"Group_{gi+1}"]["tables"]
        
        # TODO: What about not above the threshold?
        self.self_consistency_results[phase_name] = final_named_groups

    def run_schema_integration_self_consistency(self, phase_name: str, detected_tables_grouped_and_mapped: dict, **kwargs):
        integration_results = self.results[phase_name]
        if "schema_matching_phase" in self.results:
            # For each entity, choose the highest occurence across the runs
            consistent_schema_integration_results = {}
            for entity in integration_results[0].keys():
                attribute_occurences = {}
                for run in range(3):
                    for attribute_group, attribute in integration_results[run][entity].items():
                        attribute_occurences.setdefault(attribute_group, {})
                        if attribute not in attribute_occurences[attribute_group]:
                            attribute_occurences[attribute_group][attribute] = 0
                        attribute_occurences[attribute_group][attribute] += 1
                # Choose the attribute that occurs the most
                for attribute_group, attribute_occurences in attribute_occurences.items():
                    for attribute, occurences in attribute_occurences.items():
                        if occurences >= 2: # If it occurs in at least 2 runs, consider it consistent
                            consistent_schema_integration_results.setdefault(entity, {})
                            consistent_schema_integration_results[entity][attribute_group] = attribute
                    if not any(occurences >= 2 for occurences in attribute_occurences.values()):
                        # If no consistent attribute, choose the one with the highest occurences
                        most_common_attribute = max(attribute_occurences, key=attribute_occurences.get)
                        consistent_schema_integration_results.setdefault(entity, {})
                        consistent_schema_integration_results[entity][attribute_group] = most_common_attribute
        else:
            # Implicit schema matching setup
            # For each table, choose the mapping that appears the most
            consistent_schema_integration_results = {}
            for entity in detected_tables_grouped_and_mapped:
                for table_name in detected_tables_grouped_and_mapped[entity]:
                    for table_split in detected_tables_grouped_and_mapped[entity][table_name].values():
                        mapping_counts = {}
                        for run in range(self.num_runs):
                            mappings = integration_results[run][entity]["mappings_of_old_attributes"][table_split["table_name"]]
                            for original_attr, mapped_attr in mappings.items():
                                mapping_counts.setdefault(original_attr, {}).setdefault(mapped_attr, 0)
                                mapping_counts[original_attr][mapped_attr] += 1

                        # For this table, add the mappings that appear the most to the consistent mappings
                        consistent_schema_integration_results.setdefault(entity, {}).setdefault(table_split["table_name"], {})
                        for original_attr, mapped_attrs in mapping_counts.items():
                            most_common_mapped_attr = max(mapped_attrs, key=mapped_attrs.get)
                            consistent_schema_integration_results[entity][table_split["table_name"]][original_attr] = most_common_mapped_attr
        
        self.self_consistency_results[phase_name] = consistent_schema_integration_results


    def run_final_integration_self_consistency(self, phase_name: str, schema_integration_results: dict, **kwargs):
        final_integration_results = self.results[phase_name]

        # Check additions and deletions of attributes for each entity
        entity_additions = {}
        entity_deletions = {}
        for run in range(3):
            for entity, entity_info in final_integration_results[run].items():
                if entity not in schema_integration_results:
                    print("Added entity:", entity)
                    if len([entity for run_results in final_integration_results.values() if entity in run_results.keys()])>=self.minimum_occurence:
                        # print(f"Run {run}: Entity '{entity}' was added in the final integration phase.")
                        # Record entity additions
                        for attribute in entity_info["attributes"]:
                            entity_additions.setdefault(entity, {}).setdefault(attribute, 0)
                            entity_additions[entity][attribute] += 1
                else:
                    for attribute in entity_info["attributes"]:
                        if attribute not in schema_integration_results[entity]:
                            # print(f"Run {run}: Attribute '{attribute}' for entity '{entity}' was added in the final integration phase.")
                            # Record attribute additions
                            entity_additions.setdefault(entity, {}).setdefault(attribute, 0)
                            entity_additions[entity][attribute] += 1
                
                    for attribute in schema_integration_results[entity]:
                        if attribute not in final_integration_results[run][entity]["attributes"]:
                            # Record attribute deletions
                            entity_deletions.setdefault(entity, {}).setdefault(attribute, 0)
                            entity_deletions[entity][attribute] += 1

        final_consistent_schema = {}
        foreign_key_occurences = {}

        for entity, attributes in schema_integration_results.items():
            # if not any(entity in run_results.keys() for run_results in final_integration_results.values()):
            if len([entity for run_results in final_integration_results.values() if entity not in run_results.keys()])>=self.minimum_occurence:
                # If the entity is not present in most of the runs, skip it
                print(entity)
                entity_deletions[entity] = {attribute: len([entity for run_results in final_integration_results.values() if entity not in run_results.keys()])  for attribute in attributes}  # Mark all attributes as deleted in all runs

            # All attributes that are not in deletions add from the previously found merged schemas
            final_attributes = [attribute for attribute in attributes if attribute not in entity_deletions.get(entity, {}) or entity_deletions[entity][attribute] < 2] # if attribute is deleted in more than 2 runs, don't add

            # Add attributes that are added on half or more of the runs
            for attribute, count in entity_additions.get(entity, {}).items():
                if count >= self.minimum_occurence:  # Added in at least 2 out of 3 runs
                    final_attributes.append(attribute)

            if final_attributes:
                final_consistent_schema[entity] = {"attributes": final_attributes}

        for entity, attributes in entity_additions.items():
            if entity not in final_consistent_schema and entity not in schema_integration_results:
                final_consistent_schema.setdefault(entity, {"attributes": []})
                for attribute, count in attributes.items():
                    if count >= self.minimum_occurence:  # Added in at least 2 out of 3 runs
                        final_consistent_schema[entity]["attributes"].append(attribute)
        # try:
        for entity, entity_info in final_consistent_schema.items():
            # For foreign keys too, count the occurences of the attributes that show up in the final consitent schema
            for run in range(3):
                if entity in final_integration_results[run]:
                    for attribute, reference in final_integration_results[run][entity].get("foreign_keys", {}).items():
                        if attribute in entity_info["attributes"]:
                            for ft, fa in reference.items():
                                # print(ft, fa)
                                if ft in final_consistent_schema and fa in final_consistent_schema[ft]["attributes"]:
                                    # Count this occurence of the foreign key reference
                                    foreign_key_occurences.setdefault(entity, {}).setdefault(attribute, {}).setdefault(ft, {}).setdefault(fa, 0)
                                    foreign_key_occurences[entity][attribute][ft][fa] += 1
                else:
                    print(f"Entity {entity} was probably misgenerated in one of the runs.")
        # except Exception as e:
        #     pdb.set_trace()
            
        # For each entity, add foreign keys that show up in at least 2 runs
        for entity, entity_info in final_consistent_schema.items():
            for attribute, references in foreign_key_occurences.get(entity, {}).items():
                for ft, fa_dict in references.items():
                    for fa, count in fa_dict.items():
                        if count >= self.minimum_occurence:  # Foreign key reference appears in at least 2 runs
                            entity_info.setdefault("foreign_keys", {}).setdefault(attribute, {})[ft] = fa
            if not len(foreign_key_occurences.get(entity, {})):
                entity_info["foreign_keys"] = {}
        
        self.self_consistency_results[phase_name] = final_consistent_schema