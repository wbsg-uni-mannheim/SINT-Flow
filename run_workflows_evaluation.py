import pdb
from evaluation import SchemaIntegrationEvaluation
import sienna
from utils import format_time
import os

if __name__ == "__main__":
    # Specify benchmark
    # benchmark = "Real Benchmark"
    benchmark = "SINT-Benchmark"
    # benchmark = "GOBY"

    # Run evaluation
    for file_name in os.listdir("logs/"):
        # if "GOBY_100" in file_name:
        # if "GOBY_" in file_name:
        log_file_name = "logs/" + file_name.replace("_evaluation.json", "").replace(".json", "")
        schema_evaluator = SchemaIntegrationEvaluation(tables_path=f"data/selected-tables/{benchmark}", log_file_name=log_file_name)
        schema_evaluator.run_evaluation()
        schema_evaluator.save_results(folder_name="evaluation_json", summary_folder_name="evaluation_summary")

        # To print the time taken for the workflow to run:
        # use_case_run_info = {}
        # for run in schema_evaluator.all_predictions:
        #     for phase in schema_evaluator.all_predictions[run]["time_info"]:
        #         if phase not in use_case_run_info:
        #             use_case_run_info[phase] = []
        #         use_case_run_info[phase].append(schema_evaluator.all_predictions[run]["time_info"][phase])
        # print(f"Use case run info for {log_file_name}:")
        # print("Overall time", format_time(sum([sum(use_case_run_info[phase]) for phase in use_case_run_info])), "seconds")

        # For evaluating specific workflows, specfy sequence of phases
        # if schema_evaluator.run_info["sequence_of_phases"] == [ "detect_tables_phase", "schema_matching_phase", "grouping_phase", "schema_integration_phase", "final_integration_phase"]:# and schema_evaluator.self_consistency:
        # if schema_evaluator.run_info["sequence_of_phases"] == [ "detect_tables_phase", "grouping_phase", "schema_integration_phase", "final_integration_phase"]: # and schema_evaluator.self_consistency
        # if schema_evaluator.run_info["sequence_of_phases"] == ["schema_matching_phase", "schema_integration_phase", "detect_tables_phase", "final_integration_phase"]:
        # if schema_evaluator.run_info["sequence_of_phases"] == ["schema_matching_phase"]:
            # schema_evaluator = SchemaIntegrationEvaluation(tables_path=f"data/selected-tables/{benchmark}", log_file_name=log_file_name)
            # schema_evaluator.run_evaluation()
            # schema_evaluator.save_results(folder_name="evaluation_json", summary_folder_name="evaluation_summary")
    
    # Aggregate evaluation results for all runs of a specific model and prompt
    all_eval_results = {}
    aggregated_eval_results = {}
    for file_name in os.listdir("evaluation_summary/"):
        if "GOBY_" in file_name:
            continue
        eval_file = sienna.load(f"evaluation_summary/{file_name}")
        for key in eval_file:
            if key not in all_eval_results:
                all_eval_results[key] = {}
            for phase in eval_file[key]:
                if phase not in all_eval_results[key]:
                    all_eval_results[key][phase] = {}
                for metric in eval_file[key][phase]:
                    if metric not in all_eval_results[key][phase]:
                        all_eval_results[key][phase][metric] = []
                    all_eval_results[key][phase][metric].append(eval_file[key][phase][metric])

    for key in all_eval_results:
        if key not in aggregated_eval_results:
            aggregated_eval_results[key] = {}
        for phase in all_eval_results[key]:
            if phase not in aggregated_eval_results[key]:
                aggregated_eval_results[key][phase] = {}
            for metric in all_eval_results[key][phase]:
                aggregated_eval_results[key][phase][metric] = sum(all_eval_results[key][phase][metric]) / len(all_eval_results[key][phase][metric])
    
    sienna.save(aggregated_eval_results, "evaluation_summary/aggregated_evaluation_results.json")
