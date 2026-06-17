import os
import pandas as pd
from valentine import valentine_match
import pdb
from valentine.algorithms import Coma, Cupid, JaccardDistanceMatcher, ComaPy
from valentine.algorithms.jaccard_distance import StringDistanceFunction
from sklearn.metrics import f1_score, precision_recall_fscore_support
import tqdm
import sienna

# benchmark = "Real Benchmark"
benchmark = "SINT-Benchmark"

input_table_folder = f"alite-main/codes/align_utilities/Real Benchmark/" if benchmark == "Real Benchmark" else f"data/selected-tables/SINT-Benchmark/"

# folders = [, "wikidbs_paintings-collection", "wikidbs_horror-film-character", "wikidbs_monuments", "volcanic-eruptions", "games", "goby_selected", "park_events", "car-listings", "airline-data"]
folders = ["wikidbs_research-articles"] #"wikidbs_monuments""airline-data""car-listings""volcanic-eruptions"
# folders = os.listdir(f"alite-main/codes/align_utilities/{benchmark}/")


for folder in folders:
    gt = pd.read_csv(f"data/coma_input/{benchmark}/{folder}_gt_mappings.csv")

    matchers = {
        "Coma": ComaPy(use_instances=False),
        "Coma-instances-schema": ComaPy(use_instances=True, use_schema=True),
        "Coma-instances": ComaPy(use_instances=True, use_schema=False),
        # "Coma-original": Coma(use_instances=True)
        # "Cupid": Cupid(th_accept=0.5),
        # "JaccardL": JaccardDistanceMatcher(distance_fun=StringDistanceFunction.Levenshtein)
    }

    matches = {
        "Coma": [],
        "Coma-instances-schema": [],
        "Coma-instances": [],
        # "Coma-original": []
        # "Cupid": [],
        # "JaccardL": []
    }

    matches_mapped = {
        "Coma": [],
        "Coma-instances-schema": [],
        "Coma-instances": [],
        # "Coma-original": []
        # "Cupid": [],
        # "JaccardL": []
    }

    matches_predictions = {
        "Coma": [],
        "Coma-instances-schema": [],
        "Coma-instances": [],
        # "Coma-original": []
        # "Cupid": [],
        # "JaccardL": []
    }

    matches_cm = {
        "Coma": {"tp": 0, "fp": 0, "fn": 0},
        "Coma-instances-schema": {"tp": 0, "fp": 0, "fn": 0},
        "Coma-instances": {"tp": 0, "fp": 0, "fn": 0},
        # "Coma-original": {"tp": 0, "fp": 0, "fn": 0}
        # "Cupid": [],
        # "JaccardL": []
    }

    matchers_evaluation = {}

    labels = gt["label"].to_list()
    predictions = []

    gt_nr = 0

    for table_name in tqdm.tqdm(gt["table_name_left"].unique(), total=len(gt["table_name_left"].unique())):
        df1 = pd.read_csv(f"{input_table_folder}{folder}/{table_name}")
        gt_table = gt[gt["table_name_left"]==table_name]

        for table_name_right in tqdm.tqdm(gt_table["table_name_right"].unique(), total=len(gt_table["table_name_right"].unique()), leave=False):

            df2 = pd.read_csv(f"{input_table_folder}{folder}/{table_name_right}")

            for matcher in matchers:
                try:
                    # matches[matcher].append(valentine_match(df1[:500], df2[:500], matchers[matcher]))
                    found_matches = valentine_match(df1[:500], df2[:500], matchers[matcher])
                    matches[matcher].append(found_matches)
                except Exception:
                    # matches[matcher].append([])
                    print(f"Error for {matcher} in table pair {table_name}, {table_name_right}")
                # Postprocess found matches to be in the same format as gt
                # Replace table_1 and table_2 with actual table names
                if found_matches:
                    matches_mapped[matcher].append([((match[0][0].replace("table_1", table_name), match[0][1]), (match[1][0].replace("table_2", table_name_right), match[1][1])) for match in found_matches])
                

            # pdb.set_trace() 
            for _, correspondences in gt_table[gt_table["table_name_right"]==table_name_right].iterrows():
                for matcher in matchers:
                    if (('table_1', correspondences["column_name_left"]), ('table_2', correspondences["column_name_right"])) in matches[matcher][gt_nr]:
                        matches_predictions[matcher].append(1)
                        # Correct match
                        matches_cm[matcher]["tp"] += 1
                    else:
                        matches_predictions[matcher].append(0)
                        # Missed match
                        matches_cm[matcher]["fn"] += 1
            
            # Check for false positives
            for matcher in matchers:
                for match in matches[matcher][gt_nr]:
                    # pdb.set_trace()
                    if (match[0][1], match[1][1]) not in gt_table[gt_table["table_name_right"]==table_name_right][["column_name_left", "column_name_right"]].apply(tuple, axis=1).tolist():
                        # False positive
                        matches_cm[matcher]["fp"] += 1
                        # pdb.set_trace()

            gt_nr += 1

    for matcher in matches_predictions:
        tp = matches_cm[matcher]["tp"]
        fp = matches_cm[matcher]["fp"]
        fn = matches_cm[matcher]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        matchers_evaluation[matcher] = {}
        matchers_evaluation[matcher]["overall_F1"] = f1
        matchers_evaluation[matcher]["overall_precision"] = precision
        matchers_evaluation[matcher]["overall_recall"] = recall

        print(f"{matcher}_overall_f1: {f1}")


    output_file = {
        "matches": {matches_matcher: [[ list(m)+[match[m]] for m in match ] for match in matches[matches_matcher] ]  for matches_matcher in matches },
        "matchers_predictions": matches_predictions,
        "matchers_cm": matches_cm,
        "matchers_mapped": matches_mapped
    }

    output_evaluation = {
        "matchers_evaluation": matchers_evaluation
    }

    sienna.save(output_file, f"coma_results/{benchmark}/{folder}_coma_results.json")
    sienna.save(output_evaluation, f"coma_results/{benchmark}/{folder}_coma_evaluation.json")

