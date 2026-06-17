import sienna
import os
import json
import pandas as pd

def load_ground_truth(folder = None, benchmark = "SINT-Benchmark"):
    # Read from file
    gt_file = sienna.load(f"data/gt/{benchmark}/{folder}/{folder}_gt.json" )
    gt_mappings = sienna.load(f"data/gt/{benchmark}/{folder}/{folder}_gt_mappings_with_values.json" )
    gt_final_integrated_schema = sienna.load(f"data/gt/{benchmark}/{folder}/{folder}_gt_integration.json")
    gt_file_names = list(gt_file["schema_by_entity"].keys())

    return gt_file_names, gt_file, gt_mappings, gt_final_integrated_schema

def table_serialization(df, nr_rows=10, format="markdown", headers=True, show_cols=False):
    # Remove \n and \r characters from the dataframe
    df = df.replace({r'\n': ' ', r'\r': ' '}, regex=True)
    # Sort the rows of the dataframe by the number of null values in each row
    df = df.iloc[df.isnull().sum(1).sort_values(ascending=True).index]

    if not headers:
        # Remove the column headers
        df.columns = [f"Column_{i}" for i in range(len(df.columns))]

    # Trim values in the dataframe to 50 words
    df = df.map(lambda x: x if pd.isnull(x) else (" ".join(str(x).split()[:50]) + '...' if len(str(x).split()) > 50 else x))
    
    if format=="markdown":
        if not show_cols:
            return df[:nr_rows].to_markdown(index=False)
        else:
            cols = {ci: col for ci, col in enumerate(df.columns)}
            return f"The column headers along with their indices are the following: {cols}\n\n" + df[:nr_rows].to_markdown(index=False)
    elif format=="json":
        return "{\n"+"\n".join([f"   {rec}" for rec in json.loads(df[:nr_rows].to_json(orient="records"))])+"\n}"
    else:
        raise ValueError("Unsupported format. Use 'markdown' or 'json'.")