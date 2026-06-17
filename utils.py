import pickle
import re
import pandas as pd
import json
import os
import pdb
import json
from sklearn.metrics.pairwise import cosine_similarity

def parse_json(json_string):
    # Parse JSON string
    if "```json" in json_string:
        json_string = json_string.replace("```json\n", "").replace("\n```", "")

    try:
        pred_j = json.loads(json_string)
    except Exception:
        try:
            pred_j = json.loads(("{"+json_string.split("{")[1].split("}")[0].strip()+"}").replace(",}", "}").replace("} \n{",", ").replace("}; {",", "))
        except Exception:
            try:
                pred_j = json.loads(json_string.replace("'", "\"").replace("} \n{",", ").replace("}; {",", "))
            except Exception:
                try:
                    pred_j = json.loads(json_string.replace('"', '&&&').replace("'", "\"").replace("} \n{",", ").replace("}; {",", ").replace('&&&', "'"))
                except Exception:
                    pred_j = {}
    return pred_j

from pandas.api.types import (
    is_datetime64_any_dtype,
    is_object_dtype,
    is_string_dtype,
)

from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from dateutil.parser import parse

DATE_HINT_PATTERN = re.compile(
    r"""
    (
        [/-] |
        : |
        T | Z |
        \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

NUMERIC_PATTERN = re.compile(
    r"""
    ^
    [-+]?
    (
        \d+([.,]\d+)*
    )
    $
    """,
    re.VERBOSE,
)

ISO_PATTERN = re.compile(
    r"""
    ^
    [+-]?\d{4,}
    -\d{2}
    -\d{2}
    (
        [T\s]
        \d{2}:\d{2}
        (:\d{2}(\.\d+)?)?
        (Z|[+-]\d{2}:?\d{2})?
    )?
    $
    """,
    re.VERBOSE,
)


def is_date_like(value):

    if pd.isna(value):
        return False

    value = str(value).strip()

    if NUMERIC_PATTERN.match(value):
        return False

    if ISO_PATTERN.match(value):
        return True

    if not DATE_HINT_PATTERN.search(value):
        return False

    try:
        pd.to_datetime(value, errors="raise")
        return True
    except Exception:
        pass

    try:
        parse(value, fuzzy=False, dayfirst=True)
        return True
    except Exception:
        return False

def detect_column_types(
    df: pd.DataFrame,
    threshold: float = 0.8
):
    """
    Detect temporal, numerical, and textual columns.

    Returns
    -------
    dict
        {
            "temporal": [...],
            "numerical": [...],
            "textual": [...]
        }
    """

    temporal_cols = []
    numerical_cols = []
    textual_cols = []

    for col in df.columns:
        s = df[col]

        if is_datetime64_any_dtype(s):
            temporal_cols.append(col)
            continue

        if is_numeric_dtype(s):
            numerical_cols.append(col)
            continue

        if is_object_dtype(s) or is_string_dtype(s):

            sample = s.dropna()

            if len(sample) == 0:
                textual_cols.append(col)
                continue

            temporal_ratio = sample.apply(is_date_like).mean()

            if temporal_ratio >= threshold:
                temporal_cols.append(col)
                continue

            num_parsed = pd.to_numeric(sample, errors="coerce")
            numeric_ratio = num_parsed.notna().mean()

            if numeric_ratio >= threshold:
                numerical_cols.append(col)
                continue

            textual_cols.append(col)

        else:
            textual_cols.append(col)

    return {
        "temporal": temporal_cols,
        "numerical": numerical_cols,
        "textual": textual_cols,
    }

def langchain_message_to_dict(message):
    if isinstance(message, list):
        return [langchain_message_to_dict(m) for m in message]
    return {attribute: value for attribute, value in vars(message).items() if value}

def parse_model_response(response):
    # If reasoning is included
    if isinstance(response.content, list):
        # if len(response.content) < 2:
        #     return parse_json(response.content[0]["text"])
        try:
            return parse_json(response.content[1]["text"])
        except Exception:
            return parse_json(response.content[0]["text"])
    else:
        return parse_json(response.content)

#Function to return a flattened list
def flatten_list(original_list):
    flat_list = []
    for item in original_list:
        if isinstance(item, list):
            flat_list = flat_list + item
        else:
            flat_list.append(item)
    return flat_list