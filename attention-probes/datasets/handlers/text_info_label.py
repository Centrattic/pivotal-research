import pandas as pd
import ast
import os

def has_positive_label(probe_classes_col):
    """
    Returns 1 if any entry in the probe_classes_col (stored as a stringified list)
    is 1, else 0.
    """
    try:
        vals = ast.literal_eval(str(probe_classes_col))
        return int(1 in vals)
    except Exception:
        return 0

def process(row, source_file_or_df):
    # Accept either a DataFrame or a file path
    if isinstance(source_file_or_df, pd.DataFrame):
        df = source_file_or_df
    else:
        # Use os.path.splitext to get extension (lowercased)
        _, ext = os.path.splitext(source_file_or_df)
        ext = ext.lower()
        if ext == ".csv":
            df = pd.read_csv(source_file_or_df)
        elif ext == ".parquet":
            df = pd.read_parquet(source_file_or_df)
        elif ext in [".arrow", ".feather"]:
            df = pd.read_feather(source_file_or_df)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    # Get columns from main.csv
    probe_from = str(row['Probe from']).strip()
    probe_to = str(row['Probe to']).strip()

    prompt_series = df[probe_from]
    target_series = df[probe_to].apply(has_positive_label)

    out_df = pd.DataFrame({
        "prompt": prompt_series,
        "prompt_len": prompt_series.str.len(),
        "target": target_series
    })
    return out_df
