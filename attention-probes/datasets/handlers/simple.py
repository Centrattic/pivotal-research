import pandas as pd
import os
import ast

def process(row, source_file_or_df):
    # Accept either a DataFrame or a file path, DataFrame from other handlers calling it
    if isinstance(source_file_or_df, pd.DataFrame):
        df = source_file_or_df
    else:
        # Load supported formats
        _, ext = os.path.splitext(source_file_or_df)
        ext = ext.lower()
        if ext == ".csv":
            df = pd.read_csv(source_file_or_df)
        elif ext == ".parquet":
            df = pd.read_parquet(source_file_or_df)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    df.to_csv("proglangtest.csv")

    probe_from = list(str(row['Probe from']).split(',')) if pd.notnull(row['Probe from']) else []
    probe_to = list(str(row['Probe to']).split(',')) if pd.notnull(row['Probe to']) else []
    probe_from_extract = row['probe from extraction']
    probe_to_extract = row['probe to extraction']

    for i in probe_from:
        probe_from_extract = probe_from_extract.replace(f"{i}", f"df['{i}']")

    for i in probe_to:
        probe_to_extract = probe_to_extract.replace(f"{i}", f"df['{i}']")

    from_series = eval(probe_from_extract)
    to_series = eval(probe_to_extract)

    out_df = pd.DataFrame({
        "prompt": from_series, 
        "prompt_len": from_series.str.len(), 
        "target": to_series
    })
    return out_df
