import pandas as pd
import ast
import numpy as np


def process(
    row,
    source_file_or_df,
):
    # Load DataFrame
    if isinstance(source_file_or_df, pd.DataFrame):
        df = source_file_or_df
    else:
        if source_file_or_df.endswith(".csv"):
            df = pd.read_csv(source_file_or_df)
        elif source_file_or_df.endswith(".parquet"):
            df = pd.read_parquet(source_file_or_df)
        elif source_file_or_df.endswith(".tsv"):
            df = pd.read_csv(source_file_or_df, sep='\t')
        else:
            raise ValueError(f"Unsupported file extension: {source_file_or_df}")

    # Get columns from probe_from and probe_to
    probe_from = [c.strip().strip("'\"") for c in str(row['Probe from']).split(',') if c.strip()]
    probe_to = [c.strip().strip("'\"") for c in str(row['Probe to']).split(',') if c.strip()]

    # Detect question, context, and answers columns
    # Convention: question, context, answers (in order in probe_from)
    if len(probe_from) == 3:
        question_col, context_col, answers_col = probe_from
    elif len(probe_from) == 2:
        question_col, answers_col = probe_from
        context_col = None
    else:
        raise ValueError("probe_from should specify at least question and answers columns")

    target_col = probe_to[0]

    # Robustly parse answers to list
    def to_list(
        val,
    ):
        if isinstance(val, list):
            return val
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, str) and (val.startswith("[") or val.startswith("(")):
            try:
                return list(ast.literal_eval(val))
            except Exception:
                return []
        if pd.notnull(val):
            return [str(val)]
        return []

    df['answers_list'] = df[answers_col].apply(to_list)

    # Build the prompt
    def make_prompt(
        row,
    ):
        q = str(row[question_col]) if question_col in row else ""
        c = str(row[context_col]) if context_col and context_col in row else ""
        choices = row['answers_list']
        prompt = (
            q + " " + c + " A: " + (choices[0] if len(choices) > 0 else "") + " B: " +
            (choices[1] if len(choices) > 1 else "") + " C: " + (choices[2] if len(choices) > 2 else "") + " D: " +
            (choices[3] if len(choices) > 3 else "")
        )
        return prompt.strip()

    prompts = df.apply(make_prompt, axis=1)

    out_df = pd.DataFrame({"prompt": prompts, "prompt_len": prompts.str.len(), "target": df[target_col]})

    out_df = out_df.drop_duplicates().reset_index(drop=True)
    return out_df
