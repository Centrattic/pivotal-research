import pandas as pd
import numpy as np
import json
import ast
import random

def parse_list(val):
    # Accept list, numpy array, or string representation of list, or nan
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    if pd.isnull(val):
        return []
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            try:
                return ast.literal_eval(val)
            except Exception:
                # Fallback: split on comma (last resort)
                return [x.strip() for x in val.split(',')]
    return []

def is_nonempty_list(val):
    # True if val is a list/array/Series with at least one non-null entry
    if isinstance(val, (list, np.ndarray, pd.Series)):
        return len(val) > 0
    if pd.isnull(val):
        return False
    # For all else (shouldn't happen), treat as non-empty if not empty string
    return bool(val)

def process(row, source_file):
    # Load file
    if source_file.endswith(".parquet"):
        df = pd.read_parquet(source_file)
    elif source_file.endswith(".jsonl") or source_file.endswith(".ndjson"):
        df = pd.read_json(source_file, lines=True)
    else:
        df = pd.read_csv(source_file)
    
    # Get probe columns
    probe_from = [x.strip() for x in str(row['Probe from']).split(',') if x.strip()]
    probe_to = [x.strip() for x in str(row['Probe to']).split(',') if x.strip()]
    if len(probe_from) < 2 or len(probe_to) < 1:
        raise ValueError("Probe from must specify question column and at least one distractor column (or a list column); Probe to must specify correct answer column (or list column).")

    question_col = probe_from[0]
    incorrect_cols = probe_from[1:]
    correct_col = probe_to[0]

    prompts = []
    targets = []

    for _, r in df.iterrows():
        question = r[question_col]

        # If there is only one incorrect_col, treat it as a list column (misconceptions style)
        if len(incorrect_cols) == 1:
            # List-of-wrongs, e.g., "wrong_list"
            wrong_list = parse_list(r[incorrect_cols[0]])
            correct_list = parse_list(r[correct_col])
            # Explicit length checks!
            choose_correct = random.random() < 0.5
            if choose_correct and is_nonempty_list(correct_list):
                answer = random.choice(correct_list)
                label = 1
            elif is_nonempty_list(wrong_list):
                answer = random.choice(wrong_list)
                label = 0
            else:
                answer = correct_list[0] if is_nonempty_list(correct_list) else ""
                label = 1
            prompt = f"Q. {question} A. {answer}"
            prompts.append(prompt)
            targets.append(label)
        else:
            # Multiple single-value columns, e.g., "distractor1,distractor2,..."
            correct_answer = r[correct_col]
            wrong_choices = [r[c] for c in incorrect_cols if pd.notnull(r[c]) and r[c] != ""]
            random_wrong = random.choice(wrong_choices) if len(wrong_choices) > 0 else ""
            # Use pd.notnull and not-empty string for correct_answer
            choose_correct = random.random() < 0.5
            if choose_correct and pd.notnull(correct_answer) and str(correct_answer) != "":
                answer = correct_answer
                label = 1
            else:
                answer = random_wrong
                label = 0
            prompt = f"Q: {question} A: {answer}"
            prompts.append(prompt)
            targets.append(label)

    out_df = pd.DataFrame({
        'prompt': prompts,
        'prompt_len': [len(x) for x in prompts],
        'target': targets
    })
    return out_df
