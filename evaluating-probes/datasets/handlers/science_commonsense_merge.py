import pandas as pd
import numpy as np
import os
import json

def process(row, source_folder):
    # Parse source files
    source_field = row['source']
    if isinstance(source_field, str):
        source_field = source_field.strip('"').strip("'")
        parts = source_field.split(',', 1)
        if len(parts) != 2:
            raise ValueError(f"Handler requires two source files separated by a comma, got: {source_field}")
        sci_file, cs_file = parts[0].strip(), parts[1].strip()
    elif isinstance(source_field, (list, tuple)) and len(source_field) == 2:
        sci_file, cs_file = source_field[0], source_field[1]
    else:
        raise ValueError(f"Handler requires two source files, got: {source_field}")
    sci_path = os.path.join(source_folder, sci_file)
    cs_path = os.path.join(source_folder, cs_file)
    save_name = str(row.get('save_name', '') or row.get('name', '')).lower()
    # Remove .csv extension if present
    if save_name.endswith('.csv'):
        save_name = save_name[:-4]
    # Accept both long and short save names
    mode1 = (
        'science_true+commonsense_false' in save_name or
        save_name == 'cs_f_sci_t'
    )
    mode2 = (
        'science_false+commonsense_true' in save_name or
        save_name == 'sci_f_cs_t'
    )
    sci_df = pd.read_parquet(sci_path)
    cs_rows = []
    with open(cs_path, 'r') as f:
        for line in f:
            cs_rows.append(json.loads(line))
    cs_df = pd.DataFrame(cs_rows)
    prompts, targets = [], []
    max_per_class = 7500
    if mode1:
        # True science (correct_answer, target=1)
        sci_true_prompts = [f"Q: {q} A: {a}" for q, a in zip(sci_df['question'], sci_df['correct_answer'])]
        sci_true_targets = [1] * len(sci_true_prompts)
        # False commonsense (answer=='no', target=0)
        cs_false = cs_df[cs_df['answer'].str.lower() == 'no']
        cs_false_prompts = [f"Q: {q} A: {a}" for q, a in zip(cs_false['question'], cs_false['answer'])]
        cs_false_targets = [0] * len(cs_false_prompts)
        # Limit to max_per_class
        if len(sci_true_prompts) > max_per_class:
            idx = np.random.RandomState(42).choice(len(sci_true_prompts), max_per_class, replace=False)
            sci_true_prompts = [sci_true_prompts[i] for i in idx]
            sci_true_targets = [sci_true_targets[i] for i in idx]
        if len(cs_false_prompts) > max_per_class:
            idx = np.random.RandomState(42).choice(len(cs_false_prompts), max_per_class, replace=False)
            cs_false_prompts = [cs_false_prompts[i] for i in idx]
            cs_false_targets = [cs_false_targets[i] for i in idx]
        prompts = sci_true_prompts + cs_false_prompts
        targets = sci_true_targets + cs_false_targets
    elif mode2:
        # False science (distractors, target=0)
        sci_false_prompts = []
        for _, r in sci_df.iterrows():
            q = str(r['question'])
            for d in ['distractor1', 'distractor2', 'distractor3']:
                sci_false_prompts.append(f"Q: {q} A: {str(r[d])}")
        sci_false_targets = [0] * len(sci_false_prompts)
        # True commonsense (answer=='yes', target=1)
        cs_true = cs_df[cs_df['answer'].str.lower() == 'yes']
        cs_true_prompts = [f"Q: {q} A: {a}" for q, a in zip(cs_true['question'], cs_true['answer'])]
        cs_true_targets = [1] * len(cs_true_prompts)
        # Limit to max_per_class
        if len(sci_false_prompts) > max_per_class:
            idx = np.random.RandomState(42).choice(len(sci_false_prompts), max_per_class, replace=False)
            sci_false_prompts = [sci_false_prompts[i] for i in idx]
            sci_false_targets = [sci_false_targets[i] for i in idx]
        if len(cs_true_prompts) > max_per_class:
            idx = np.random.RandomState(42).choice(len(cs_true_prompts), max_per_class, replace=False)
            cs_true_prompts = [cs_true_prompts[i] for i in idx]
            cs_true_targets = [cs_true_targets[i] for i in idx]
        prompts = sci_false_prompts + cs_true_prompts
        targets = sci_false_targets + cs_true_targets
    else:
        raise ValueError("Could not determine mode from save_name or name. Should contain 'science_true+commonsense_false', 'cs_f_sci_t', 'science_false+commonsense_true', or 'sci_f_cs_t'.")
    out_df = pd.DataFrame({
        'prompt': prompts,
        'prompt_len': [len(p) for p in prompts],
        'target': targets
    })
    out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return out_df 