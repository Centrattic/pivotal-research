import pandas as pd
import numpy as np
import os

def process(row, source_folder):
    # Support both: source as a single string with comma, or as a list
    source_field = row['source']
    if isinstance(source_field, str):
        source_field = source_field.strip('"').strip("'")
        parts = source_field.split(',', 1)
        if len(parts) != 2:
            raise ValueError(f"Handler requires two source files separated by a comma, got: {source_field}")
        file1, file2 = parts[0].strip(), parts[1].strip()
    elif isinstance(source_field, (list, tuple)) and len(source_field) == 2:
        file1, file2 = source_field[0], source_field[1]
    else:
        raise ValueError(f"Handler requires two source files, got: {source_field}")
    file1_path = os.path.join(source_folder, file1)
    file2_path = os.path.join(source_folder, file2)
    save_name = str(row.get('save_name', '') or row.get('name', '')).lower()
    if 'train' in save_name:
        # Train: class 1 from justice (true justice, target=1), class 1 from commonsense (false common sense, target=0)
        df1 = pd.read_csv(file1_path)
        df1 = df1[df1['label'].astype(str) == '1']
        prompts1 = df1['scenario'].astype(str)
        targets1 = np.ones(len(df1), dtype=int)
        df2 = pd.read_csv(file2_path)
        df2 = df2[df2['label'].astype(str) == '1']
        prompts2 = df2['input'].astype(str)
        targets2 = np.zeros(len(df2), dtype=int)
    elif 'eval' in save_name:
        # Eval: class 0 from justice (false justice, target=0), class 0 from commonsense (true common sense, target=1)
        df1 = pd.read_csv(file1_path)
        df1 = df1[df1['label'].astype(str) == '0']
        prompts1 = df1['scenario'].astype(str)
        targets1 = np.zeros(len(df1), dtype=int)
        df2 = pd.read_csv(file2_path)
        df2 = df2[df2['label'].astype(str) == '0']
        prompts2 = df2['input'].astype(str)
        targets2 = np.ones(len(df2), dtype=int)
    else:
        raise ValueError("Could not determine if this is train or eval from save_name or name.")
    prompts = pd.concat([prompts1, prompts2], ignore_index=True)
    targets = np.concatenate([targets1, targets2])
    out_df = pd.DataFrame({
        'prompt': prompts,
        'prompt_len': prompts.str.len(),
        'target': targets
    })
    out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return out_df 