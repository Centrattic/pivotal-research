import pandas as pd
import os


def process(
    row,
    source_folder,
):
    # Parse source files - expect three files separated by commas
    source_field = row['source']
    if isinstance(source_field, str):
        source_field = source_field.strip('"').strip("'")
        parts = [part.strip() for part in source_field.split(',')]
        if len(parts) != 3:
            raise ValueError(f"Handler requires three source files separated by commas, got: {source_field}")
        file1, file2, file3 = parts
    elif isinstance(source_field, (list, tuple)) and len(source_field) == 3:
        file1, file2, file3 = source_field
    else:
        raise ValueError(f"Handler requires three source files, got: {source_field}")

    # Load all three CSV files
    file1_path = os.path.join(source_folder, file1)
    file2_path = os.path.join(source_folder, file2)
    file3_path = os.path.join(source_folder, file3)

    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    df3 = pd.read_csv(file3_path)

    # Extract the required columns from each file
    # prompt: generation(System Prompt + User Prompt)_run1
    # target: eval_honesty_1

    prompts = []
    targets = []

    # Process file 1
    for _, r in df1.iterrows():
        prompt = str(r['generation(System Prompt + User Prompt)_run1'])
        target = r['eval_honesty_1']
        if pd.notna(prompt) and pd.notna(target):
            # Convert -1 to 0
            if target == -1:
                target = 0
            prompts.append(prompt)
            targets.append(target)

    # Process file 2
    for _, r in df2.iterrows():
        prompt = str(r['generation(System Prompt + User Prompt)_run1'])
        target = r['eval_honesty_1']
        if pd.notna(prompt) and pd.notna(target):
            # Convert -1 to 0
            if target == -1:
                target = 0
            prompts.append(prompt)
            targets.append(target)

    # Process file 3
    for _, r in df3.iterrows():
        prompt = str(r['generation(System Prompt + User Prompt)_run1'])
        target = r['eval_honesty_1']
        if pd.notna(prompt) and pd.notna(target):
            # Convert -1 to 0
            if target == -1:
                target = 0
            prompts.append(prompt)
            targets.append(target)

    # Create output DataFrame
    out_df = pd.DataFrame({'prompt': prompts, 'prompt_len': [len(p) for p in prompts], 'target': targets})

    # Remove duplicates and reset index
    out_df = out_df.drop_duplicates(keep='first').reset_index(drop=True)

    # Drop rows where prompt is duplicated (keep first occurrence)
    out_df = out_df.drop_duplicates(subset=["prompt"], keep=False).reset_index(drop=True)

    return out_df
