import pandas as pd
import json
import os
import random

def process(row, source_folder):
    source_files = [f.strip() for f in str(row['source']).split(',')]
    if len(source_files) != 2:
        raise ValueError("Handler requires two source files (data,labels) in 'source' column.")
    data_file, labels_file = source_files
    data_path = os.path.join(source_folder, data_file)
    labels_path = os.path.join(source_folder, labels_file)

    # Load JSONL data
    with open(data_path, "r", encoding="utf-8") as f:
        data_list = [json.loads(line) for line in f if line.strip()]
    data_df = pd.DataFrame(data_list)

    # Load .lst labels (as ints)
    with open(labels_path, encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    if len(labels) != len(data_df):
        raise ValueError(f"Labels ({labels_file}) has {len(labels)} lines but data file ({data_file}) has {len(data_df)} rows.")
    data_df = data_df.copy()
    data_df["label"] = labels

    probe_from = [x.strip() for x in str(row['Probe from']).split(',') if x.strip()]

    prompts = []
    targets = []

    for _, r in data_df.iterrows():
        # Always get columns as strings
        prompt = str(r[str(probe_from[0])])
        sol1 = str(r[str(probe_from[1])])
        sol2 = str(r[str(probe_from[2])])

        correct_answer = sol1 if int(r["label"]) == 0 else sol2
        wrong_answer = sol2 if int(r["label"]) == 0 else sol1

        if random.random() < 0.5:
            answer = correct_answer
            label = 1
        else:
            answer = wrong_answer
            label = 0

        formatted_prompt = f"Q. {prompt} A. {answer}"
        prompts.append(formatted_prompt)
        targets.append(label)

    out_df = pd.DataFrame({
        'prompt': prompts,
        'prompt_len': [len(x) for x in prompts],
        'target': targets
    })
    return out_df
