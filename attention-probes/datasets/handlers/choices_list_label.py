import pandas as pd
import json

def get_nested(row, key):
    """Fetch nested fields using dot notation."""
    keys = key.split('.')
    value = row[keys[0]]
    for k in keys[1:]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                raise ValueError(f"Could not parse string as JSON for key {key}: {value}")
        value = value[k]
    return value

def process(row, source_file):
    # Load the source file
    if source_file.endswith(".parquet"):
        df = pd.read_parquet(source_file)
    elif source_file.endswith(".jsonl") or source_file.endswith(".ndjson"):
        df = pd.read_json(source_file, lines=True)
    else:
        df = pd.read_csv(source_file)

    probe_from = [x.strip() for x in str(row['Probe from']).split(',') if x.strip()]
    probe_to = [x.strip() for x in str(row['Probe to']).split(',') if x.strip()]
    if len(probe_from) < 2 or len(probe_to) < 1:
        raise ValueError("Probe from must specify question stem and choices; Probe to must specify answer key.")

    question_key = probe_from[0]
    choices_key = probe_from[1]
    answer_col = probe_to[0]

    prompts = []
    targets = []

    for _, r in df.iterrows():
        # Get question stem
        question = get_nested(r, question_key)
        # Get choices list (list of dicts)
        choices = get_nested(r, choices_key)
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except Exception:
                choices = []
        choices_label = [c.get('label', '') for c in choices]
        choices_text = [c.get('text', '') for c in choices]
        formatted_choices = " ".join(f"({label}) {text}" for label, text in zip(choices_label, choices_text))
        prompt = f"{question} {formatted_choices}".strip()
        target = r[answer_col]
        prompts.append(prompt)
        targets.append(target)

    out_df = pd.DataFrame({
        'prompt': prompts,
        'prompt_len': [len(x) for x in prompts],
        'target': targets
    }).drop_duplicates().reset_index(drop=True)

    return out_df
