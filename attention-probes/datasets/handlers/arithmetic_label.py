import pandas as pd
import json

def get_nested(row, key):
    """
    Fetch nested fields from a row using dot notation only.
    If any field is a stringified JSON, it will be parsed once.
    """
    keys = key.split('.')
    value = row[keys[0]]
    for k in keys[1:]:
        # Parse stringified JSON once
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
    
    df.to_csv("test.csv")

    # Parse probe columns
    probe_from = [x.strip() for x in str(row['Probe from']).split(',') if x.strip()]
    probe_to = [x.strip() for x in str(row['Probe to']).split(',') if x.strip()]

    if len(probe_from) < 3 or len(probe_to) < 1:
        raise ValueError(
            "mcq_label handler requires at least three 'Probe from' fields (question, choices.text, choices.label) and one 'Probe to' (answer label)."
        )

    question_key, choices_text_key, choices_label_key = probe_from[:3]
    answer_col = probe_to[0]

    prompts = []
    targets = []
    for _, r in df.iterrows():
        question = get_nested(r, question_key)
        choices_text = get_nested(r, choices_text_key)
        choices_label = get_nested(r, choices_label_key)

        # Defensive: Ensure they're lists, even if single string
        if isinstance(choices_text, str):
            choices_text = [choices_text]
        if isinstance(choices_label, str):
            choices_label = [choices_label]

        formatted_choices = " ".join(
            f"{label}.{text}" for label, text in zip(choices_label, choices_text)
        )
        prompt = f"{question} {formatted_choices}"
        target = r[answer_col]

        prompts.append(prompt)
        targets.append(target)

    out_df = pd.DataFrame({
        'prompt': prompts,
        'prompt_len': [len(x) for x in prompts],
        'target': targets
    })
    return out_df
