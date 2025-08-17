import os
import pandas as pd


def _parse_sources(source_field):

    if isinstance(source_field, str):
        cleaned = source_field.strip().strip('"').strip("'")
        parts = [p.strip() for p in cleaned.split(',') if p.strip()]
        return parts
    if isinstance(source_field, (list, tuple)):
        return list(source_field)
    raise ValueError(f"Unsupported 'source' format: {source_field}")


def _load_and_filter_one(csv_path):

    df = pd.read_csv(csv_path)

    # Robustly coerce to numeric and filter to {-1, 1}
    if 'eval_honesty_1' not in df.columns:
        return pd.DataFrame(columns=['prompt', 'target'])

    honesty = pd.to_numeric(df['eval_honesty_1'], errors='coerce')
    mask = honesty.isin([-1, 1])
    df = df.loc[mask].copy()

    if 'generation(System Prompt + User Prompt)_run1' not in df.columns:
        return pd.DataFrame(columns=['prompt', 'target'])

    prompts = df['generation(System Prompt + User Prompt)_run1'].astype(str)

    # Build question = system_prompt + user_prompt
    sys_col = 'system_prompt' if 'system_prompt' in df.columns else None
    usr_col = 'user_prompt' if 'user_prompt' in df.columns else None
    if sys_col is not None and usr_col is not None:
        questions = (
            df[sys_col].fillna('').astype(str) + '\n\n' + df[usr_col].fillna('').astype(str)
        ).str.strip()
    else:
        # Fallback: just use user_prompt if available, else empty
        if usr_col is not None:
            questions = df[usr_col].fillna('').astype(str).str.strip()
        else:
            questions = pd.Series([''] * len(df), index=df.index)

    # Map: lie (-1) -> 1, truth (1) -> 0
    targets = honesty.map({-1: 1, 1: 0})

    out = pd.DataFrame({'prompt': prompts, 'question': questions, 'target': targets})
    out = out.dropna(subset=['prompt', 'target'])
    return out


def process(row, source_folder):

    sources = _parse_sources(row['source'])
    frames = []
    for src in sources:
        csv_path = os.path.join(source_folder, src)
        if not os.path.exists(csv_path):
            # Skip missing sources rather than failing the whole job
            continue
        frames.append(_load_and_filter_one(csv_path))

    if not frames:
        return pd.DataFrame(columns=['prompt', 'prompt_len', 'target'])

    combined = pd.concat(frames, axis=0, ignore_index=True)

    # Drop exact duplicate rows; then drop duplicates by (prompt, question)
    combined = combined.drop_duplicates(keep='first')
    if {'prompt', 'question'}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=['prompt', 'question'], keep='first')

    combined['prompt_len'] = combined['prompt'].astype(str).str.len()
    # Ensure column order
    cols = ['prompt', 'question', 'prompt_len', 'target'] if 'question' in combined.columns else ['prompt', 'prompt_len', 'target']
    combined = combined[cols]

    return combined


