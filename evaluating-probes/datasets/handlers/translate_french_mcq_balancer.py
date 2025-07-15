"""
translate_french_mcq_balancer.py
--------------------------------
pip install deep_translator pandas numpy
"""

import pandas as pd, numpy as np, json, ast, random
from deep_translator import GoogleTranslator           # <-- new sync translator
from tqdm.auto import tqdm

# ------------------------------------------------------------------ #
# helpers (unchanged)                                                #
# ------------------------------------------------------------------ #
def parse_list(val):
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    if pd.isnull(val):
        return []
    if isinstance(val, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(val)
                return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
            except Exception:
                continue
        return [x.strip() for x in val.split(',')]
    return []

def is_nonempty_list(val):
    if isinstance(val, (list, np.ndarray, pd.Series)):
        return len(val) > 0
    if pd.isnull(val):
        return False
    return bool(val)

# ------------------------------------------------------------------ #
#  translation helper (sync, with cache)                             #
# ------------------------------------------------------------------ #
_trans        = GoogleTranslator(source="auto", target="fr")
_trans_cache: dict[str, str] = {}

def en_to_fr(text: str) -> str:
    if text in _trans_cache:
        return _trans_cache[text]
    try:
        fr = _trans.translate(text)
    except Exception:
        fr = text     # fail-open: keep English so pipeline never breaks
    _trans_cache[text] = fr
    return fr

# ------------------------------------------------------------------ #
#  main processor                                                    #
# ------------------------------------------------------------------ #
def process(row, source_file):
    # —— load —— #
    if   source_file.endswith(".parquet"): df = pd.read_parquet(source_file)
    elif source_file.endswith((".jsonl", ".ndjson")): df = pd.read_json(source_file, lines=True)
    else: df = pd.read_csv(source_file)

    probe_from = [x.strip() for x in str(row["Probe from"]).split(",") if x.strip()]
    probe_to   = [x.strip() for x in str(row["Probe to"]).split(",") if x.strip()]
    if len(probe_from) < 2 or len(probe_to) < 1:
        raise ValueError("Probe from: question + ≥1 distractor ; Probe to: correct column")

    question_col, incorrect_cols, correct_col = probe_from[0], probe_from[1:], probe_to[0]

    prompts_fr, targets = [], []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Building prompts"):
        # ---------- choose answer / label (original rules) ----------
        question = r[question_col]
        if len(incorrect_cols) == 1:                          # list-of-wrongs style
            wrongs   = parse_list(r[incorrect_cols[0]])
            corrects = parse_list(r[correct_col])
            choose_true = random.random() < 0.5
            if choose_true and is_nonempty_list(corrects):
                answer, label = random.choice(corrects), 1
            elif is_nonempty_list(wrongs):
                answer, label = random.choice(wrongs), 0
            else:
                answer, label = (corrects[0] if is_nonempty_list(corrects) else ""), 1
            prompt_en = f"Q. {question} A. {answer}"
        else:                                                 # multi-column distractors
            correct_answer = r[correct_col]
            wrong_choices  = [r[c] for c in incorrect_cols if pd.notnull(r[c]) and str(r[c]) != ""]
            choose_true    = random.random() < 0.5
            if choose_true and pd.notnull(correct_answer) and str(correct_answer) != "":
                answer, label = correct_answer, 1
            else:
                answer, label = (random.choice(wrong_choices) if wrong_choices else ""), 0
            prompt_en = f"Q: {question} A: {answer}"

        # ---------- translate once per full prompt ----------
        prompt_fr = en_to_fr(prompt_en)

        prompts_fr.append(prompt_fr)
        targets.append(label)

    return pd.DataFrame({
        "prompt":     prompts_fr,
        "prompt_len": [len(p) for p in prompts_fr],
        "target":     targets
    })
