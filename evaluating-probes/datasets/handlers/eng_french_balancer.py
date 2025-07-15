"""
bilingual_phrase_handler.py
--------------------------------------------------------------
pip install pandas numpy tqdm
--------------------------------------------------------------
• Reads a two-column TSV of English/French phrases.
• Randomly emits one of the two phrases as `prompt`.
• target = 1  (French)   |   0  (English)
"""

import re, random
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

_ASCII_RE = re.compile(r'^[\u0000-\u007F]+$')       # crude EN check

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _is_ascii(text: str) -> bool:
    """True if text has only basic ASCII (≈ English for this dataset)."""
    return bool(_ASCII_RE.match(text.strip()))

def _load_tsv(path: str | Path) -> pd.DataFrame:
    """
    Read file as TSV *exactly two columns*, even if commas appear inside cells.
    """
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", maxsplit=1)
            if len(parts) != 2:                 # fallback: try double-space split
                parts = re.split(r"\s{2,}", line.rstrip("\n"), maxsplit=1)
            if len(parts) == 2:
                records.append(parts)
            # else: silently skip malformed line
    return pd.DataFrame(records, columns=["col1", "col2"])

def _decide_languages(df: pd.DataFrame) -> tuple[str, str]:
    """Return ('english_col', 'french_col') names based on ASCII share."""
    ascii_ratio_col1 = df["col1"].apply(_is_ascii).mean()
    ascii_ratio_col2 = df["col2"].apply(_is_ascii).mean()
    if ascii_ratio_col1 >= ascii_ratio_col2:
        return "col1", "col2"     # col1 = EN, col2 = FR
    else:
        return "col2", "col1"     # col2 = EN, col1 = FR

# ---------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------
def process(row: dict, source_file: str) -> pd.DataFrame:
    """
    Compatible with your dataset-loader framework: returns
      prompt | prompt_len | target     (French=1, English=0)
    The incoming `row` (config) is ignored for this dataset.
    """
    df_raw = _load_tsv(source_file)

    english_col, french_col = _decide_languages(df_raw)

    prompts, targets = [], []

    for en, fr in tqdm(
        df_raw[[english_col, french_col]].itertuples(index=False),
        total=len(df_raw),
        desc="Building prompts",
    ):
        if random.random() < 0.5:
            prompts.append(en)
            targets.append(0)          # English → 0
        else:
            prompts.append(fr)
            targets.append(1)          # French  → 1

    out_df = pd.DataFrame({
            "prompt":     prompts,
            "prompt_len": [len(p) for p in prompts],
            "target":     targets,
        })
    
    out_df = out_df.drop_duplicates().reset_index(drop=True)

    return out_df
