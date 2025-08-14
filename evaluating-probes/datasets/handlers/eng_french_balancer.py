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

_ASCII_RE = re.compile(r'^[\u0000-\u007F]+$')  # crude EN check


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
            if len(parts) != 2:  # fallback: try double-space split
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
        return "col1", "col2"  # col1 = EN, col2 = FR
    else:
        return "col2", "col1"  # col2 = EN, col1 = FR


def process(row: dict, source_file: str, n_en: int = 10000, n_fr: int = 10000) -> pd.DataFrame:
    """
    Returns a DataFrame with randomly chosen English/French prompts.
    Only includes prompts with > 20 characters.
    Allows user to specify number of English and French samples.

    Args:
        row: (unused, for framework compatibility)
        source_file: Path to TSV file.
        n_en: Number of English samples to include.
        n_fr: Number of French samples to include.

    Returns:
        DataFrame with columns: prompt | prompt_len | target
    """
    df_raw = _load_tsv(source_file)
    english_col, french_col = _decide_languages(df_raw)

    prompts, targets, prompt_lens = [], [], []

    # Iterate through rows, generate both prompt directions per row
    for en, fr in tqdm(
        df_raw[[english_col, french_col]].itertuples(index=False),
        total=len(df_raw),
        desc="Building prompts",
    ):
        if len(en) > 20:
            prompts.append(en)
            targets.append(0)
            prompt_lens.append(len(en))
        if len(fr) > 20:
            prompts.append(fr)
            targets.append(1)
            prompt_lens.append(len(fr))

    out_df = pd.DataFrame({
        "prompt": prompts,
        "prompt_len": prompt_lens,
        "target": targets,
    }).drop_duplicates().reset_index(drop=True)

    # Sample n_en English and n_fr French prompts
    en_df = out_df[out_df["target"] == 0].sample(n=min(n_en, (out_df["target"] == 0).sum()), random_state=42)
    fr_df = out_df[out_df["target"] == 1].sample(n=min(n_fr, (out_df["target"] == 1).sum()), random_state=42)

    out_df = pd.concat([en_df, fr_df]).reset_index(drop=True)
    out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Keep one of the complete duplicates.
    out_df = out_df.drop_duplicates(keep='first').reset_index(drop=True)

    # Drop both of the opposite duplicates, unclear which is correct.
    out_df = out_df.drop_duplicates(subset=["prompt"], keep=False).reset_index(drop=True)

    return out_df
