from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Core loader
@dataclass
class DataLoader:
    tag: str                     # e.g. "7_nyc_lat"
    data_root: str = "data/cleaned"
    target_col: str = "target"
    text_col: str = "prompt"
    # To Do: update to pull num from main.csv, not in tag

    def __post_init__(self):
        path = os.path.join(self.data_root, f"{self.tag}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self.df: pd.DataFrame = pd.read_csv(path)
        if self.text_col not in self.df.columns or self.target_col not in self.df.columns:
            raise ValueError(f"CSV must contain '{self.text_col}' and '{self.target_col}' columns")

    @property
    def prompts(self) -> pd.Series:  # noqa: D401
        "Return text column."  # simple docstring
        return self.df[self.text_col]

    @property
    def labels(self) -> np.ndarray:  # noqa: D401
        return self.df[self.target_col].to_numpy()

    def split(self, test_size: float = 0.2, seed: int = 42):
        idx_train, idx_test = train_test_split(
            np.arange(len(self.df)), test_size=test_size, stratify=self.labels, random_state=seed)
        return idx_train, idx_test

    def features(
        self,
        extractor,
        layer: int,
        component: str = "resid_post",
        agg: str = "mean",
        test_size: float = 0.2,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convenience: run `extractor.features` then split.

        Returns `(X_train, y_train, X_test, y_test)` ready for probes.
        """
        idx_train, idx_test = self.split(test_size=test_size, seed=seed)
        feats = extractor.features(
            self.prompts.tolist(), layer=layer, component=component, agg=agg)
        return feats[idx_train], self.labels[idx_train], feats[idx_test], self.labels[idx_test]

# Functional shortcut
def split_dataset(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Return train/test *dataframes* with stratification by `target`."""
    idx_train, idx_test = train_test_split(
        df.index, test_size=test_size, stratify=df["target"], random_state=seed)
    return df.loc[idx_train], df.loc[idx_test]
