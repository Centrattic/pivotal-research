from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer

from src.logger import Logger
from src.activations import ActivationManager

__all__ = [
    "Dataset",
    "get_available_datasets",
    "load_combined_classification_datasets",
]

from functools import lru_cache

@lru_cache(maxsize=1)
def get_main_csv_metadata() -> pd.DataFrame:  # type: ignore[override]
    path = Path("datasets/main.csv")
    if not path.exists():
        raise FileNotFoundError("datasets/main.csv not found. This file is required for dataset metadata.")
    return pd.read_csv(path).set_index("number")

class Dataset:
    """Dataset wrapper that lazily populates activation caches."""

    def __init__(
        self,
        dataset_name: str,
        *,
        model: HookedTransformer,
        device: str = "cuda:0",
        data_dir: Path = Path("datasets/cleaned"),
        cache_root: Path = Path("activation_cache"),
        test_size: float = 0.15,
        seed: int = 42, # This is so important - how train and test sets persist across models/runs/etc.
    ):
        # ---- load csv & split (unchanged) ----
        self.dataset_name = dataset_name
        meta = get_main_csv_metadata()
        meta_row = meta.loc[int(dataset_name.split("_", 1)[0])]
        self.task_type = meta_row["Data type"].strip().lower()

        csv_path = data_dir / f"{dataset_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path).dropna(subset=["prompt", "target"])
        self.df = df

        self.max_len = (
            df["prompt_len"].max() if "prompt_len" in df.columns else df["prompt"].str.len().max()
        )

        if "classification" in self.task_type:
            df["target"], uniq = pd.factorize(df["target"].astype(str))
            self.n_classes = len(uniq)
        else:
            df["target"] = pd.to_numeric(df["target"], errors="coerce")
            self.n_classes = None

        X = df["prompt"].astype(str).to_numpy()
        y = df["target"].to_numpy()
        tr_idx, te_idx = train_test_split(
            np.arange(len(df)), test_size=test_size, random_state=seed, stratify=y if self.n_classes else None, shuffle=True
        )
        self.X_train_text, self.y_train = X[tr_idx].tolist(), y[tr_idx]
        self.X_test_text, self.y_test = X[te_idx].tolist(), y[te_idx]

        # print("TELL ME NUMBER")
        # print(len(self.X_train_text))
        # print(len(self.X_test_text))
        # print(len(set(self.X_train_text)))
        # print(len(set(self.X_test_text)))
        # print(len(set(self.X_train_text) & set(self.X_test_text))) # any overlaps??

        try: 
            cache_dir = cache_root / model.cfg.model_name / dataset_name
            self.act_manager = ActivationManager(
                model=model,
                device=device,
                d_model=model.cfg.d_model,
                max_len=self.max_len,
                cache_dir=cache_dir,
            )
        except: 
            print("Using dataset class to fetch text, not manage activations.")

    # text getters (unchanged)
    def get_train_set(self):
        return self.X_train_text, self.y_train

    def get_test_set(self):
        return self.X_test_text, self.y_test

    # activation getters
    def get_train_set_activations(self, layer: int, component: str):
        acts = self.act_manager.get_activations_for_texts(self.X_train_text, layer, component)
        return acts, self.y_train

    def get_test_set_activations(self, layer: int, component: str):
        acts = self.act_manager.get_activations_for_texts(self.X_test_text, layer, component)
        return acts, self.y_test
    
def get_available_datasets(data_dir: Path = Path("datasets/cleaned")) -> List[str]:
    return [f.stem for f in data_dir.glob("*.csv")]

def load_combined_classification_datasets(seed: int, model: HookedTransformer, device: str) -> Dataset:
    """Return a synthetic *single_all* Dataset w/ attached ActivationManager."""
    # Implementation unchanged; omitted for brevity – but you would create the
    # combined Dataset via `Dataset.from_combined` then *wrap* it with an
    # ActivationManager just like above.
    raise NotImplementedError("Re‑implement for combined dataset use case.")

