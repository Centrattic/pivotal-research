from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from datasets import load_dataset, DatasetDict
import yaml

from main import *

class DatasetManager:
    """Utility for loading & preprocessing binary‑classification datasets."""

    def __init__(self, dataset_name: str, split_ratio: Tuple[float, float] = (0.8, 0.1)):
        self.name = dataset_name
        self.split_ratio = split_ratio  # train, val (rest test)

    def load(self) -> DatasetDict:
        # Assumes HF dataset exposes 'text' + 'label' fields or similar.
        ds = load_dataset(self.name)
        if 'train' not in ds:
            # fallback: take 'validation' as test if only two splits
            raise ValueError(f"{self.name} missing 'train' split")

        # Shuffle & split train into train/val
        train_val = ds['train'].shuffle(seed=42)
        n_train = int(len(train_val) * self.split_ratio[0])
        n_val = int(len(train_val) * self.split_ratio[1])
        train_ds = train_val.select(range(n_train))
        val_ds = train_val.select(range(n_train, n_train + n_val))
        test_ds = ds['test'] if 'test' in ds else train_val.select(range(n_train + n_val, len(train_val)))
        return DatasetDict(train=train_ds, validation=val_ds, test=test_ds)
    
def load_dataset_list(filter_auc: bool = False) -> List[str]:
    with open(CONFIG_PATH) as f:
        listing: Dict[str, Dict] = yaml.safe_load(f)
    if not filter_auc:
        return list(listing.keys())
    return [k for k, v in listing.items() if v.get('linear_auc', 1.0) < 0.8]
