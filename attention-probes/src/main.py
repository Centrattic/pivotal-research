from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from datasets import load_dataset, DatasetDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import transformer_lens as tl
import yaml

from data import *
from extract import *
from probe import *
from params import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="Single HF dataset id to run")
parser.add_argument("--all_datasets", action="store_true")
parser.add_argument("--cache_dir", type=Path, default=Path.home() / ".cache/attn_probes")
args = parser.parse_args()

args.cache_dir.mkdir(parents=True, exist_ok=True)

targets = [args.dataset] if args.dataset else load_dataset_list()
if args.all_datasets:
    targets = load_dataset_list(filter_auc=True)

results = {}
for name in targets:
    auc, acc = train_probe(name, args.cache_dir)
    results[name] = dict(auc=auc, acc=acc)

# save summary
out_path = args.cache_dir / "results.csv"
with open(out_path, "w") as f:
    f.write("dataset,auc,acc\n")
    for n, metrics in results.items():
        f.write(f"{n},{metrics['auc']:.3f},{metrics['acc']:.3f}\n")
print(f"Saved results to {out_path}")
