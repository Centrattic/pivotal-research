from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from data import *
from extract import *
from probe import *
from params import *

class LinearAttentionProbe:
    def __init__(self, l1: float = 0.001):
        self.clf = LogisticRegression(penalty='l1', solver='saga', C=1.0 / l1, max_iter=4000, n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

def train_probe(dataset_name: str, cache_dir: Path):
    ds_mgr = DatasetManager(dataset_name)
    dsets = ds_mgr.load()

    extractor = AttentionFeatureExtractor()

    def to_features(split):
        cache_path = cache_dir / f"{dataset_name}_{split}.pt"
        if cache_path.exists():
            return torch.load(cache_path)
        loader = DataLoader(dsets[split], batch_size=DEFAULT_BATCH_SIZE)
        feats, labels = [], []
        for batch in tqdm(loader, desc=f"{split}"):
            feats.append(extractor.encode(batch['text']))
            labels.append(torch.tensor(batch['label']))
        X = torch.cat(feats).numpy()
        y = torch.cat(labels).numpy()
        torch.save((X, y), cache_path)
        return X, y

    X_train, y_train = to_features('train')
    X_val, y_val = to_features('validation')
    X_test, y_test = to_features('test')

    probe = LinearAttentionProbe()
    probe.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))

    y_pred = probe.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred > 0.5)
    print(f"{dataset_name}: AUC={auc:.3f}  Acc={acc:.3f}")
    return auc, acc
