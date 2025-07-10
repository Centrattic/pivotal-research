from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Literal, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import json
import time

# ================= BASE CLASSES =================

class BaseProbe:
    name: str = "base_probe"

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray, prob: bool = False) -> np.ndarray:
        lgts = self.predict_logits(X)
        if prob:
            return 1 / (1 + np.exp(-lgts))
        return (lgts > 0).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_prob = self.predict(X, prob=True)
        y_hat = (y_prob > 0.5).astype(int)
        return {
            "acc": float(accuracy_score(y, y_hat)),
            "auc": float(roc_auc_score(y, y_prob)),
        }

    def save_results(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        model_name: str,
        dataset: str,
        layer: int,
        component: str,
        out_dir: str = "results",
    ) -> str:
        os.makedirs(out_dir, exist_ok=True)
        metrics = self.score(X_test, y_test)
        logits  = self.predict_logits(X_test).tolist()
        payload = {
            "probe": self.name,
            "model": model_name,
            "dataset": dataset,
            "layer": layer,
            "component": component,
            "timestamp": int(time.time()),
            "metrics": metrics,
            "logits": logits,
        }
        fname = f"{model_name}__{dataset}__L{layer}_{component}__{self.name}.json"
        path  = os.path.join(out_dir, fname)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def _check_dims(self, X: np.ndarray, y: np.ndarray):  # noqa: D401
        assert X.ndim == 2, "X must be 2‑D (N, d)"
        assert y.ndim == 1, "y must be 1‑D"
        assert X.shape[0] == y.shape[0], "N mismatch"

# ================= FEATURE-BASED PROBES =================

@dataclass
class LRConfig:
    penalty: Literal["l1", "l2"] = "l2"
    C: float = 1.0
    solver: str = "liblinear"
    max_iter: int = 1000

class LogisticRegressionProbe(BaseProbe):
    name: str = "logreg"
    def __init__(self, cfg: LRConfig | None = None):
        self.cfg = cfg or LRConfig()
        self._clf: Optional[LogisticRegression] = None
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._check_dims(X, y)
        self._clf = LogisticRegression(**asdict(self.cfg))
        self._clf.fit(X, y)
        return self
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        assert self._clf is not None, "Probe not fitted yet"
        probs = self._clf.predict_proba(X)[:, 1]
        return np.log(probs / (1.0 - probs + 1e-9))

@dataclass
class MMConfig:
    tilt: bool = True
    reg: float = 1e-6

class MassMeanProbe(BaseProbe):
    name: str = "mass_mean"
    def __init__(self, cfg: MMConfig | None = None):
        self.cfg = cfg or MMConfig()
        self.theta: Optional[np.ndarray] = None
        self.bias: float = 0.0
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._check_dims(X, y)
        pos, neg = X[y == 1], X[y == 0]
        mu_pos, mu_neg = pos.mean(0), neg.mean(0)
        diff = mu_pos - mu_neg
        if self.cfg.tilt:
            centred = np.concatenate([pos - mu_pos, neg - mu_neg], axis=0)
            Σ = centred.T @ centred / centred.shape[0]
            Σ += np.eye(Σ.shape[0]) * self.cfg.reg
            diff = np.linalg.solve(Σ, diff)
        self.theta = diff
        prj_pos = mu_pos @ self.theta
        prj_neg = mu_neg @ self.theta
        self.bias = -0.5 * (prj_pos + prj_neg)
        return self
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        assert self.theta is not None, "Probe not fitted yet"
        return X @ self.theta + self.bias

# ================= SEQUENCE PROBES (NO LOGREG) =================

class MeanProbe(BaseProbe):
    name = "mean"
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.theta = np.zeros(d_model)
        self.bias = 0.0
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, lr=1e-2, epochs=100):
        # X: (N, S, D), y: (N,)
        N, S, D = X.shape
        device = "cuda" if torch.cuda.is_available() else "cpu"
        theta = torch.nn.Parameter(torch.zeros(D, device=device))
        bias  = torch.nn.Parameter(torch.zeros(1, device=device))
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)
        optimizer = torch.optim.Adam([theta, bias], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            per_token = torch.einsum('nsd,d->ns', X_tensor, theta)  # (N, S)
            logits = per_token.mean(dim=1) + bias  # (N,)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tensor)
            loss.backward()
            optimizer.step()
        self.theta = theta.detach().cpu().numpy()
        self.bias  = float(bias.detach().cpu().numpy())
        self._fitted = True
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        # X: (batch, seq_len, d_model)
        per_token = X @ self.theta  # (N, S)
        logits = per_token.mean(axis=1) + self.bias
        return logits

class MaxProbe(BaseProbe):
    name = "max"
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.theta = np.zeros(d_model)
        self.bias = 0.0
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, lr=1e-2, epochs=100):
        # X: (N, S, D), y: (N,)
        N, S, D = X.shape
        device = "cuda" if torch.cuda.is_available() else "cpu"
        theta = torch.nn.Parameter(torch.zeros(D, device=device))
        bias  = torch.nn.Parameter(torch.zeros(1, device=device))
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)
        optimizer = torch.optim.Adam([theta, bias], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            per_token = torch.einsum('nsd,d->ns', X_tensor, theta)  # (N, S)
            logits = per_token.max(dim=1).values + bias  # (N,)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tensor)
            loss.backward()
            optimizer.step()
        self.theta = theta.detach().cpu().numpy()
        self.bias  = float(bias.detach().cpu().numpy())
        self._fitted = True
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        per_token = X @ self.theta  # (N, S)
        logits = per_token.max(axis=1) + self.bias
        return logits

class LastTokenProbe(BaseProbe):
    name = "last_token"
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.theta = np.zeros(d_model)
        self.bias = 0.0
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, lr=1e-2, epochs=100):
        # X: (N, S, D), y: (N,)
        N, S, D = X.shape
        device = "cuda" if torch.cuda.is_available() else "cpu"
        theta = torch.nn.Parameter(torch.zeros(D, device=device))
        bias  = torch.nn.Parameter(torch.zeros(1, device=device))
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)
        optimizer = torch.optim.Adam([theta, bias], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            last_token = X_tensor[:, -1, :]  # (N, D)
            logits = last_token @ theta + bias  # (N,)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tensor)
            loss.backward()
            optimizer.step()
        self.theta = theta.detach().cpu().numpy()
        self.bias  = float(bias.detach().cpu().numpy())
        self._fitted = True
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        last_token = X[:, -1, :]
        logits = last_token @ self.theta + self.bias
        return logits

class SoftmaxProbe(BaseProbe):
    name = "softmax"
    def __init__(self, d_model: int, temperature: float = 1.0):
        self.d_model = d_model
        self.theta = np.zeros(d_model)
        self.bias = 0.0
        self.temperature = temperature
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, lr=1e-2, epochs=100):
        # X: (N, S, D), y: (N,)
        N, S, D = X.shape
        device = "cuda" if torch.cuda.is_available() else "cpu"
        theta = torch.nn.Parameter(torch.zeros(D, device=device))
        bias  = torch.nn.Parameter(torch.zeros(1, device=device))
        temp  = torch.tensor(self.temperature, device=device)
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)
        optimizer = torch.optim.Adam([theta, bias], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            scores = torch.einsum('nsd,d->ns', X_tensor, theta)  # (N, S)
            attn = torch.softmax(scores / temp, dim=1)  # (N, S)
            logits = torch.sum(attn * scores, dim=1) + bias  # (N,)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tensor)
            loss.backward()
            optimizer.step()
        self.theta = theta.detach().cpu().numpy()
        self.bias  = float(bias.detach().cpu().numpy())
        self._fitted = True
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        scores = X @ self.theta  # (N, S)
        attn = np.exp(scores / self.temperature)
        attn = attn / (np.sum(attn, axis=1, keepdims=True) + 1e-9)
        logits = np.sum(attn * scores, axis=1) + self.bias
        return logits

class AttentionProbe(BaseProbe):
    name = "attention"
    def __init__(self, d_model: int, temperature: float = 1.0):
        self.d_model = d_model
        self.theta_q = np.random.randn(d_model) * 0.01
        self.theta_v = np.random.randn(d_model) * 0.01
        self.bias = 0.0
        self.temperature = temperature
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 1e-2, epochs: int = 100):
        N, S, D = X.shape
        device = "cuda" if torch.cuda.is_available() else "cpu"
        theta_q = torch.nn.Parameter(torch.from_numpy(self.theta_q).float().to(device))
        theta_v = torch.nn.Parameter(torch.from_numpy(self.theta_v).float().to(device))
        bias    = torch.nn.Parameter(torch.zeros(1, device=device))
        temp    = torch.tensor(self.temperature, device=device)
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).float().to(device)
        optimizer = torch.optim.Adam([theta_q, theta_v, bias], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            q = (X_tensor @ theta_q) / temp  # (N, S)
            attn = torch.softmax(q, dim=1)   # (N, S)
            v = (X_tensor @ theta_v)         # (N, S)
            logits = torch.sum(attn * v, dim=1) + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tensor)
            loss.backward()
            optimizer.step()
        self.theta_q = theta_q.detach().cpu().numpy()
        self.theta_v = theta_v.detach().cpu().numpy()
        self.bias = float(bias.detach().cpu().numpy())
        self._fitted = True
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        q = (X @ self.theta_q) / self.temperature  # (batch, seq_len)
        attn = np.exp(q)
        attn = attn / (np.sum(attn, axis=1, keepdims=True) + 1e-9)
        v = X @ self.theta_v  # (batch, seq_len)
        logits = np.sum(attn * v, axis=1) + self.bias
        return logits
