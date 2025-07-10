# src/probes.py
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, Literal, Optional

from sklearn.metrics import accuracy_score, roc_auc_score

class BaseProbe:
    """Abstract base class for probes that learn a projection vector."""
    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray, aggregation: str, **kwargs):
        """Trains the probe's projection vector(s) theta."""
        raise NotImplementedError

    def predict_logits(self, X: np.ndarray, aggregation: str) -> np.ndarray:
        """Projects activations to scores, aggregates them, and returns a final logit."""
        raise NotImplementedError

    def score(self, X: np.ndarray, y: np.ndarray, aggregation: str) -> Dict[str, float]:
        """Calculates accuracy and AUC for the probe."""
        logits = self.predict_logits(X, aggregation=aggregation)
        y_prob = 1 / (1 + np.exp(-logits))
        y_hat = (y_prob > 0.5).astype(int)

        auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.5
        return {"acc": float(accuracy_score(y, y_hat)), "auc": auc}

    def save_state(self, path: Path):
        """Saves the learned probe parameters."""
        raise NotImplementedError

class LinearProbe(BaseProbe):
    """
    A standard linear probe that learns a single projection vector θ.
    It implements the 'project-then-aggregate' methodology.
    """
    name: str = "linear"

    def __init__(self, d_model: int):
        self.d_model = d_model
        self.theta: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def _aggregate_scores(self, scores: torch.Tensor, aggregation: str) -> torch.Tensor:
        if aggregation == "mean":
            return scores.mean(dim=1)
        if aggregation == "max":
            return scores.max(dim=1).values
        if aggregation == "last_token":
            return scores[:, -1]
        if aggregation == "softmax":
            # Softmax is applied to the scores to form a weighted average of the scores
            attn = torch.softmax(scores, dim=1)
            return torch.sum(attn * scores, dim=1)
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    def fit(self, X: np.ndarray, y: np.ndarray, aggregation: str, lr: float = 0.01, epochs: int = 100, weight_decay: float = 1e-2):
        self.aggregation = aggregation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize parameters
        theta = torch.nn.Parameter(torch.randn(self.d_model, device=device) * 0.02)
        bias = torch.nn.Parameter(torch.zeros(1, device=device))
        
        optimizer = torch.optim.AdamW([theta, bias], lr=lr, weight_decay=weight_decay)
        
        X_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(y).float().to(device)

        for _ in range(epochs):
            optimizer.zero_grad()
            
            # 1. Project all activation vectors to scalar scores
            scores = torch.einsum('nsd,d->ns', X_t, theta)
            
            # 2. Aggregate scores
            aggregated_score = self._aggregate_scores(scores, aggregation)
            
            # 3. Add bias to get final logits
            logits = aggregated_score + bias
            
            # 4. Calculate loss and update
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), y_t)
            loss.backward()
            optimizer.step()

        self.theta = theta.detach().cpu().numpy()
        self.bias = float(bias.detach().cpu().numpy())
        return self

    def predict_logits(self, X: np.ndarray, aggregation: str) -> np.ndarray:
        assert self.theta is not None, "Probe not fitted yet"
        scores = X @ self.theta
        
        # Use torch for aggregation functions for consistency
        scores_t = torch.from_numpy(scores)
        aggregated_score = self._aggregate_scores(scores_t, aggregation).numpy()
        
        return aggregated_score + self.bias

    def save_state(self, path: Path):
        np.savez(path, theta=self.theta, bias=self.bias, name=self.name)


class AttentionProbe(BaseProbe):
    """
    An attention probe that learns separate query (θq) and value (θv) vectors.
    """
    name: str = "attention"
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.theta_q: Optional[np.ndarray] = None
        self.theta_v: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, aggregation: str = "attention", lr: float = 0.01, epochs: int = 100, weight_decay: float = 1e-2):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        theta_q = torch.nn.Parameter(torch.randn(self.d_model, device=device) * 0.02)
        theta_v = torch.nn.Parameter(torch.randn(self.d_model, device=device) * 0.02)
        bias = torch.nn.Parameter(torch.zeros(1, device=device))
        
        optimizer = torch.optim.AdamW([theta_q, theta_v, bias], lr=lr, weight_decay=weight_decay)
        X_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(y).float().to(device)

        for _ in range(epochs):
            optimizer.zero_grad()
            
            # 1. Project to get attention scores and value scores
            attn_scores = torch.einsum('nsd,d->ns', X_t, theta_q)
            value_scores = torch.einsum('nsd,d->ns', X_t, theta_v)
            
            # 2. Compute attention weights
            attn_weights = torch.softmax(attn_scores, dim=1)
            
            # 3. Aggregate value scores using attention weights
            aggregated_score = torch.sum(attn_weights * value_scores, dim=1)
            
            logits = aggregated_score + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), y_t)
            loss.backward()
            optimizer.step()

        self.theta_q = theta_q.detach().cpu().numpy()
        self.theta_v = theta_v.detach().cpu().numpy()
        self.bias = float(bias.detach().cpu().numpy())
        return self

    def predict_logits(self, X: np.ndarray, aggregation: str = "attention") -> np.ndarray:
        assert self.theta_q is not None and self.theta_v is not None, "Probe not fitted yet"
        attn_scores = X @ self.theta_q
        value_scores = X @ self.theta_v
        
        # Softmax in numpy
        attn_weights = np.exp(attn_scores) / np.sum(np.exp(attn_scores), axis=1, keepdims=True)
        
        aggregated_score = np.sum(attn_weights * value_scores, axis=1)
        return aggregated_score + self.bias

    def save_state(self, path: Path):
        np.savez(path, theta_q=self.theta_q, theta_v=self.theta_v, bias=self.bias, name=self.name)