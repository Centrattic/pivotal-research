from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Literal, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

class BaseProbe:
    """Abstract convenience wrapper (not strictly necessary)."""

    name: str = "base_probe"

    def fit(self, X: np.ndarray, y: np.ndarray):  # noqa: D401 (simple docstring)
        raise NotImplementedError

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Return 1‑D logits (real‑valued) for class *1*."""
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

    def _check_dims(self, X: np.ndarray, y: np.ndarray):  # noqa: D401
        assert X.ndim == 2, "X must be 2‑D (N, d)"
        assert y.ndim == 1, "y must be 1‑D"
        assert X.shape[0] == y.shape[0], "N mismatch"

# Log-Reg Probe
@dataclass
class LRConfig:
    penalty: Literal["l1", "l2"] = "l2"
    C: float = 1.0               # inverse regularisation
    solver: str = "liblinear"    # supports both l1 & l2
    max_iter: int = 1000


class LogisticRegressionProbe(BaseProbe):
    """Thin wrapper around sklearn LogisticRegression."""

    name: str = "logreg"

    def __init__(self, cfg: LRConfig | None = None):
        self.cfg = cfg or LRConfig()
        self._clf: Optional[LogisticRegression] = None

    # --------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):  # noqa: D401 (docstring)
        self._check_dims(X, y)
        self._clf = LogisticRegression(**asdict(self.cfg))
        self._clf.fit(X, y)
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        assert self._clf is not None, "Probe not fitted yet"
        # sklearn returns 2‑D probas -> take positive‑class column
        probs = self._clf.predict_proba(X)[:, 1]
        # Convert to logits for consistency
        return np.log(probs / (1.0 - probs + 1e-9))

# Mass‑Mean Probe (mean diff + optional LDA tilt)
@dataclass
class MMConfig:
    tilt: bool = True      # True  -> use Σ^{-1} tilt (LDA style)
    reg: float = 1e-6      # Ridge for Σ inversion stability


class MassMeanProbe(BaseProbe):
    """Implements θ_mm = μ+ − μ−  with optional Σ^{-1} tilt.

    The decision rule is σ(θᵀx) where θ = Σ^{-1}(μ+ − μ−) if *tilt*
    else (μ+ − μ−).  The bias term is automatically set so that the
    threshold 0.5 lies halfway between projected means.
    """

    name: str = "mass_mean"

    def __init__(self, cfg: MMConfig | None = None):
        self.cfg = cfg or MMConfig()
        self.theta: Optional[np.ndarray] = None  # (d,)
        self.bias: float = 0.0

    # -------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._check_dims(X, y)
        pos, neg = X[y == 1], X[y == 0]
        mu_pos, mu_neg = pos.mean(0), neg.mean(0)
        diff = mu_pos - mu_neg  # θ_mm raw

        if self.cfg.tilt:
            # Shrink‑regularised covariance of *class‑centred* data
            centred = np.concatenate([pos - mu_pos, neg - mu_neg], axis=0)
            Σ = centred.T @ centred / centred.shape[0]
            Σ += np.eye(Σ.shape[0]) * self.cfg.reg
            diff = np.linalg.solve(Σ, diff)  # Σ^{-1} (μ+−μ−)

        self.theta = diff
        # bias so that σ=0.5 mid‑point between class means in θ‑space
        prj_pos = mu_pos @ self.theta
        prj_neg = mu_neg @ self.theta
        self.bias = -0.5 * (prj_pos + prj_neg)
        return self

    # -----------------------------------------------
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        assert self.theta is not None, "Probe not fitted yet"
        return X @ self.theta + self.bias


# Helper: flatten sequence into features if needed
def aggregate_sequence(t: torch.Tensor, how: str = "mean") -> np.ndarray:
    """Convert **(batch, seq_len, d_model)** → **(batch, d_model)**.

    *how* ∈ {"mean", "first", "last", "max"}.
    """
    if t.ndim != 3:
        raise ValueError("expected 3‑D tensor (batch, seq, d)")
    if how == "mean":
        out = t.mean(1)
    elif how == "first":
        out = t[:, 0, :]
    elif how == "last":
        out = t[:, -1, :]
    elif how == "max":
        out = t.max(1).values
    else:
        raise ValueError(f"Unknown agg '{how}'.")
    return out.cpu().numpy()
