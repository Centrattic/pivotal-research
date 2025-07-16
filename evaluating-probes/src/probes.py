import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Any, Optional

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score
from src.logger import Logger
import sys

def numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """A numerically stable softmax implementation using NumPy."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class BaseProbe:
    """Abstract base class for all probes."""
    name: str = "base"
    task_type: str
    model: Any # Will hold the sklearn model instance

    def fit(self, X, y, **kwargs):
        raise NotImplementedError
        
    def save_state(self, path: Path):
        raise NotImplementedError

    def load_state(self, path: Path, logger: Logger):
        raise NotImplementedError

    def score(self, X: np.ndarray, y: np.ndarray, aggregation: str) -> dict[str, float]:
        predictions = self.predict(X, aggregation)
        
        if self.task_type == 'regression':
            return {"r2_score": float(r2_score(y, predictions)), "mse": float(mean_squared_error(y, predictions))}
        
        is_multiclass = predictions.ndim == 2 and predictions.shape[1] > 1
        if not is_multiclass: # Binary
            y_prob = 1 / (1 + np.exp(-predictions.squeeze()))
            y_hat = (y_prob > 0.5).astype(int)
            auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.5
        else: # Multiclass
            y_prob = numpy_softmax(predictions, axis=1)
            y_hat = np.argmax(predictions, axis=1)
            auc = roc_auc_score(y, y_prob, multi_class='ovr')

        return {"acc": float(accuracy_score(y, y_hat)), "auc": auc}

    def predict(self, X: np.ndarray, aggregation: str) -> np.ndarray:
        """Predicts either class logits or continuous values."""
        raise NotImplementedError

class LinearProbe(BaseProbe):
    """
    Token-level linear / logistic probe.

    * Stage-1:  train on **non-padded tokens**  (N×S_valid, D)
    * Stage-2:  aggregate token logits back to sequence level
    """

    name = "linear"

    # --------------------------------------------------------------------- #
    def __init__(self, d_model: int, device: str):
        self.d_model = d_model
        self.device  = device

        self.model:  Optional[LogisticRegression | LinearRegression] = None
        self.scaler: Optional[StandardScaler]                        = None
        self.task_type: str = ""

    # --------------------------------------------------------------------- #
    @staticmethod
    def _pad_mask(X: np.ndarray) -> np.ndarray:
        """
        Return a bool mask  (N, S)  where True means “this token row is *not*
        padding” (i.e. at least one dimension is non-zero).
        """
        return (X != 0).any(axis=2)

    # --------------------------------------------------------------------- #
    def _aggregate(self, logits: np.ndarray, mode: str) -> np.ndarray:
        if mode == "mean":
            return logits.mean(axis=1)
        if mode == "max":
            return logits.max(axis=1)
        if mode == "last":
            return logits[np.arange(len(logits)), (logits != 0).sum(1) - 1]
        if mode == "softmax":
            w = np.exp(logits - logits.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True) + 1e-9
            return (logits * w).sum(axis=1)
        raise ValueError(f"Unknown aggregation '{mode}'")

    # --------------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray, **lr_kwargs):
        """
        X : (N, S, D)  activations
        y : (N,)       integer labels or floats
        """
        N, S, D = X.shape
        assert D == self.d_model, "d_model mismatch"

        pad_mask   = self._pad_mask(X)            # (N, S)
        mask_flat  = pad_mask.reshape(-1)         # (N·S,)
        X_tokens   = X.reshape(-1, D)[mask_flat]  # keep only real tokens
        y_tokens   = np.repeat(y, S)[mask_flat]

        # scale
        self.scaler = StandardScaler(with_mean=False)   # sparse-friendly
        X_scaled    = self.scaler.fit_transform(X_tokens)

        # model
        if np.issubdtype(y.dtype, np.integer):
            self.task_type = "classification"
            self.model = LogisticRegression(
                max_iter=10_000,
                class_weight="balanced",
                random_state=42,
                **lr_kwargs,
            )
        else:
            self.task_type = "regression"
            self.model = LinearRegression(**lr_kwargs)

        self.model.fit(X_scaled, y_tokens)
        return self

    # --------------------------------------------------------------------- #
    def predict(self, X: np.ndarray, aggregation: str = "mean") -> np.ndarray:
        assert self.model is not None and self.scaler is not None

        N, S, D      = X.shape
        pad_mask      = self._pad_mask(X)
        mask_flat     = pad_mask.reshape(-1)
        X_tokens_flat = X.reshape(-1, D)[mask_flat]

        # scale + token-level logits
        X_scaled      = self.scaler.transform(X_tokens_flat)
        if self.task_type == "classification":
            if len(getattr(self.model, "classes_", [0, 1])) == 2:
                tok_logits = self.model.decision_function(X_scaled)
            else:
                probs      = self.model.predict_proba(X_scaled)[:, 1]
                tok_logits = np.log(probs / (1 - probs + 1e-9))
        else:
            tok_logits = self.model.predict(X_scaled)

        # put logits back into full (N,S) array, zeros for padding
        token_logits = np.zeros((N * S,), dtype=tok_logits.dtype)
        token_logits[mask_flat] = tok_logits
        token_logits = token_logits.reshape(N, S)

        # sequence-level aggregation
        return self._aggregate(token_logits, aggregation)


# --------------------------------------------------------------------------- #
# DEBUG AttentionProbe with infinite prints
# --------------------------------------------------------------------------- #
import pandas as pd
import sys

class AttentionProbe(BaseProbe):
    """
    Two-stage logistic probe with *very noisy* debugging prints.
    DO NOT use in production – this is only for inspecting the data flow.
    """

    name: str = "attention"

    def __init__(self, d_model: int, device: str):
        self.d_model = d_model
        self.device = device

        # stage-1
        self.scaler_tok: Optional[StandardScaler] = None
        self.lr_tok: Optional[LogisticRegression | LinearRegression] = None

        # stage-2
        self.scaler_seq: Optional[StandardScaler] = None
        self.lr_seq: Optional[LogisticRegression | LinearRegression] = None

        self.task_type: str = ""
        self.loss_history = []

    # --------------------------------------------------------------------- #
    def _print_df(self, arr: np.ndarray, name: str, head_rows: int = 5, head_cols: int = 5):
        """Pretty-print a small slice of an ndarray as a DataFrame."""
        r, c = arr.shape
        df = pd.DataFrame(arr[:head_rows, :head_cols])
        print(f"\n===== {name}  shape={arr.shape}  =====", file=sys.stdout, flush=True)
        print(df, file=sys.stdout, flush=True)

    # --------------------------------------------------------------------- #
    def fit(
        self,
        X: np.ndarray,  # (N, S, D)
        y: np.ndarray,  # (N,)
        *,
        stage1_kwargs: dict[str, Any] | None = None,
        stage2_kwargs: dict[str, Any] | None = None,
    ):
        stage1_kwargs = stage1_kwargs or {}
        stage2_kwargs = stage2_kwargs or {}

        N, S, D = X.shape
        is_classification = np.issubdtype(y.dtype, np.integer)

        # ------------- Stage-1: token-level --------------------------------
        X_tok = X.reshape(N * S, D)
        y_tok = np.repeat(y, S)

        self._print_df(X_tok, "RAW  X_tok")

        self.scaler_tok = StandardScaler()
        X_tok_scaled = self.scaler_tok.fit_transform(X_tok)

        self._print_df(X_tok_scaled, "SCALED  X_tok")

        if is_classification:
            self.task_type = "classification"
            self.lr_tok = LogisticRegression(max_iter=10_000, **stage1_kwargs)
        else:
            self.task_type = "regression"
            self.lr_tok = LinearRegression(**stage1_kwargs)

        self.lr_tok.fit(X_tok_scaled, y_tok)

        # token logits
        if is_classification and len(self.lr_tok.classes_) == 2:
            token_logits = self.lr_tok.decision_function(X_tok_scaled).reshape(N, S)
        elif is_classification:
            probs = self.lr_tok.predict_proba(X_tok_scaled).reshape(N, S, -1)
            token_logits = np.log(probs / (1 - probs + 1e-9))[..., 1]
        else:
            token_logits = self.lr_tok.predict(X_tok_scaled).reshape(N, S)

        self._print_df(token_logits, "TOKEN  logits")

        # ------------- Stage-2: sequence-level -----------------------------
        self.scaler_seq = StandardScaler()
        X_seq_scaled = self.scaler_seq.fit_transform(token_logits)

        self._print_df(X_seq_scaled, "SCALED  X_seq (token logits)")

        if is_classification:
            self.lr_seq = LogisticRegression(max_iter=10_000, **stage2_kwargs)
        else:
            self.lr_seq = LinearRegression(**stage2_kwargs)

        self.lr_seq.fit(X_seq_scaled, y)

        # ---- final diagnostics ------------------------------------------
        if is_classification:
            final_logits = self.lr_seq.decision_function(X_seq_scaled)
            y_prob = 1 / (1 + np.exp(-final_logits))
            print(
                "\n===== FINAL sequence logits  (first 10) =====",
                file=sys.stdout,
                flush=True,
            )
            print(final_logits[:10], file=sys.stdout, flush=True)
            print(
                "===== FINAL sequence probs   (first 10) =====",
                file=sys.stdout,
                flush=True,
            )
            print(y_prob[:10], file=sys.stdout, flush=True)
        else:
            preds = self.lr_seq.predict(X_seq_scaled)
            print(
                "\n===== FINAL sequence preds   (first 10) =====",
                file=sys.stdout,
                flush=True,
            )
            print(preds[:10], file=sys.stdout, flush=True)

        return self

    # --------------------------------------------------------------------- #
    def predict(self, X: np.ndarray, aggregation: str = "two_stage") -> np.ndarray:
        # (same as previous version – no extra prints here)
        assert (
            self.lr_tok is not None
            and self.scaler_tok is not None
            and self.lr_seq is not None
            and self.scaler_seq is not None
        ), "Probe not fitted."

        N, S, D = X.shape
        X_tok = X.reshape(N * S, D)
        X_tok_scaled = self.scaler_tok.transform(X_tok)

        if self.task_type == "classification" and len(self.lr_tok.classes_) == 2:
            token_logits = self.lr_tok.decision_function(X_tok_scaled).reshape(N, S)
        elif self.task_type == "classification":
            probs = self.lr_tok.predict_proba(X_tok_scaled).reshape(N, S, -1)
            token_logits = np.log(probs / (1 - probs + 1e-9))[..., 1]
        else:
            token_logits = self.lr_tok.predict(X_tok_scaled).reshape(N, S)

        X_seq_scaled = self.scaler_seq.transform(token_logits)
        if self.task_type == "classification":
            return self.lr_seq.decision_function(X_seq_scaled)
        else:
            return self.lr_seq.predict(X_seq_scaled)

    # save_state / load_state identical to previous two-stage version …


    # --------------------------------------------------------------------- #
    def save_state(self, path: Path):
        assert (
            self.lr_tok is not None
            and self.lr_seq is not None
            and self.scaler_tok is not None
            and self.scaler_seq is not None
        )
        np.savez(
            path,
            task_type=self.task_type,
            # stage‑1
            tok_coef=self.lr_tok.coef_,
            tok_int=self.lr_tok.intercept_,
            tok_classes=getattr(self.lr_tok, "classes_", None),
            tok_mu=self.scaler_tok.mean_,
            tok_sigma=self.scaler_tok.scale_,
            # stage‑2
            seq_coef=self.lr_seq.coef_,
            seq_int=self.lr_seq.intercept_,
            seq_classes=getattr(self.lr_seq, "classes_", None),
            seq_mu=self.scaler_seq.mean_,
            seq_sigma=self.scaler_seq.scale_,
        )

    # --------------------------------------------------------------------- #
    def load_state(self, path: Path, logger: Logger):
        data = np.load(path, allow_pickle=True)
        self.task_type = str(data["task_type"])

        # stage‑1
        if self.task_type == "classification":
            self.lr_tok = LogisticRegression()
            self.lr_tok.classes_ = data["tok_classes"]
        else:
            self.lr_tok = LinearRegression()
        self.lr_tok.coef_ = data["tok_coef"]
        self.lr_tok.intercept_ = data["tok_int"]

        self.scaler_tok = StandardScaler()
        self.scaler_tok.mean_ = data["tok_mu"]
        self.scaler_tok.scale_ = data["tok_sigma"]

        # stage‑2
        if self.task_type == "classification":
            self.lr_seq = LogisticRegression()
            self.lr_seq.classes_ = data["seq_classes"]
        else:
            self.lr_seq = LinearRegression()
        self.lr_seq.coef_ = data["seq_coef"]
        self.lr_seq.intercept_ = data["seq_int"]

        self.scaler_seq = StandardScaler()
        self.scaler_seq.mean_ = data["seq_mu"]
        self.scaler_seq.scale_ = data["seq_sigma"]

        logger.log(f"  - Loaded two‑stage AttentionProbe from {path}")
