import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Any, Optional
import pandas as pd
import sys

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
            y_prob = predictions # 1 / (1 + np.exp(-predictions.squeeze()))
            print(f"PROBS", file=sys.stdout, flush=True)
            print(y_prob, file=sys.stdout, flush=True)
            y_hat = (y_prob > 0.5).astype(int)
            print(y_hat, file=sys.stdout, flush=True)
            print(y, file=sys.stdout, flush=True)
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
    Token-level (one agg option) linear / logistic probe.
    """
    name = "linear"

    def __init__(self, d_model: int, device: str):
        self.d_model = d_model
        self.device  = device

        self.model:  Optional[LogisticRegression | LinearRegression] = None
        self.scaler: Optional[StandardScaler]                        = None
        self.task_type: str = ""
        self.loss_history = []

    def aggregate(self, logits: np.ndarray, mode: str) -> np.ndarray:
        # Applies to sequences one at a time
        if mode == "mean":
            return logits.mean(axis=0)
        if mode == "max":
            return logits.max(axis=0)
        if mode == "last":
            return logits[-1]
        if mode == "softmax":
            w = np.exp(logits - logits.max(axis=0, keepdims=True))
            w /= w.sum(axis=0, keepdims=True) + 1e-9
            return (logits * w).sum(axis=0)
        raise ValueError(f"Unknown aggregation '{mode}'")

    def _print_df(self, arr: np.ndarray, name: str, head_rows: int = 5, head_cols: int = 5):
        import pandas as pd
        df = pd.DataFrame(arr[:head_rows, -head_cols:])
        print(f"\n===== {name}  shape={arr.shape}  =====", file=sys.stdout, flush=True)
        print(df, file=sys.stdout, flush=True)

    def fit(self, X: np.ndarray, y: np.ndarray, agg: str, **lr_kwargs):
        N, S, D = X.shape
        assert D == self.d_model, "d_model mismatch"

        print(f"\n===== LinearProbe.fit: X shape={X.shape}, y shape={y.shape} =====", file=sys.stdout, flush=True)

        X_seq = np.zeros((N, D), dtype=X.dtype)
        for i in range(N):
            # print(f"HELLLP + {X[i]}", file=sys.stdout, flush=True)
            X_seq[i] = self.aggregate(X[i], mode=agg) # ensure same agg!

        self._print_df(X_seq, "Sequence-level mean (X_seq)", head_cols=min(40, D))

        # scale
        self.scaler = StandardScaler(with_mean=True)
        X_scaled    = self.scaler.fit_transform(X_seq)
        print(f"\nScaled X_seq: shape={X_scaled.shape}", file=sys.stdout, flush=True)
        self._print_df(X_scaled, "X_scaled", head_cols=min(10, D))

        # model
        if np.issubdtype(y.dtype, np.integer):
            self.task_type = "classification"
            self.model = LogisticRegression(
                verbose=1,
                **lr_kwargs,
            )
        else:
            self.task_type = "regression"
            self.model = LinearRegression(**lr_kwargs)

        print(f"\nFitting {self.task_type} model on {X_scaled.shape[0]} sequences...", file=sys.stdout, flush=True)
        self.model.fit(X_scaled, y)
        print("Model fitting done.", file=sys.stdout, flush=True)
        return self

    def predict(self, X: np.ndarray, agg: str = "mean") -> np.ndarray:
        assert self.model is not None and self.scaler is not None

        N, S, D = X.shape
        print(f"\n===== LinearProbe.predict: X shape={X.shape} =====", file=sys.stdout, flush=True)

        X_seq = np.zeros((N, D), dtype=X.dtype)
        for i in range(N):
            X_seq[i] = self.aggregate(X[i], mode=agg) # [pad_mask[i]].mean(axis=0) oh i see why that was failing, diff aggs

        print(f"X_seq (predict): shape={X_seq.shape}", file=sys.stdout, flush=True)

        X_scaled = self.scaler.transform(X_seq)
        print(f"X_scaled (predict): shape={X_scaled.shape}", file=sys.stdout, flush=True)

        if self.task_type == "classification":
            if len(getattr(self.model, "classes_", [0, 1])) == 2:
                # seq_logits = self.model.decision_function(X_scaled)
                seq_logits = self.model.predict_proba(X_scaled)[:, 1]
            else:
                probs = self.model.predict_proba(X_scaled)[:, 1]
                seq_logits = np.log(probs / (1 - probs + 1e-9))
        else:
            seq_logits = self.model.predict(X_scaled)

        print(f"Sequence-level logits/preds: shape={seq_logits.shape}", file=sys.stdout, flush=True)
        print(f"First 10 sequence outputs: {seq_logits[:10]}", file=sys.stdout, flush=True)
        return seq_logits
    
    def save_state(self, path: Path):
        """Save model and scaler parameters to disk."""
        assert self.model is not None and self.scaler is not None

        # Prepare dictionary for saving
        save_dict = dict(
            task_type=self.task_type,
            coef=self.model.coef_,
            intercept=self.model.intercept_,
            mu = self.scaler.mean_,
            sigma=self.scaler.scale_,
        )
        if self.task_type == "classification" and hasattr(self.model, "classes_"):
            save_dict["classes"] = self.model.classes_

        np.savez(path, **save_dict)
        print(f"Saved LinearProbe to {path}", file=sys.stdout, flush=True)

    def load_state(self, path: Path, logger: Logger = None):
        """Load model and scaler parameters from disk."""
        data = np.load(path, allow_pickle=True)
        self.task_type = str(data["task_type"])

        # Restore model
        if self.task_type == "classification":
            self.model = LogisticRegression()
            if "classes" in data:
                self.model.classes_ = data["classes"]
        else:
            self.model = LinearRegression()

        self.model.coef_ = data["coef"]
        self.model.intercept_ = data["intercept"]

        # Restore scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = data["mu"]
        self.scaler.scale_ = data["sigma"]

        if logger:
            logger.log(f"  - Loaded LinearProbe from {path}")
        else:
            print(f"Loaded LinearProbe from {path}", file=sys.stdout, flush=True)

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

    def _print_df(self, arr: np.ndarray, name: str, head_rows: int = 5, head_cols: int = 5):
        """Pretty-print a small slice of an ndarray as a DataFrame."""
        r, c = arr.shape
        df = pd.DataFrame(arr[:head_rows, :head_cols])
        print(f"\n===== {name}  shape={arr.shape}  =====", file=sys.stdout, flush=True)
        print(df, file=sys.stdout, flush=True)

    def fit(
        self,
        X: np.ndarray,  # (N, S, D)
        y: np.ndarray,  # (N,)
        agg: str = "attention",
        *,
        stage1_kwargs: dict[str, Any] | None = None,
        stage2_kwargs: dict[str, Any] | None = None,
    ):
        stage1_kwargs = stage1_kwargs or {}
        stage2_kwargs = stage2_kwargs or {}

        N, S, D = X.shape
        is_classification = np.issubdtype(y.dtype, np.integer)

        X_tok = X.reshape(N * S, D)
        y_tok = np.repeat(y, S)

        self._print_df(X_tok, "RAW  X_tok")

        self.scaler_tok = StandardScaler()
        X_tok_scaled = self.scaler_tok.fit_transform(X_tok)

        self._print_df(X_tok_scaled, "SCALED  X_tok")

        if is_classification:
            self.task_type = "classification"
            self.lr_tok = LogisticRegression(**stage1_kwargs)
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

        # Stage-2: sequence-level
        self.scaler_seq = StandardScaler()
        X_seq_scaled = self.scaler_seq.fit_transform(token_logits)

        self._print_df(X_seq_scaled, "SCALED  X_seq (token logits)")

        if is_classification:
            self.lr_seq = LogisticRegression(verbose=1, **stage2_kwargs)
        else:
            self.lr_seq = LinearRegression(**stage2_kwargs)

        self.lr_seq.fit(X_seq_scaled, y)

        # final diagnostics
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

    def predict(self, X: np.ndarray, aggregation: str = "two_stage") -> np.ndarray:
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