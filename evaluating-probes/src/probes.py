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
    Linear / logistic probe that **trains on token-level activations**.

    * Training set:          (N, S, D) → (N × S, D)
    * Prediction procedure:  (N, S, D) → token logits → aggregate back to (N,…)
    """

    name: str = "linear"

    def __init__(self, d_model: int, device: str):
        self.d_model = d_model
        self.device = device

        self.model: Optional[LogisticRegression | LinearRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.loss_history: list[float] = []          # still tracked by runner

    def aggregate_logits(self, logits: np.ndarray, aggregation: str) -> np.ndarray:
        """
        Aggregates token-level logits back to sequence-level.

        `logits` shape:
            • binary   → (N, S)
            • multicls → (N, S, C)
        """
        if aggregation == "mean":
            return logits.mean(axis=1)
        if aggregation == "max":
            return logits.max(axis=1)
        if aggregation == "last":
            return logits[:, -1]
        if aggregation == "softmax":
            w = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            w = w / (w.sum(axis=1, keepdims=True) + 1e-9)
            return (logits * w).sum(axis=1)

        raise ValueError(f"Unknown aggregation: {aggregation}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        X : ndarray (N, S, D)
        y : ndarray (N,)   – integer classes or floats
        """
        is_classification = np.issubdtype(y.dtype, np.integer)

        N, S, D = X.shape
        assert D == self.d_model, "d_model mismatch"

        # flatten tokens into rows
        X_tokens = X.reshape(N * S, D)               # (N·S, D)
        y_tokens = np.repeat(y, S)                   # duplicate labels

        # scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_tokens)

        # model 
        if is_classification:
            self.task_type = "classification"
            print(kwargs, file=sys.stdout, flush=True)
            self.model = LogisticRegression(verbose=3, random_state=42, **kwargs)
        else:
            self.task_type = "regression"
            self.model = LinearRegression()

        self.model.fit(X_scaled, y_tokens)
        self.theta_ = self.model.coef_.squeeze()     # saved for analysis
        return self

    def predict(self, X: np.ndarray, aggregation: str = "mean") -> np.ndarray:
        """
        Returns sequence-level logits / predictions after aggregating over
        tokens with `aggregation`.
        """
        assert self.model is not None and self.scaler is not None, "Probe not trained."

        N, S, D = X.shape
        X_tokens = X.reshape(N * S, D)
        X_scaled = self.scaler.transform(X_tokens)

        # token-level logits / outputs
        if isinstance(self.model, LogisticRegression):
            if len(self.model.classes_) > 2:             # multiclass
                probs = self.model.predict_proba(X_scaled)
                token_logits = np.log(probs / (1 - probs + 1e-9))  # (N·S, C)
            else:                                        # binary
                token_logits = self.model.decision_function(X_scaled)  # (N·S,)
        elif isinstance(self.model, LinearRegression):
            token_logits = self.model.predict(X_scaled)  # (N·S,) or (N·S, C?)
        else:
            raise TypeError(f"Unsupported model: {type(self.model)}")

        # reshape back to (N, S, …) then aggregate
        if token_logits.ndim == 1:                       # binary scalar logit
            token_logits = token_logits.reshape(N, S)
        else:                                            # (N·S, C)
            token_logits = token_logits.reshape(N, S, -1)

        return self.aggregate_logits(token_logits, aggregation)

    def save_state(self, path: Path):
        assert self.model is not None and self.scaler is not None
        np.savez(
            path,
            coef=self.model.coef_,
            intercept=self.model.intercept_,
            task_type=self.task_type,
            classes=getattr(self.model, "classes_", None),
            scaler_mean=self.scaler.mean_,
            scaler_scale=self.scaler.scale_,
        )

    def load_state(self, path: Path, logger: "Logger"):
        data = np.load(path, allow_pickle=True)

        self.task_type = str(data["task_type"])
        if self.task_type == "classification":
            self.model = LogisticRegression()
            self.model.classes_ = data["classes"]
        else:
            self.model = LinearRegression()

        self.model.coef_ = data["coef"]
        self.model.intercept_ = data["intercept"]

        # rebuild scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = data["scaler_mean"]
        self.scaler.scale_ = data["scaler_scale"]

        logger.log(f"  - Loaded probe (token-level) from {path}")


class AttentionProbe(BaseProbe):
    """
    Single-head attention probe matching the SAE-Probe implementation.
    Trains on token-level activations and records per-epoch losses.

    Key choices:
      • token-wise layer-norm on inputs
      • softmax temperature = 1/√d
      • Adam (no weight-decay) with optional warm-up
      • mini-batch training
    """
    name: str = "attention"

    def __init__(self, d_model: int, device):
        self.d_model = d_model
        self.device = device
        self.theta_q: Optional[np.ndarray] = None
        self.theta_v: Optional[np.ndarray] = None
        self.bias:    Optional[np.ndarray] = None
        self.loss_history: list[float] = []

    def fit(  # noqa: PLR0913, N802
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 30,
        batch_size: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
    ):
        """Token-level mini-batch training with Adam."""
        N, S, D = X.shape
        classification = np.issubdtype(y.dtype, np.integer)

        # task-specific head
        if classification:
            n_classes = len(np.unique(y))
            out_features = n_classes if n_classes > 2 else 1
            loss_fn = (
                torch.nn.CrossEntropyLoss()
                if n_classes > 2
                else torch.nn.BCEWithLogitsLoss()
            )
        else:
            out_features = 1
            loss_fn = torch.nn.MSELoss()
            self.task_type = "regression"

        if classification:
            self.task_type = "classification"
            self.n_classes = out_features if out_features > 1 else 2
        else:
            self.n_classes = None

        # parameters
        device = self.device if torch.cuda.is_available() else "cpu"
        theta_q = torch.nn.Parameter(torch.randn(D, device=device) * 0.02)
        theta_v = torch.nn.Parameter(torch.randn(out_features, D, device=device) * 0.02)

        p = y.mean()         # empirical positive rate
        bias_val = np.log(p / (1 - p + 1e-9))  # avoid div-by-zero
        bias = torch.nn.Parameter(torch.full((out_features,), bias_val, device=device))

        optim = torch.optim.Adam(
            [theta_q, theta_v, bias], lr=lr, weight_decay=weight_decay
        )
        sched = (
            torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.0, total_iters=warmup_steps)
            if warmup_steps > 0
            else None
        )

        X_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(y).to(device)
        sqrt_d = np.sqrt(D)

        self.loss_history = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for idx in torch.randperm(N).split(batch_size):
                optim.zero_grad()

                x_b = torch.nn.functional.layer_norm(X_t[idx], (D,))
                y_b = y_t[idx]

                attn_scores = torch.einsum("bsd,d->bs", x_b, theta_q) / sqrt_d
                attn_w = torch.softmax(attn_scores, dim=1)

                value_scores = torch.einsum("bsd,cd->bsc", x_b, theta_v)
                agg = torch.einsum("bs,bsc->bc", attn_w, value_scores)
                preds = (agg + bias).squeeze()

                loss = loss_fn(
                    preds,
                    y_b.long() if classification and out_features > 1 else y_b.float(),
                )
                loss.backward()
                optim.step()
                if sched:
                    sched.step()

                epoch_loss += loss.item() * idx.size(0)

            self.loss_history.append(epoch_loss / N)

        # save learned weights
        self.theta_q = theta_q.detach().cpu().numpy()
        self.theta_v = theta_v.detach().cpu().numpy()
        self.bias = bias.detach().cpu().numpy()
        return self

    def predict(self, X: np.ndarray, aggregation: str = "attention") -> np.ndarray:
        """
        NumPy inference that mirrors the PyTorch forward-pass.
        `aggregation` is ignored (only one mode).
        """
        assert (
            self.theta_q is not None and self.theta_v is not None and self.bias is not None
        ), "Probe not fitted"

        # token-wise LN in NumPy
        μ = X.mean(axis=-1, keepdims=True)
        σ = X.std(axis=-1, keepdims=True) + 1e-6
        X_norm = (X - μ) / σ                                                  # (N, S, D)

        attn_scores = np.einsum("nsd,d->ns", X_norm, self.theta_q) / np.sqrt(self.d_model)
        attn_w = numpy_softmax(attn_scores, axis=1)                           # (N, S)

        if self.task_type == "classification" and self.theta_v.shape[0] > 1:
            value_scores = np.einsum("nsd,cd->nsc", X_norm, self.theta_v)     # (N,S,C)
            agg_scores = np.einsum("ns,nsc->nc", attn_w, value_scores)        # (N,C)
        else:
            value_scores = np.einsum("nsd,d->ns", X_norm, self.theta_v.squeeze())  # (N,S)
            agg_scores = np.einsum("ns,ns->n", attn_w, value_scores)               # (N,)

        return (agg_scores + self.bias).squeeze()

    # --------------------------------------------------------------------- #
    def save_state(self, path: Path):
        np.savez(
            path,
            theta_q=self.theta_q,
            theta_v=self.theta_v,
            bias=self.bias,
            name=self.name,
            task_type=self.task_type,
            n_classes=(self.theta_v.shape[0] if self.task_type == "classification" else -1),
        )

    def load_state(self, path: Path, logger: Logger):
        data = np.load(path)
        self.theta_q = data["theta_q"]
        self.theta_v = data["theta_v"]
        self.bias = data["bias"]
        self.task_type = str(data["task_type"])
        self.n_classes = (
            int(data["n_classes"]) if self.task_type == "classification" else None
        )
        logger.log(f"  - Loaded '{self.name}' probe ({self.task_type}) from {path}")
