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
    A probe using a linear model from scikit-learn. It automatically chooses
    Logistic Regression for classification and Linear Regression for regression.
    """
    name: str = "linear"

    def __init__(self, d_model: int, device):
        self.d_model = d_model
        self.model: Optional[LogisticRegression | LinearRegression] = None
        self.device = device
        self.loss_history: list[float] = []

    def aggregate_activations(self, logits: np.ndarray, aggregation: str) -> np.ndarray:
        """Aggregates activations from (N, S, D) to (N, D) before feeding to sklearn."""
        # Now aggregate over sequence axis (axis=1)
        if aggregation == "mean":
            return logits.mean(axis=1)
        elif aggregation == "max":
            return logits.max(axis=1)
        elif aggregation == "softmax":
            # Softmax pooling across sequence axis
            w = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            w = w / (w.sum(axis=1, keepdims=True) + 1e-9)
            return (logits * w).sum(axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        is_classification = np.issubdtype(y.dtype, np.integer)
        
        # Training flattens across sequence length
        N, S, D = X.shape
        X_flat = X.reshape(N, S*D)

        self.scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = self.scaler.fit_transform(X_flat)

        if is_classification:
            self.task_type = 'classification'
            print(**kwargs, file=sys.stdout, flush=True)
            self.model = LogisticRegression(verbose=3, random_state=42, **kwargs)
        else: # Regression
            self.task_type = 'regression'
            self.model = LinearRegression()

        self.model.fit(X_scaled, y)
        # Store for softmax aggregation during prediction
        self.theta_ = self.model.coef_.squeeze()
        return self

    def predict(self, X: np.ndarray, aggregation: str) -> np.ndarray:
        assert self.model is not None, "Probe has not been trained."

        N, S, D = X.shape
        X_flat = X.reshape(N, S*D)
        X_scaled = self.scaler.transform(X_flat)
        
        if isinstance(self.model, LogisticRegression):
            # For binary, decision_function gives logits. For multiclass, predict_proba is used.
            if len(self.model.classes_) > 2:
                probs = self.model.predict_proba(X_scaled)
                # Convert probabilities to logits
                logits = np.log(probs / (1 - probs + 1e-9))
                print(logits)
            else:
                logits = self.model.decision_function(X_scaled)
        elif isinstance(self.model, LinearRegression):
            logits = self.model.predict(X_scaled)
        else:
            raise TypeError(f"Unsupported model type for prediction: {type(self.model)}")
    
        # final_score = self._aggregate_activations(logits, aggregation) # not even doing an aggregation right now
        return logits
    
    def save_state(self, path: Path):
        assert self.model is not None

        np.savez(
            path,
            coef=self.model.coef_, intercept=self.model.intercept_,
            task_type=self.task_type,
            classes=getattr(self.model, "classes_", None),
            scaler_mean=self.scaler.mean_,
            scaler_scale=self.scaler.scale_,
        )

    def load_state(self, path: Path, logger: Logger):
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
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.scaler.mean_  = data["scaler_mean"]
        self.scaler.scale_ = data["scaler_scale"]
        logger.log(f"  - Loaded probe (with StandardScaler) from {path}")


class AttentionProbe(BaseProbe):
    """
    Single-head attention probe matching the settings from SAE-Probes
    Key choices:
      • token-wise layer-norm on inputs
      • softmax temperature = 1/√d, as SAE paper recommended
      • Adam (no weight-decay) with optional warm-up
      • mini-batch training
    """
    name: str = "attention"

    def __init__(self, d_model: int, device):
        self.d_model = d_model

        # learnable parameters (populated in `fit`)
        self.theta_q: Optional[np.ndarray] = None
        self.theta_v: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.device = device

        # recorded every *epoch*  → dumped by runner
        self.loss_history: list[float] = []

    def fit(  # noqa: PLR0913, N802
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 30,
        batch_size: int = 512,
        lr: float = 1e-3,
        warmup_steps: int = 0,
    ):
        """
        Parameters
        warmup_steps : int
            If >0, use linear LR warm-up for this many *optimizer steps*.
        """
        N, S, D = X.shape
        assert D == self.d_model, "d_model mismatch"

        classification = np.issubdtype(y.dtype, np.integer)
        if classification:
            self.task_type = "classification"
            n_classes = len(np.unique(y))
            out_features = n_classes if n_classes > 2 else 1
            loss_fn = (
                torch.nn.CrossEntropyLoss()
                if n_classes > 2
                else torch.nn.BCEWithLogitsLoss()
            )
        else:
            self.task_type = "regression"
            out_features = 1
            loss_fn = torch.nn.MSELoss()

        device = self.device if torch.cuda.is_available() else "cpu"
        theta_q = torch.nn.Parameter(
            torch.randn(D, device=device) * 0.02
        )                                              # (D,)
        theta_v = torch.nn.Parameter(
            torch.randn(out_features, D, device=device) * 0.02
        )                                              # (C, D) or (1, D)
        bias = torch.nn.Parameter(torch.zeros(out_features, device=device))  # (C,) or (1,)

        optim = torch.optim.Adam([theta_q, theta_v, bias], lr=lr, weight_decay=0.0)

        if warmup_steps > 0:
            sched = torch.optim.lr_scheduler.LinearLR(
                optim, start_factor=0.0, total_iters=warmup_steps
            )
        else:
            sched = None

        # tensors
        X_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(y).to(device)

        sqrt_d = np.sqrt(D)
        self.loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            # shuffle indices each epoch
            for idx in torch.randperm(N).split(batch_size):
                optim.zero_grad()

                x_b = X_t[idx]                                     # (B, S, D)
                y_b = y_t[idx]

                # --- SAE trick: token-wise layer-norm --------------------
                x_b = torch.nn.functional.layer_norm(x_b, (D,))

                # attention
                attn_scores = torch.einsum("bsd,d->bs", x_b, theta_q) / sqrt_d
                attn_weights = torch.softmax(attn_scores, dim=1)          # (B, S)

                value_scores = torch.einsum("bsd,cd->bsc", x_b, theta_v)   # (B, S, C)
                aggregated = torch.einsum("bs,bsc->bc", attn_weights, value_scores)

                preds = (aggregated + bias).squeeze()                     # (B,) or (B, C)

                loss = loss_fn(
                    preds,
                    y_b.long() if classification and out_features > 1 else y_b.float(),
                )
                loss.backward()
                optim.step()
                if sched:
                    sched.step()

                epoch_loss += loss.item() * idx.size(0)

            # mean loss for the epoch
            self.loss_history.append(epoch_loss / N)

        # ---------------------------------------------------------------- #
        # Save learned weights in NumPy format for CPU inference
        # ---------------------------------------------------------------- #
        self.theta_q = theta_q.detach().cpu().numpy()
        self.theta_v = theta_v.detach().cpu().numpy()
        self.bias = bias.detach().cpu().numpy()

        return self

    # --------------------------------------------------------------------- #
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
