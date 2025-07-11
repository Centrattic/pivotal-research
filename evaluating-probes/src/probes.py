import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score
from src.logger import Logger

def numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """A numerically stable softmax implementation using NumPy."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class BaseProbe:
    """Abstract base class for all probes."""
    name: str = "base"

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict_logits(self, X: np.ndarray, aggregation: str) -> np.ndarray:
        raise NotImplementedError
        
    def save_state(self, path: Path):
        raise NotImplementedError

    def load_state(self, path: Path, logger: Logger):
        raise NotImplementedError

    def score(self, X: np.ndarray, y: np.ndarray, aggregation: str) -> dict[str, float]:
        """Calculates metrics, automatically handling binary vs. multiclass cases."""
        logits = self.predict_logits(X, aggregation=aggregation)
        is_multiclass = logits.ndim == 2 and logits.shape[1] > 1

        if not is_multiclass:
            # Binary case
            y_prob = 1 / (1 + np.exp(-logits.squeeze()))
            y_hat = (y_prob > 0.5).astype(int)
            auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.5
        else:
            # Multiclass case
            y_prob = numpy_softmax(logits, axis=1)
            y_hat = np.argmax(logits, axis=1)
            auc = roc_auc_score(y, y_prob, multi_class='ovr')

        return {"acc": float(accuracy_score(y, y_hat)), "auc": auc}

class LinearProbe(BaseProbe):
    """A linear probe that learns a projection vector θ for each class."""
    name: str = "linear"

    def __init__(self, d_model: int, n_classes: int):
        self.d_model = d_model
        self.n_classes = n_classes
        self.theta: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def _aggregate_scores(self, scores: torch.Tensor, aggregation: str) -> torch.Tensor:
        if aggregation == "mean":
            return scores.mean(dim=1)
        if aggregation == "max":
            return scores.max(dim=1).values
        if aggregation == "last_token":
            return scores[:, -1]
        if aggregation == "softmax":
            if scores.ndim == 3: # (N, S, K) for multiclass
                norm_scores = torch.linalg.norm(scores, dim=2)
                attn = torch.softmax(norm_scores, dim=1).unsqueeze(-1)
            else: # (N, S) for binary
                attn = torch.softmax(scores, dim=1)
            return torch.sum(attn * scores, dim=1)
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    def fit(self, X: np.ndarray, y: np.ndarray, aggregation: str, lr: float = 0.01, epochs: int = 100, weight_decay: float = 1e-2):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        out_features = self.n_classes if self.n_classes > 2 else 1
        
        theta = torch.nn.Parameter(torch.randn(out_features, self.d_model, device=device) * 0.02)
        bias = torch.nn.Parameter(torch.zeros(out_features, device=device))
        
        optimizer = torch.optim.AdamW([theta, bias], lr=lr, weight_decay=weight_decay)
        X_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(y).long().to(device)
        loss_fn = torch.nn.CrossEntropyLoss() if self.n_classes > 2 else torch.nn.BCEWithLogitsLoss()

        for _ in tqdm(range(epochs), desc="  - Fitting Probe", leave=False):
            optimizer.zero_grad()
            scores = torch.einsum('nsd,cd->nsc', X_t, theta)
            aggregated_scores = self._aggregate_scores(scores, aggregation)
            logits = aggregated_scores + bias
            
            loss = loss_fn(logits.squeeze(), y_t.float() if self.n_classes <= 2 else y_t)
            loss.backward()
            optimizer.step()

        self.theta = theta.detach().cpu().numpy()
        self.bias = bias.detach().cpu().numpy()
        return self

    def predict_logits(self, X: np.ndarray, aggregation: str) -> np.ndarray:
        assert self.theta is not None and self.bias is not None, "Probe not fitted yet"
        
        if self.n_classes <= 2:
            scores = np.einsum('nsd,d->ns', X, self.theta.squeeze())
        else:
            scores = np.einsum('nsd,cd->nsc', X, self.theta)
        
        scores_t = torch.from_numpy(scores)
        aggregated_scores = self._aggregate_scores(scores_t, aggregation).numpy()
        
        return (aggregated_scores + self.bias).squeeze()
    
    def save_state(self, path: Path):
        np.savez(path, theta=self.theta, bias=self.bias, name=self.name, n_classes=self.n_classes)

    def load_state(self, path: Path, logger: Logger): # ✨ FIX: Added logger parameter
        data = np.load(path)
        self.theta = data['theta']
        self.bias = data['bias']
        self.n_classes = int(data['n_classes'])
        logger.log(f"  - Loaded pre-trained '{self.name}' probe state from {path}") # ✨ FIX: Used logger

class AttentionProbe(BaseProbe):
    """An attention probe that learns one query vector and K value vectors."""
    name: str = "attention"
    
    def __init__(self, d_model: int, n_classes: int):
        self.d_model = d_model
        self.n_classes = n_classes
        self.theta_q: np.ndarray | None = None
        self.theta_v: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, aggregation: str = "attention", lr: float = 0.01, epochs: int = 100, weight_decay: float = 1e-2):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        out_features = self.n_classes if self.n_classes > 2 else 1

        theta_q = torch.nn.Parameter(torch.randn(self.d_model, device=device) * 0.02)
        theta_v = torch.nn.Parameter(torch.randn(out_features, self.d_model, device=device) * 0.02)
        bias = torch.nn.Parameter(torch.zeros(out_features, device=device))
        
        optimizer = torch.optim.AdamW([theta_q, theta_v, bias], lr=lr, weight_decay=weight_decay)
        X_t = torch.from_numpy(X).float().to(device)
        y_t = torch.from_numpy(y).long().to(device)
        loss_fn = torch.nn.CrossEntropyLoss() if self.n_classes > 2 else torch.nn.BCEWithLogitsLoss()

        for _ in tqdm(range(epochs), desc="  - Fitting Probe", leave=False):
            optimizer.zero_grad()
            attn_scores = torch.einsum('nsd,d->ns', X_t, theta_q)
            attn_weights = torch.softmax(attn_scores, dim=1)
            
            value_scores = torch.einsum('nsd,cd->nsc', X_t, theta_v)
            aggregated_scores = torch.einsum('ns,nsc->nc', attn_weights, value_scores)
            
            logits = aggregated_scores + bias
            loss = loss_fn(logits.squeeze(), y_t.float() if self.n_classes <= 2 else y_t)
            loss.backward()
            optimizer.step()

        self.theta_q = theta_q.detach().cpu().numpy()
        self.theta_v = theta_v.detach().cpu().numpy()
        self.bias = bias.detach().cpu().numpy()
        return self

    def predict_logits(self, X: np.ndarray, aggregation: str = "attention") -> np.ndarray:
        assert self.theta_q is not None and self.theta_v is not None and self.bias is not None, "Probe not fitted yet"
        
        attn_scores = np.einsum('nsd,d->ns', X, self.theta_q)
        attn_weights = numpy_softmax(attn_scores, axis=1)
        
        if self.n_classes <= 2:
            value_scores = np.einsum('nsd,d->ns', X, self.theta_v.squeeze())
            aggregated_scores = np.einsum('ns,ns->n', attn_weights, value_scores)
        else:
            value_scores = np.einsum('nsd,cd->nsc', X, self.theta_v)
            aggregated_scores = np.einsum('ns,nsc->nc', attn_weights, value_scores)
        
        return (aggregated_scores + self.bias).squeeze()

    def save_state(self, path: Path):
        np.savez(path, theta_q=self.theta_q, theta_v=self.theta_v, bias=self.bias, name=self.name, n_classes=self.n_classes)

    def load_state(self, path: Path, logger:Logger): 
        data = np.load(path)
        self.theta_q = data['theta_q']
        self.theta_v = data['theta_v']
        self.bias = data['bias']
        self.n_classes = int(data['n_classes'])
        logger.log(f"  - Loaded pre-trained '{self.name}' probe state from {path}") # ✨ FIX: Used logger

class RidgeProbe(BaseProbe):
    """A probe for regression tasks using Ridge regression."""
    name: str = "ridge"

    def __init__(self, d_model: int, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.theta: Optional[np.ndarray] = None
        self.bias: Optional[float] = None

    def _aggregate_activations(self, X: np.ndarray, aggregation: str) -> np.ndarray:
        """For regression, we aggregate the vectors *before* fitting."""
        if aggregation == "mean":
            return X.mean(axis=1)
        if aggregation == "last_token":
            return X[:, -1, :]
        raise ValueError(f"Aggregation '{aggregation}' not supported for RidgeProbe")

    def fit(self, X: np.ndarray, y: np.ndarray, aggregation: str, **kwargs):
        X_agg = self._aggregate_activations(X, aggregation)
        self.model.fit(X_agg, y)
        self.theta = self.model.coef_
        self.bias = self.model.intercept_
        return self

    def predict(self, X: np.ndarray, aggregation: str) -> np.ndarray:
        """Predicts continuous values."""
        assert self.model is not None, "Probe not fitted yet"
        X_agg = self._aggregate_activations(X, aggregation)
        return self.model.predict(X_agg)

    def score(self, X: np.ndarray, y: np.ndarray, aggregation: str) -> dict[str, float]:
        y_pred = self.predict(X, aggregation)
        return {
            "r2_score": float(r2_score(y, y_pred)),
            "mse": float(mean_squared_error(y, y_pred))
        }

    def save_state(self, path: Path):
        np.savez(path, theta=self.theta, bias=self.bias, name=self.name)

    def load_state(self, path: Path, logger:Logger):
        data = np.load(path)
        self.theta = data['theta']
        self.bias = data['bias']
        self.model = Ridge()
        self.model.coef_ = self.theta
        self.model.intercept_ = self.bias
        assert self.theta != None
        self.model.n_features_in_ = self.theta.shape[0]
        logger.log(f"  - Loaded pre-trained '{self.name}' probe state from {path}")
