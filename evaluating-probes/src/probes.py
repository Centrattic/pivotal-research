import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Any, Optional

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score

class Logger:
    def log(self, message: Any):
        pass

def numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """A numerically stable softmax implementation using NumPy."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class BaseProbe:
    """Abstract base class for all probes."""
    name: str = "base"
    task_type: str
    n_classes: Optional[int]

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
        if not is_multiclass:
            y_prob = 1 / (1 + np.exp(-predictions.squeeze()))
            y_hat = (y_prob > 0.5).astype(int)
            auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.5
        else:
            y_prob = numpy_softmax(predictions, axis=1)
            y_hat = np.argmax(predictions, axis=1)
            auc = roc_auc_score(y, y_prob, multi_class='ovr')

        return {"acc": float(accuracy_score(y, y_hat)), "auc": auc}

    def predict(self, X: np.ndarray, aggregation: str) -> np.ndarray:
        raise NotImplementedError

class LinearProbe(BaseProbe):
    """A linear probe that learns a projection vector θ for each class or for regression."""
    name: str = "linear"

    def __init__(self, d_model: int):
        self.d_model = d_model
        self.theta: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def _aggregate_scores(self, scores: torch.Tensor, aggregation: str) -> torch.Tensor:
        if aggregation == "mean": return scores.mean(dim=1)
        if aggregation == "max": return scores.max(dim=1).values
        if aggregation == "last_token": return scores[:, -1]
        if aggregation == "softmax":
            if scores.ndim == 3:
                norm_scores = torch.linalg.norm(scores, dim=2)
                attn = torch.softmax(norm_scores, dim=1).unsqueeze(-1)
            else:
                attn = torch.softmax(scores, dim=1)
            return torch.sum(attn * scores, dim=1)
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    def fit(self, X: np.ndarray, y: np.ndarray, aggregation: str, lr: float = 0.01, epochs: int = 100, weight_decay: float = 1e-2):
        is_classification = np.issubdtype(y.dtype, np.integer)
        if is_classification:
            self.task_type = 'classification'
            self.n_classes = len(np.unique(y))
            out_features = self.n_classes if self.n_classes > 2 else 1
            loss_fn = torch.nn.CrossEntropyLoss() if self.n_classes > 2 else torch.nn.BCEWithLogitsLoss()
        else:
            self.task_type = 'regression'
            self.n_classes = None
            out_features = 1
            loss_fn = torch.nn.MSELoss()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        theta = torch.nn.Parameter(torch.randn(out_features, self.d_model, device=device) * 0.02)
        bias = torch.nn.Parameter(torch.zeros(out_features, device=device))
        optimizer = torch.optim.AdamW([theta, bias], lr=lr, weight_decay=weight_decay)
        X_t, y_t = torch.from_numpy(X).float().to(device), torch.from_numpy(y).to(device)

        for _ in tqdm(range(epochs), desc="  - Fitting Probe", leave=False):
            optimizer.zero_grad()
            scores = torch.einsum('nsd,cd->nsc', X_t, theta)
            aggregated_scores = self._aggregate_scores(scores, aggregation)
            predictions = (aggregated_scores + bias).squeeze()
            loss = loss_fn(predictions, y_t.long() if is_classification and self.n_classes > 2 else y_t.float())
            loss.backward()
            optimizer.step()

        self.theta, self.bias = theta.detach().cpu().numpy(), bias.detach().cpu().numpy()
        return self

    def predict(self, X: np.ndarray, aggregation: str) -> np.ndarray:
        assert self.theta is not None and self.bias is not None, "Probe not fitted yet"
        
        is_multiclass = hasattr(self, 'task_type') and self.task_type == 'classification' and self.n_classes > 2
        scores = np.einsum('nsd,cd->nsc', X, self.theta) if is_multiclass else np.einsum('nsd,d->ns', X, self.theta.squeeze())
        
        scores_t = torch.from_numpy(scores)
        aggregated_scores = self._aggregate_scores(scores_t, aggregation).numpy()
        
        return (aggregated_scores + self.bias).squeeze()
    
    def save_state(self, path: Path):
        np.savez(path, theta=self.theta, bias=self.bias, name=self.name, task_type=self.task_type, n_classes=self.n_classes or -1)

    def load_state(self, path: Path, logger: Logger):
        data = np.load(path)
        self.theta, self.bias = data['theta'], data['bias']
        self.task_type = str(data['task_type'])
        self.n_classes = int(data['n_classes']) if self.task_type == 'classification' else None
        logger.log(f"  - Loaded pre-trained '{self.name}' probe ({self.task_type}) from {path}")

class AttentionProbe(BaseProbe):
    """An attention probe that learns one query vector and K value vectors."""
    name: str = "attention"
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.theta_q: np.ndarray | None = None
        self.theta_v: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, aggregation: str = "attention", lr: float = 0.01, epochs: int = 100, weight_decay: float = 1e-2):
        is_classification = np.issubdtype(y.dtype, np.integer)
        if is_classification:
            self.task_type = 'classification'
            self.n_classes = len(np.unique(y))
            out_features = self.n_classes if self.n_classes > 2 else 1
            loss_fn = torch.nn.CrossEntropyLoss() if self.n_classes > 2 else torch.nn.BCEWithLogitsLoss()
        else:
            self.task_type = 'regression'
            self.n_classes = None
            out_features = 1
            loss_fn = torch.nn.MSELoss()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        theta_q = torch.nn.Parameter(torch.randn(self.d_model, device=device) * 0.02)
        theta_v = torch.nn.Parameter(torch.randn(out_features, self.d_model, device=device) * 0.02)
        bias = torch.nn.Parameter(torch.zeros(out_features, device=device))
        optimizer = torch.optim.AdamW([theta_q, theta_v, bias], lr=lr, weight_decay=weight_decay)
        X_t, y_t = torch.from_numpy(X).float().to(device), torch.from_numpy(y).to(device)

        for _ in tqdm(range(epochs), desc="  - Fitting Probe", leave=False):
            optimizer.zero_grad()
            attn_scores = torch.einsum('nsd,d->ns', X_t, theta_q)
            attn_weights = torch.softmax(attn_scores, dim=1)
            value_scores = torch.einsum('nsd,cd->nsc', X_t, theta_v)
            aggregated_scores = torch.einsum('ns,nsc->nc', attn_weights, value_scores)
            predictions = (aggregated_scores + bias).squeeze()
            loss = loss_fn(predictions, y_t.long() if is_classification and self.n_classes > 2 else y_t.float())
            loss.backward()
            optimizer.step()

        self.theta_q, self.theta_v, self.bias = theta_q.detach().cpu().numpy(), theta_v.detach().cpu().numpy(), bias.detach().cpu().numpy()
        return self

    def predict(self, X: np.ndarray, aggregation: str = "attention") -> np.ndarray:
        assert self.theta_q is not None and self.theta_v is not None and self.bias is not None, "Probe not fitted yet"
        
        attn_scores = np.einsum('nsd,d->ns', X, self.theta_q)
        attn_weights = numpy_softmax(attn_scores, axis=1)
        
        is_multiclass = hasattr(self, 'task_type') and self.task_type == 'classification' and self.n_classes > 2
        if is_multiclass:
            value_scores = np.einsum('nsd,cd->nsc', X, self.theta_v)
            aggregated_scores = np.einsum('ns,nsc->nc', attn_weights, value_scores)
        else:
            value_scores = np.einsum('nsd,d->ns', X, self.theta_v.squeeze())
            aggregated_scores = np.einsum('ns,ns->n', attn_weights, value_scores)
        
        return (aggregated_scores + self.bias).squeeze()

    def save_state(self, path: Path):
        np.savez(path, theta_q=self.theta_q, theta_v=self.theta_v, bias=self.bias, name=self.name, task_type=self.task_type, n_classes=self.n_classes or -1)

    def load_state(self, path: Path, logger: Logger):
        data = np.load(path)
        self.theta_q, self.theta_v, self.bias = data['theta_q'], data['theta_v'], data['bias']
        self.task_type = str(data['task_type'])
        self.n_classes = int(data['n_classes']) if self.task_type == 'classification' else None
        logger.log(f"  - Loaded pre-trained '{self.name}' probe ({self.task_type}) from {path}")