import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Any, Optional
import yaml

from sklearn.linear_model import LogisticRegression, LinearRegression
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

    def __init__(self, d_model: int):
        self.d_model = d_model
        self.model: Optional[LogisticRegression | LinearRegression] = None

    def _aggregate_activations(self, logits: np.ndarray, aggregation: str) -> np.ndarray:
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

        # Normalize
        mu_  = X_flat.mean(axis=0)
        sig_ = X_flat.std(axis=0) + 1e-6
        X_norm = (X_flat - mu_) / sig_

        if is_classification:
            self.task_type = 'classification'
            self.model = LogisticRegression(verbose=3, random_state=42, **kwargs)
        else: # Regression
            self.task_type = 'regression'
            self.model = LinearRegression()

        self.model.fit(X_norm, y)
        # Store for softmax aggregation during prediction
        self.theta_ = self.model.coef_.squeeze()
        return self

    def predict(self, X: np.ndarray, aggregation: str) -> np.ndarray:
        assert self.model is not None, "Probe has not been trained."

        N, S, D = X.shape
        X_flat = X.reshape(N, S*D)
        mu_  = X_flat.mean(axis=0)
        sig_ = X_flat.std(axis=0) + 1e-6
        X_flat = (X_flat - mu_) / sig_
        
        if isinstance(self.model, LogisticRegression):
            # For binary, decision_function gives logits. For multiclass, predict_proba is used.
            if len(self.model.classes_) > 2:
                probs = self.model.predict_proba(X_flat)
                # Convert probabilities to logits
                logits = np.log(probs / (1 - probs + 1e-9))
                print(logits)
            else:
                logits = self.model.decision_function(X_flat)
        elif isinstance(self.model, LinearRegression):
            logits = self.model.predict(X_flat)
        else:
            raise TypeError(f"Unsupported model type for prediction: {type(self.model)}")
    
        # final_score = self._aggregate_activations(logits, aggregation) # not even doing an aggregation right now
        return logits
    
    def save_state(self, path: Path):
        if isinstance(self.model, LogisticRegression):
            np.savez(path, coef=self.model.coef_, intercept=self.model.intercept_,
                    task_type=self.task_type, classes=getattr(self.model, 'classes_', None))
        else:
            raise TypeError(f"Unsupported model type for prediction: {type(self.model)}")


    def load_state(self, path: Path, logger: Logger):
        data = np.load(path, allow_pickle=True)
        self.task_type = str(data['task_type'])
        
        if self.task_type == 'classification':
            self.model = LogisticRegression()
            self.model.classes_ = data['classes']
        else:
            self.model = LinearRegression()
        
        self.model.coef_ = data['coef']
        self.model.intercept_ = data['intercept']
        self.theta_ = self.model.coef_.squeeze()
        logger.log(f"  - Loaded pre-trained '{self.name}' probe ({self.task_type}) from {path}")

class AttentionProbe(BaseProbe):
    # This class remains a custom PyTorch implementation as requested.
    # Its internal logic is unchanged from the previous version.
    name: str = "attention"
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.theta_q: np.ndarray | None = None
        self.theta_v: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        with open("configs/french_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        self.device = config['device']

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 100, weight_decay: float = 1e-2):
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

        device = self.device if torch.cuda.is_available() else "cpu" # funny logic lol, but correct effect
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
            loss = loss_fn(predictions, y_t.long() if is_classification and self.n_classes > 2 else y_t.float()) # not issue at runtime, for type safety should fix
            loss.backward()
            optimizer.step()

        self.theta_q, self.theta_v, self.bias = theta_q.detach().cpu().numpy(), theta_v.detach().cpu().numpy(), bias.detach().cpu().numpy()
        return self

    def predict(self, X: np.ndarray, aggregation: str = "attention") -> np.ndarray:
        assert self.theta_q is not None and self.theta_v is not None and self.bias is not None, "Probe not fitted yet"
        
        attn_scores = np.einsum('nsd,d->ns', X, self.theta_q)
        attn_weights = numpy_softmax(attn_scores, axis=1)
        
        is_multiclass = self.task_type == 'classification' and self.n_classes > 2
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
