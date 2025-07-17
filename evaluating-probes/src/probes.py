import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Any, Optional
import json
import math

from src.logger import Logger

class BaseProbe:
    """
    General wrapper for PyTorch-based probes. Handles training, evaluation, and saving/loading.
    """
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", aggregation: str = "mean"):
        self.d_model = d_model
        self.device = device
        self.task_type = task_type  # 'classification' or 'regression'
        self.aggregation = aggregation
        self.model: Optional[nn.Module] = None
        self.loss_history = []
        self._init_model()

    def _init_model(self):
        raise NotImplementedError("Subclasses must implement _init_model to set self.model")

    def fit(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, epochs: int = 20, lr: float = 1e-3, batch_size: int = 64, weight_decay: float = 0.0, verbose: bool = True):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32 if self.task_type == "regression" else torch.long, device=self.device)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        else:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=self.device)

        dataset = torch.utils.data.TensorDataset(X, y, mask)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if self.task_type == "classification":
            criterion = nn.BCEWithLogitsLoss() if y.ndim == 1 or y.shape[1] == 1 else nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb, mb in loader:
                optimizer.zero_grad()
                logits = self.model(xb, mb)
                if self.task_type == "classification":
                    if yb.ndim == 1 or yb.shape[-1] == 1:
                        yb = yb.float()
                        loss = criterion(logits, yb)
                    else:
                        loss = criterion(logits, yb)
                else:
                    loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            avg_loss = epoch_loss / len(dataset)
            self.loss_history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        return self

    def predict(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        else:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.model(X, mask)
            if self.task_type == "classification":
                if logits.ndim == 1 or logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long().cpu().numpy()
                else:
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
            else:
                preds = logits.cpu().numpy()
        return preds

    def predict_proba(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        else:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.model(X, mask)
            if self.task_type == "classification":
                if logits.ndim == 1 or logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits).cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                probs = logits.cpu().numpy()
        return probs

    def score(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error, precision_score, recall_score, confusion_matrix
        preds = self.predict(X, mask)
        if self.task_type == "regression":
            return {
                "r2_score": float(r2_score(y, preds)),
                "mse": float(mean_squared_error(y, preds)),
            }
        else:
            y_true = y
            y_prob = self.predict_proba(X, mask)
            if y_prob.ndim == 1 or y_prob.shape[-1] == 1:
                auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
                acc = accuracy_score(y_true, preds)
                # Calculate additional metrics for binary classification
                precision = precision_score(y_true, preds, zero_division=0)
                recall = recall_score(y_true, preds, zero_division=0)
                # Calculate FPR from confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                acc = accuracy_score(y_true, preds)
                # For multiclass, calculate macro-averaged metrics
                precision = precision_score(y_true, preds, average='macro', zero_division=0)
                recall = recall_score(y_true, preds, average='macro', zero_division=0)
                # For multiclass FPR, we'll use 1 - specificity (macro-averaged)
                fpr = 1 - recall_score(y_true, preds, average='macro', zero_division=0)
            return {"acc": float(acc), "auc": float(auc), "precision": float(precision), "recall": float(recall), "fpr": float(fpr)}

    def save_state(self, path: Path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'd_model': self.d_model,
            'task_type': self.task_type,
            'aggregation': self.aggregation,
        }, path)
        print(f"Saved probe to {path}")
        # Save training info (loss history, etc.)
        log_path = path.with_name(path.stem + "_train_log.json")
        train_info = {
            "loss_history": self.loss_history,
            # Add more training info here if needed
        }
        with open(log_path, "w") as f:
            json.dump(train_info, f, indent=2)
        print(f"Saved training log to {log_path}")

    def load_state(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.d_model = checkpoint['d_model']
        self.task_type = checkpoint['task_type']
        self.aggregation = checkpoint['aggregation']
        self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded probe from {path}")

class LinearProbeNet(nn.Module):
    def __init__(self, d_model: int, aggregation: str = "mean", device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.aggregation = aggregation
        self.device = device
        self.linear = nn.Linear(d_model, 1).to(self.device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model), mask: (batch, seq)
        if self.aggregation == "mean":
            x = x * mask.unsqueeze(-1)
            x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        elif self.aggregation == "max":
            x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            x, _ = x.max(dim=1)
        elif self.aggregation == "last":
            idx = mask.sum(dim=1) - 1
            idx = idx.clamp(min=0)
            x = x[torch.arange(x.size(0)), idx]
        elif self.aggregation == "softmax":
            attn_scores = x.mean(dim=-1)  # (batch, seq)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)
            x = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        logits = self.linear(x).squeeze(-1)
        return logits

class LinearProbe(BaseProbe):
    def _init_model(self):
        self.model = LinearProbeNet(self.d_model, aggregation=self.aggregation, device=self.device)

class AttentionProbeNet(nn.Module):
    def __init__(self, d_model: int, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.scale = math.sqrt(d_model)
        self.context_query = nn.Linear(d_model, 1)
        self.classifier = nn.Linear(d_model, 1)
        # Move the model to the specified device
        self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model), mask: (batch, seq)
        attn_scores = self.context_query(x).squeeze(-1) / self.scale
        masked_attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(masked_attn_scores, dim=-1)
        token_logits = self.classifier(x).squeeze(-1)
        token_logits = token_logits.masked_fill(~mask, 0)

        # Compute weighted context
        context = torch.einsum("bs,bse->be", attn_weights, x)
        sequence_logits = self.classifier(context).squeeze(-1)
        
        return sequence_logits

class AttentionProbe(BaseProbe):
    def _init_model(self):
        self.model = AttentionProbeNet(self.d_model, device=self.device)

# Example usage:
# class MyProbe(BaseProbe):
#     def _init_model(self):
#         self.model = LinearProbe(self.d_model, aggregation=self.aggregation).to(self.device)

# To use:
# probe = MyProbe(d_model=768, device="cuda", task_type="classification", aggregation="mean")
# probe.fit(X_train, y_train, mask=mask_train)
# acc = probe.score(X_val, y_val, mask=mask_val) 