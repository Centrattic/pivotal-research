import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional
import json
import optuna

from src.probes.base_probe import BaseProbe

class LinearProbeNet(nn.Module):
    def __init__(self, d_model: int, aggregation: str = "mean", device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.aggregation = aggregation
        self.device = device
        self.linear = nn.Linear(d_model, 1).to(self.device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model), mask: (batch, seq)
        # Optimize aggregation operations for better performance
        if self.aggregation == "mean":
            # Use more efficient mean pooling
            mask_expanded = mask.unsqueeze(-1)  # (batch, seq, 1)
            masked_sum = (x * mask_expanded).sum(dim=1)  # (batch, d_model)
            mask_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
            x = masked_sum / mask_counts
        elif self.aggregation == "max":
            # Use masked_fill for efficient max pooling
            x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            x, _ = x.max(dim=1)
        elif self.aggregation == "last":
            # Optimize last token selection
            idx = mask.sum(dim=1) - 1
            idx = idx.clamp(min=0)
            x = x[torch.arange(x.size(0), device=x.device), idx]
        elif self.aggregation == "softmax":
            # Optimize softmax attention
            attn_scores = x.mean(dim=-1)  # (batch, seq)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)
            x = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Ensure x is contiguous for better performance
        x = x.contiguous()
        logits = self.linear(x).squeeze(-1)
        return logits

class LinearProbe(BaseProbe):
    def __init__(self, d_model: int, device: str = "cpu", aggregation: str = "mean", **kwargs):
        # Store any additional config parameters for this probe
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.aggregation = aggregation
        super().__init__(d_model, device, task_type="classification")  # Binary classification only

    def _init_model(self):
        self.model = LinearProbeNet(self.d_model, aggregation=self.aggregation, device=self.device)
        # Ensure model is on the correct device and in training mode
        self.model = self.model.to(self.device)
        self.model.train()

    def find_best_fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, mask_train: Optional[np.ndarray] = None, mask_val: Optional[np.ndarray] = None, 
                    n_trials: int = 10, direction: str = None, verbose: bool = True, weighting_method: str = 'weighted_loss', metric: str = 'acc', fpr_threshold: float = 0.01, 
                    probe_save_dir: Optional[Path] = None, probe_filename_base: Optional[str] = None):
        """
        Hyperparameter tuning for the probe. If weighting_method is 'pcngd', tune using fit_pcngd, else use fit.
        metric: 'acc' (default), 'auc', or 'fpr_recall'.
        If 'fpr_recall', minimize FPR, but if FPR <= fpr_threshold, maximize Recall.
        """
        import json
        best_score = None
        best_params = None
        def objective(trial):
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-2)
            epochs = 50
            # Re-init probe for each trial - pass all config parameters
            config_params = {k: v for k, v in self.__dict__.items() if k not in ['d_model', 'device', 'aggregation', 'model', 'loss_history']}
            probe = LinearProbe(self.d_model, device=self.device, aggregation=self.aggregation, **config_params)
            if weighting_method == 'pcngd':
                probe.fit_pcngd(X_train, y_train, mask=mask_train, epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=False)
            else:
                probe.fit(X_train, y_train, mask=mask_train, epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=False, use_weighted_loss=(weighting_method=='weighted_loss'), use_weighted_sampler=(weighting_method=='weighted_sampler'))
            metrics = probe.score(X_val, y_val, mask=mask_val)
            if metric == 'auc':
                return -metrics.get('auc', 0.0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        # Dump best_params to a dedicated file
        if probe_save_dir is not None and probe_filename_base is not None:
            best_hparams_path = probe_save_dir / f"{probe_filename_base}_best_hparams.json"
            with open(best_hparams_path, 'w') as f:
                json.dump(best_params, f, indent=2)
        return best_params
    # Inherits predict_logits from BaseProbe 