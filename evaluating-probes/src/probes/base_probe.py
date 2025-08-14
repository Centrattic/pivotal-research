import os
# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional, List
import json
import math
import gc
from tqdm import tqdm
import optuna
from src.logger import Logger


class BaseProbe:
    """
    General wrapper for PyTorch-based probes. Handles training, evaluation, and saving/loading.
    """

    def __init__(
        self,
        d_model: int,
        device: str = "cpu",
        task_type: str = "classification",
    ):
        self.d_model = d_model
        self.device = device
        self.task_type = task_type  # Keep for future extensibility
        self.model: Optional[nn.Module] = None
        self.loss_history = []
        self._init_model()
        self.fit_patience = 15

    def _init_model(
        self,
    ):
        raise NotImplementedError("Subclasses must implement _init_model to set self.model")

    def fit(self, X: List[np.ndarray], y: np.ndarray, **kwargs) -> None:
        """Fit the probe to the data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """Predict binary labels. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict method")

    def predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """Predict probabilities. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict_proba method")

    def predict_logits(self, X: List[np.ndarray]) -> np.ndarray:
        """Predict logits. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict_logits method")

    def score(self, X: List[np.ndarray], y: np.ndarray) -> dict[str, float]:
        """Calculate accuracy score. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement score method")

    def save_state(
        self,
        path: Path,
    ):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'd_model': self.d_model,
            'task_type': self.task_type,  # Keep for future extensibility
        }
        # Only save aggregation if it exists (for backward compatibility)
        if hasattr(self, 'aggregation'):
            save_dict['aggregation'] = self.aggregation
        torch.save(save_dict, path)
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

    def load_state(
        self,
        path: Path,
    ):
        checkpoint = torch.load(path, map_location=self.device)
        self.d_model = checkpoint['d_model']
        self.task_type = checkpoint.get('task_type', 'classification')  # Keep for future extensibility
        # Load aggregation if it exists (for backward compatibility)
        if 'aggregation' in checkpoint:
            self.aggregation = checkpoint['aggregation']
        self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded probe from {path}")
