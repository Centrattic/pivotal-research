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
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification"):
        self.d_model = d_model
        self.device = device
        self.task_type = task_type  # Keep for future extensibility
        self.model: Optional[nn.Module] = None
        self.loss_history = []
        self._init_model()
        self.fit_patience=15

    def _init_model(self):
        raise NotImplementedError("Subclasses must implement _init_model to set self.model")

    def _aggregate_activations_with_masks(self, activations: np.ndarray, masks: np.ndarray, aggregation: str = "mean") -> np.ndarray:
        """
        Aggregate activations across sequence dimension using attention masks.
        
        Args:
            activations: Activation array with shape (N, seq_len, d_model)
            masks: Attention mask array with shape (N, seq_len) where True indicates actual tokens
            aggregation: Aggregation method ("mean", "max", "last", "softmax")
            
        Returns:
            Aggregated activations, shape (N, d_model)
        """
        if activations.size == 0:
            return np.empty((0, self.d_model), dtype=np.float16)
        
        N, seq_len, d_model = activations.shape
        
        if aggregation == "mean":
            # Mean pooling across sequence dimension, ignoring padding
            # Use masks to compute mean only over actual tokens
            masked_activations = activations * masks[:, :, np.newaxis]  # Broadcast mask to d_model dimension
            token_counts = masks.sum(axis=1, keepdims=True)  # Count of actual tokens per sequence
            # Avoid division by zero
            token_counts = np.maximum(token_counts, 1)
            result = masked_activations.sum(axis=1) / token_counts
            
        elif aggregation == "max":
            # Max pooling across sequence dimension, ignoring padding
            # Set padding tokens to -inf so they don't affect max
            masked_activations = activations.copy()
            masked_activations[~masks] = -np.inf
            result = np.max(masked_activations, axis=1)
            
        elif aggregation == "last":
            # Take the last token (since we pad on the left, this is always the actual last token)
            result = activations[:, -1, :]
            
        elif aggregation == "softmax":
            # Softmax-weighted mean, ignoring padding
            # Apply softmax only to actual tokens
            masked_activations = activations.copy()
            masked_activations[~masks] = -np.inf  # Set padding to -inf for softmax
            from scipy.special import softmax
            softmax_weights = softmax(masked_activations, axis=1)
            
            # Weighted mean
            result = (softmax_weights * activations).sum(axis=1)
            
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        return result

    def fit(self, X: List[np.ndarray], y: np.ndarray, **kwargs) -> None:
        """Fit the probe to the data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement fit method")

    def find_best_fit(self, X_train: List[np.ndarray], y_train: np.ndarray, X_val: List[np.ndarray], y_val: np.ndarray,
                      **kwargs) -> dict:
        """Find best hyperparameters. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement find_best_fit method")

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

    def score_filtered(self, X: np.ndarray, y: np.ndarray, masks: np.ndarray = None, 
                      dataset_name: str = None, results_dir: Path = None, 
                      seed: int = None, test_size: float = 0.15) -> dict[str, float]:
        """
        Calculate metrics only on examples where the model's logit_diff indicates correct prediction.
        Reads the CSV file from runthrough folder and filters based on logit_diff values.
        
        Args:
            X: Activations array, shape (N, seq_len, d_model)
            y: Labels, shape (N,)
            masks: Attention masks, shape (N, seq_len) where True indicates actual tokens
            dataset_name: Name of the dataset for finding the CSV file
            results_dir: Results directory to find the runthrough folder
            seed: Seed for reproducibility
            test_size: Test size for dataset splitting
            
        Returns:
            Dictionary of metrics on filtered data
        """
        if dataset_name is None or results_dir is None:
            # Fallback to regular scoring if we can't find the CSV
            return self.score(X, y, masks)
        
        # Find the runthrough directory
        parent_dir = results_dir.parent
        runthrough_dir = parent_dir / f"runthrough_{dataset_name}"
        
        if not runthrough_dir.exists():
            return self.score(X, y, masks)
        
        # Look for CSV files with logit_diff in the filename
        csv_files = list(runthrough_dir.glob("*logit_diff*.csv"))
        if not csv_files:
            return self.score(X, y, masks)
        
        csv_path = csv_files[0]
        try:
            df = pd.read_csv(csv_path)
            
            # Check if we have the required columns
            if 'logit_diff' not in df.columns or 'label' not in df.columns:
                return self.score(X, y, masks)
            
            # Calculate use_in_filtered_scoring based on logit_diff and true label
            # For class 0 samples: use if logit_diff < 0 (model correctly predicts class 0)
            # For class 1 samples: use if logit_diff > 0 (model correctly predicts class 1)
            use_in_filtered = []
            for _, row in df.iterrows():
                true_class = row['label']
                logit_diff = row['logit_diff']
                
                if true_class == 0:
                    # Class 0: use if logit_diff < 0 (model correctly predicts class 0)
                    use_in_filtered.append(1 if logit_diff < 0 else 0)
                elif true_class == 1:
                    # Class 1: use if logit_diff > 0 (model correctly predicts class 1)
                    use_in_filtered.append(1 if logit_diff > 0 else 0)
                else:
                    # Fallback for any other class
                    use_in_filtered.append(0)
            
            # Check if the number of samples matches
            if len(use_in_filtered) != len(y):
                return self.score(X, y, masks)
            
            # Filter based on use_in_filtered_scoring
            filtered_indices = [i for i, use in enumerate(use_in_filtered) if use == 1]
            
            if len(filtered_indices) == 0:
                return self.score(X, y, masks)
            
            # Apply the filter to X, y, and masks
            X_filtered = X[filtered_indices]
            y_filtered = y[filtered_indices]
            masks_filtered = masks[filtered_indices] if masks is not None else None
            
            # Calculate metrics on filtered data
            result = self.score(X_filtered, y_filtered, masks_filtered)
            
            # Add filter info to result
            result["filtered"] = True
            result["filter_method"] = "logit_diff_based"
            result["original_size"] = len(y)
            result["filtered_size"] = len(filtered_indices)
            result["removed_count"] = len(y) - len(filtered_indices)
            
            return result
            
        except Exception as e:
            # If anything goes wrong, fall back to regular scoring
            return self.score(X, y, masks)

    def save_state(self, path: Path):
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

    def load_state(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.d_model = checkpoint['d_model']
        self.task_type = checkpoint.get('task_type', 'classification')  # Keep for future extensibility
        # Load aggregation if it exists (for backward compatibility)
        if 'aggregation' in checkpoint:
            self.aggregation = checkpoint['aggregation']
        self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded probe from {path}")

