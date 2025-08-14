import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import joblib
from joblib import Parallel, delayed
from pathlib import Path
import json
from typing import Optional, List

from src.probes.base_probe import BaseProbe


class SklearnLinearProbe(BaseProbe):
    """
    Sklearn-based linear probe using LogisticRegression.
    Inherits from BaseProbe for compatibility with existing infrastructure.
    """

    def __init__(
        self,
        d_model: int,
        device: str = "cpu",
        task_type: str = "classification",
        aggregation: str = "mean",
        solver: str = "liblinear",
        C: float = 1.0,
        max_iter: int = 1500,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.aggregation = aggregation
        self.solver = solver
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        # Initialize sklearn model
        self.sklearn_model = LogisticRegression(
            solver=solver,
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=1  # Ensure single-threaded execution
        )

        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()

        # Call parent constructor
        super().__init__(d_model=d_model, device=device, task_type=task_type)

    def _init_model(
        self,
    ):
        """Override to create a dummy model for compatibility with BaseProbe."""
        # Create a dummy linear layer for compatibility with BaseProbe methods
        self.model = nn.Linear(self.d_model, 1)
        # We won't actually use this for training, but it's needed for compatibility

    def fit(self, X: np.ndarray, y: np.ndarray, masks: np.ndarray = None) -> 'SklearnLinearProbe':
        """
        Fit the sklearn linear probe to the data.
        
        Args:
            X: Pre-aggregated activations array, shape (N, d_model)
            y: Labels, shape (N,)
            masks: Ignored (kept for compatibility)
        """

        X_aggregated = X
        X_scaled = self.scaler.fit_transform(X_aggregated)
        self.sklearn_model.fit(X_scaled, y)

        return self

    def _validate_scaled_data(
        self,
        X_scaled,
    ):
        """Validate scaled data."""
        print(f"[DEBUG] Scaled X shape: {X_scaled.shape}, dtype: {X_scaled.dtype}")
        print(f"[DEBUG] Scaled X range: [{np.min(X_scaled):.6f}, {np.max(X_scaled):.6f}]")

        # Check for infinity in scaled data
        if np.any(np.isinf(X_scaled)):
            inf_count = np.sum(np.isinf(X_scaled))
            total_elements = X_scaled.size
            print(f"[WARNING] Found {inf_count}/{total_elements} infinity values in scaled X")
            X_scaled[np.isinf(X_scaled)] = np.sign(X_scaled[np.isinf(X_scaled)]) * 1e10

        # Check for NaN in scaled data
        if np.any(np.isnan(X_scaled)):
            nan_count = np.sum(np.isnan(X_scaled))
            total_elements = X_scaled.size
            print(f"[WARNING] Found {nan_count}/{total_elements} NaN values in scaled X")
            X_scaled[np.isnan(X_scaled)] = 0.0

    def predict(self, X: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """
        Get predictions.
        
        Args:
            X: Pre-aggregated activations array, shape (N, d_model)
            masks: Ignored (kept for compatibility)
            
        Returns:
            Predictions, shape (N,)
        """
        # X should already be pre-aggregated, no need to aggregate
        X_aggregated = X

        # Scale features
        X_scaled = self.scaler.transform(X_aggregated)

        # Make predictions
        return self.sklearn_model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Pre-aggregated activations array, shape (N, d_model)
            masks: Ignored (kept for compatibility)
            
        Returns:
            Probabilities, shape (N,)
        """
        # X should already be pre-aggregated, no need to aggregate
        X_aggregated = X

        # Scale features
        X_scaled = self.scaler.transform(X_aggregated)

        # Get probabilities (sklearn returns (N, 2) for binary classification)
        probas = self.sklearn_model.predict_proba(X_scaled)
        # Return probability of positive class
        return probas[:, 1]

    def predict_logits(self, X: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """
        Get raw logits (decision function values).
        
        Args:
            X: Pre-aggregated activations array, shape (N, d_model)
            masks: Ignored (kept for compatibility)
            
        Returns:
            Logits, shape (N,)
        """
        # X should already be pre-aggregated, no need to aggregate
        X_aggregated = X

        # Scale features
        X_scaled = self.scaler.transform(X_aggregated)

        # Get decision function values (logits)
        return self.sklearn_model.decision_function(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray, masks: np.ndarray = None) -> dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            X: Pre-aggregated activations array, shape (N, d_model)
            y: Labels, shape (N,)
            masks: Ignored (kept for compatibility)
            
        Returns:
            Dictionary of metrics
        """
        preds = self.predict(X, masks)
        y_true = y
        y_prob = self.predict_proba(X, masks)

        # Binary classification metrics
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        acc = accuracy_score(y_true, preds)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "acc": float(acc),
            "auc": float(auc),
            "precision": float(precision),
            "recall": float(recall),
            "fpr": float(fpr)
        }

    def save_state(
        self,
        path: Path,
    ):
        """Save the sklearn probe state in .npz format."""
        # Extract model coefficients and intercept
        coef = self.sklearn_model.coef_
        intercept = self.sklearn_model.intercept_

        # Extract scaler parameters
        scaler_mean = self.scaler.mean_
        scaler_scale = self.scaler.scale_
        scaler_var = self.scaler.var_

        # Save all parameters
        np.savez_compressed(
            path,
            # Model parameters
            coef=coef,
            intercept=intercept,
            # Scaler parameters
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            scaler_var=scaler_var,
            # Metadata
            d_model=self.d_model,
            task_type=self.task_type,
            aggregation=self.aggregation,
            solver=self.solver,
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def load_state(
        self,
        path: Path,
    ):
        """Load the sklearn probe state from .npz format."""
        save_dict = np.load(path, allow_pickle=True)

        # Load metadata
        self.d_model = int(save_dict['d_model'])
        self.task_type = str(save_dict['task_type'])
        self.aggregation = str(save_dict['aggregation'])
        self.solver = str(save_dict['solver'])
        self.C = float(save_dict['C'])
        self.max_iter = int(save_dict['max_iter'])
        self.class_weight = str(save_dict['class_weight'])
        self.random_state = int(save_dict['random_state'])

        # Update existing sklearn model parameters
        self.sklearn_model.coef_ = save_dict['coef']
        self.sklearn_model.intercept_ = save_dict['intercept']
        self.sklearn_model.classes_ = np.array([0, 1])  # Binary classification

        # Update existing scaler parameters
        self.scaler.mean_ = save_dict['scaler_mean']
        self.scaler.scale_ = save_dict['scaler_scale']
        self.scaler.var_ = save_dict['scaler_var']

        # Recreate dummy model for compatibility
        self._init_model()
