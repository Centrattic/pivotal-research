import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import joblib
from pathlib import Path
import json
from typing import Optional, List

from src.probes.base_probe import BaseProbe


class SklearnLinearProbe(BaseProbe):
    """
    Sklearn-based linear probe using LogisticRegression.
    Inherits from BaseProbe for compatibility with existing infrastructure.
    """
    
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", 
                 aggregation: str = "mean", solver: str = "liblinear", C: float = 1.0, 
                 max_iter: int = 1000, class_weight: str = "balanced", random_state: int = 42):
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
            random_state=random_state
        )
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Call parent constructor
        super().__init__(d_model=d_model, device=device, task_type=task_type)
    
    def _init_model(self):
        """Override to create a dummy model for compatibility with BaseProbe."""
        # Create a dummy linear layer for compatibility with BaseProbe methods
        self.model = nn.Linear(self.d_model, 1)
        # We won't actually use this for training, but it's needed for compatibility
    
    def fit(self, X: np.ndarray, y: np.ndarray, masks: np.ndarray = None, **kwargs):
        """
        Fit the sklearn linear probe.
        
        Args:
            X: Activations array, shape (N, seq_len, d_model)
            y: Labels, shape (N,)
            masks: Attention masks, shape (N, seq_len) where True indicates actual tokens
        """
        # Aggregate activations using masks
        if masks is not None:
            X_aggregated = self._aggregate_activations_with_masks(X, masks, self.aggregation)
        else:
            raise ValueError("Masks are required for sklearn probe aggregation")
        
        # Fit scaler and transform features
        X_scaled = self.scaler.fit_transform(X_aggregated)
        
        # Fit the sklearn model
        self.sklearn_model.fit(X_scaled, y)
        
        return self

    def predict(self, X: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """
        Make predictions using the sklearn linear probe.
        
        Args:
            X: Activations array, shape (N, seq_len, d_model)
            masks: Attention masks, shape (N, seq_len) where True indicates actual tokens
            
        Returns:
            Predictions, shape (N,)
        """
        # Aggregate activations using masks
        if masks is not None:
            X_aggregated = self._aggregate_activations_with_masks(X, masks, self.aggregation)
        else:
            raise ValueError("Masks are required for sklearn probe aggregation")
        
        # Scale features
        X_scaled = self.scaler.transform(X_aggregated)
        
        # Make predictions
        return self.sklearn_model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Activations array, shape (N, seq_len, d_model)
            masks: Attention masks, shape (N, seq_len) where True indicates actual tokens
            
        Returns:
            Probabilities, shape (N,)
        """
        # Aggregate activations using masks
        if masks is not None:
            X_aggregated = self._aggregate_activations_with_masks(X, masks, self.aggregation)
        else:
            raise ValueError("Masks are required for sklearn probe aggregation")
        
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
            X: Activations array, shape (N, seq_len, d_model)
            masks: Attention masks, shape (N, seq_len) where True indicates actual tokens
            
        Returns:
            Logits, shape (N,)
        """
        # Aggregate activations using masks
        if masks is not None:
            X_aggregated = self._aggregate_activations_with_masks(X, masks, self.aggregation)
        else:
            raise ValueError("Masks are required for sklearn probe aggregation")
        
        # Scale features
        X_scaled = self.scaler.transform(X_aggregated)
        
        # Get decision function values (logits)
        return self.sklearn_model.decision_function(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray, masks: np.ndarray = None) -> dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            X: Activations array, shape (N, seq_len, d_model)
            y: Labels, shape (N,)
            masks: Attention masks, shape (N, seq_len) where True indicates actual tokens
            
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
        
        return {"acc": float(acc), "auc": float(auc), "precision": float(precision), "recall": float(recall), "fpr": float(fpr)}

    def find_best_fit(self, X_train: List[np.ndarray], y_train: np.ndarray, X_val: List[np.ndarray], y_val: np.ndarray,
                     n_trials: int = 20, direction: str = None, verbose: bool = True,
                     probe_save_dir: Path = None, probe_filename_base: str = None) -> dict:
        """
        Find best hyperparameters using Optuna.
        For sklearn probes, we use training accuracy as the selection criterion.
        
        Args:
            X_train: Training features (list of activation arrays)
            y_train: Training labels
            X_val: Validation features (list of activation arrays) - not used for sklearn
            y_val: Validation labels - not used for sklearn
            n_trials: Number of trials
            direction: Optimization direction
            verbose: Verbosity
            probe_save_dir: Directory to save best hyperparameters
            probe_filename_base: Base filename for saving
            
        Returns:
            Best hyperparameters
        """
        import optuna
        
        def objective(trial):
            # Define hyperparameter search space
            C = trial.suggest_float('C', 1e-4, 1e2, log=True)
            solver = trial.suggest_categorical('solver', ['liblinear'])
            class_weight = trial.suggest_categorical('class_weight', ['balanced'])
            
            # Create probe with trial hyperparameters
            trial_probe = SklearnLinearProbe(
                d_model=self.d_model,
                device=self.device,
                task_type=self.task_type,
                aggregation=self.aggregation,
                solver=solver,
                C=C,
                max_iter=self.max_iter,
                class_weight=class_weight,
                random_state=self.random_state
            )
            
            # Fit and evaluate on training set
            trial_probe.fit(X_train, y_train)
            metrics = trial_probe.score(X_train, y_train)
            
            return metrics['auc']  # Use training AUC as selection criterion
        
        # Create study to maximize training accuracy
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
        
        # Get best parameters
        best_params = study.best_params
        
        if verbose:
            print(f"Best hyperparameters found:")
            print(f"  C: {best_params['C']:.2e}")
            print(f"  solver: {best_params['solver']}")
            print(f"  class_weight: {best_params['class_weight']}")
            print(f"  Best training accuracy: {study.best_value:.6f}")
        
        # Update current probe with best parameters
        self.C = best_params['C']
        self.solver = best_params['solver']
        self.class_weight = best_params['class_weight']
        
        # Recreate sklearn model with best parameters
        self.sklearn_model = LogisticRegression(
            solver=self.solver,
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        
        # Save best hyperparameters if directory provided
        if probe_save_dir is not None and probe_filename_base is not None:
            best_hparams_path = probe_save_dir / f"{probe_filename_base}_best_hparams.json"
            with open(best_hparams_path, 'w') as f:
                json.dump(best_params, f, indent=2)
        
        return best_params

    def save_state(self, path: Path):
        """Save the sklearn probe state."""
        save_dict = {
            'd_model': self.d_model,
            'task_type': self.task_type,
            'aggregation': self.aggregation,
            'solver': self.solver,
            'C': self.C,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'sklearn_model': self.sklearn_model,
            'scaler': self.scaler,
        }
        joblib.dump(save_dict, path)

    def load_state(self, path: Path):
        """Load the sklearn probe state."""
        save_dict = joblib.load(path)
        
        self.d_model = save_dict['d_model']
        self.task_type = save_dict['task_type']
        self.aggregation = save_dict['aggregation']
        self.solver = save_dict['solver']
        self.C = save_dict['C']
        self.max_iter = save_dict['max_iter']
        self.class_weight = save_dict['class_weight']
        self.random_state = save_dict['random_state']
        self.sklearn_model = save_dict['sklearn_model']
        self.scaler = save_dict['scaler']
        
        # Recreate dummy model for compatibility
        self._init_model()
