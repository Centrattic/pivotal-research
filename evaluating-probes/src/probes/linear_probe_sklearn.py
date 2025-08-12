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
            random_state=random_state,
            n_jobs=1  # Ensure single-threaded execution
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

    def find_best_fit(self, X_train: List[np.ndarray], y_train: np.ndarray, masks_train: np.ndarray = None,
                     C_values: List[float] = None, fit_patience: None = None, verbose: bool = True,
                     probe_save_dir: Path = None, probe_filename_base: str = None, 
                     n_jobs: int = -1) -> dict:
        """
        Find best hyperparameters using parallelized grid search over C values.
        Saves all probes to hyperparameter_sweep folder and selects best based on training loss.
        
        Args:
            X_train: Training features (list of activation arrays)
            y_train: Training labels
            masks_train: Training masks for aggregation (optional)
            C_values: List of C values to try (default: logspace from 1e-4 to 1e2)
            fit_patience: Number of epochs to average training loss over (not used for sklearn)
            verbose: Verbosity
            probe_save_dir: Directory to save probes (will create hyperparameter_sweep subfolder)
            probe_filename_base: Base filename for saving
            n_jobs: Number of jobs for parallelization (-1 for all CPUs)
            
        Returns:
            Best hyperparameters
        """
        # Set default C values if not provided
        if C_values is None:
            C_values = np.logspace(-4, 2, 20).tolist()
        
        def train_and_evaluate_probe(C):
            """Train a probe with given C value and return training loss."""
            # Create probe with current C value (n_jobs=1 ensures single-threaded execution)
            trial_probe = SklearnLinearProbe(
                d_model=self.d_model,
                device=self.device,
                task_type=self.task_type,
                aggregation=self.aggregation,
                solver=self.solver,
                C=C,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
            
            # Fit the probe
            trial_probe.fit(X_train, y_train, masks_train)
            
            # Calculate training loss (using negative log likelihood)
            y_pred_proba = trial_probe.predict_proba(X_train, masks_train)
            # Clip probabilities to avoid log(0)
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            # Calculate negative log likelihood
            loss = -np.mean(y_train * np.log(y_pred_proba) + (1 - y_train) * np.log(1 - y_pred_proba))
            
            # Save probe if directory provided
            if probe_save_dir is not None and probe_filename_base is not None:
                # Create hyperparameter_sweep subfolder
                sweep_dir = probe_save_dir / "hyperparameter_sweep"
                sweep_dir.mkdir(exist_ok=True)
                
                # Create filename with C value
                C_str = f"{C:.2e}".replace("+", "").replace(".", "p")
                probe_filename = f"{probe_filename_base}_C_{C_str}_state.npz"
                probe_path = sweep_dir / probe_filename
                
                # Save probe in .npz format
                trial_probe.save_state(probe_path)
            
            return C, loss, trial_probe
        
        # Parallelize the grid search
        if verbose:
            print(f"Running parallelized grid search over {len(C_values)} C values...")
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(train_and_evaluate_probe)(C) for C in C_values
        )
        
        # Extract results
        C_loss_pairs = [(C, loss) for C, loss, _ in results]
        probes = [probe for _, _, probe in results]
        
        # Find best C value (minimum loss)
        best_idx = np.argmin([loss for _, loss in C_loss_pairs])
        best_C, best_loss = C_loss_pairs[best_idx]
        best_probe = probes[best_idx]
        
        if verbose:
            print(f"Grid search results:")
            for C, loss in C_loss_pairs:
                print(f"  C={C:.2e}: loss={loss:.6f}")
            print(f"\nBest hyperparameters:")
            print(f"  C: {best_C:.2e}")
            print(f"  Best training loss: {best_loss:.6f}")
        
        # Update current probe with best parameters
        self.C = best_C
        self.sklearn_model = best_probe.sklearn_model
        self.scaler = best_probe.scaler
        
        # Save best hyperparameters if directory provided
        if probe_save_dir is not None and probe_filename_base is not None:
            best_hparams_path = probe_save_dir / f"{probe_filename_base}_best_hparams.json"
            best_params = {
                'C': best_C,
                'solver': self.solver,
                'class_weight': self.class_weight,
                'max_iter': self.max_iter,
                'random_state': self.random_state,
                'best_training_loss': best_loss,
                'all_results': {f"C_{C:.2e}": loss for C, loss in C_loss_pairs}
            }
            with open(best_hparams_path, 'w') as f:
                json.dump(best_params, f, indent=2)
        
        return {'C': best_C, 'training_loss': best_loss}

    def save_state(self, path: Path):
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

    def load_state(self, path: Path):
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
