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
                 max_iter: int = 1500, class_weight: str = "balanced", random_state: int = 42):
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
    
    def fit(self, X: np.ndarray, y: np.ndarray, masks: np.ndarray = None) -> 'SklearnLinearProbe':
        """
        Fit the sklearn linear probe to the data.
        
        Args:
            X: Pre-aggregated activations array, shape (N, d_model)
            y: Labels, shape (N,)
            masks: Ignored (kept for compatibility)
        """
        # Validate input data
        self._validate_input_data(X, y, masks)
        
        # X should already be pre-aggregated, no need to aggregate
        X_aggregated = X
        
        # Validate aggregated data
        self._validate_aggregated_data(X_aggregated, y)
        
        # Fit scaler and transform features
        X_scaled = self.scaler.fit_transform(X_aggregated)
        
        # Validate scaled data
        self._validate_scaled_data(X_scaled)
        
        # Fit the sklearn model
        self.sklearn_model.fit(X_scaled, y)
        
        return self

    def _validate_input_data(self, X, y, masks):
        """Validate input data for numerical issues."""
        print(f"[DEBUG] Input X shape: {X.shape}, dtype: {X.dtype}")
        print(f"[DEBUG] Input y shape: {y.shape}, dtype: {y.dtype}")
        if masks is not None:
            print(f"[DEBUG] Input masks shape: {masks.shape}, dtype: {masks.dtype}")
        
        # Check for infinity in X
        if np.any(np.isinf(X)):
            inf_count = np.sum(np.isinf(X))
            total_elements = X.size
            print(f"[WARNING] Found {inf_count}/{total_elements} infinity values in input X")
            # Replace infinity with large finite values
            X[np.isinf(X)] = np.sign(X[np.isinf(X)]) * 1e10
            print(f"[INFO] Replaced infinity values with ±1e10")
        
        # Check for NaN in X
        if np.any(np.isnan(X)):
            nan_count = np.sum(np.isnan(X))
            total_elements = X.size
            print(f"[WARNING] Found {nan_count}/{total_elements} NaN values in input X")
            # Replace NaN with zeros
            X[np.isnan(X)] = 0.0
            print(f"[INFO] Replaced NaN values with 0.0")
        
        # Check for extremely large values
        max_abs_val = np.max(np.abs(X))
        if max_abs_val > 1e10:
            print(f"[WARNING] Found extremely large values in input X: max_abs={max_abs_val}")
            # Clip to reasonable range
            X = np.clip(X, -1e10, 1e10)
            print(f"[INFO] Clipped values to range [-1e10, 1e10]")
        
        # Validate input shape (should be pre-aggregated: N, d_model)
        if len(X.shape) != 2:
            raise ValueError(f"Expected pre-aggregated activations with shape (N, d_model), got {X.shape}")
        
        if X.shape[1] != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {X.shape[1]}")

    def _validate_aggregated_data(self, X_aggregated, y):
        """Validate aggregated data."""
        print(f"[DEBUG] Aggregated X shape: {X_aggregated.shape}, dtype: {X_aggregated.dtype}")
        print(f"[DEBUG] Aggregated X range: [{np.min(X_aggregated):.6f}, {np.max(X_aggregated):.6f}]")
        
        # Check for infinity in aggregated data
        if np.any(np.isinf(X_aggregated)):
            inf_count = np.sum(np.isinf(X_aggregated))
            total_elements = X_aggregated.size
            print(f"[WARNING] Found {inf_count}/{total_elements} infinity values in aggregated X")
            X_aggregated[np.isinf(X_aggregated)] = np.sign(X_aggregated[np.isinf(X_aggregated)]) * 1e10
        
        # Check for NaN in aggregated data
        if np.any(np.isnan(X_aggregated)):
            nan_count = np.sum(np.isnan(X_aggregated))
            total_elements = X_aggregated.size
            print(f"[WARNING] Found {nan_count}/{total_elements} NaN values in aggregated X")
            X_aggregated[np.isnan(X_aggregated)] = 0.0

    def _validate_scaled_data(self, X_scaled):
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
        
        return {"acc": float(acc), "auc": float(auc), "precision": float(precision), "recall": float(recall), "fpr": float(fpr)}

    def find_best_fit(self, X_train: np.ndarray, y_train: np.ndarray,
                     C_values: List[float] = None, fit_patience: None = None, verbose: bool = True,
                     probe_save_dir: Path = None, probe_filename_base: str = None, 
                     n_jobs: int = -1) -> dict:
        """
        Find best hyperparameters using parallelized grid search over C values.
        Saves all probes to hyperparameter_sweep folder and selects best based on training loss.
        
        Args:
            X_train: Pre-aggregated training activations, shape (N_train, d_model)
            y_train: Training labels, shape (N_train,)
            X_val: Pre-aggregated validation activations, shape (N_val, d_model) (optional)
            y_val: Validation labels, shape (N_val,) (optional)
            C_values: List of C values to try
            fit_patience: Ignored (kept for compatibility)
            verbose: Whether to print progress
            probe_save_dir: Directory to save probes
            probe_filename_base: Base filename for saving
            n_jobs: Number of jobs for parallelization (-1 for all CPUs)
            
        Returns:
            Best hyperparameters
        """
        # Set default C values if not provided
        if C_values is None:
            C_values = np.logspace(-5, 5, 20).tolist()
        
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
            
            # Fit the probe (X_train is already pre-aggregated)
            trial_probe.fit(X_train, y_train)
            
            # Calculate training loss (using negative log likelihood)
            y_pred_proba = trial_probe.predict_proba(X_train)
            # Clip probabilities to avoid log(0)
            y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            # Calculate negative log likelihood
            loss = -np.mean(y_train * np.log(y_pred_proba) + (1 - y_train) * np.log(1 - y_pred_proba))
            
            # Save probe if directory provided
            if probe_save_dir is not None and probe_filename_base is not None:
                # Ensure parent directory exists first
                probe_save_dir.mkdir(parents=True, exist_ok=True)
                
                # Create hyperparameter_sweep subfolder
                sweep_dir = probe_save_dir / "hyperparameter_sweep"
                sweep_dir.mkdir(exist_ok=True)
                
                print(f"[DEBUG] Created sweep directory: {sweep_dir}")
                
                # Create filename with C value
                C_str = f"{C:.2e}".replace("+", "").replace(".", "p")
                probe_filename = f"{probe_filename_base}_C_{C_str}_state.npz"
                probe_path = sweep_dir / probe_filename
                
                # Save probe in .npz format
                trial_probe.save_state(probe_path)
                print(f"[DEBUG] Saved hyperparameter sweep probe: {probe_path}")
            else:
                print(f"[DEBUG] Not saving hyperparameter sweep probe - probe_save_dir={probe_save_dir}, probe_filename_base={probe_filename_base}")
            
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
            # Ensure parent directory exists first
            probe_save_dir.mkdir(parents=True, exist_ok=True)
            
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
            print(f"[DEBUG] Saved best hyperparameters: {best_hparams_path}")
        else:
            print(f"[DEBUG] Not saving best hyperparameters - probe_save_dir={probe_save_dir}, probe_filename_base={probe_filename_base}")
        
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
