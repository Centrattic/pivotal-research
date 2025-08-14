import numpy as np
import torch
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List


class BaseProbeNonTrainable:
    """
    Base class for probes that don't require training. Handles evaluation and saving/loading.
    """

    def __init__(
        self,
        d_model: int,
        device: str = "cpu",
        task_type: str = "classification",
        aggregation: str = "mean",
    ):
        self.d_model = d_model
        self.device = device
        self.task_type = task_type  # Keep for future extensibility
        self.aggregation = aggregation  # 'mean', 'max', 'last', 'softmax', or None for probes that don't need aggregation
        self.model = None  # No trainable model for these probes
        self.loss_history = []  # Empty for non-trainable probes

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the non-trainable probe to the data."""
        # X should already be pre-aggregated activations (N, d_model)
        # Store the activations and labels for later use
        self.X_aggregated = X
        self.y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        logits = self.predict_logits(X)
        return (logits > 0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        logits = self.predict_logits(X)
        probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Predict logits."""
        # X should already be pre-aggregated activations (N, d_model)
        # For non-trainable probes, we need to implement the specific prediction logic
        # This is a placeholder - subclasses should override this method
        raise NotImplementedError("Subclasses must implement predict_logits")

    def score(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Calculate accuracy and AUC scores."""
        from sklearn.metrics import roc_auc_score

        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)

        # Calculate AUC using probabilities
        try:
            y_prob = self.predict_proba(X)
            # For binary classification, use the positive class probability
            if y_prob.shape[1] == 2:
                y_prob_positive = y_prob[:, 1]  # Probability of positive class
            else:
                y_prob_positive = y_prob.flatten()

            auc = roc_auc_score(y, y_prob_positive) if len(np.unique(y)) > 1 else 0.5
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            auc = 0.5

        return {"accuracy": accuracy, "auc": float(auc)}

    def score_filtered(
        self,
        X,
        y: np.ndarray,
        dataset_name: str = None,
        results_dir: Path = None,
        seed: int = None,
        test_size: float = 0.15
    ) -> dict[str,
              float]:
        """
        Calculate metrics only on examples where the model's logit_diff indicates correct prediction.
        Reads the CSV file from runthrough folder and filters based on logit_diff values.
        
        Args:
            X: Pre-aggregated activations, shape (N, d_model)
            y: Labels, shape (N,)
            dataset_name: Name of the dataset for finding the CSV file
            results_dir: Results directory to find the runthrough folder
            seed: Seed for reproducibility
            test_size: Test size for dataset splitting
            
        Returns:
            Dictionary of metrics on filtered data
        """
        if dataset_name is None or results_dir is None:
            # Fallback to regular scoring if we can't find the CSV
            return self.score(X, y)

        # Find the runthrough directory
        parent_dir = results_dir.parent
        runthrough_dir = parent_dir / f"runthrough_{dataset_name}"

        if not runthrough_dir.exists():
            print(f"[DEBUG] Runthrough directory not found: {runthrough_dir}")
            return self.score(X, y)

        # Look for CSV files with logit_diff in the filename
        csv_files = list(runthrough_dir.glob("*logit_diff*.csv"))
        if not csv_files:
            print(f"[DEBUG] No logit_diff CSV files found in: {runthrough_dir}")
            return self.score(X, y)

        csv_path = csv_files[0]
        try:
            df = pd.read_csv(csv_path)

            # Check if we have the required columns
            if 'logit_diff' not in df.columns or 'label' not in df.columns:
                print(f"[DEBUG] Missing required columns. Available columns: {list(df.columns)}")
                return self.score(X, y)

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
                print(f"[DEBUG] Sample count mismatch: CSV has {len(use_in_filtered)} samples, but y has {len(y)}")
                return self.score(X, y)

            # Filter based on use_in_filtered_scoring
            filtered_indices = [i for i, use in enumerate(use_in_filtered) if use == 1]

            if len(filtered_indices) == 0:
                print(f"[DEBUG] No samples passed the filter criteria")
                return self.score(X, y)

            # Apply the filter to X and y
            # Handle both numpy arrays and lists of arrays
            if isinstance(X, np.ndarray):
                X_filtered = X[filtered_indices]
            elif isinstance(X, list):
                X_filtered = [X[i] for i in filtered_indices]
            else:
                print(f"[DEBUG] Unexpected X type: {type(X)}")
                return self.score(X, y)

            y_filtered = y[filtered_indices]

            # Calculate metrics on filtered data
            result = self.score(X_filtered, y_filtered)

            # Add filter info to result
            result["filtered"] = True
            result["filter_method"] = "logit_diff_based"
            result["original_size"] = len(y)
            result["filtered_size"] = len(filtered_indices)
            result["removed_count"] = len(y) - len(filtered_indices)

            return result

        except Exception as e:
            # If anything goes wrong, fall back to regular scoring
            print(f"[DEBUG] Exception in score_filtered: {e}")
            return self.score(X, y)

    def save_state(
        self,
        path: Path,
    ):
        """
        Save the probe state (parameters, not model weights).
        """
        save_dict = {
            'd_model': self.d_model,
            'task_type': self.task_type,
            'aggregation': self.aggregation,
        }
        # Add probe-specific parameters
        probe_params = self._get_probe_parameters()
        save_dict.update(probe_params)

        torch.save(save_dict, path)

        # Save training info (empty for non-trainable probes)
        log_path = path.with_name(path.stem + "_train_log.json")
        train_info = {"loss_history": self.loss_history, "probe_type": "non_trainable"}
        with open(log_path, "w") as f:
            json.dump(train_info, f, indent=2)

    def _get_probe_parameters(self) -> dict:
        """
        Return probe-specific parameters to save. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement _get_probe_parameters")

    def load_state(
        self,
        path: Path,
    ):
        """
        Load the probe state (parameters, not model weights).
        """
        # Use weights_only=False to allow loading numpy arrays and other objects
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.d_model = checkpoint['d_model']
        self.task_type = checkpoint['task_type']
        self.aggregation = checkpoint['aggregation']

        # Load probe-specific parameters
        self._load_probe_parameters(checkpoint)

    def _load_probe_parameters(
        self,
        checkpoint: dict,
    ):
        """
        Load probe-specific parameters. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement _load_probe_parameters")
