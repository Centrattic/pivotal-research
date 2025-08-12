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
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", 
                aggregation: str = "mean"):
        self.d_model = d_model
        self.device = device
        self.task_type = task_type  # Keep for future extensibility
        self.aggregation = aggregation  # 'mean', 'max', 'last', 'softmax', or None for probes that don't need aggregation
        self.model = None  # No trainable model for these probes
        self.loss_history = []  # Empty for non-trainable probes

    def _aggregate_activations(self, activations: List[np.ndarray], aggregation: str = "mean") -> np.ndarray:
        """
        Aggregate activations across sequence dimension.
        
        Args:
            activations: List of activation arrays, each with shape (seq_len, d_model) where seq_len varies
            aggregation: Aggregation method ("mean", "max", "last", "mass_mean")
            
        Returns:
            Aggregated activations, shape (N, d_model)
        """
        if not activations:
            return np.empty((0, self.d_model), dtype=np.float16)
        
        aggregated = []
        
        for act in activations:
            if act.size == 0:
                # Handle empty activations
                aggregated.append(np.zeros(self.d_model, dtype=np.float16))
                continue
                
            if aggregation == "mean":
                # Mean pooling across sequence dimension
                result = np.mean(act, axis=0)
            elif aggregation == "max":
                # Max pooling across sequence dimension
                result = np.max(act, axis=0)
            elif aggregation == "last":
                # Take the last token
                result = act[-1]
            elif aggregation == "mass_mean":
                # Mass mean aggregation (specific to mass mean probe)
                # This is a placeholder - actual implementation depends on the specific probe
                result = np.mean(act, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            aggregated.append(result)
        
        return np.stack(aggregated)

    def fit(self, X: List[np.ndarray], y: np.ndarray) -> None:
        """Fit the non-trainable probe to the data."""
        if not hasattr(self, 'aggregation'):
            raise ValueError("Probe must have aggregation attribute set")
        
        # Aggregate activations
        X_aggregated = self._aggregate_activations(X, self.aggregation)
        
        # Store the aggregated activations and labels for later use
        self.X_aggregated = X_aggregated
        self.y = y

    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """Predict binary labels."""
        logits = self.predict_logits(X)
        return (logits > 0).astype(int)

    def predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """Predict probabilities."""
        logits = self.predict_logits(X)
        probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])

    def predict_logits(self, X: List[np.ndarray]) -> np.ndarray:
        """Predict logits."""
        if not hasattr(self, 'aggregation'):
            raise ValueError("Probe must have aggregation attribute set")
        
        # Aggregate activations
        X_aggregated = self._aggregate_activations(X, self.aggregation)
        
        # For non-trainable probes, we need to implement the specific prediction logic
        # This is a placeholder - subclasses should override this method
        raise NotImplementedError("Subclasses must implement predict_logits")

    def score(self, X: List[np.ndarray], y: np.ndarray) -> dict[str, float]:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return {"accuracy": accuracy}

    def score_filtered(self, X: List[np.ndarray], y: np.ndarray, dataset_name: str = None, 
                      results_dir: Path = None, seed: int = None, test_size: float = 0.15) -> dict[str, float]:
        """
        Calculate metrics only on examples where the model's logit_diff indicates correct prediction.
        Reads the CSV file from runthrough folder and filters based on logit_diff values.
        
        Args:
            X: List of activation arrays, each with shape (seq_len, d_model)
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
            return self.score(X, y)
        
        # Look for CSV files with logit_diff in the filename
        csv_files = list(runthrough_dir.glob("*logit_diff*.csv"))
        if not csv_files:
            return self.score(X, y)
        
        csv_path = csv_files[0]
        try:
            df = pd.read_csv(csv_path)
            
            # Check if we have the required columns
            if 'logit_diff' not in df.columns or 'label' not in df.columns:
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
                return self.score(X, y)
            
            # Filter based on use_in_filtered_scoring
            filtered_indices = [i for i, use in enumerate(use_in_filtered) if use == 1]
            
            if len(filtered_indices) == 0:
                return self.score(X, y)
            
            # Apply the filter to X and y
            X_filtered = [X[i] for i in filtered_indices]
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
            return self.score(X, y)

    def save_state(self, path: Path):
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
        train_info = {
            "loss_history": self.loss_history,
            "probe_type": "non_trainable"
        }
        with open(log_path, "w") as f:
            json.dump(train_info, f, indent=2)

    def _get_probe_parameters(self) -> dict:
        """
        Return probe-specific parameters to save. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement _get_probe_parameters")

    def load_state(self, path: Path):
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

    def _load_probe_parameters(self, checkpoint: dict):
        """
        Load probe-specific parameters. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement _load_probe_parameters")
