import numpy as np
import torch
from pathlib import Path
from typing import Optional
import json

from src.probes.base_probe import BaseProbeNonTrainable

class MassMeanProbe(BaseProbeNonTrainable):
    """
    Mass Mean probe that computes the direction between class means.
    This probe requires no training - it's computed analytically.
    """
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", use_iid: bool = False, **kwargs):
        # Mass-mean probes always use mean aggregation (no sequence aggregation needed)
        super().__init__(d_model, device, task_type, aggregation="mass_mean")
        # IID functionality disabled due to numerical instability with small sample sizes
        # self.use_iid = use_iid
        self.use_iid = False  # Always use basic mass-mean
        self.theta_mm = None  # θ_mm = μ_+ - μ_-
        # self.sigma_inv = None  # Σ^(-1) for IID version (disabled)
        self.mu_plus = None   # μ_+ (mean of positive class)
        self.mu_minus = None  # μ_- (mean of negative class)

    def fit(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 1000, **kwargs):
        """
        Compute the mass-mean direction using batched processing to save memory.
        
        Args:
            X: Input features, shape (N, seq, d_model)
            y: Labels, shape (N,) with binary values {0, 1}
            mask: Optional mask, shape (N, seq)
            batch_size: Batch size for processing to avoid memory issues
        """
        print(f"\n=== MASS MEAN PROBE COMPUTATION ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Use IID: {self.use_iid}")
        print(f"Batch size: {batch_size}")
        
        # Process data in batches to avoid memory issues
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Initialize accumulators for computing means
        pos_sum = None
        neg_sum = None
        pos_count = 0
        neg_count = 0
        
        # Process batches
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            batch_mask = mask[start_idx:end_idx] if mask is not None else None
            
            # Check that we have binary classification
            unique_labels = np.unique(batch_y)
            if len(unique_labels) != 2:
                raise ValueError(f"Mass-mean probe requires binary classification. Found {len(unique_labels)} classes: {unique_labels}")
            
            # Ensure labels are 0 and 1
            if not np.all((unique_labels == 0) | (unique_labels == 1)):
                # Remap labels to 0 and 1
                batch_y = (batch_y == unique_labels[1]).astype(int)
                if i == 0:
                    print(f"Remapped labels: {unique_labels[0]} -> 0, {unique_labels[1]} -> 1")
            
            # Aggregate this batch (mean over sequence dimension)
            batch_aggregated = self._aggregate_activations(batch_X, batch_mask)
            
            # Separate positive and negative samples
            pos_mask = batch_y == 1
            neg_mask = batch_y == 0
            
            # Accumulate positive samples
            if pos_mask.sum() > 0:
                pos_batch_sum = batch_aggregated[pos_mask].sum(axis=0)
                if pos_sum is None:
                    pos_sum = pos_batch_sum
                else:
                    pos_sum += pos_batch_sum
                pos_count += pos_mask.sum()
            
            # Accumulate negative samples
            if neg_mask.sum() > 0:
                neg_batch_sum = batch_aggregated[neg_mask].sum(axis=0)
                if neg_sum is None:
                    neg_sum = neg_batch_sum
                else:
                    neg_sum += neg_batch_sum
                neg_count += neg_mask.sum()
            
            if i % 10 == 0 or i == n_batches - 1:
                print(f"Processed batch {i+1}/{n_batches} ({start_idx}-{end_idx})")
        
        # Compute final means
        if pos_count > 0:
            self.mu_plus = pos_sum / pos_count
        else:
            self.mu_plus = np.zeros(X.shape[2])
        
        if neg_count > 0:
            self.mu_minus = neg_sum / neg_count
        else:
            self.mu_minus = np.zeros(X.shape[2])
        
        # Compute mass-mean direction
        self.theta_mm = self.mu_plus - self.mu_minus
        
        return self
    
    # IID functionality disabled due to numerical instability with small sample sizes
    # def _compute_iid_version(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray], batch_size: int):
    #     """
    #     Compute the IID version (Fisher's LDA) using batched processing.
    #     """
    #     # ... entire method commented out ...

    def _compute_logits(self, processed_X: np.ndarray) -> np.ndarray:
        """Compute logits for mass-mean probe using PyTorch for GPU acceleration."""
        if self.theta_mm is None:
            raise ValueError("Mass-mean direction not computed. Call fit() first.")
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(processed_X, dtype=torch.float16, device=self.device)
        theta_tensor = torch.tensor(self.theta_mm, dtype=torch.float16, device=self.device)
        
        # IID functionality disabled due to numerical instability with small sample sizes
        # Always use basic version: θ_mm^T x
        logits = torch.matmul(X_tensor, theta_tensor)
        
        return logits.cpu().numpy()

    def _compute_predictions(self, processed_X: np.ndarray) -> np.ndarray:
        """Compute predictions for mass-mean probe."""
        logits = self._compute_logits(processed_X)
        return (logits > 0).astype(int)

    def _compute_probabilities(self, processed_X: np.ndarray) -> np.ndarray:
        """Compute probabilities for mass-mean probe."""
        logits = self._compute_logits(processed_X)
        
        # Use PyTorch sigmoid for GPU acceleration
        logits_tensor = torch.tensor(logits, dtype=torch.float16, device=self.device)
        probs = torch.sigmoid(logits_tensor).cpu().numpy()
        
        # For binary classification, return just the positive class probability
        # This matches the expected format for sklearn's roc_auc_score
        return probs  # Shape: (N,) - probability of positive class

    def _get_probe_parameters(self) -> dict:
        """Get probe parameters for saving."""
        return {
            'theta_mm': self.theta_mm,
            # 'sigma_inv': self.sigma_inv,  # IID functionality disabled
            'mu_plus': self.mu_plus,
            'mu_minus': self.mu_minus,
            'use_iid': self.use_iid,
        }

    def _load_probe_parameters(self, checkpoint: dict):
        """Load probe parameters from checkpoint."""
        self.theta_mm = checkpoint['theta_mm']
        # self.sigma_inv = checkpoint['sigma_inv']  # IID functionality disabled
        self.mu_plus = checkpoint['mu_plus']
        self.mu_minus = checkpoint['mu_minus']
        # Handle backward compatibility - use_iid maybe was added later
        # self.use_iid = checkpoint.get('use_iid', False)  # IID functionality disabled
        self.use_iid = False  # Always use basic mass-mean
        
        # Mass-mean probes should always use mass_mean aggregation, regardless of saved state
        # This fixes the issue where old checkpoints have the wrong aggregation method
        self.aggregation = "mass_mean" 