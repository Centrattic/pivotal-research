import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
import json

from src.probes.base_probe import BaseProbe

class MassMeanProbeNet(nn.Module):
    """
    Mass Mean probe implementation.
    
    Implements the mass-mean probe as described in the paper:
    - θ_mm = μ_+ - μ_- where μ_+, μ_- are the means of positive and negative labeled datapoints
    - p_mm(x) = σ(θ_mm^T x) for basic mass-mean
    - p_iid_mm(x) = σ(θ_mm^T Σ^(-1) x) for IID version (Fisher's LDA)
    """
    def __init__(self, d_model: int, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.device = device
        
        # These will be computed during fit
        self.theta_mm = None  # θ_mm = μ_+ - μ_-
        self.sigma_inv = None  # Σ^(-1) for IID version
        self.mu_plus = None   # μ_+ (mean of positive class)
        self.mu_minus = None  # μ_- (mean of negative class)
        
        self.to(device)

    def compute_mass_mean_direction(self, X: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, use_iid: bool = False):
        """
        Compute the mass-mean direction θ_mm = μ_+ - μ_-
        
        Args:
            X: Input features, shape (N, seq, d_model)
            y: Labels, shape (N,) with binary values {0, 1}
            mask: Mask, shape (N, seq)
            use_iid: Whether to use IID version (Fisher's LDA)
        """
        # Apply aggregation first
        X_agg = self._aggregate(X, mask)  # (N, d_model)
        
        # Separate positive and negative samples
        pos_mask = (y == 1)
        neg_mask = (y == 0)
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            raise ValueError("Need samples from both positive and negative classes")
        
        # Compute means
        X_pos = X_agg[pos_mask]  # (N_pos, d_model)
        X_neg = X_agg[neg_mask]  # (N_neg, d_model)
        
        self.mu_plus = X_pos.mean(dim=0)   # (d_model,)
        self.mu_minus = X_neg.mean(dim=0)  # (d_model,)
        
        # Compute mass-mean direction
        self.theta_mm = self.mu_plus - self.mu_minus  # (d_model,)
        
        # For IID version, compute covariance matrix and its inverse
        if use_iid:
            # Center the data: subtract μ_+ from positive samples, μ_- from negative samples
            X_pos_centered = X_pos - self.mu_plus.unsqueeze(0)   # (N_pos, d_model)
            X_neg_centered = X_neg - self.mu_minus.unsqueeze(0)  # (N_neg, d_model)
            
            # Combine centered data
            X_centered = torch.cat([X_pos_centered, X_neg_centered], dim=0)  # (N, d_model)
            
            # Compute covariance matrix
            sigma = torch.cov(X_centered.T)  # (d_model, d_model)
            
            # Add small regularization to ensure invertibility
            sigma_reg = sigma + 1e-6 * torch.eye(sigma.shape[0], device=sigma.device)
            
            # Compute inverse
            try:
                self.sigma_inv = torch.inverse(sigma_reg)
            except:
                # If inverse fails, use pseudo-inverse
                self.sigma_inv = torch.pinverse(sigma_reg)

    def _aggregate(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mean aggregation to sequence dimension."""
        x = x * mask.unsqueeze(-1)
        x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for mass-mean probe.
        
        Args:
            x: Input features, shape (batch, seq, d_model)
            mask: Mask, shape (batch, seq)
            
        Returns:
            logits: shape (batch,)
        """
        if self.theta_mm is None:
            raise ValueError("Mass-mean direction not computed. Call fit() first.")
        
        # Apply aggregation
        x_agg = self._aggregate(x, mask)  # (batch, d_model)
        
        if self.sigma_inv is not None:
            # IID version: θ_mm^T Σ^(-1) x
            logits = torch.matmul(x_agg, torch.matmul(self.sigma_inv, self.theta_mm))
        else:
            # Basic version: θ_mm^T x
            logits = torch.matmul(x_agg, self.theta_mm)
        
        return logits

class MassMeanProbe(BaseProbe):
    """
    Mass Mean probe that computes the direction between class means.
    This probe requires no training - it's computed analytically.
    """
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", use_iid: bool = False, **kwargs):
        # Store any additional config parameters for this probe
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.use_iid = use_iid
        self.aggregation = "mean"  # Mass-mean probes always use mean aggregation
        super().__init__(d_model, device, task_type)

    def _init_model(self):
        self.model = MassMeanProbeNet(self.d_model, device=self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, **kwargs):
        """
        Compute the mass-mean direction. No training needed - computed analytically.
        
        Args:
            X: Input features, shape (N, seq, d_model)
            y: Labels, shape (N,) with binary values {0, 1}
            mask: Optional mask, shape (N, seq)
            **kwargs: Ignored for mass-mean probe
        """
        use_iid = getattr(self, 'use_iid', False)
        print(f"\n=== MASS MEAN PROBE COMPUTATION ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Mask shape: {mask.shape if mask is not None else 'None'}")
        print(f"Use IID: {use_iid}")
        print(f"Aggregation: mean (fixed)")
        
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        else:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=self.device)
        
        # Check that we have binary classification
        unique_labels = torch.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(f"Mass-mean probe requires binary classification. Found {len(unique_labels)} classes: {unique_labels}")
        
        # Ensure labels are 0 and 1
        if not torch.all((unique_labels == 0) | (unique_labels == 1)):
            # Remap labels to 0 and 1
            y = (y == unique_labels[1]).long()
            print(f"Remapped labels: {unique_labels[0]} -> 0, {unique_labels[1]} -> 1")
        
        # Compute mass-mean direction
        self.model.compute_mass_mean_direction(X, y, mask, use_iid=use_iid)
        
        # Compute some statistics for debugging
        pos_count = (y == 1).sum().item()
        neg_count = (y == 0).sum().item()
        print(f"Positive samples: {pos_count}, Negative samples: {neg_count}")
        print(f"θ_mm norm: {torch.norm(self.model.theta_mm).item():.4f}")
        if use_iid and self.model.sigma_inv is not None:
            print(f"Σ^(-1) computed successfully")
        
        print(f"=== MASS MEAN PROBE COMPUTATION COMPLETE ===\n")
        return self

    def find_best_fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                     mask_train: Optional[np.ndarray] = None, mask_val: Optional[np.ndarray] = None, 
                     n_trials: int = 10, direction: str = None, verbose: bool = True, 
                     weighting_method: str = 'weighted_loss', metric: str = 'acc', fpr_threshold: float = 0.01, 
                     probe_save_dir: Optional[Path] = None, probe_filename_base: Optional[str] = None):
        """
        For mass-mean probe, we don't need hyperparameter tuning since it's computed analytically.
        We can try different aggregation methods and IID vs non-IID versions.
        """
        print(f"Mass-mean probe doesn't require hyperparameter tuning - computed analytically")
        return {"method": "mass_mean_analytical"}

    def save_state(self, path: Path):
        """Override save_state to save mass-mean specific parameters."""
        # Save the computed mass-mean parameters
        mass_mean_state = {
            'theta_mm': self.model.theta_mm.cpu() if self.model.theta_mm is not None else None,
            'sigma_inv': self.model.sigma_inv.cpu() if self.model.sigma_inv is not None else None,
            'mu_plus': self.model.mu_plus.cpu() if self.model.mu_plus is not None else None,
            'mu_minus': self.model.mu_minus.cpu() if self.model.mu_minus is not None else None,
            'd_model': self.d_model,
            'task_type': self.task_type,
            'aggregation': self.aggregation,
        }
        torch.save(mass_mean_state, path)
        print(f"Saved mass-mean probe to {path}")
        
        # Save training info (loss history, etc.)
        log_path = path.with_name(path.stem + "_train_log.json")
        train_info = {
            "loss_history": self.loss_history,
            "method": "mass_mean_analytical",
        }
        with open(log_path, "w") as f:
            json.dump(train_info, f, indent=2)
        print(f"Saved training log to {log_path}")

    def load_state(self, path: Path):
        """Override load_state to load mass-mean specific parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.d_model = checkpoint['d_model']
        self.task_type = checkpoint['task_type']
        self.aggregation = checkpoint['aggregation']
        
        # Reinitialize the model
        self._init_model()
        
        # Load the computed parameters
        if checkpoint['theta_mm'] is not None:
            self.model.theta_mm = checkpoint['theta_mm'].to(self.device)
        if checkpoint['sigma_inv'] is not None:
            self.model.sigma_inv = checkpoint['sigma_inv'].to(self.device)
        if checkpoint['mu_plus'] is not None:
            self.model.mu_plus = checkpoint['mu_plus'].to(self.device)
        if checkpoint['mu_minus'] is not None:
            self.model.mu_minus = checkpoint['mu_minus'].to(self.device)
        
        print(f"Loaded mass-mean probe from {path}") 