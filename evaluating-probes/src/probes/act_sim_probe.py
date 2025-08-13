import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from pathlib import Path
import json

from src.probes.base_probe_non_trainable import BaseProbeNonTrainable

class ActivationSimilarityProbe(BaseProbeNonTrainable):
    """
    Non-trainable probe that computes cosine similarity between activations and class prototypes.
    
    The probe score is: cosine_sim(activation, positive_prototype) - cosine_sim(activation, negative_prototype)
    
    Expects pre-aggregated activations of shape (N, d_model) where N is the number of samples.
    """
    
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", aggregation: str = "mean"):
        super().__init__(d_model, device, task_type, aggregation)
        self.positive_prototype = None
        self.negative_prototype = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Compute class prototypes from pre-aggregated training activations.
        
        Args:
            X: Pre-aggregated input features, shape (N, d_model)
            y: Labels, shape (N,) or (N, num_classes)
        """
        print(f"\n=== ACTIVATION SIMILARITY PROBE FITTING ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Task type: {self.task_type}")
        print(f"Aggregation: {self.aggregation}")
        
        # Store aggregated activations and labels
        self.X_aggregated = X
        self.y = y
        
        # Binary classification only
        y_flat = y.flatten() if y.ndim > 1 else y
        unique_classes = np.unique(y_flat)
        
        if len(unique_classes) == 2:
            # Binary classification
            pos_mask = y_flat == unique_classes[1]  # Assume class 1 is positive
            neg_mask = y_flat == unique_classes[0]  # Assume class 0 is negative
            
            if pos_mask.sum() > 0:
                self.positive_prototype = X[pos_mask].mean(axis=0)
            else:
                self.positive_prototype = np.zeros(X.shape[1])
            
            if neg_mask.sum() > 0:
                self.negative_prototype = X[neg_mask].mean(axis=0)
            else:
                self.negative_prototype = np.zeros(X.shape[1])
            
            print(f"Binary classification - Positive class: {unique_classes[1]}, Negative class: {unique_classes[0]}")
            print(f"Positive samples: {pos_mask.sum()}, Negative samples: {neg_mask.sum()}")
        else:
            # Single class (edge case)
            self.positive_prototype = X.mean(axis=0)
            self.negative_prototype = np.zeros(X.shape[1])
            print(f"Warning: Expected 2 classes for binary classification, got {len(unique_classes)}")
            print(f"Single class detected: {unique_classes[0]}")
        
        print(f"=== ACTIVATION SIMILARITY PROBE FITTING COMPLETE ===\n")
        return self
    
    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """
        Compute logits (similarity scores) from pre-aggregated activations (binary classification only).
        
        Args:
            X: Pre-aggregated activations, shape (N, d_model)
            
        Returns:
            Logits: shape (N,)
        """
        # Binary classification only: positive similarity - negative similarity
        # Check prototypes for NaN
        if np.isnan(self.positive_prototype).any():
            print(f"Warning: NaN detected in positive_prototype")
            # Replace NaN with zeros
            self.positive_prototype = np.nan_to_num(self.positive_prototype, nan=0.0)
        if np.isnan(self.negative_prototype).any():
            print(f"Warning: NaN detected in negative_prototype")
            # Replace NaN with zeros
            self.negative_prototype = np.nan_to_num(self.negative_prototype, nan=0.0)
        
        # Check if prototypes are all zeros
        if np.all(self.positive_prototype == 0):
            print(f"Warning: positive_prototype is all zeros")
        if np.all(self.negative_prototype == 0):
            print(f"Warning: negative_prototype is all zeros")
        
        pos_sim = self._cosine_similarity(X, self.positive_prototype)
        neg_sim = self._cosine_similarity(X, self.negative_prototype)
        logits = pos_sim - neg_sim
        
        # Check for NaN in logits
        if np.isnan(logits).any():
            print(f"Warning: NaN detected in logits computation")
            print(f"pos_sim range: [{pos_sim.min():.4f}, {pos_sim.max():.4f}]")
            print(f"neg_sim range: [{neg_sim.min():.4f}, {neg_sim.max():.4f}]")
            print(f"logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            # Replace NaN with zeros as fallback
            logits = np.nan_to_num(logits, nan=0.0)
        
        return logits
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions from pre-aggregated activations (binary classification only).
        
        Args:
            X: Pre-aggregated activations, shape (N, d_model)
            
        Returns:
            Predictions: shape (N,)
        """
        logits = self.predict_logits(X)
        
        # Binary classification only: threshold at 0
        return (logits > 0).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute probabilities from pre-aggregated activations (binary classification only).
        
        Args:
            X: Pre-aggregated activations, shape (N, d_model)
            
        Returns:
            Probabilities: shape (N, 2) - [P(class 0), P(class 1)]
        """
        logits = self.predict_logits(X)
        
        # Binary classification only: use PyTorch sigmoid for GPU acceleration
        # Check for NaN in input logits
        if np.isnan(logits).any():
            print(f"Warning: NaN detected in logits before sigmoid")
            # Replace NaN with 0 for sigmoid computation
            logits = np.nan_to_num(logits, nan=0.0)
        
        logits_tensor = torch.tensor(logits, dtype=torch.float32, device=self.device)
        probs_positive = torch.sigmoid(logits_tensor).cpu().numpy()
        probs_negative = 1 - probs_positive
        
        # Check for NaN in output probabilities
        if np.isnan(probs_positive).any():
            print(f"Warning: NaN detected in probabilities after sigmoid")
        
        return np.column_stack([probs_negative, probs_positive])
    
    def _cosine_similarity(self, X: np.ndarray, prototype: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between activations and prototype using PyTorch's built-in function.
        
        Args:
            X: Shape (N, d_model)
            prototype: Shape (d_model,)
            
        Returns:
            Similarities: Shape (N,)
        """
        import torch.nn.functional as F
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        prototype_tensor = torch.tensor(prototype, dtype=torch.float32, device=self.device)
        
        # Replace any infinity values with zeros
        X_tensor = torch.nan_to_num(X_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        prototype_tensor = torch.nan_to_num(prototype_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Expand prototype to match X shape for broadcasting
        # prototype_tensor: (d_model,) -> (1, d_model)
        prototype_expanded = prototype_tensor.unsqueeze(0)
        
        # Compute cosine similarity using PyTorch's built-in function
        # This handles zero vectors and numerical stability automatically
        similarities = F.cosine_similarity(X_tensor, prototype_expanded, dim=1)
        
        # Clamp to valid range [-1, 1] to handle any numerical errors
        similarities = torch.clamp(similarities, min=-1.0, max=1.0)
        
        return similarities.cpu().numpy()
    
    def _get_probe_parameters(self) -> dict:
        """
        Return probe-specific parameters to save.
        """
        params = {}
        if self.positive_prototype is not None:
            params['positive_prototype'] = self.positive_prototype
        if self.negative_prototype is not None:
            params['negative_prototype'] = self.negative_prototype
        return params
    
    def _load_probe_parameters(self, checkpoint: dict):
        """
        Load probe-specific parameters.
        """
        if 'positive_prototype' in checkpoint:
            self.positive_prototype = checkpoint['positive_prototype']
        if 'negative_prototype' in checkpoint:
            self.negative_prototype = checkpoint['negative_prototype']
