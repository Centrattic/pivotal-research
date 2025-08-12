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
    """
    
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", aggregation: str = "mean"):
        super().__init__(d_model, device, task_type, aggregation)
        self.positive_prototype = None
        self.negative_prototype = None
        self.class_prototypes = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 1000, **kwargs):
        """
        Compute class prototypes from training data using batched processing to save memory.
        
        Args:
            X: Input features, shape (N, seq, d_model)
            y: Labels, shape (N,) or (N, num_classes)
            mask: Optional mask, shape (N, seq)
            batch_size: Batch size for processing to avoid memory issues
        """
        print(f"\n=== ACTIVATION SIMILARITY PROBE FITTING ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Task type: {self.task_type}")
        print(f"Aggregation: {self.aggregation}")
        print(f"Batch size: {batch_size}")
        
        # Process activations in batches to avoid memory issues
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Initialize storage for aggregated activations
        if self.aggregation == "mass_mean":
            # For mass-mean, we don't aggregate, so we need to store the full activations
            # This is memory-intensive, so we'll process class by class
            print("Processing class by class to save memory...")
            self._fit_class_by_class(X, y, mask, batch_size)
            return self
        else:
            # For other aggregation methods, we can aggregate in batches
            aggregated_X = np.zeros((n_samples, X.shape[2]), dtype=np.float32)
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_X = X[start_idx:end_idx]
                batch_mask = mask[start_idx:end_idx] if mask is not None else None
                
                batch_aggregated = self._aggregate_activations(batch_X, batch_mask)
                aggregated_X[start_idx:end_idx] = batch_aggregated
                
                if i % 10 == 0 or i == n_batches - 1:
                    print(f"Processed batch {i+1}/{n_batches} ({start_idx}-{end_idx})")
        
        print(f"Aggregated X shape: {aggregated_X.shape}")
        
        # Binary classification only
        y_flat = y.flatten() if y.ndim > 1 else y
        unique_classes = np.unique(y_flat)
        
        if len(unique_classes) == 2:
            # Binary classification
            pos_mask = y_flat == unique_classes[1]  # Assume class 1 is positive
            neg_mask = y_flat == unique_classes[0]  # Assume class 0 is negative
            
            if pos_mask.sum() > 0:
                self.positive_prototype = aggregated_X[pos_mask].mean(axis=0)
            else:
                self.positive_prototype = np.zeros(aggregated_X.shape[1])
            
            if neg_mask.sum() > 0:
                self.negative_prototype = aggregated_X[neg_mask].mean(axis=0)
            else:
                self.negative_prototype = np.zeros(aggregated_X.shape[1])
            
            print(f"Binary classification - Positive class: {unique_classes[1]}, Negative class: {unique_classes[0]}")
            print(f"Positive samples: {pos_mask.sum()}, Negative samples: {neg_mask.sum()}")
        else:
            # Single class (edge case)
            self.positive_prototype = aggregated_X.mean(axis=0)
            self.negative_prototype = np.zeros(aggregated_X.shape[1])
            print(f"Warning: Expected 2 classes for binary classification, got {len(unique_classes)}")
            print(f"Single class detected: {unique_classes[0]}")
        
        print(f"=== ACTIVATION SIMILARITY PROBE FITTING COMPLETE ===\n")
        return self
        
        print(f"=== ACTIVATION SIMILARITY PROBE FITTING COMPLETE ===\n")
        return self
    
    def _fit_class_by_class(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 1000):
        """
        Fit the probe by processing each class separately to save memory (binary classification only).
        """
        # Binary classification only
        y_flat = y.flatten() if y.ndim > 1 else y
        unique_classes = np.unique(y_flat)
        
        if len(unique_classes) == 2:
            # Process positive class
            pos_mask = y_flat == unique_classes[1]
            pos_indices = np.where(pos_mask)[0]
            self.positive_prototype = self._compute_class_prototype(X, pos_indices, mask, batch_size)
            
            # Process negative class
            neg_mask = y_flat == unique_classes[0]
            neg_indices = np.where(neg_mask)[0]
            self.negative_prototype = self._compute_class_prototype(X, neg_indices, mask, batch_size)
            
            print(f"Binary classification - Positive class: {unique_classes[1]}, Negative class: {unique_classes[0]}")
            print(f"Positive samples: {pos_mask.sum()}, Negative samples: {neg_mask.sum()}")
        else:
            # Single class (edge case)
            self.positive_prototype = self._compute_class_prototype(X, np.arange(len(X)), mask, batch_size)
            self.negative_prototype = np.zeros(X.shape[2])
            print(f"Warning: Expected 2 classes for binary classification, got {len(unique_classes)}")
            print(f"Single class detected: {unique_classes[0]}")
    
    def _compute_class_prototype(self, X: np.ndarray, indices: np.ndarray, mask: Optional[np.ndarray], batch_size: int) -> np.ndarray:
        """
        Compute prototype for a specific class using batched processing.
        """
        if len(indices) == 0:
            return np.zeros(X.shape[2])
        
        # Process class samples in batches
        n_batches = (len(indices) + batch_size - 1) // batch_size
        class_sum = np.zeros(X.shape[2], dtype=np.float32)
        total_count = 0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            batch_X = X[batch_indices]
            batch_mask = mask[batch_indices] if mask is not None else None
            
            # Aggregate this batch
            batch_aggregated = self._aggregate_activations(batch_X, batch_mask)
            class_sum += batch_aggregated.sum(axis=0)
            total_count += len(batch_indices)
        
        return class_sum / total_count
    
    def _compute_logits(self, processed_X: np.ndarray) -> np.ndarray:
        """
        Compute logits (similarity scores) from processed activations (binary classification only).
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
        
        pos_sim = self._cosine_similarity(processed_X, self.positive_prototype)
        neg_sim = self._cosine_similarity(processed_X, self.negative_prototype)
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
    
    def _compute_predictions(self, processed_X: np.ndarray) -> np.ndarray:
        """
        Compute predictions from processed activations (binary classification only).
        """
        logits = self._compute_logits(processed_X)
        
        # Binary classification only: threshold at 0
        return (logits > 0).astype(int)
    
    def _compute_probabilities(self, processed_X: np.ndarray) -> np.ndarray:
        """
        Compute probabilities from processed activations (binary classification only).
        """
        logits = self._compute_logits(processed_X)
        
        # Binary classification only: use PyTorch sigmoid for GPU acceleration
        # Check for NaN in input logits
        if np.isnan(logits).any():
            print(f"Warning: NaN detected in logits before sigmoid")
            # Replace NaN with 0 for sigmoid computation
            logits = np.nan_to_num(logits, nan=0.0)
        
        logits_tensor = torch.tensor(logits, dtype=torch.bfloat16, device=self.device)
        # Convert bfloat16 to float32 only for numpy conversion
        probs = torch.sigmoid(logits_tensor).float().cpu().numpy()
        
        # Check for NaN in output probabilities
        if np.isnan(probs).any():
            print(f"Warning: NaN detected in probabilities after sigmoid")
        
        return probs
    
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
        X_tensor = torch.tensor(X, dtype=torch.bfloat16, device=self.device)
        prototype_tensor = torch.tensor(prototype, dtype=torch.bfloat16, device=self.device)
        
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
        
        # Convert bfloat16 to float32 only for numpy conversion
        return similarities.float().cpu().numpy()
    
    def _get_probe_parameters(self) -> dict:
        """
        Return probe-specific parameters to save.
        """
        params = {}
        if self.positive_prototype is not None:
            params['positive_prototype'] = self.positive_prototype
        if self.negative_prototype is not None:
            params['negative_prototype'] = self.negative_prototype
        if self.class_prototypes is not None:
            params['class_prototypes'] = self.class_prototypes
        return params
    
    def _load_probe_parameters(self, checkpoint: dict):
        """
        Load probe-specific parameters.
        """
        if 'positive_prototype' in checkpoint:
            self.positive_prototype = checkpoint['positive_prototype']
        if 'negative_prototype' in checkpoint:
            self.negative_prototype = checkpoint['negative_prototype']
        if 'class_prototypes' in checkpoint:
            self.class_prototypes = checkpoint['class_prototypes']
