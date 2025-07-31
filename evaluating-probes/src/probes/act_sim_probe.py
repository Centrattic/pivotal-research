import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from pathlib import Path
import json

from src.probes.base_probe import BaseProbeNonTrainable

class ActivationSimilarityProbe(BaseProbeNonTrainable):
    """
    Non-trainable probe that computes cosine similarity between activations and class prototypes.
    
    The probe score is: cosine_sim(activation, positive_prototype) - cosine_sim(activation, negative_prototype)
    """
    
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", aggregation: str = "mean"):
        super().__init__(d_model, device, task_type, aggregation)
        self.positive_prototype = None
        self.negative_prototype = None
        self.class_prototypes = None  # For multiclass
        
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
        
        if self.task_type == "classification":
            # Handle binary and multiclass classification
            if y.ndim == 1 or y.shape[1] == 1:
                # Binary classification
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
                    print(f"Single class detected: {unique_classes[0]}")
            else:
                # Multiclass classification
                y_flat = np.argmax(y, axis=1) if y.ndim > 1 else y
                unique_classes = np.unique(y_flat)
                n_classes = len(unique_classes)
                
                self.class_prototypes = {}
                for cls in unique_classes:
                    cls_mask = y_flat == cls
                    if cls_mask.sum() > 0:
                        self.class_prototypes[cls] = aggregated_X[cls_mask].mean(axis=0)
                    else:
                        self.class_prototypes[cls] = np.zeros(aggregated_X.shape[1])
                
                print(f"Multiclass classification - {n_classes} classes: {unique_classes}")
                for cls in unique_classes:
                    cls_mask = y_flat == cls
                    print(f"  Class {cls}: {cls_mask.sum()} samples")
        else:
            # Regression - not really applicable for similarity probes
            raise ValueError("ActivationSimilarityProbe is designed for classification tasks only")
        
        print(f"=== ACTIVATION SIMILARITY PROBE FITTING COMPLETE ===\n")
        return self
    
    def _fit_class_by_class(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 1000):
        """
        Fit the probe by processing each class separately to save memory.
        """
        if self.task_type == "classification":
            if y.ndim == 1 or y.shape[1] == 1:
                # Binary classification
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
                    print(f"Single class detected: {unique_classes[0]}")
            else:
                # Multiclass classification
                y_flat = np.argmax(y, axis=1) if y.ndim > 1 else y
                unique_classes = np.unique(y_flat)
                n_classes = len(unique_classes)
                
                self.class_prototypes = {}
                for cls in unique_classes:
                    cls_indices = np.where(y_flat == cls)[0]
                    self.class_prototypes[cls] = self._compute_class_prototype(X, cls_indices, mask, batch_size)
                    print(f"  Class {cls}: {len(cls_indices)} samples")
                
                print(f"Multiclass classification - {n_classes} classes: {unique_classes}")
        else:
            raise ValueError("ActivationSimilarityProbe is designed for classification tasks only")
    
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
        Compute logits (similarity scores) from processed activations.
        """
        if self.task_type == "classification":
            if self.class_prototypes is not None:
                # Multiclass: compute similarity to each class prototype
                n_samples = processed_X.shape[0]
                n_classes = len(self.class_prototypes)
                logits = np.zeros((n_samples, n_classes))
                
                for i, cls in enumerate(sorted(self.class_prototypes.keys())):
                    prototype = self.class_prototypes[cls]
                    similarities = self._cosine_similarity(processed_X, prototype)
                    logits[:, i] = similarities
                
                return logits
            else:
                # Binary: positive similarity - negative similarity
                pos_sim = self._cosine_similarity(processed_X, self.positive_prototype)
                neg_sim = self._cosine_similarity(processed_X, self.negative_prototype)
                return pos_sim - neg_sim
        else:
            raise ValueError("ActivationSimilarityProbe is designed for classification tasks only")
    
    def _compute_predictions(self, processed_X: np.ndarray) -> np.ndarray:
        """
        Compute predictions from processed activations.
        """
        logits = self._compute_logits(processed_X)
        
        if self.task_type == "classification":
            if self.class_prototypes is not None:
                # Multiclass: argmax
                return np.argmax(logits, axis=1)
            else:
                # Binary: threshold at 0
                return (logits > 0).astype(int)
        else:
            raise ValueError("ActivationSimilarityProbe is designed for classification tasks only")
    
    def _compute_probabilities(self, processed_X: np.ndarray) -> np.ndarray:
        """
        Compute probabilities from processed activations.
        """
        logits = self._compute_logits(processed_X)
        
        if self.task_type == "classification":
            if self.class_prototypes is not None:
                # Multiclass: softmax
                exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            else:
                # Binary: sigmoid of logits
                return 1 / (1 + np.exp(-logits))
        else:
            raise ValueError("ActivationSimilarityProbe is designed for classification tasks only")
    
    def _cosine_similarity(self, X: np.ndarray, prototype: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between activations and prototype.
        
        Args:
            X: Shape (N, d_model)
            prototype: Shape (d_model,)
            
        Returns:
            Similarities: Shape (N,)
        """
        # Normalize both vectors
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        prototype_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(X_norm, prototype_norm)
        return similarities
    
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
