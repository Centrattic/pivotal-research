import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import json
import warnings
from tqdm import tqdm
import optuna
import psutil
import gc

from sae_lens import SAE
from src.probes.base_probe import BaseProbe
from src.logger import Logger

def get_memory_usage():
    """Get current memory usage for debugging."""
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3  # GB
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"RAM: {ram_usage:.2f}GB, GPU_allocated: {gpu_allocated:.2f}GB, GPU_reserved: {gpu_reserved:.2f}GB"
    else:
        return f"RAM: {ram_usage:.2f}GB"

class SAEProbeNet(nn.Module):
    """
    Neural network for SAE-based probing.
    Takes SAE-encoded activations and applies a linear classifier.
    """
    def __init__(self, sae_feature_dim: int, aggregation: str = "mean", device: str = "cpu"):
        super().__init__()
        self.sae_feature_dim = sae_feature_dim
        self.aggregation = aggregation
        self.device = device
        # Create linear layer with float32 dtype for mixed precision training
        self.linear = nn.Linear(sae_feature_dim, 1, dtype=torch.float32).to(self.device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, sae_feature_dim), mask: (batch, seq)
        if self.aggregation == "mean":
            x = x * mask.unsqueeze(-1)
            x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        elif self.aggregation == "max":
            x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            x, _ = x.max(dim=1)
        elif self.aggregation == "last":
            idx = mask.sum(dim=1) - 1
            idx = idx.clamp(min=0)
            x = x[torch.arange(x.size(0)), idx]
        elif self.aggregation == "softmax":
            attn_scores = x.mean(dim=-1)  # (batch, seq)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)
            x = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        logits = self.linear(x).squeeze(-1)
        return logits

class SAEProbe(BaseProbe):
    """
    SAE-based probe that loads pre-trained SAEs and trains linear classifiers on their features.
    Supports pooled aggregation across sequence dimensions and feature selection.
    """
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", 
                 aggregation: str = "mean", model_name: str = "gemma-2-9b", layer: int = 20,
                 sae_id: Optional[str] = None, top_k_features: int = 128, 
                 sae_cache_dir: Optional[Path] = None, encoding_batch_size: int = 256, 
                 training_batch_size: int = 64, **kwargs):
        """
        Initialize SAE probe.
        
        Args:
            d_model: Original model hidden dimension
            device: Device to use
            task_type: 'classification' or 'regression'
            aggregation: How to aggregate across sequence dimension ('mean', 'max', 'last', 'softmax')
            model_name: Name of the base model (e.g., 'gemma-2-9b')
            layer: Layer number to extract activations from
            sae_id: Specific SAE ID to use (must be provided)
            top_k_features: Number of top features to select using difference of means
            sae_cache_dir: Directory to cache SAE models
            batch_size: Batch size for SAE encoding and probe training
        """
        if sae_id is None:
            raise ValueError("sae_id must be provided. Use specific SAE probe configurations from configs/probes.py")
        
        # Store SAE-specific parameters
        self.model_name = model_name
        self.layer = layer
        self.sae_id = sae_id
        self.top_k_features = top_k_features
        self.aggregation = aggregation
        self.sae_cache_dir = sae_cache_dir or Path("sae_cache")
        self.sae_cache_dir.mkdir(exist_ok=True)
        self.encoding_batch_size = encoding_batch_size
        self.training_batch_size = training_batch_size
        
        # Store any additional config parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize SAE and feature selection
        self.sae = None
        self.feature_indices = None
        self.sae_feature_dim = None
        
        # Call parent constructor
        super().__init__(d_model, device, task_type)

    def _init_model(self):
        """Initialize the neural network model."""
        # This will be set after SAE is loaded and features are selected
        if self.sae_feature_dim is not None:
            self.model = SAEProbeNet(self.sae_feature_dim, aggregation=self.aggregation, device=self.device)
        else:
            # Placeholder - will be set in fit method
            self.model = None

    def _load_sae(self) -> SAE:
        """Load the SAE model."""
        if self.sae is not None:
            return self.sae
        
        print(f"Loading SAE: {self.sae_id}")
        
        # Load from sae_lens using the specific SAE ID
        # Note: We don't cache SAE objects because they contain lambda functions that can't be pickled
        self.sae, _, _ = SAE.from_pretrained(
            release=self._get_sae_release(),
            sae_id=self.sae_id,
            device=self.device,
        )
        print(f"Loaded SAE: {self.sae_id}")
        
        return self.sae

    def _get_sae_release(self) -> str:
        """Get the SAE release name for the model."""
        if self.model_name == "gemma-2-9b":
            return "gemma-scope-9b-pt-res"
        elif self.model_name == "llama-3.1-8b":
            return "llama_scope_lxr_32x"
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def _encode_activations(self, activations: np.ndarray) -> np.ndarray:
        """Encode raw activations through the SAE using configurable batch size."""
        print(f"[DEBUG] Input activations shape: {activations.shape}")
        
        # For now, only process the last sequence position to reduce memory usage
        print(f"[INFO] Processing only the last sequence position for now to reduce memory usage")
        activations = activations[:, -1:, :]  # Take only the last position: (batch, 1, d_model)
        print(f"[DEBUG] After taking last position: {activations.shape}")
        
        sae = self._load_sae()
        
        # Get original shape and flatten batch and sequence dimensions
        original_shape = activations.shape
        flattened = activations.reshape(-1, activations.shape[-1])  # (batch*seq, d_model)
        print(f"[DEBUG] Flattened shape: {flattened.shape}")
        print(f"[DEBUG] This means {original_shape[0]} samples × {original_shape[1]} sequence positions = {flattened.shape[0]} total activations")
        
        # Encode in batches using configurable batch size
        encoded_list = []
        total_batches = (len(flattened) + self.encoding_batch_size - 1) // self.encoding_batch_size
        print(f"[DEBUG] Processing {total_batches} batches of size {self.encoding_batch_size}")
        
        for i in tqdm(range(0, len(flattened), self.encoding_batch_size), desc="Encoding activations"):
            batch = flattened[i:i+self.encoding_batch_size]
            # Convert batch to tensor and move to device only for this batch
            batch_tensor = torch.tensor(batch, dtype=torch.float16, device=self.device)
            encoded_batch = sae.encode(batch_tensor).cpu().detach().numpy()
            encoded_list.append(encoded_batch)
        
        encoded = np.concatenate(encoded_list, axis=0)
        
        # Reshape back to original batch and sequence dimensions
        encoded = encoded.reshape(original_shape[0], original_shape[1], -1)
        print(f"[DEBUG] Final encoded shape: {encoded.shape}")
        
        return encoded

    def _select_top_features(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Select top-k features using the difference of means method.
        
        Args:
            X_train: SAE-encoded activations, shape (N, seq, sae_features)
            y_train: Labels, shape (N,)
        
        Returns:
            Indices of top-k features
        """
        print(f"[DEBUG] Feature selection input shapes: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"[DEBUG] Memory at start of feature selection: {get_memory_usage()}")
        
        X_agg = X_train.mean(axis=1) # agg doesnt matter rn
        
        print(f"[DEBUG] Aggregated X shape: {X_agg.shape}")
        print(f"[DEBUG] Memory after aggregation: {get_memory_usage()}")
        
        # Calculate difference of means
        pos_mask = y_train == 1
        neg_mask = y_train == 0
        
        if not pos_mask.any() or not neg_mask.any():
            raise ValueError("Need both positive and negative samples for feature selection")
        
        pos_mean = X_agg[pos_mask].mean(axis=0)
        neg_mean = X_agg[neg_mask].mean(axis=0)
        diff = pos_mean - neg_mean
        
        print(f"[DEBUG] Difference vector shape: {diff.shape}")
        print(f"[DEBUG] Memory after difference calculation: {get_memory_usage()}")
        
        # Select top-k features by absolute difference
        sorted_indices = np.argsort(np.abs(diff))[::-1]
        top_k_indices = sorted_indices[:self.top_k_features]
        
        print(f"Selected top {self.top_k_features} features from {len(diff)} total features")
        print(f"[DEBUG] Memory at end of feature selection: {get_memory_usage()}")
        
        return top_k_indices

    def fit(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, 
            epochs: int = 20, lr: float = 1e-3, batch_size: int = None, weight_decay: float = 0.0, 
            verbose: bool = True, early_stopping: bool = True, patience: int = 20, min_delta: float = 0.005,
            use_weighted_loss: bool = True, use_weighted_sampler: bool = False, **kwargs):
        """
        Train the SAE probe.
        
        Args:
            X: Raw model activations, shape (N, seq, d_model)
            y: Labels, shape (N,)
            mask: Optional mask, shape (N, seq)
            batch_size: Override the default batch size for training (uses self.batch_size if None)
            **kwargs: Additional arguments passed to parent fit method
        """
        print(f"\n=== SAE PROBE TRAINING START ===")
        print(f"Model: {self.model_name}, Layer: {self.layer}")
        print(f"SAE ID: {self.sae_id}")
        print(f"Aggregation: {self.aggregation}")
        print(f"Top-k features: {self.top_k_features}")
        print(f"SAE encoding batch size: {self.encoding_batch_size}")
        print(f"Training batch size: {batch_size if batch_size is not None else self.training_batch_size}")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        
        # Step 1: Encode activations through SAE
        print("Encoding activations through SAE...")
        print(f"[DEBUG] Before encoding. Memory: {get_memory_usage()}")
        X_encoded = self._encode_activations(X)
        print(f"Encoded X shape: {X_encoded.shape}")
        print(f"[DEBUG] After encoding. Memory: {get_memory_usage()}")
        
        # Step 2: Select top features
        print("Selecting top features...")
        print(f"[DEBUG] Before feature selection. Memory: {get_memory_usage()}")
        self.feature_indices = self._select_top_features(X_encoded, y)
        X_selected = X_encoded[:, :, self.feature_indices]
        print(f"Selected X shape: {X_selected.shape}")
        print(f"[DEBUG] After feature selection. Memory: {get_memory_usage()}")
        
        # Step 3: Initialize model with correct feature dimension
        self.sae_feature_dim = self.top_k_features
        self._init_model()
        
        # Step 4: Call parent fit method with selected features
        # Use provided batch_size or fall back to self.training_batch_size
        training_batch_size = batch_size if batch_size is not None else self.training_batch_size
        print(f"Training batch size: {training_batch_size}")
        super().fit(X_selected, y, mask, epochs, lr, training_batch_size, weight_decay, 
                   verbose, early_stopping, patience, min_delta, 
                   use_weighted_loss, use_weighted_sampler, **kwargs)

    def predict(self, X: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = None) -> np.ndarray:
        """Predict using the SAE probe."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Encode and select features
        X_encoded = self._encode_activations(X)
        X_selected = X_encoded[:, :, self.feature_indices]
        
        # Use provided batch_size or fall back to self.training_batch_size
        predict_batch_size = batch_size if batch_size is not None else self.training_batch_size
        return super().predict(X_selected, mask, predict_batch_size)

    def predict_proba(self, X: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = None) -> np.ndarray:
        """Predict probabilities using the SAE probe."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Encode and select features
        X_encoded = self._encode_activations(X)
        X_selected = X_encoded[:, :, self.feature_indices]
        
        # Use provided batch_size or fall back to self.training_batch_size
        predict_batch_size = batch_size if batch_size is not None else self.training_batch_size
        return super().predict_proba(X_selected, mask, predict_batch_size)

    def predict_logits(self, X: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = None) -> np.ndarray:
        """Predict logits using the SAE probe."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Encode and select features
        X_encoded = self._encode_activations(X)
        X_selected = X_encoded[:, :, self.feature_indices]
        
        # Use provided batch_size or fall back to self.training_batch_size
        predict_batch_size = batch_size if batch_size is not None else self.training_batch_size
        return super().predict_logits(X_selected, mask, predict_batch_size)

    def save_state(self, path: Path):
        """Save the probe state including SAE info and feature indices."""
        state = {
            'model_state': self.model.state_dict() if self.model else None,
            'loss_history': self.loss_history,
            'model_name': self.model_name,
            'layer': self.layer,
            'sae_id': self.sae_id,
            'top_k_features': self.top_k_features,
            'aggregation': self.aggregation,
            'encoding_batch_size': self.encoding_batch_size,
            'training_batch_size': self.training_batch_size,
            'feature_indices': self.feature_indices,
            'sae_feature_dim': self.sae_feature_dim,
            'task_type': self.task_type,
        }
        # Use weights_only=False to ensure compatibility with numpy arrays and other objects
        torch.save(state, path, _use_new_zipfile_serialization=False)

    def load_state(self, path: Path):
        """Load the probe state."""
        try:
            # Try with weights_only=False for backward compatibility with saved states
            state = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            # If that fails, try with weights_only=True (new PyTorch 2.6+ default)
            print(f"Warning: Failed to load with weights_only=False, trying with weights_only=True: {e}")
            state = torch.load(path, map_location=self.device, weights_only=True)
        
        # Restore attributes
        self.model_name = state['model_name']
        self.layer = state['layer']
        self.sae_id = state['sae_id']
        self.top_k_features = state['top_k_features']
        self.aggregation = state['aggregation']
        # Handle backward compatibility for old batch_size parameter
        if 'batch_size' in state:
            self.encoding_batch_size = state['batch_size']
            self.training_batch_size = state['batch_size']
        else:
            self.encoding_batch_size = state.get('encoding_batch_size', 256)
            self.training_batch_size = state.get('training_batch_size', 64)
        self.feature_indices = state['feature_indices']
        self.sae_feature_dim = state['sae_feature_dim']
        self.task_type = state['task_type']
        
        # Initialize model and load state
        self._init_model()
        if self.model and state['model_state']:
            self.model.load_state_dict(state['model_state'])
        
        self.loss_history = state['loss_history']

    def find_best_fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                     mask_train: Optional[np.ndarray] = None, mask_val: Optional[np.ndarray] = None, 
                     n_trials: int = 10, direction: str = None, verbose: bool = True, 
                     metric: str = 'acc', **kwargs):
        """
        Hyperparameter tuning for the SAE probe.
        """
        def objective(trial):
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-2)
            epochs = 50
            
            # Re-init probe for each trial
            config_params = {k: v for k, v in self.__dict__.items() 
                           if k not in ['d_model', 'device', 'task_type', 'model', 'loss_history', 'sae']}
            probe = SAEProbe(self.d_model, device=self.device, **config_params)
            
            probe.fit(X_train, y_train, mask=mask_train, epochs=epochs, lr=lr, 
                     weight_decay=weight_decay, verbose=False)
            metrics = probe.score(X_val, y_val, mask=mask_val)
            
            if metric == 'acc':
                return -metrics.get('accuracy', 0.0)
            elif metric == 'auc':
                return -metrics.get('auc', 0.0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params