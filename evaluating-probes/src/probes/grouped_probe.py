import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from tqdm import tqdm
import gc

from src.probes.base_probe import BaseProbe
from src.utils import get_probe_architecture
from configs.probes import PROBE_CONFIGS
from dataclasses import asdict


class GroupedProbe:
    """
    A probe that manages multiple probes and trains them simultaneously on the same batches
    to reduce GPU memory transfers. This is especially useful when training many probes
    on the same dataset.
    """
    
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification"):
        self.d_model = d_model
        self.device = device
        self.task_type = task_type
        self.probes: Dict[str, BaseProbe] = {}
        self.probe_configs: Dict[str, Dict] = {}
        self.probe_optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.probe_criterions: Dict[str, nn.Module] = {}
        self.probe_class_weights: Dict[str, Optional[torch.Tensor]] = {}
        self.probe_loss_histories: Dict[str, List[float]] = {}

        
    def add_probe(self, probe_name: str, architecture_name: str, config_name: str):
        """
        Add a probe to the group.
        
        Args:
            probe_name: Unique name for this probe
            architecture_name: Name of the probe architecture (e.g., 'linear', 'attention')
            config_name: Name of the probe configuration from PROBE_CONFIGS
        """
        if probe_name in self.probes:
            raise ValueError(f"Probe '{probe_name}' already exists in the group")
            
        # Get probe configuration
        if config_name not in PROBE_CONFIGS:
            raise ValueError(f"Unknown config_name: {config_name}")
            
        config = asdict(PROBE_CONFIGS[config_name])
        self.probe_configs[probe_name] = config
        
        # Create the probe
        probe = get_probe_architecture(architecture_name, d_model=self.d_model, device=self.device, config=config)
        self.probes[probe_name] = probe
        
        # Debug: Check if probe.model exists
        if probe.model is None:
            print(f"Warning: probe.model is None for {probe_name}. Probe type: {type(probe)}")
            # For probes like SAE that need to defer model initialization, we'll create optimizer later
            self.probe_optimizers[probe_name] = None
        else:
            # Initialize optimizer for probes with available models
            optimizer = torch.optim.Adam(probe.model.parameters(), 
                                       lr=config.get('lr', 1e-3), 
                                       weight_decay=config.get('weight_decay', 0.0))
            self.probe_optimizers[probe_name] = optimizer
        
        # Initialize loss function
        self.probe_criterions[probe_name] = nn.BCEWithLogitsLoss()
        
        # Initialize loss history
        self.probe_loss_histories[probe_name] = []
        
        print(f"Added probe '{probe_name}' ({architecture_name}) with config '{config_name}'")
        
    def fit(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, 
            epochs: int = 20, lr: float = 1e-3, batch_size: int = 1, weight_decay: float = 0.0, 
            verbose: bool = True, early_stopping: bool = True, patience: int = 10, min_delta: float = 0.005,
            use_weighted_loss: bool = True, use_weighted_sampler: bool = False):
        """
        Train all probes in the group simultaneously on the same batches.
        """
        if not self.probes:
            raise ValueError("No probes added to the group. Call add_probe() first.")
            
        print(f"\n=== GROUPED TRAINING START ===")
        print(f"Training {len(self.probes)} probes simultaneously")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Mask shape: {mask.shape if mask is not None else 'None'}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Create tensors efficiently
        X = torch.tensor(X, dtype=torch.bfloat16, device="cpu")
        y = torch.tensor(y, dtype=torch.long, device="cpu")
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool, device="cpu")
        else:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device="cpu")
            
        # Compute class weights for each probe if needed
        if use_weighted_loss:
            try:
                from sklearn.utils.class_weight import compute_class_weight
                y_unique, y_counts = torch.unique(y, return_counts=True)
                classes = y_unique.cpu().numpy()
                y_counts_np = y_counts.cpu().numpy()
                
                if len(classes) == 2:
                    class_weights_np = compute_class_weight('balanced', classes=classes, y=y.cpu().numpy())
                    class_weights = torch.tensor(class_weights_np, dtype=torch.bfloat16, device=self.device)
                    
                    for probe_name in self.probes:
                        if use_weighted_loss:
                            pos_weight = class_weights[1] / class_weights[0]
                            self.probe_criterions[probe_name] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                            self.probe_class_weights[probe_name] = class_weights
                        else:
                            self.probe_criterions[probe_name] = nn.BCEWithLogitsLoss()
                            self.probe_class_weights[probe_name] = None
                            
                    print(f"Class weights computed: {class_weights}")
                else:
                    print(f"Warning: Expected 2 classes for binary classification, got {len(classes)}")
                    for probe_name in self.probes:
                        self.probe_criterions[probe_name] = nn.BCEWithLogitsLoss()
                        self.probe_class_weights[probe_name] = None
                        
            except Exception as e:
                print(f"Could not compute class weights: {e}")
                for probe_name in self.probes:
                    self.probe_criterions[probe_name] = nn.BCEWithLogitsLoss()
                    self.probe_class_weights[probe_name] = None
        else:
            for probe_name in self.probes:
                self.probe_criterions[probe_name] = nn.BCEWithLogitsLoss()
                self.probe_class_weights[probe_name] = None
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X, y, mask)
        
        # Handle weighted sampling if requested
        sampler = None
        if use_weighted_sampler:
            y_unique, y_counts = torch.unique(y, return_counts=True)
            classes = y_unique.cpu().numpy()
            y_counts_np = y_counts.cpu().numpy()
            
            if len(classes) == 2:
                weight = 1. / y_counts_np
                samples_weight = torch.zeros_like(y, dtype=torch.bfloat16)
                for i, cls in enumerate(classes):
                    samples_weight[y == cls] = weight[i]
                
                from torch.utils.data import WeightedRandomSampler
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
                print(f"Using WeightedRandomSampler for class balancing.")
        
        # Create dataloader
        num_workers = min(16, batch_size)
        
        # SAE probes now store their own data, so we can always shuffle
        if sampler is not None:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                sampler=sampler, pin_memory=True, num_workers=num_workers,
                                                persistent_workers=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                pin_memory=True, num_workers=num_workers,
                                                persistent_workers=True)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of batches: {len(loader)}")
        print(f"Using {num_workers} workers for data loading")
        
        # No SAE preprocessing needed - SAE probes are trained individually
        
        # Ensure all models are on the correct device and in training mode
        # Also create/recreate optimizers for all probes (some may have been reinitialized)
        for probe_name, probe in self.probes.items():
            if probe.model is not None:
                # Always recreate optimizer in case model was reinitialized (like SAE probes)
                config = self.probe_configs[probe_name]
                optimizer = torch.optim.Adam(probe.model.parameters(), 
                                           lr=config.get('lr', 1e-3), 
                                           weight_decay=config.get('weight_decay', 0.0))
                self.probe_optimizers[probe_name] = optimizer
                print(f"Created/recreated optimizer for {probe_name}")
                
                probe.model = probe.model.to(self.device)
                probe.model.train()
            else:
                print(f"Warning: probe.model is still None for {probe_name}")
        
        # Check if any probes still don't have optimizers
        probes_without_optimizers = [name for name, opt in self.probe_optimizers.items() if opt is None]
        if probes_without_optimizers:
            raise ValueError(f"Probes without optimizers: {probes_without_optimizers}. Their models may not be initialized properly.")
        
        # Training loop
        best_losses = {probe_name: float('inf') for probe_name in self.probes}
        best_model_states = {probe_name: None for probe_name in self.probes}
        epochs_no_improve = {probe_name: 0 for probe_name in self.probes}
        stop_epoch = None
        
        # Track recent losses for early stopping
        recent_losses = {probe_name: [] for probe_name in self.probes}
        recent_model_states = {probe_name: [] for probe_name in self.probes}
        window_size = 20
        
        for epoch in tqdm(range(epochs), desc="Training probes"):
            epoch_losses = {probe_name: 0.0 for probe_name in self.probes}
            batch_count = 0
            
            # Print memory usage every 5 epochs
            if epoch % 5 == 0 and verbose:
                if torch.cuda.is_available() and "cuda" in self.device:
                    print(f"Epoch {epoch}: GPU memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
            
            for batch_idx, (xb, yb, mb) in enumerate(loader):
                # Move batch to device once for all probes
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                mb = mb.to(self.device, non_blocking=True)
                yb = yb.bfloat16()
                
                # Train each probe on this batch
                for probe_name, probe in self.probes.items():
                    optimizer = self.probe_optimizers[probe_name]
                    criterion = self.probe_criterions[probe_name]
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Forward pass
                    logits = probe.model(xb, mb)
                    
                    # Compute loss
                    loss = criterion(logits, yb)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Accumulate loss
                    epoch_losses[probe_name] += loss.detach().item() * xb.size(0)
                
                batch_count += 1
            
            # Compute average losses and update histories
            for probe_name in self.probes:
                avg_loss = epoch_losses[probe_name] / len(dataset)
                self.probe_loss_histories[probe_name].append(avg_loss)
                recent_losses[probe_name].append(avg_loss)
                recent_model_states[probe_name].append({key: value.clone() for key, value in self.probes[probe_name].model.state_dict().items()})
                
                # Keep only the last window_size epochs
                if len(recent_losses[probe_name]) > window_size:
                    recent_losses[probe_name].pop(0)
                    recent_model_states[probe_name].pop(0)
            
            if verbose:
                loss_str = ", ".join([f"{name}: {loss:.4f}" for name, loss in epoch_losses.items()])
                print(f"Epoch {epoch+1}/{epochs} - Losses: {loss_str}")
            
            # Early stopping logic for each probe
            if early_stopping:
                all_stopped = True
                for probe_name in self.probes:
                    if len(recent_losses[probe_name]) >= window_size:
                        # Find best loss in the recent window
                        min_loss_idx = np.argmin(recent_losses[probe_name])
                        current_best_loss = recent_losses[probe_name][min_loss_idx]
                        
                        if current_best_loss < best_losses[probe_name] - min_delta:
                            improvement = best_losses[probe_name] - current_best_loss
                            best_losses[probe_name] = current_best_loss
                            best_model_states[probe_name] = recent_model_states[probe_name][min_loss_idx]
                            epochs_no_improve[probe_name] = 0
                            if verbose:
                                print(f"  {probe_name}: New best loss: {best_losses[probe_name]:.4f} (improved by {improvement:.4f})")
                        else:
                            epochs_no_improve[probe_name] += 1
                            if verbose:
                                print(f"  {probe_name}: No improvement for {epochs_no_improve[probe_name]}/{patience} epochs")
                        
                        if epochs_no_improve[probe_name] < patience:
                            all_stopped = False
                    else:
                        all_stopped = False
                
                if all_stopped:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    # Restore best model states
                    for probe_name in self.probes:
                        if best_model_states[probe_name] is not None:
                            self.probes[probe_name].model.load_state_dict(best_model_states[probe_name])
                            print(f"Restored {probe_name} to best state (loss: {best_losses[probe_name]:.4f})")
                    stop_epoch = epoch + 1
                    break
        
        print(f"=== GROUPED TRAINING COMPLETE ===")
        if stop_epoch is not None:
            print(f"Training stopped early at epoch {stop_epoch}.")
        else:
            print(f"Training completed for all {epochs} epochs.")
            
        # Copy loss histories to individual probes for compatibility
        for probe_name, probe in self.probes.items():
            probe.loss_history = self.probe_loss_histories[probe_name].copy()
    
    def get_probe(self, probe_name: str) -> BaseProbe:
        """Get a specific probe from the group."""
        if probe_name not in self.probes:
            raise ValueError(f"Probe '{probe_name}' not found in the group")
        return self.probes[probe_name]
    
    def get_all_probes(self) -> Dict[str, BaseProbe]:
        """Get all probes in the group."""
        return self.probes.copy()
    
    def save_state(self, path: Path, architecture_configs: list = None, train_dataset_name: str = None, 
                   layer: int = None, component: str = None, rebuild_config: dict = None, 
                   contrast_fn: Any = None):
        """
        Save the state of all probes in the group.
        If architecture_configs is provided, saves each probe with its proper individual filename.
        Otherwise, saves with generic names.
        """
        if architecture_configs is not None:
            # Save each probe with its proper individual filename
            from src.utils import get_probe_filename_prefix, rebuild_suffix
            
            for i, arch_config in enumerate(architecture_configs):
                architecture_name = arch_config['name']
                config_name = arch_config.get('config_name')
                
                # Create individual probe filename (same as train_probe would use)
                individual_probe_filename_base = get_probe_filename_prefix(train_dataset_name, architecture_name, layer, component, config_name, contrast_fn)
                
                if rebuild_config is not None:
                    suffix = rebuild_suffix(rebuild_config)
                    individual_probe_state_path = path.parent / f"{individual_probe_filename_base}_{suffix}_state.npz"
                else:
                    individual_probe_state_path = path.parent / f"{individual_probe_filename_base}_state.npz"
                
                # Get the corresponding probe from the group
                probe_name = f"{architecture_name}_{config_name}_{i}"
                if probe_name in self.probes:
                    individual_probe = self.probes[probe_name]
                    individual_probe.save_state(individual_probe_state_path)
        else:
            # Original behavior - save with generic names
            state = {
                'probe_states': {},
                'probe_configs': self.probe_configs,
                'probe_loss_histories': self.probe_loss_histories,
                'd_model': self.d_model,
                'device': self.device,
                'task_type': self.task_type
            }
            
            # Save each probe's state
            for probe_name, probe in self.probes.items():
                probe_state_path = path.parent / f"{probe_name}_state.npz"
                probe.save_state(probe_state_path)
                state['probe_states'][probe_name] = str(probe_state_path)
            
            # Save the group state
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
    
    def load_state(self, path: Path):
        """Load the state of all probes in the group."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.probe_configs = state['probe_configs']
        self.probe_loss_histories = state['probe_loss_histories']
        
        # Load each probe's state
        for probe_name, probe_state_path in state['probe_states'].items():
            if probe_name in self.probes:
                probe_state_path = Path(probe_state_path)
                self.probes[probe_name].load_state(probe_state_path)
    
 