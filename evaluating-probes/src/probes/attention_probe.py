import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List
import json
import math

from src.probes.base_probe import BaseProbe

class AttentionProbeNet(nn.Module):
    def __init__(self, d_model: int, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.scale = math.sqrt(d_model)
        # Create linear layers with bfloat16 dtype for mixed precision training
        self.context_query = nn.Linear(d_model, 1, dtype=torch.bfloat16)
        self.classifier = nn.Linear(d_model, 1, dtype=torch.bfloat16)
        # Move the model to the specified device
        self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model), mask: (batch, seq)
        attn_scores = self.context_query(x).squeeze(-1) / self.scale
        masked_attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(masked_attn_scores, dim=-1)
        token_logits = self.classifier(x).squeeze(-1)
        token_logits = token_logits.masked_fill(~mask, 0)

        # Compute weighted context
        context = torch.einsum("bs,bse->be", attn_weights, x)
        sequence_logits = self.classifier(context).squeeze(-1)
        
        return sequence_logits

class AttentionProbe(BaseProbe):
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", **kwargs):
        # Store any additional config parameters for this probe
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().__init__(d_model, device, task_type)

    def _init_model(self):
        self.model = AttentionProbeNet(self.d_model, device=self.device)

    def find_best_fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                     epochs: int = 100, n_trials: int = 20, direction: str = None, verbose: bool = True, metric: str = 'acc',
                     probe_save_dir: Path = None, probe_filename_base: str = None) -> dict:
        """
        Find best hyperparameters by training multiple probes simultaneously on each batch.
        This is more efficient than optuna as it avoids repeated GPU data transfers.
        Saves all probes to hyperparameter_sweep folder and selects best based on training loss.
        """
        # Define hyperparameter search space
        lr_values = np.logspace(-5, -2, 8)  # 8 learning rates from 1e-5 to 1e-2
        weight_decay_values = np.logspace(-6, -2, 5)  # 5 weight decay values from 1e-6 to 1e-2
        
        # Create all combinations
        hyperparam_combinations = []
        for lr in lr_values:
            for weight_decay in weight_decay_values:
                hyperparam_combinations.append((lr, weight_decay))
        
        if verbose:
            print(f"Training {len(hyperparam_combinations)} probes simultaneously...")
            print(f"Learning rates: {lr_values}")
            print(f"Weight decay values: {weight_decay_values}")
        
        # Convert data to torch tensors once
        X_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        mask = torch.ones(X_train.shape[:2], dtype=torch.bool, device=self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, mask)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # Initialize all probes
        probes = []
        optimizers = []
        loss_histories = []
        
        for lr, weight_decay in hyperparam_combinations:
            # Create probe with current hyperparameters
            config_params = {k: v for k, v in self.__dict__.items() 
                           if k not in ['d_model', 'device', 'task_type', 'aggregation', 'model', 'loss_history']}
            probe = AttentionProbe(
                self.d_model, 
                device=self.device, 
                **config_params
            )
            probe._init_model()  # Initialize the model
            
            # Setup optimizer
            optimizer = torch.optim.Adam(probe.model.parameters(), lr=lr, weight_decay=weight_decay)
            
            probes.append(probe)
            optimizers.append(optimizer)
            loss_histories.append([])
        
        # Training loop - all probes train simultaneously on each batch
        criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            # Set all models to training mode
            for probe in probes:
                probe.model.train()
            
            epoch_losses = [0.0] * len(probes)
            num_batches = 0
            
            for batch_X, batch_y, batch_mask in dataloader:
                num_batches += 1
                
                # Train all probes on this batch
                for i, (probe, optimizer) in enumerate(zip(probes, optimizers)):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = probe.model(batch_X, batch_mask)
                    loss = criterion(outputs.squeeze(), batch_y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses[i] += loss.item()
            
            # Record average losses for this epoch
            for i in range(len(probes)):
                avg_loss = epoch_losses[i] / num_batches
                loss_histories[i].append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                for i, (lr, weight_decay) in enumerate(hyperparam_combinations):
                    print(f"  lr={lr:.2e}, wd={weight_decay:.2e}: loss={epoch_losses[i]/num_batches:.4f}")
        
        # Calculate final average losses (using last fit_patience epochs for stability)
        final_losses = []
        for i, loss_history in enumerate(loss_histories):
            if len(loss_history) >= self.fit_patience:
                final_loss = np.mean(loss_history[-self.fit_patience:])
            else:
                final_loss = np.mean(loss_history) if loss_history else float('inf')
            final_losses.append(final_loss)
        
        # Find best probe
        best_idx = np.argmin(final_losses)
        best_lr, best_weight_decay = hyperparam_combinations[best_idx]
        best_loss = final_losses[best_idx]
        best_probe = probes[best_idx]
        
        if verbose:
            print(f"\nHyperparameter sweep results:")
            for i, (lr, weight_decay) in enumerate(hyperparam_combinations):
                print(f"  lr={lr:.2e}, wd={weight_decay:.2e}: final_loss={final_losses[i]:.6f}")
            print(f"\nBest hyperparameters:")
            print(f"  Learning rate: {best_lr:.2e}")
            print(f"  Weight decay: {best_weight_decay:.2e}")
            print(f"  Best final loss: {best_loss:.6f}")
        
        # Save all probes if directory provided
        if probe_save_dir is not None and probe_filename_base is not None:
            # Create hyperparameter_sweep subfolder
            sweep_dir = probe_save_dir / "hyperparameter_sweep"
            sweep_dir.mkdir(exist_ok=True)
            
            for i, (lr, weight_decay) in enumerate(hyperparam_combinations):
                # Create filename with hyperparameters
                lr_str = f"{lr:.2e}".replace("+", "").replace(".", "p")
                wd_str = f"{weight_decay:.2e}".replace("+", "").replace(".", "p")
                probe_filename = f"{probe_filename_base}_lr_{lr_str}_wd_{wd_str}_state.pt"
                probe_path = sweep_dir / probe_filename
                
                # Save probe state
                probes[i].save_state(probe_path)
        
        # Update current probe with best parameters
        self.model = best_probe.model
        self.loss_history = loss_histories[best_idx]
        
        # Save best hyperparameters if directory provided
        if probe_save_dir is not None and probe_filename_base is not None:
            best_hparams_path = probe_save_dir / f"{probe_filename_base}_best_hparams.json"
            best_params = {
                'lr': best_lr,
                'weight_decay': best_weight_decay,
                'best_final_loss': best_loss,
                'all_results': {f"lr_{lr:.2e}_wd_{wd:.2e}": loss 
                               for (lr, wd), loss in zip(hyperparam_combinations, final_losses)}
            }
            with open(best_hparams_path, 'w') as f:
                json.dump(best_params, f, indent=2)
        
        return {'lr': best_lr, 'weight_decay': best_weight_decay, 'final_loss': best_loss}

    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100, lr: float = 5e-4, batch_size: int = 1024, 
            weight_decay: float = 0.0, verbose: bool = True, early_stopping: bool = True,
            patience: int = 10, min_delta: float = 0.005, use_weighted_sampler: bool = True) -> None:
        """
        Fit the attention probe to the data.
        
        Args:
            X: Fixed-length activations, shape (N, seq_len, d_model)
            y: Target labels, shape (N,)
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size
            weight_decay: Weight decay
            verbose: Whether to print progress
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            use_weighted_sampler: Whether to use weighted sampling for class imbalance
        """
        # Convert to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # Create attention mask (all tokens are valid for fixed-length activations)
        mask = torch.ones(X.shape[:2], dtype=torch.bool, device=self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, mask)
        
        if use_weighted_sampler:
            # Calculate class weights for weighted sampling
            class_counts = np.bincount(y.astype(int))
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[y.astype(int)]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=sampler
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # Training loop
        self.loss_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y, batch_mask in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X, batch_mask)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Early stopping
            if early_stopping:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

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
        # Convert to torch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Create attention mask (all tokens are valid for fixed-length activations)
        mask = torch.ones(X.shape[:2], dtype=torch.bool, device=self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor, mask).cpu().numpy().squeeze()
        
        return logits

    def score(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return {"accuracy": accuracy} 