import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List
import json
import math
import optuna

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
        Find best hyperparameters by sweeping over weight decay and learning rate values.
        Chooses the model with the lowest averaged training loss over the last self.fit_patience epochs.
        """
        import optuna
        
        def objective(trial):
            # Define hyperparameter search space
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            
            # Create a copy of the current probe with trial hyperparameters
            config_params = {k: v for k, v in self.__dict__.items() 
                           if k not in ['d_model', 'device', 'task_type', 'aggregation', 'model', 'loss_history']}
            trial_probe = AttentionProbe(
                self.d_model, 
                device=self.device, 
                **config_params
            )
            
            # Train with trial hyperparameters
            try:
                trial_probe.fit(
                    X_train, y_train,
                    epochs=epochs,  # Use epochs parameter
                    lr=lr,
                    weight_decay=weight_decay,
                    batch_size=1024,  # Fixed batch size
                    verbose=False,  # Reduce output during hyperparameter search
                    use_weighted_sampler=True
                )
                
                # Get the last self.fit_patience epochs of training loss for stability assessment
                if len(trial_probe.loss_history) >= self.fit_patience:
                    last_losses = trial_probe.loss_history[-self.fit_patience:]
                    avg_loss = np.mean(last_losses)
                else:
                    # If not enough epochs, use all available losses
                    avg_loss = np.mean(trial_probe.loss_history) if trial_probe.loss_history else float('inf')
                
                return avg_loss
                
            except Exception as e:
                if verbose:
                    print(f"Trial failed with lr={lr}, weight_decay={weight_decay}: {e}")
                return float('inf')
        
        # Create study to minimize the average training loss
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
        
        # Get best parameters
        best_params = study.best_params
        
        if verbose:
            print(f"Best hyperparameters found:")
            print(f"  Learning rate: {best_params['lr']:.2e}")
            print(f"  Weight decay: {best_params['weight_decay']:.2e}")
            print(f"  Best average training loss: {study.best_value:.6f}")
        
        # Save best hyperparameters if directory provided
        if probe_save_dir is not None and probe_filename_base is not None:
            best_hparams_path = probe_save_dir / f"{probe_filename_base}_best_hparams.json"
            with open(best_hparams_path, 'w') as f:
                json.dump(best_params, f, indent=2)
        
        return best_params

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