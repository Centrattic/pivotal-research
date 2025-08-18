import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List
import json
import math

from src.probes.base_probe import BaseProbe


class AttentionProbeNet(nn.Module):

    def __init__(
        self,
        d_model: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.scale = math.sqrt(d_model)
        # Create linear layers with bfloat16 dtype
        self.context_query = nn.Linear(d_model, 1, dtype=torch.bfloat16)
        self.classifier = nn.Linear(d_model, 1, dtype=torch.bfloat16)
        # Move the model to the specified device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model) - all positions are valid, let attention learn importance
        attn_scores = self.context_query(x).squeeze(-1) / self.scale
        # No masking - let attention learn which positions are important
        if x.device.type == 'cpu':
            attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(attn_scores.dtype)
        else:
            attn_weights = torch.softmax(attn_scores, dim=-1)
        token_logits = self.classifier(x).squeeze(-1)
        # No masking of token logits - let the model learn

        # Compute weighted context (promote to float32 on CPU for op support)
        if x.device.type == 'cpu':
            context = torch.einsum("bs,bse->be", attn_weights.float(), x.float()).to(x.dtype)
        else:
            context = torch.einsum("bs,bse->be", attn_weights, x)
        sequence_logits = self.classifier(context).squeeze(-1)

        return sequence_logits


class AttentionProbe(BaseProbe):

    def __init__(
        self,
        d_model: int,
        device: str = "cpu",
        task_type: str = "classification",
        **kwargs,
    ):
        # Store any additional config parameters for this probe
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().__init__(d_model, device, task_type)

    def _init_model(
        self,
    ):
        self.model = AttentionProbeNet(self.d_model, device=self.device)

    def load_state(
        self,
        path: Path,
    ):
        """Load probe state and convert back to bfloat16 for training."""
        super().load_state(path)
        # Convert model back to bfloat16 for training efficiency
        try:
            self.model = self.model.to(torch.bfloat16)
            print(f"Converted attention probe back to bfloat16 for training")
        except Exception as e:
            print(f"Warning: Could not convert to bfloat16, keeping as float32: {e}")
            # Keep as float32 if bfloat16 is not supported

    def find_best_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        masks_train: np.ndarray = None,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        masks_val: np.ndarray = None,
        epochs: int = 100,
        n_trials: int = 20,
        direction: str = None,
        verbose: bool = True,
        metric: str = 'acc',
        probe_save_dir: Path = None,
        probe_filename_base: str = None
    ) -> dict:
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

        # Convert data to torch tensors once - match model dtype (bfloat16)
        # First convert to float32, then to bfloat16 to handle any potential issues
        X_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device).to(torch.bfloat16)
        y_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)  # Keep labels as float32 for loss

        # Create dataset and dataloader (no masks needed for attention probe)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

        # Initialize all probes
        probes = []
        optimizers = []
        loss_histories = []

        for lr, weight_decay in hyperparam_combinations:
            # Create probe with current hyperparameters
            config_params = {
                k: v
                for k, v in self.__dict__.items()
                if k not in ['d_model', 'device', 'task_type', 'aggregation', 'model', 'loss_history']
            }
            probe = AttentionProbe(self.d_model, device=self.device, **config_params)
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

            for batch_X, batch_y in dataloader:
                num_batches += 1

                # Train all probes on this batch
                for i, (probe, optimizer) in enumerate(zip(probes, optimizers)):
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = probe.model(batch_X)
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
                # Group by weight decay for cleaner output
                wd_to_losses = {}
                for i, (lr, weight_decay) in enumerate(hyperparam_combinations):
                    if weight_decay not in wd_to_losses:
                        wd_to_losses[weight_decay] = []
                    wd_to_losses[weight_decay].append((lr, epoch_losses[i] / num_batches))

                for weight_decay in sorted(wd_to_losses.keys()):
                    losses = wd_to_losses[weight_decay]
                    best_lr, best_loss = min(losses, key=lambda x: x[1])
                    print(f"  wd={weight_decay:.2e}: best lr={best_lr:.2e} (loss={best_loss:.4f})")

        # Calculate final average losses (using last fit_patience epochs for stability)
        final_losses = []
        for i, loss_history in enumerate(loss_histories):
            if len(loss_history) >= self.fit_patience:
                final_loss = np.mean(loss_history[-self.fit_patience:])
            else:
                final_loss = np.mean(loss_history) if loss_history else float('inf')
            final_losses.append(final_loss)

        # Find best probe among those with weight_decay = 0.0
        zero_wd_indices = [i for i, (lr, wd) in enumerate(hyperparam_combinations) if wd == 0.0]

        if zero_wd_indices:
            # Select best among zero weight decay probes
            zero_wd_losses = [final_losses[i] for i in zero_wd_indices]
            best_zero_wd_idx = zero_wd_indices[np.argmin(zero_wd_losses)]
            best_idx = best_zero_wd_idx
            best_lr, best_weight_decay = hyperparam_combinations[best_idx]
            best_loss = final_losses[best_idx]
            best_probe = probes[best_idx]

            if verbose:
                print(f"\nSelected best probe among weight_decay=0.0 probes:")
                print(f"  Learning rate: {best_lr:.2e}")
                print(f"  Weight decay: {best_weight_decay:.2e}")
                print(f"  Best final loss: {best_loss:.6f}")
        else:
            # Fallback: no zero weight decay probes found, use overall best
            best_idx = np.argmin(final_losses)
            best_lr, best_weight_decay = hyperparam_combinations[best_idx]
            best_loss = final_losses[best_idx]
            best_probe = probes[best_idx]

            if verbose:
                print(f"\nNo weight_decay=0.0 probes found, using overall best:")
                print(f"  Learning rate: {best_lr:.2e}")
                print(f"  Weight decay: {best_weight_decay:.2e}")
                print(f"  Best final loss: {best_loss:.6f}")

        if verbose:
            print(f"\nHyperparameter sweep results:")
            for i, (lr, weight_decay) in enumerate(hyperparam_combinations):
                marker = " *" if weight_decay == 0.0 else ""
                print(f"  lr={lr:.2e}, wd={weight_decay:.2e}: final_loss={final_losses[i]:.6f}{marker}")
            print(f"  * = weight_decay=0.0 (preferred for selection)")

        # Save best probe per weight decay if directory provided
        if probe_save_dir is not None and probe_filename_base is not None:
            # Create hyperparameter_sweep subfolder
            sweep_dir = probe_save_dir / "hyperparameter_sweep"
            sweep_dir.mkdir(exist_ok=True)

            # Group by weight decay and find best for each
            wd_to_results = {}
            for i, (lr, weight_decay) in enumerate(hyperparam_combinations):
                if weight_decay not in wd_to_results:
                    wd_to_results[weight_decay] = []
                wd_to_results[weight_decay].append((i, lr, final_losses[i]))

            # Save best probe for each weight decay
            for weight_decay, results in wd_to_results.items():
                # Find best (lowest loss) for this weight decay
                best_idx, best_lr, best_loss = min(results, key=lambda x: x[2])

                # Create filename with best learning rate for this weight decay
                lr_str = f"{best_lr:.2e}".replace("+", "").replace(".", "p")
                wd_str = f"{weight_decay:.2e}".replace("+", "").replace(".", "p")
                probe_filename = f"{probe_filename_base}_best_lr_{lr_str}_wd_{wd_str}_state.pt"
                probe_path = sweep_dir / probe_filename

                # Save the best probe for this weight decay
                # Convert to float32 for saving to avoid bfloat16 compatibility issues
                # original_dtype = next(probes[best_idx].model.parameters()).dtype
                # if original_dtype != torch.float32:
                #     probes[best_idx].model = probes[best_idx].model.to(torch.float32)
                #     probes[best_idx].save_state(probe_path)
                #     # Convert back to original dtype
                #     probes[best_idx].model = probes[best_idx].model.to(original_dtype)
                # else:
                #     # Already float32, save directly
                probes[best_idx].save_state(probe_path)

                if verbose:
                    print(f"  Saved best probe for wd={weight_decay:.2e}: lr={best_lr:.2e}, loss={best_loss:.6f}")

        # Update current probe with best parameters
        self.model = best_probe.model
        self.loss_history = loss_histories[best_idx]

        # Save best hyperparameters if directory provided
        if probe_save_dir is not None and probe_filename_base is not None:
            best_hparams_path = probe_save_dir / f"{probe_filename_base}_best_hparams.json"

            # Create summary of best learning rate for each weight decay
            best_per_wd = {}
            for weight_decay, results in wd_to_results.items():
                best_idx, best_lr, best_loss = min(results, key=lambda x: x[2])
                best_per_wd[f"wd_{weight_decay:.2e}"] = {
                    'best_lr': best_lr, 'best_loss': best_loss, 'all_lrs':
                    {f"lr_{lr:.2e}": loss
                     for _, lr, loss in results}
                }

            best_params = {
                'selected_lr': best_lr, 'selected_weight_decay': best_weight_decay, 'selected_final_loss': best_loss,
                'selection_criteria':
                'best_loss_among_weight_decay_0.0' if best_weight_decay == 0.0 else 'overall_best',
                'best_per_weight_decay': best_per_wd, 'all_results':
                {f"lr_{lr:.2e}_wd_{wd:.2e}": loss
                 for (lr, wd), loss in zip(hyperparam_combinations, final_losses)}
            }
            with open(best_hparams_path, 'w') as f:
                json.dump(best_params, f, indent=2)

        return {'lr': best_lr, 'weight_decay': best_weight_decay, 'final_loss': best_loss}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        masks: np.ndarray = None,
        epochs: int = 100,
        lr: float = 5e-4,
        batch_size: int = 1024,
        weight_decay: float = 0.0,
        verbose: bool = True,
        early_stopping: bool = True,
        patience: int = 10,
        min_delta: float = 0.005,
        use_weighted_sampler: bool = True
    ) -> None:
        """
        Fit the attention probe to the data.
        
        Args:
            X: Fixed-length activations, shape (N, seq_len, d_model)
            y: Target labels, shape (N,)
            masks: Attention masks (ignored - attention probe learns to attend to important positions)
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
        # Convert to torch tensors - match model dtype (bfloat16)
        X_tensor = torch.tensor(X, dtype=torch.bfloat16, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.bfloat16, device=self.device)  # Keep labels as float32 for loss

        # Create dataset and dataloader (no masks needed for attention probe)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

        if use_weighted_sampler:
            # Calculate class weights for weighted sampling
            class_counts = np.bincount(y.astype(int))
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[y.astype(int)]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)
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
        # Ensure logits are float32 for stable numpy ops
        logits = self.predict_logits(X).astype(np.float32, copy=False)
        probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Predict logits. Cast to float32 on CPU for eval to avoid bfloat16 issues."""
        is_cpu = (self.device == 'cpu') or (isinstance(self.device, torch.device) and self.device.type == 'cpu')
        if is_cpu:
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            # Temporarily cast model to float32 for CPU inference
            original_dtype = next(self.model.parameters()).dtype
            self.model = self.model.to(torch.float32)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(X_tensor).cpu().numpy().squeeze()
            # Restore original dtype
            self.model = self.model.to(original_dtype)
            return logits
        else:
            # GPU path keeps bfloat16 for speed
            X_tensor = torch.tensor(X, dtype=torch.bfloat16, device=self.device)
            self.model.eval()
            with torch.no_grad():
                # Cast to float32 before moving to CPU/numpy to avoid bf16 numpy ops later
                logits = self.model(X_tensor).to(torch.float32).cpu().numpy().squeeze()
            return logits

    def score(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Calculate accuracy score. Ensure eval in float32 on CPU."""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return {"accuracy": accuracy}
