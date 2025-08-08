import os
# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional
import json
import math
import gc
from tqdm import tqdm
import optuna

from src.logger import Logger

class BaseProbe:
    """
    General wrapper for PyTorch-based probes. Handles training, evaluation, and saving/loading.
    """
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification"):
        self.d_model = d_model
        self.device = device
        self.task_type = task_type  # Keep for future extensibility
        self.model: Optional[nn.Module] = None
        self.loss_history = []
        self._init_model()

    def _init_model(self):
        raise NotImplementedError("Subclasses must implement _init_model to set self.model")

    def fit(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, 
            epochs: int = 20, lr: float = 1e-3, batch_size: int = 1, weight_decay: float = 0.0, 
            verbose: bool = True, early_stopping: bool = True, patience: int = 10, min_delta: float = 0.005,
            use_weighted_loss: bool = True, use_weighted_sampler: bool = False):
        """
        Train the probe model.
        Args:
            X: Input features, shape (N, seq, d_model)
            y: Labels, shape (N,) or (N, num_classes)
            mask: Optional mask, shape (N, seq)
            epochs: Number of epochs
            lr: Learning rate
            batch_size: Batch size
            weight_decay: Weight decay
            verbose: Print progress
            early_stopping: Use early stopping
            patience: Early stopping patience
            min_delta: Early stopping min delta
            use_weighted_loss: If True, use class-weighted loss for classification
            use_weighted_sampler: If True, use WeightedRandomSampler for class balancing
        """
        print(f"\n=== TRAINING START ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Mask shape: {mask.shape if mask is not None else 'None'}")
        print(f"Task type: {self.task_type}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}, LR: {lr}, Batch size: {batch_size}, Weight decay: {weight_decay}")
        print(f"Weighted loss: {use_weighted_loss}, Weighted sampler: {use_weighted_sampler}")
        
        # Create tensors efficiently with minimal memory overhead
        print("Creating tensors efficiently...")
        
        # Keep data on CPU and move batches to GPU as needed to avoid GPU memory overflow
        X = torch.tensor(X, dtype=torch.bfloat16, device="cpu")
        y = torch.tensor(y, dtype=torch.long, device="cpu")
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool, device="cpu")
        else:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device="cpu")
        
        print(f"Tensor X shape: {X.shape}")
        print(f"Tensor y shape: {y.shape}")
        print(f"Tensor mask shape: {mask.shape}")

        # Compute class weights for weighted loss (binary classification only)
        class_weights = None
        if use_weighted_loss:
            try:
                from sklearn.utils.class_weight import compute_class_weight
                # Compute class weights directly on GPU to avoid CPU transfer
                y_unique, y_counts = torch.unique(y, return_counts=True)
                classes = y_unique.cpu().numpy()
                y_counts_np = y_counts.cpu().numpy()
                
                if len(classes) != 2:
                    print(f"Warning: Expected 2 classes for binary classification, got {len(classes)}")
                
                # Compute class weights using sklearn
                class_weights_np = compute_class_weight('balanced', classes=classes, y=y.cpu().numpy())
                class_weights = torch.tensor(class_weights_np, dtype=torch.bfloat16, device=self.device)
                
                # Print class sample counts and weights for debugging
                class_sample_counts = {int(cls): int(count) for cls, count in zip(classes, y_counts_np)}
                print(f"Class sample counts: {class_sample_counts}")
                print(f"Class weights: {class_weights}")
            except Exception as e:
                print(f"Could not compute class weights: {e}")
                class_weights = None

        # Create dataset with memory-efficient tensors
        dataset = torch.utils.data.TensorDataset(X, y, mask)
        
        # Print memory usage after dataset creation
        import psutil
        process = psutil.Process()
        print(f"Memory after dataset creation: {process.memory_info().rss / 1024**3:.2f} GB")

        # WeightedRandomSampler for class balancing (binary classification only)
        sampler = None
        if use_weighted_sampler:
            # Compute weights directly on GPU to avoid CPU transfer
            y_unique, y_counts = torch.unique(y, return_counts=True)
            classes = y_unique.cpu().numpy()
            y_counts_np = y_counts.cpu().numpy()
            
            if len(classes) != 2:
                print(f"Warning: Expected 2 classes for binary classification, got {len(classes)}")
            
            # Compute sample weights
            weight = 1. / y_counts_np
            samples_weight = torch.zeros_like(y, dtype=torch.bfloat16)
            for i, cls in enumerate(classes):
                samples_weight[y == cls] = weight[i]
            
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
            print(f"Using WeightedRandomSampler for class balancing.")

        # Use DataLoader with proper CPU-GPU batch transfers
        # This keeps data on CPU and moves batches to GPU as needed
        # Use multiple workers for efficient data loading
        num_workers = min(16, batch_size)  # Use up to 4 workers, but not more than batch_size
        
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
        print(f"Batch size: {batch_size}")
 
        # Print memory usage info
        if torch.cuda.is_available() and "cuda" in self.device:
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved(self.device) / 1024**3:.2f} GB")

        # Ensure model is on the correct device
        self.model = self.model.to(self.device)
        self.model.train()
        
        # Use mixed precision optimizer for better performance
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Binary classification only
        if use_weighted_loss and class_weights is not None:
            # For BCEWithLogitsLoss, use pos_weight for positive class
            # pos_weight should be a single value: weight for positive class / weight for negative class
            if len(class_weights) == 2:
                pos_weight = class_weights[1] / class_weights[0]
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight}")
            else:
                criterion = nn.BCEWithLogitsLoss()
                print(f"Using BCEWithLogitsLoss (no pos_weight, class_weights length != 2)")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print(f"Using BCEWithLogitsLoss (no class weights)")
        
        best_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        stop_epoch = None
        # Track losses from last 20 epochs for best loss calculation
        recent_losses = []
        recent_model_states = []
        window_size = 20

        # Remove GradScaler and autocast for bfloat16

        for epoch in tqdm(range(epochs)):
            epoch_loss = 0.0
            batch_count = 0
            
            # Clear cache periodically to prevent memory buildup
            # if epoch % 5 == 0:  # More frequent cache clearing
            #     if torch.cuda.is_available() and "cuda" in self.device:
            #         torch.cuda.empty_cache()
            #     gc.collect()  # Force CPU garbage collection
                
            # Print memory usage every 5 epochs
            if epoch % 5 == 0:
                if torch.cuda.is_available() and "cuda" in self.device:
                    print(f"Epoch {epoch}: GPU memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
                    # Try to get GPU utilization (if available)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        print(f"Epoch {epoch}: GPU utilization: {util.gpu}%")
                    except:
                        print(f"Epoch {epoch}: GPU utilization: Unable to measure")
                # Print CPU memory usage
                import psutil
                process = psutil.Process()
                print(f"Epoch {epoch}: CPU memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
            for xb, yb, mb in loader:
                # Move batch to device with non_blocking for efficiency
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                mb = mb.to(self.device, non_blocking=True)
                yb = yb.bfloat16()  # Keep as bfloat16 for mixed precision training
                # continue

                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                logits = self.model(xb, mb)
                
                if batch_count == 0 and epoch == 0:
                    print(f"First batch - xb shape: {xb.shape}, yb shape: {yb.shape}, mb shape: {mb.shape}")
                    print(f"First batch - logits shape: {logits.shape}")
                
                # Binary classification only
                loss = criterion(logits, yb)
                
                if batch_count == 0 and epoch == 0:
                    print(f"First batch - loss: {loss.item():.4f}")
                
                loss.backward()
                optimizer.step() 

                # Use detach() to prevent memory leaks - keep on GPU to avoid CPU transfer
                epoch_loss += loss.detach().item() * xb.size(0)
                batch_count += 1
                
                # Don't delete variables immediately - let GPU handle cleanup
                # This reduces CPU-GPU synchronization overhead

            avg_loss = epoch_loss / len(dataset)
            self.loss_history.append(avg_loss)
            
            # Track recent losses and model states
            recent_losses.append(avg_loss)
            recent_model_states.append({key: value.clone() for key, value in self.model.state_dict().items()})
            
            # Keep only the last window_size epochs
            if len(recent_losses) > window_size:
                recent_losses.pop(0)
                recent_model_states.pop(0)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            # Early stopping logic - only consider recent epochs for best loss
            if early_stopping and len(recent_losses) >= window_size:
                # Find best loss in the recent window
                min_loss_idx = np.argmin(recent_losses)
                current_best_loss = recent_losses[min_loss_idx]
                
                if current_best_loss < best_loss - min_delta:
                    improvement = best_loss - current_best_loss
                    best_loss = current_best_loss
                    # Save the model state from the best epoch in recent window
                    best_model_state = recent_model_states[min_loss_idx]
                    epochs_no_improve = 0
                    if verbose:
                        print(f"  New best loss from recent {window_size} epochs: {best_loss:.4f} (improved by {improvement:.4f})")
                else:
                    epochs_no_improve += 1
                    if verbose:
                        improvement_needed = best_loss - current_best_loss
                        print(f"  No improvement: current best in window={current_best_loss:.4f}, overall best={best_loss:.4f}, need >{min_delta:.4f}, got {improvement_needed:.4f}, patience={epochs_no_improve}/{patience}")
                
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best loss from recent {window_size} epochs: {best_loss:.4f}")
                    # Restore the best model state
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                        print(f"Restored model to best state (loss: {best_loss:.4f})")
                    stop_epoch = epoch + 1
                    break
            elif early_stopping:
                # Not enough epochs yet, just track without early stopping
                if verbose:
                    print(f"  Building up recent loss window ({len(recent_losses)}/{window_size} epochs)")
        if stop_epoch is not None:
            print(f"Training stopped early at epoch {stop_epoch}.")
        else:
            # If training completed without early stopping, restore best model if we have one
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                print(f"Training completed. Restored model to best state from recent {window_size} epochs (loss: {best_loss:.4f})")
        print(f"=== TRAINING COMPLETE ===\n")
        return self
    
    def find_best_fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, mask_train: Optional[np.ndarray] = None, 
                    mask_val: Optional[np.ndarray] = None, n_trials: int = 10, direction: str = None, verbose: bool = True):
        raise NotImplementedError("Subclasses must implement find_best_fit to be used in runner.py")

    def predict(self, X: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 1) -> np.ndarray:
        print(f"\n=== PREDICTION START ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input mask shape: {mask.shape if mask is not None else 'None'}")
        print(f"Batch size: {batch_size}")
        
        self.model.eval()
        
        # Convert to tensors on CPU and transfer batches to GPU as needed
        # Use bfloat16 for GPU operations, convert to float32 only for numpy
        X_tensor = torch.tensor(X, dtype=torch.bfloat16, device="cpu")
        if mask is not None:
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device="cpu")
        else:
            mask_tensor = torch.ones(X.shape[:2], dtype=torch.bool, device="cpu")
        
        all_preds = []
        num_batches = (len(X) + batch_size - 1) // batch_size
        print(f"Processing {num_batches} batches")
        
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i+batch_size].to(self.device, non_blocking=True)
            batch_mask = mask_tensor[i:i+batch_size].to(self.device, non_blocking=True)
            
            if i == 0:
                print(f"First batch - batch_X shape: {batch_X.shape}, batch_mask shape: {batch_mask.shape}")
            
            with torch.no_grad():
                logits = self.model(batch_X, batch_mask)
                
                if i == 0:
                    print(f"First batch - logits shape: {logits.shape}")
                
                # Binary classification only
                probs = torch.sigmoid(logits)
                # Convert bfloat16 to float32 before numpy conversion to avoid BFloat16 unsupported error
                preds = (probs > 0.5).long().cpu().numpy()
                
                if i == 0:
                    print(f"First batch - preds shape: {preds.shape}")
                
                all_preds.append(preds)
        
        result = np.concatenate(all_preds, axis=0)
        print(f"Final predictions shape: {result.shape}")
        print(f"=== PREDICTION COMPLETE ===\n")
        return result

    def predict_proba(self, X: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 1) -> np.ndarray:
        print(f"\n=== PROBABILITY PREDICTION START ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input mask shape: {mask.shape if mask is not None else 'None'}")
        print(f"Batch size: {batch_size}")
        
        self.model.eval()
        
        # Convert to tensors on CPU and transfer batches to GPU as needed
        # Use bfloat16 for GPU operations, convert to float32 only for numpy
        X_tensor = torch.tensor(X, dtype=torch.bfloat16, device="cpu")
        if mask is not None:
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device="cpu")
        else:
            mask_tensor = torch.ones(X.shape[:2], dtype=torch.bool, device="cpu")
        
        all_probs = []
        num_batches = (len(X) + batch_size - 1) // batch_size
        print(f"Processing {num_batches} batches")
        
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i+batch_size].to(self.device, non_blocking=True)
            batch_mask = mask_tensor[i:i+batch_size].to(self.device, non_blocking=True)
            
            if i == 0:
                print(f"First batch - batch_X shape: {batch_X.shape}, batch_mask shape: {batch_mask.shape}")
            
            with torch.no_grad():
                logits = self.model(batch_X, batch_mask)
                
                if i == 0:
                    print(f"First batch - logits shape: {logits.shape}")
                
                # Binary classification only
                # Convert bfloat16 to float32 before numpy conversion to avoid BFloat16 unsupported error
                probs = torch.sigmoid(logits).float().cpu().numpy()
                
                if i == 0:
                    print(f"First batch - probs shape: {probs.shape}")
                
                all_probs.append(probs)
        
        result = np.concatenate(all_probs, axis=0)
        print(f"Final probabilities shape: {result.shape}")
        print(f"=== PROBABILITY PREDICTION COMPLETE ===\n")
        return result

    def predict_logits(self, X: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 1) -> np.ndarray:
        """
        Returns the raw logits (pre-sigmoid/softmax) for the input X.
        """
        self.model.eval()
        
        # Convert to tensors on CPU and transfer batches to GPU as needed
        # Convert activations to bfloat16 to match model parameters
        X_tensor = torch.tensor(X, dtype=torch.bfloat16, device="cpu")
        if mask is not None:
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device="cpu")
        else:
            mask_tensor = torch.ones(X.shape[:2], dtype=torch.bool, device="cpu")
        
        all_logits = []
        num_batches = (len(X) + batch_size - 1) // batch_size
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i+batch_size].to(self.device, non_blocking=True)
            batch_mask = mask_tensor[i:i+batch_size].to(self.device, non_blocking=True)
            with torch.no_grad():
                logits = self.model(batch_X, batch_mask)
                if logits.ndim == 1:
                    logits = logits[:, None]  # shape (batch, 1)
                # Convert bfloat16 to float32 before numpy conversion to avoid BFloat16 unsupported error
                all_logits.append(logits.float().cpu().numpy())
        result = np.concatenate(all_logits, axis=0)
        return result

    def score(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 200) -> dict[str, float]:
        print(f"\n=== SCORING START ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Input mask shape: {mask.shape if mask is not None else 'None'}")
        print(f"Task type: {self.task_type}")
        
        from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error, precision_score, recall_score, confusion_matrix
        preds = self.predict(X, mask, batch_size=batch_size)
        print(f"Predictions shape: {preds.shape}")
        print(f"Predictions unique values: {np.unique(preds)}")
        
        # Binary classification only
        y_true = y
        y_prob = self.predict_proba(X, mask, batch_size=batch_size)
        print(f"Probabilities shape: {y_prob.shape}")
        print(f"True labels unique values: {np.unique(y_true)}")
        
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        acc = accuracy_score(y_true, preds)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        print(f"Binary classification metrics:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  FPR: {fpr:.4f}")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        result = {"acc": float(acc), "auc": float(auc), "precision": float(precision), "recall": float(recall), "fpr": float(fpr)}
        
        print(f"=== SCORING COMPLETE ===\n")
        return result

    def score_filtered(self, X: np.ndarray, y: np.ndarray, dataset_name: str, results_dir: Path, 
                      seed: int, threshold_class_0: float = 1.0, threshold_class_1: float = 1, 
                      test_size: float = 0.15, mask: Optional[np.ndarray] = None) -> dict[str, float]:
        """
        Calculate metrics only on examples where the model's logit_diff is above threshold.
        Filters out examples where abs(logit_diff) <= threshold from the CSV file.
        
        Args:
            threshold_class_0: Threshold for class 0 samples
            threshold_class_1: Threshold for class 1 samples
        """
        print(f"\n=== FILTERED SCORING START ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Class 0 threshold: {threshold_class_0}")
        print(f"Class 1 threshold: {threshold_class_1}")
        
        # Read the CSV file to get logit_diff values
        # Runthrough directory is always in the parent directory (results/{experiment_name}/)
        parent_dir = results_dir.parent
        runthrough_dir = parent_dir / f"runthrough_{dataset_name}"
        
        if not runthrough_dir.exists():
            print(f"Warning: No runthrough directory found at {runthrough_dir}. Using unfiltered scoring.")
            return self.score(X, y, mask)
        
        csv_files = list(runthrough_dir.glob("*logit_diff*.csv"))
        if not csv_files:
            print(f"Warning: No logit_diff*.csv file found in {runthrough_dir}. Using unfiltered scoring.")
            return self.score(X, y, mask)
        
        csv_path = csv_files[0]  # Use the first matching file
        print(f"Using logit_diff file: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            if 'logit_diff' not in df.columns:
                print(f"Warning: 'logit_diff' column not found in {csv_path}. Using unfiltered scoring.")
                return self.score(X, y, mask)
            
            # Validate that CSV has the same number of rows as our test set
            if len(df) != len(y):
                print(f"Warning: CSV has {len(df)} rows but test set has {len(y)} examples. Dataset mismatch detected!")
                print("This could indicate that the model_check and evaluation are using different datasets.")
                print("Using unfiltered scoring.")
                return self.score(X, y, mask)
            
            # Get the test set texts to match with CSV
            from src.data import Dataset
            # recreate the dataset to get the test texts using the same method as evaluation
            temp_ds = Dataset(dataset_name, model=None, device=self.device, seed=seed)
            temp_ds = Dataset.build_imbalanced_train_balanced_eval(temp_ds, test_size=test_size, seed=seed)
            test_texts = temp_ds.get_test_set()[0]
            
            # Validate that the recreated test set matches our current test set size
            if len(test_texts) != len(y):
                print(f"Warning: Recreated test set has {len(test_texts)} examples but current test set has {len(y)} examples.")
                print("This indicates a dataset creation inconsistency.")
                print("Using unfiltered scoring.")
                return self.score(X, y, mask)
            
            # Filter based on class-specific logit_diff thresholds
            logit_diff_values = df['logit_diff'].values
            
            # Initialize mask filter
            mask_filter = np.zeros(len(logit_diff_values), dtype=bool)
            
            # Apply class-specific thresholds
            for i, (logit_diff, true_label) in enumerate(zip(logit_diff_values, y)):
                if true_label == 0:
                    threshold = threshold_class_0
                elif true_label == 1:
                    threshold = threshold_class_1
                
                mask_filter[i] = np.abs(logit_diff) > threshold
            
            # Get the filtered indices
            filtered_indices = np.where(mask_filter)[0]
            
            print(f"Original test set size: {len(test_texts)}")
            print(f"Filtered test set size: {len(filtered_indices)}")
            print(f"Removed {len(test_texts) - len(filtered_indices)} examples")
            
            if len(filtered_indices) == 0:
                print("Warning: No examples remain after filtering. Using unfiltered scoring.")
                return self.score(X, y, mask)
            
            # Apply the filter to X, y, and mask
            X_filtered = X[filtered_indices]
            y_filtered = y[filtered_indices]
            mask_filtered = mask[filtered_indices] if mask is not None else None
            
            print(f"Filtered X shape: {X_filtered.shape}")
            print(f"Filtered y shape: {y_filtered.shape}")
            
            # Calculate metrics on filtered data
            result = self.score(X_filtered, y_filtered, mask_filtered)
            
            # Add filter info to result
            result["filtered"] = True
            result["threshold_class_0"] = threshold_class_0
            result["threshold_class_1"] = threshold_class_1
            result["original_size"] = len(test_texts)
            result["filtered_size"] = len(filtered_indices)
            result["removed_count"] = len(test_texts) - len(filtered_indices)
            
            print(f"=== FILTERED SCORING COMPLETE ===\n")
            return result
            
        except Exception as e:
            print(f"Error in filtered scoring: {e}. Using unfiltered scoring.")
            return self.score(X, y, mask)

    def score_with_filtered(self, X: np.ndarray, y: np.ndarray, dataset_name: str, results_dir: Path,
                           seed: int, threshold_class_0: float = 2.0, threshold_class_1: float = 2.0, 
                           test_size: float = 0.15, mask: Optional[np.ndarray] = None) -> dict[str, dict]:
        """
        Calculate both regular and filtered metrics, returning them in a combined dictionary.
        """
        print(f"\n=== COMBINED SCORING START ===")
        
        # Get regular metrics
        regular_metrics = self.score(X, y, mask)
        regular_metrics["filtered"] = False
        
        # Get filtered metrics
        filtered_metrics = self.score_filtered(X, y, dataset_name, results_dir, seed, threshold_class_0, 
                                             threshold_class_1, test_size, mask)
        
        # Combine results
        combined_results = {
            "all_examples": regular_metrics,
            "filtered_examples": filtered_metrics
        }
        
        print(f"=== COMBINED SCORING COMPLETE ===\n")
        return combined_results

    def save_state(self, path: Path):
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'd_model': self.d_model,
            'task_type': self.task_type,  # Keep for future extensibility
        }
        # Only save aggregation if it exists (for backward compatibility)
        if hasattr(self, 'aggregation'):
            save_dict['aggregation'] = self.aggregation
        torch.save(save_dict, path)
        print(f"Saved probe to {path}")
        # Save training info (loss history, etc.)
        log_path = path.with_name(path.stem + "_train_log.json")
        train_info = {
            "loss_history": self.loss_history,
            # Add more training info here if needed
        }
        with open(log_path, "w") as f:
            json.dump(train_info, f, indent=2)
        print(f"Saved training log to {log_path}")

    def load_state(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.d_model = checkpoint['d_model']
        self.task_type = checkpoint.get('task_type', 'classification')  # Keep for future extensibility
        # Load aggregation if it exists (for backward compatibility)
        if 'aggregation' in checkpoint:
            self.aggregation = checkpoint['aggregation']
        self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded probe from {path}")


class BaseProbeNonTrainable:
    """
    Base class for probes that don't require training. Handles evaluation and saving/loading.
    """
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", 
                aggregation: str = "mean"):
        self.d_model = d_model
        self.device = device
        self.task_type = task_type  # Keep for future extensibility
        self.aggregation = aggregation  # 'mean', 'max', 'last', 'softmax', or None for probes that don't need aggregation
        self.model = None  # No trainable model for these probes
        self.loss_history = []  # Empty for non-trainable probes

    def _aggregate_activations(self, activations: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Aggregate activations across sequence dimension based on self.aggregation.
        
        Args:
            activations: Shape (N, seq_len, d_model)
            mask: Optional mask, shape (N, seq_len)
            
        Returns:
            Aggregated activations, shape (N, d_model) or original shape if no aggregation needed
        """
        # If aggregation is None, return original activations
        if self.aggregation is None:
            return activations
        
        # For mass_mean, we still need to aggregate (mean over sequence)
        if self.aggregation == "mass_mean":
            if mask is None:
                mask = np.ones(activations.shape[:2], dtype=bool)
            
            # Use mean aggregation for mass_mean
            mask_expanded = mask[:, :, None]  # Shape: (N, seq_len, 1)
            masked_sum = np.einsum('nsl,nsd->nd', mask_expanded, activations)
            mask_counts = mask.sum(axis=1, keepdims=True)  # Shape: (N, 1)
            return masked_sum / (mask_counts + 1e-8)  # Add small epsilon to avoid division by zero
        
        if mask is None:
            mask = np.ones(activations.shape[:2], dtype=bool)
        
        if self.aggregation == "mean":
            # Optimized mean pooling with mask - only consider non-zero tokens
            # Use einsum for efficient masked sum
            mask_expanded = mask[:, :, None]  # Shape: (N, seq_len, 1)
            masked_sum = np.einsum('nsl,nsd->nd', mask_expanded, activations)
            mask_counts = mask.sum(axis=1, keepdims=True)  # Shape: (N, 1)
            
            # Ensure we don't divide by zero and handle edge cases
            # If all tokens are padding (mask_counts == 0), use zeros
            valid_mask = mask_counts > 0
            result = np.zeros_like(masked_sum)
            
            # Fix the broadcasting issue by properly indexing
            valid_indices = np.where(valid_mask.flatten())[0]
            if len(valid_indices) > 0:
                # masked_sum[valid_indices] has shape (n_valid, d_model)
                # mask_counts[valid_indices] needs to be (n_valid, 1) for broadcasting
                result[valid_indices] = masked_sum[valid_indices] / mask_counts[valid_indices]
            
            # Check for NaN in mean aggregation
            if np.isnan(result).any():
                print(f"ERROR: NaN detected in mean aggregation!")
                print(f"  activations shape: {activations.shape}")
                print(f"  activations has NaN: {np.isnan(activations).any()}")
                print(f"  activations min: {activations.min()}, max: {activations.max()}")
                print(f"  mask shape: {mask.shape}")
                print(f"  mask has NaN: {np.isnan(mask).any()}")
                print(f"  mask_counts shape: {mask_counts.shape}")
                print(f"  mask_counts has NaN: {np.isnan(mask_counts).any()}")
                print(f"  mask_counts min: {mask_counts.min()}, max: {mask_counts.max()}")
                print(f"  masked_sum shape: {masked_sum.shape}")
                print(f"  masked_sum has NaN: {np.isnan(masked_sum).any()}")
                print(f"  masked_sum min: {masked_sum.min()}, max: {masked_sum.max()}")
                print(f"  result has NaN: {np.isnan(result).any()}")
                print(f"  result min: {result.min()}, max: {result.max()}")
                raise ValueError("NaN detected in mean aggregation - investigate einsum or division")
            
            return result
        elif self.aggregation == "max":
            # Optimized max pooling with mask
            # Create a copy and set masked values to -inf
            masked_activations = activations.copy()
            # For each sample, set masked positions to -inf
            for i in range(len(activations)):
                masked_activations[i, ~mask[i]] = -np.inf
            return masked_activations.max(axis=1)
        elif self.aggregation == "last":
            # Last token (assuming mask is True for valid tokens)
            last_indices = mask.sum(axis=1) - 1
            return activations[np.arange(len(activations)), last_indices]
        elif self.aggregation == "softmax":
            # Softmax-weighted average
            # For simplicity, use uniform weights if no attention mechanism
            return activations.mean(axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def fit(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, **kwargs):
        """
        For non-trainable probes, this method computes the probe parameters from the training data.
        Subclasses must implement this to compute their specific parameters.
        """
        raise NotImplementedError("Subclasses must implement fit to compute probe parameters")

    def predict(self, X: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 1) -> np.ndarray:
        """
        Make predictions using the non-trainable probe with batched processing.
        """
        # Process data in batches to avoid memory issues
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        all_predictions = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_X = X[start_idx:end_idx]
            batch_mask = mask[start_idx:end_idx] if mask is not None else None
            
            # Process this batch
            batch_processed_X = self._aggregate_activations(batch_X, batch_mask)
            batch_predictions = self._compute_predictions(batch_processed_X)
            all_predictions.append(batch_predictions)
        
        # Concatenate all predictions
        return np.concatenate(all_predictions, axis=0)

    def _compute_predictions(self, processed_X: np.ndarray) -> np.ndarray:
        """
        Compute predictions from processed activations. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement _compute_predictions")

    def predict_proba(self, X: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 1) -> np.ndarray:
        """
        Compute prediction probabilities for non-trainable probes with batched processing.
        """
        # Process data in batches to avoid memory issues
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        all_probabilities = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_X = X[start_idx:end_idx]
            batch_mask = mask[start_idx:end_idx] if mask is not None else None
            
            # Process this batch
            batch_processed_X = self._aggregate_activations(batch_X, batch_mask)
            batch_probabilities = self._compute_probabilities(batch_processed_X)
            all_probabilities.append(batch_probabilities)
        
        # Concatenate all probabilities
        return np.concatenate(all_probabilities, axis=0)

    def _compute_probabilities(self, processed_X: np.ndarray) -> np.ndarray:
        """
        Compute probabilities from processed activations. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement _compute_probabilities")

    def predict_logits(self, X: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 1) -> np.ndarray:
        """
        Returns the raw scores (logits) for the input X with batched processing.
        """
        # Process data in batches to avoid memory issues
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        all_logits = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_X = X[start_idx:end_idx]
            batch_mask = mask[start_idx:end_idx] if mask is not None else None
            
            # Process this batch
            batch_processed_X = self._aggregate_activations(batch_X, batch_mask)
            batch_logits = self._compute_logits(batch_processed_X)
            all_logits.append(batch_logits)
        
        # Concatenate all logits
        return np.concatenate(all_logits, axis=0)

    def _compute_logits(self, processed_X: np.ndarray) -> np.ndarray:
        """
        Compute logits from processed activations. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement _compute_logits")

    def score(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 200) -> dict[str, float]:
        """
        Calculate performance metrics for the non-trainable probe (binary classification only).
        """
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
        
        preds = self.predict(X, mask, batch_size=batch_size)
        y_true = y
        y_prob = self.predict_proba(X, mask, batch_size=batch_size)
        
        # Binary classification only
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        acc = accuracy_score(y_true, preds)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        result = {"acc": float(acc), "auc": float(auc), "precision": float(precision), "recall": float(recall), "fpr": float(fpr)}
        
        return result

    def score_filtered(self, X: np.ndarray, y: np.ndarray, dataset_name: str, results_dir: Path, 
                      seed: int, threshold_class_0: float = 2.0, threshold_class_1: float = 2.0, 
                      test_size: float = 0.15, mask: Optional[np.ndarray] = None) -> dict[str, float]:
        """
        Calculate metrics only on examples where the model's logit_diff is above threshold.
        """
        # Read the CSV file to get logit_diff values
        parent_dir = results_dir.parent
        runthrough_dir = parent_dir / f"runthrough_{dataset_name}"
        
        if not runthrough_dir.exists():
            return self.score(X, y, mask)
        
        csv_files = list(runthrough_dir.glob("*logit_diff*.csv"))
        if not csv_files:
            return self.score(X, y, mask)
        
        csv_path = csv_files[0]
        try:
            df = pd.read_csv(csv_path)
            if 'logit_diff' not in df.columns:
                return self.score(X, y, mask)
            
            if len(df) != len(y):
                return self.score(X, y, mask)
            
            # Get the test set texts to match with CSV
            from src.data import Dataset
            temp_ds = Dataset(dataset_name, model=None, device=self.device, seed=seed)
            temp_ds = Dataset.build_imbalanced_train_balanced_eval(temp_ds, test_size=test_size, seed=seed)
            test_texts = temp_ds.get_test_set()[0]
            
            if len(test_texts) != len(y):
                return self.score(X, y, mask)
            
            # Filter based on class-specific logit_diff thresholds
            logit_diff_values = df['logit_diff'].values
            
            # Initialize mask filter
            mask_filter = np.zeros(len(logit_diff_values), dtype=bool)
            
            # Apply class-specific thresholds
            for i, (logit_diff, true_label) in enumerate(zip(logit_diff_values, y)):
                if true_label == 0:
                    threshold = threshold_class_0
                elif true_label == 1:
                    threshold = threshold_class_1
                
                mask_filter[i] = np.abs(logit_diff) > threshold
            
            filtered_indices = np.where(mask_filter)[0]
            
            if len(filtered_indices) == 0:
                return self.score(X, y, mask)
            
            # Apply the filter to X, y, and mask
            X_filtered = X[filtered_indices]
            y_filtered = y[filtered_indices]
            mask_filtered = mask[filtered_indices] if mask is not None else None
            
            # Calculate metrics on filtered data
            result = self.score(X_filtered, y_filtered, mask_filtered)
            
            # Add filter info to result
            result["filtered"] = True
            result["threshold_class_0"] = threshold_class_0
            result["threshold_class_1"] = threshold_class_1
            result["original_size"] = len(test_texts)
            result["filtered_size"] = len(filtered_indices)
            result["removed_count"] = len(test_texts) - len(filtered_indices)
            
            return result
            
        except Exception as e:
            return self.score(X, y, mask)

    def score_with_filtered(self, X: np.ndarray, y: np.ndarray, dataset_name: str, results_dir: Path,
                           seed: int, threshold_class_0: float = 2.0, threshold_class_1: float = 2.0, 
                           test_size: float = 0.15, mask: Optional[np.ndarray] = None) -> dict[str, dict]:
        """
        Calculate both regular and filtered metrics, returning them in a combined dictionary.
        """
        # Get regular metrics
        regular_metrics = self.score(X, y, mask)
        regular_metrics["filtered"] = False
        
        # Get filtered metrics
        filtered_metrics = self.score_filtered(X, y, dataset_name, results_dir, seed, threshold_class_0, 
                                             threshold_class_1, test_size, mask)
        
        # Combine results
        combined_results = {
            "all_examples": regular_metrics,
            "filtered_examples": filtered_metrics
        }
        
        return combined_results

    def save_state(self, path: Path):
        """
        Save the probe state (parameters, not model weights).
        """
        save_dict = {
            'd_model': self.d_model,
            'task_type': self.task_type,
            'aggregation': self.aggregation,
        }
        # Add probe-specific parameters
        probe_params = self._get_probe_parameters()
        save_dict.update(probe_params)
        
        torch.save(save_dict, path)
        
        # Save training info (empty for non-trainable probes)
        log_path = path.with_name(path.stem + "_train_log.json")
        train_info = {
            "loss_history": self.loss_history,
            "probe_type": "non_trainable"
        }
        with open(log_path, "w") as f:
            json.dump(train_info, f, indent=2)

    def _get_probe_parameters(self) -> dict:
        """
        Return probe-specific parameters to save. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement _get_probe_parameters")

    def load_state(self, path: Path):
        """
        Load the probe state (parameters, not model weights).
        """
        # Use weights_only=False to allow loading numpy arrays and other objects
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.d_model = checkpoint['d_model']
        self.task_type = checkpoint['task_type']
        self.aggregation = checkpoint['aggregation']
        
        # Load probe-specific parameters
        self._load_probe_parameters(checkpoint)

    def _load_probe_parameters(self, checkpoint: dict):
        """
        Load probe-specific parameters. Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement _load_probe_parameters")

