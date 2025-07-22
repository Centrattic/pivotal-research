import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional
import json
import math
from tqdm import tqdm
import optuna

from src.logger import Logger

class BaseProbe:
    """
    General wrapper for PyTorch-based probes. Handles training, evaluation, and saving/loading.
    """
    def __init__(self, d_model: int, device: str = "cpu", task_type: str = "classification", aggregation: str = "mean"):
        self.d_model = d_model
        self.device = device
        self.task_type = task_type  # 'classification' or 'regression'
        self.aggregation = aggregation
        self.model: Optional[nn.Module] = None
        self.loss_history = []
        self._init_model()

    def _init_model(self):
        raise NotImplementedError("Subclasses must implement _init_model to set self.model")

    def fit(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None, 
            epochs: int = 20, lr: float = 1e-3, batch_size: int = 64, weight_decay: float = 0.0, 
            verbose: bool = True, early_stopping: bool = True, patience: int = 5, min_delta: float = 0.001,
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
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32 if self.task_type == "regression" else torch.long)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool)
        else:
            mask = torch.ones(X.shape[:2], dtype=torch.bool)
        
        print(f"Tensor X shape: {X.shape}")
        print(f"Tensor y shape: {y.shape}")
        print(f"Tensor mask shape: {mask.shape}")

        # Compute class weights for weighted loss (classification only)
        class_weights = None
        if self.task_type == "classification" and use_weighted_loss:
            try:
                from sklearn.utils.class_weight import compute_class_weight
                y_np = y.cpu().numpy() if hasattr(y, 'cpu') else y.numpy()
                if y_np.ndim > 1 and y_np.shape[1] == 1:
                    y_np = y_np.squeeze(-1)
                classes = np.unique(y_np)
                class_weights_np = compute_class_weight('balanced', classes=classes, y=y_np)
                class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=self.device)
                print(f"Class weights: {class_weights}")
            except Exception as e:
                print(f"Could not compute class weights: {e}")
                class_weights = None

        dataset = torch.utils.data.TensorDataset(X, y, mask)

        # WeightedRandomSampler for class balancing (classification only)
        sampler = None
        if self.task_type == "classification" and use_weighted_sampler:
            y_np = y.cpu().numpy() if hasattr(y, 'cpu') else y.numpy()
            if y_np.ndim > 1 and y_np.shape[1] == 1:
                y_np = y_np.squeeze(-1)
            class_sample_count = np.array([len(np.where(y_np == t)[0]) for t in np.unique(y_np)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[int(t)] for t in y_np])
            samples_weight = torch.from_numpy(samples_weight).float()
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
            print(f"Using WeightedRandomSampler for class balancing.")

        if sampler is not None:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of batches: {len(loader)}")

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if self.task_type == "classification":
            if y.ndim == 1 or y.shape[1] == 1:
                # Binary classification
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
            else:
                # Multiclass
                if use_weighted_loss and class_weights is not None:
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    print(f"Using CrossEntropyLoss with class weights")
                else:
                    criterion = nn.CrossEntropyLoss()
                    print(f"Using CrossEntropyLoss (no class weights)")
        else:
            criterion = nn.MSELoss()
            print(f"Using MSELoss (regression)")
        
        best_loss = float('inf')
        epochs_no_improve = 0
        stop_epoch = None

        for epoch in tqdm(range(epochs)):
            epoch_loss = 0.0
            batch_count = 0
            for xb, yb, mb in loader:
                # Only move one batch to the device at a time
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mb = mb.to(self.device)

                optimizer.zero_grad()
                logits = self.model(xb, mb)
                
                if batch_count == 0 and epoch == 0:
                    print(f"First batch - xb shape: {xb.shape}, yb shape: {yb.shape}, mb shape: {mb.shape}")
                    print(f"First batch - logits shape: {logits.shape}")
                
                if self.task_type == "classification":
                    if yb.ndim == 1 or yb.shape[-1] == 1:
                        yb = yb.float()
                        loss = criterion(logits, yb)
                    else:
                        loss = criterion(logits, yb)
                else:
                    loss = criterion(logits, yb)
                
                if batch_count == 0 and epoch == 0:
                    print(f"First batch - loss: {loss.item():.4f}")
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                batch_count += 1
            avg_loss = epoch_loss / len(dataset)
            self.loss_history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            # Early stopping logic
            if early_stopping:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Best loss: {best_loss:.4f}")
                    stop_epoch = epoch + 1
                    break
        if stop_epoch is not None:
            print(f"Training stopped early at epoch {stop_epoch}.")
        print(f"=== TRAINING COMPLETE ===\n")
        return self
    
    def find_best_fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, mask_train: Optional[np.ndarray] = None, 
                    mask_val: Optional[np.ndarray] = None, n_trials: int = 20, direction: str = None, verbose: bool = True):
        raise NotImplementedError("Subclasses must implement find_best_fit to be used in runner.py")

    def predict(self, X: np.ndarray, mask: Optional[np.ndarray] = None, batch_size: int = 1) -> np.ndarray:
        print(f"\n=== PREDICTION START ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input mask shape: {mask.shape if mask is not None else 'None'}")
        print(f"Batch size: {batch_size}")
        
        self.model.eval()
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        else:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=self.device)
        
        all_preds = []
        num_batches = (len(X) + batch_size - 1) // batch_size
        print(f"Processing {num_batches} batches")
        
        for i in range(0, len(X), batch_size):
            batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=self.device)
            batch_mask = mask[i:i+batch_size]
            
            if i == 0:
                print(f"First batch - batch_X shape: {batch_X.shape}, batch_mask shape: {batch_mask.shape}")
            
            with torch.no_grad():
                logits = self.model(batch_X, batch_mask)
                
                if i == 0:
                    print(f"First batch - logits shape: {logits.shape}")
                
                if self.task_type == "classification":
                    if logits.ndim == 1 or logits.shape[-1] == 1:
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).long().cpu().numpy()
                    else:
                        preds = torch.argmax(logits, dim=-1).cpu().numpy()
                else:
                    preds = logits.cpu().numpy()
                
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
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        else:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=self.device)
        
        all_probs = []
        num_batches = (len(X) + batch_size - 1) // batch_size
        print(f"Processing {num_batches} batches")
        
        for i in range(0, len(X), batch_size):
            batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=self.device)
            batch_mask = mask[i:i+batch_size]
            
            if i == 0:
                print(f"First batch - batch_X shape: {batch_X.shape}, batch_mask shape: {batch_mask.shape}")
            
            with torch.no_grad():
                logits = self.model(batch_X, batch_mask)
                
                if i == 0:
                    print(f"First batch - logits shape: {logits.shape}")
                
                if self.task_type == "classification":
                    if logits.ndim == 1 or logits.shape[-1] == 1:
                        probs = torch.sigmoid(logits).cpu().numpy()
                    else:
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                else:
                    probs = logits.cpu().numpy()
                
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
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        else:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=self.device)
        all_logits = []
        num_batches = (len(X) + batch_size - 1) // batch_size
        for i in range(0, len(X), batch_size):
            batch_X = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=self.device)
            batch_mask = mask[i:i+batch_size]
            with torch.no_grad():
                logits = self.model(batch_X, batch_mask)
                if logits.ndim == 1:
                    logits = logits[:, None]  # shape (batch, 1)
                all_logits.append(logits.cpu().numpy())
        result = np.concatenate(all_logits, axis=0)
        return result

    def score(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None) -> dict[str, float]:
        print(f"\n=== SCORING START ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Input mask shape: {mask.shape if mask is not None else 'None'}")
        print(f"Task type: {self.task_type}")
        
        from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error, precision_score, recall_score, confusion_matrix
        preds = self.predict(X, mask)
        print(f"Predictions shape: {preds.shape}")
        print(f"Predictions unique values: {np.unique(preds)}")
        
        if self.task_type == "regression":
            r2 = float(r2_score(y, preds))
            mse = float(mean_squared_error(y, preds))
            print(f"Regression metrics - R²: {r2:.4f}, MSE: {mse:.4f}")
            result = {
                "r2_score": r2,
                "mse": mse,
            }
        else:
            y_true = y
            y_prob = self.predict_proba(X, mask)
            print(f"Probabilities shape: {y_prob.shape}")
            print(f"True labels unique values: {np.unique(y_true)}")
            
            if y_prob.ndim == 1 or y_prob.shape[-1] == 1:
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
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                acc = accuracy_score(y_true, preds)
                precision = precision_score(y_true, preds, average='macro', zero_division=0)
                recall = recall_score(y_true, preds, average='macro', zero_division=0)
                fpr = 1 - recall_score(y_true, preds, average='macro', zero_division=0)
                print(f"Multiclass classification metrics:")
                print(f"  Accuracy: {acc:.4f}")
                print(f"  AUC: {auc:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  FPR: {fpr:.4f}")
            
            result = {"acc": float(acc), "auc": float(auc), "precision": float(precision), "recall": float(recall), "fpr": float(fpr)}
        
        print(f"=== SCORING COMPLETE ===\n")
        return result

    def score_filtered(self, X: np.ndarray, y: np.ndarray, dataset_name: str, results_dir: Path, 
                      seed: int, logit_diff_threshold: float = 1.0, test_size: float = 0.15, mask: Optional[np.ndarray] = None) -> dict[str, float]:
        """
        Calculate metrics only on examples where the model's logit_diff is above threshold.
        Filters out examples where abs(logit_diff) <= threshold from the CSV file.
        """
        print(f"\n=== FILTERED SCORING START ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Logit diff threshold: {logit_diff_threshold}")
        
        # Read the CSV file to get logit_diff values
        runthrough_dir = results_dir / f"runthrough_{dataset_name}"
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
            
            # Get the test set texts to match with CSV
            from src.data import Dataset
            # We need to recreate the dataset to get the test texts
            # This is a bit hacky but necessary to match activations with CSV rows
            temp_ds = Dataset(dataset_name, model=None, device=self.device)
            temp_ds.split_data(test_size=test_size, seed=42)
            test_texts = temp_ds.get_test_set()[0]
            
            # Filter based on logit_diff threshold
            mask_filter = np.abs(df['logit_diff'].values) > logit_diff_threshold
            
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
            result["logit_diff_threshold"] = logit_diff_threshold
            result["original_size"] = len(test_texts)
            result["filtered_size"] = len(filtered_indices)
            result["removed_count"] = len(test_texts) - len(filtered_indices)
            
            print(f"=== FILTERED SCORING COMPLETE ===\n")
            return result
            
        except Exception as e:
            print(f"Error in filtered scoring: {e}. Using unfiltered scoring.")
            return self.score(X, y, mask)

    def score_with_filtered(self, X: np.ndarray, y: np.ndarray, dataset_name: str, results_dir: Path,
                           seed: int, logit_diff_threshold: float = 1.0, test_size: float = 0.15, mask: Optional[np.ndarray] = None) -> dict[str, dict]:
        """
        Calculate both regular and filtered metrics, returning them in a combined dictionary.
        """
        print(f"\n=== COMBINED SCORING START ===")
        
        # Get regular metrics
        regular_metrics = self.score(X, y, mask)
        regular_metrics["filtered"] = False
        
        # Get filtered metrics
        filtered_metrics = self.score_filtered(X, y, dataset_name, results_dir, seed, logit_diff_threshold, test_size, mask)
        
        # Combine results
        combined_results = {
            "all_examples": regular_metrics,
            "filtered_examples": filtered_metrics
        }
        
        print(f"=== COMBINED SCORING COMPLETE ===\n")
        return combined_results

    def save_state(self, path: Path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'd_model': self.d_model,
            'task_type': self.task_type,
            'aggregation': self.aggregation,
        }, path)
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
        self.task_type = checkpoint['task_type']
        self.aggregation = checkpoint['aggregation']
        self._init_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded probe from {path}")

    def fit_pcngd(self, X: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None,
                  epochs: int = 20, lr: float = 1e-3, weight_decay: float = 0.0, verbose: bool = True, early_stopping: bool = True, patience: int = 5, min_delta: float = 0.0001, eps: float = 1e-6):
        """
        Train the probe model using Per-Class Normalized Gradient Descent (PCNGD).
        Only supports classification tasks for now.
        Args:
            X: Input features, shape (N, seq, d_model)
            y: Labels, shape (N,) or (N, num_classes)
            mask: Optional mask, shape (N, seq)
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay (not used in PCNGD, but included for API compatibility)
            verbose: Print progress
            early_stopping: Use early stopping
            patience: Early stopping patience
            min_delta: Early stopping min delta
            eps: Small constant for numerical stability
        """
        print(f"\n=== PCNGD TRAINING START ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Mask shape: {mask.shape if mask is not None else 'None'}")
        print(f"Task type: {self.task_type}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}, LR: {lr}")
        if self.task_type != "classification":
            raise NotImplementedError("PCNGD is only implemented for classification tasks.")
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        else:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=self.device)
        # Get unique classes
        y_np = y.cpu().numpy() if hasattr(y, 'cpu') else y.numpy()
        if y_np.ndim > 1 and y_np.shape[1] == 1:
            y_np = y_np.squeeze(-1)
        classes = np.unique(y_np)
        n_classes = len(classes)
        # Build per-class indices
        class_indices = {int(cls): np.where(y_np == cls)[0] for cls in classes}
        best_loss = float('inf')
        epochs_no_improve = 0
        stop_epoch = None
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            grads = []
            norms = []
            epoch_loss = 0.0
            # (a) Compute per-class gradients
            for l in classes:
                idxs = class_indices[int(l)]
                xb = X[idxs]
                yb = y[idxs]
                mb = mask[idxs]
                # Forward
                logits = self.model(xb, mb)
                # If binary, logits shape (N,) or (N,1), yb shape (N,)
                if logits.ndim == 1 or logits.shape[-1] == 1:
                    logits = logits.view(-1, 1)
                # CrossEntropyLoss expects (N, C) and y as (N,)
                if logits.shape[-1] == 1 and n_classes == 2:
                    # Convert logits to (N, 2) for binary
                    logits = torch.cat([-logits, logits], dim=1)
                loss_l = criterion(logits, yb)
                grad_l = torch.autograd.grad(loss_l, self.model.parameters(), retain_graph=True, create_graph=False)
                flat_grad = torch.cat([g.flatten() for g in grad_l])
                norm_l = flat_grad.norm() + eps
                grads.append(grad_l)
                norms.append(norm_l)
                epoch_loss += loss_l.item() * len(idxs)
            # (b) Combine normalized gradients
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            for grad_l, norm_l in zip(grads, norms):
                for p, g in zip(self.model.parameters(), grad_l):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    p.grad.add_(g / norm_l)
            # (c) Manual SGD step
            for p in self.model.parameters():
                p.data = p.data - lr * p.grad
            avg_loss = epoch_loss / X.shape[0]
            self.loss_history.append(avg_loss)
            if verbose:
                print(f"PCNGD Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            # Early stopping logic
            if early_stopping:
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"PCNGD Early stopping triggered at epoch {epoch+1}. Best loss: {best_loss:.4f}")
                    stop_epoch = epoch + 1
                    break
        if stop_epoch is not None:
            print(f"PCNGD Training stopped early at epoch {stop_epoch}.")
        print(f"=== PCNGD TRAINING COMPLETE ===\n")
        return self

class LinearProbeNet(nn.Module):
    def __init__(self, d_model: int, aggregation: str = "mean", device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.aggregation = aggregation
        self.device = device
        self.linear = nn.Linear(d_model, 1).to(self.device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model), mask: (batch, seq)
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

class LinearProbe(BaseProbe):
    def _init_model(self):
        self.model = LinearProbeNet(self.d_model, aggregation=self.aggregation, device=self.device)

    def find_best_fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, mask_train: Optional[np.ndarray] = None, mask_val: Optional[np.ndarray] = None, 
                    n_trials: int = 15, direction: str = None, verbose: bool = True, weighting_method: str = 'weighted_loss', metric: str = 'acc', fpr_threshold: float = 0.01):
        """
        Hyperparameter tuning for the probe. If weighting_method is 'pcngd', tune using fit_pcngd, else use fit.
        metric: 'acc' (default), 'auc', or 'fpr_recall'.
        If 'fpr_recall', minimize FPR, but if FPR <= fpr_threshold, maximize Recall.
        """
        best_score = None
        best_params = None
        best_probe = None
        def objective(trial):
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-2)
            epochs = 50
            # Re-init probe for each trial
            probe = LinearProbe(self.d_model, device=self.device, aggregation=self.aggregation)
            if weighting_method == 'pcngd':
                probe.fit_pcngd(X_train, y_train, mask=mask_train, epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=False)
            else:
                probe.fit(X_train, y_train, mask=mask_train, epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=False, use_weighted_loss=(weighting_method=='weighted_loss'), use_weighted_sampler=(weighting_method=='weighted_sampler'))
            metrics = probe.score(X_val, y_val, mask=mask_val)
            if metric == 'auc':
                return -metrics.get('auc', 0.0)
            elif metric == 'fpr_recall':
                fpr = metrics.get('fpr', 1.0)
                recall = metrics.get('recall', 0.0)
                # If FPR > threshold, minimize FPR; else, maximize recall
                if fpr > fpr_threshold:
                    return fpr  # minimize FPR
                else:
                    return -recall  # maximize recall
            else:
                raise ValueError(f"Unknown metric: {metric}")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        # Train final probe with best params
        best_probe = LinearProbe(self.d_model, device=self.device, aggregation=self.aggregation)
        if weighting_method == 'pcngd':
            best_probe.fit_pcngd(X_train, y_train, mask=mask_train, epochs=30, lr=best_params['lr'], weight_decay=best_params['weight_decay'], verbose=verbose)
        else:
            best_probe.fit(X_train, y_train, mask=mask_train, epochs=30, lr=best_params['lr'], weight_decay=best_params['weight_decay'], verbose=verbose, use_weighted_loss=(weighting_method=='weighted_loss'), use_weighted_sampler=(weighting_method=='weighted_sampler'))
        return best_probe, best_params
    # Inherits predict_logits from BaseProbe

class AttentionProbeNet(nn.Module):
    def __init__(self, d_model: int, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.scale = math.sqrt(d_model)
        self.context_query = nn.Linear(d_model, 1)
        self.classifier = nn.Linear(d_model, 1)
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
    def _init_model(self):
        self.model = AttentionProbeNet(self.d_model, device=self.device)

    def find_best_fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, mask_train: Optional[np.ndarray] = None, mask_val: Optional[np.ndarray] = None, 
                    n_trials: int = 15, direction: str = None, verbose: bool = True, weighting_method: str = 'weighted_loss', metric: str = 'acc', fpr_threshold: float = 0.01):
        """
        Hyperparameter tuning for the attention probe. If weighting_method is 'pcngd', tune using fit_pcngd, else use fit.
        metric: 'acc' (default), 'auc', or 'fpr_recall'.
        If 'fpr_recall', minimize FPR, but if FPR <= fpr_threshold, maximize Recall.
        """
        best_score = None
        best_params = None
        best_probe = None
        def objective(trial):
            lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-2)
            epochs = 50
            probe = AttentionProbe(self.d_model, device=self.device)
            if weighting_method == 'pcngd':
                probe.fit_pcngd(X_train, y_train, mask=mask_train, epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=False)
            else:
                probe.fit(X_train, y_train, mask=mask_train, epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=False, use_weighted_loss=(weighting_method=='weighted_loss'), use_weighted_sampler=(weighting_method=='weighted_sampler'))
            metrics = probe.score(X_val, y_val, mask=mask_val)
            if metric == 'auc':
                return -metrics.get('auc', 0.0)
            elif metric == 'fpr_recall':
                fpr = metrics.get('fpr', 1.0)
                recall = metrics.get('recall', 0.0)
                if fpr > fpr_threshold:
                    return fpr
                else:
                    return -recall
            else:
                raise ValueError(f"Unknown metric: {metric}")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_probe = AttentionProbe(self.d_model, device=self.device)
        if weighting_method == 'pcngd':
            best_probe.fit_pcngd(X_train, y_train, mask=mask_train, epochs=30, lr=best_params['lr'], weight_decay=best_params['weight_decay'], verbose=verbose)
        else:
            best_probe.fit(X_train, y_train, mask=mask_train, epochs=30, lr=best_params['lr'], weight_decay=best_params['weight_decay'], verbose=verbose, use_weighted_loss=(weighting_method=='weighted_loss'), use_weighted_sampler=(weighting_method=='weighted_sampler'))
        return best_probe, best_params
    # Inherits predict_logits from BaseProbe

# Example usage:
# class MyProbe(BaseProbe):
#     def _init_model(self):
#         self.model = LinearProbe(self.d_model, aggregation=self.aggregation).to(self.device)

# To use:
# probe = MyProbe(d_model=768, device="cuda", task_type="classification", aggregation="mean")
# probe.fit(X_train, y_train, mask=mask_train)
# acc = probe.score(X_val, y_val, mask=mask_val) 