import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import json
from tqdm import tqdm
import psutil

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

from sae_lens import SAE
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as safetensors_load_file

from src.probes.base_probe import BaseProbe


def get_memory_usage():
    """Get current memory usage for debugging."""
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3  # GB
    if torch.cuda.is_available():
        try:
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            return f"RAM: {ram_usage:.2f}GB, GPU_allocated: {gpu_allocated:.2f}GB, GPU_reserved: {gpu_reserved:.2f}GB"
        except:
            return f"RAM: {ram_usage:.2f}GB, GPU: unavailable"
    else:
        return f"RAM: {ram_usage:.2f}GB"


class SAEProbeNet(nn.Module):
    """
    Neural network for SAE-based probing.
    Takes SAE-encoded activations and applies a linear classifier.
    """

    def __init__(
        self,
        sae_feature_dim: int,
        aggregation: str = "mean",
        device: str = "cpu",
    ):
        super().__init__()
        self.sae_feature_dim = sae_feature_dim
        self.aggregation = aggregation
        self.device = device
        # Create linear layer with bfloat16 dtype for mixed precision training
        self.linear = nn.Linear(sae_feature_dim, 1, dtype=torch.bfloat16).to(self.device)

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

    def __init__(
        self,
        d_model: int,
        device: str = "cpu",
        task_type: str = "classification",
        aggregation: str = "mean",
        model_name: str = "gemma-2-9b",
        layer: int = 20,
        sae_id: Optional[str] = None,
        top_k_features: int = 128,
        sae_cache_dir: Optional[Path] = None,
        encoding_batch_size: int = 256,
        training_batch_size: int = 64,
        # sklearn hyperparams
        solver: str = "liblinear",
        C: float = 1.0,
        max_iter: int = 1500,
        class_weight: str = "balanced",
        random_state: int = 42,
        **kwargs,
    ):
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
        # For llama 3.3 70B Goodfire we do not require sae_id since the repo is fixed
        if sae_id is None and model_name not in ["meta-llama/Llama-3.3-70B-Instruct"]:
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

        # sklearn components
        self.solver = solver
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.sklearn_model = LogisticRegression(
            solver=self.solver,
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.scaler = StandardScaler()

        # Store any additional config parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Initialize SAE and feature selection
        self.sae = None
        self.feature_indices = None
        self.sae_feature_dim = None

        # Goodfire HF SAE repo (used when model_name is llama 3.3 70B)
        self.hf_sae_repo: Optional[str] = "Goodfire/Llama-3.3-70B-Instruct-SAE-l50" if (
            self.model_name == "meta-llama/Llama-3.3-70B-Instruct"
        ) else None

        # Call parent constructor
        super().__init__(d_model, device, task_type)

        # Flag to track if preprocessing has been done
        self._preprocessing_done = False

    def _init_model(
        self,
    ):
        """Initialize the neural network model."""
        # Create a tiny dummy module for BaseProbe compatibility; sklearn is used for training
        in_features = max(1, self.top_k_features)
        self.model = nn.Linear(in_features, 1)

    def _load_sae(self):
        """Load the SAE model (sae_lens for Gemma; Goodfire HF for Llama-3.3-70B)."""
        if self.sae is not None:
            return self.sae

        if self.model_name == "meta-llama/Llama-3.3-70B-Instruct":
            # Load Goodfire SAE from Hugging Face
            if self.hf_sae_repo is None:
                raise ValueError("HF SAE repo is not set for Llama-3.3-70B.")
            local_dir = self.sae_cache_dir / "goodfire_llama33b_l50"
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"Loading Goodfire SAE from HF repo: {self.hf_sae_repo}")
            repo_path = snapshot_download(
                repo_id=self.hf_sae_repo, local_dir=str(local_dir), local_dir_use_symlinks=False
            )

            # Find a safetensors file
            repo_path = Path(repo_path)
            candidates = list(repo_path.glob("*.safetensors"))
            if not candidates:
                # fallback to common filename inside subfolders
                candidates = list(repo_path.rglob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No .safetensors file found in {repo_path}")

            weights = safetensors_load_file(str(candidates[0]))

            # Heuristically determine encoder weight/bias as the 2D tensor with shape (F, D)
            # where D == d_model, and the largest F is selected
            enc_weight_key = None
            enc_weight = None
            for k, v in weights.items():
                if v.dim() == 2 and v.shape[1] == self.d_model:
                    if enc_weight is None or v.shape[0] > enc_weight.shape[0]:
                        enc_weight = v
                        enc_weight_key = k
            if enc_weight is None:
                # try transposed case (D, F)
                for k, v in weights.items():
                    if v.dim() == 2 and v.shape[0] == self.d_model:
                        if enc_weight is None or v.shape[1] > enc_weight.shape[1]:
                            enc_weight = v.T
                            enc_weight_key = k + ".T"
            if enc_weight is None:
                raise ValueError("Could not locate encoder weight of shape (features, d_model) in Goodfire SAE")

            # Find bias
            bias = None
            for k, v in weights.items():
                if v.dim() == 1 and v.shape[0] == enc_weight.shape[0]:
                    bias = v
                    break
            if bias is None:
                bias = torch.zeros(enc_weight.shape[0])

            class _SimpleHFSAE:

                def __init__(self, W: torch.Tensor, b: torch.Tensor, device: str):
                    self.W = W.to(device)
                    self.b = b.to(device)
                    self.device = device

                def encode(self, x: torch.Tensor) -> torch.Tensor:
                    # x: (N, D)
                    return F.relu(torch.addmm(self.b, x, self.W.t()))

            self.sae = _SimpleHFSAE(enc_weight, bias, self.device)
            print(
                f"Loaded Goodfire SAE ({enc_weight.shape[0]} features, d_model={self.d_model}) from {candidates[0].name}"
            )
            return self.sae
        else:
            # Use sae_lens for other supported models
            print(f"Loading SAE via sae_lens: {self.sae_id}")
        self.sae, _, _ = SAE.from_pretrained(
            release=self._get_sae_release(),
            sae_id=self.sae_id,
            device=self.device,
        )
        print(f"Loaded SAE: {self.sae_id}")
        return self.sae

    def _get_sae_release(self) -> str:
        """Get the SAE release name for the model."""
        # Handle full model names with prefixes and suffixes
        if self.model_name == "google/gemma-2-9b":
            return "gemma-scope-9b-pt-res"
        elif self.model_name == "google/gemma-2-9b-it":
            return "gemma-scope-9b-it-res-canonical"
        elif self.model_name == "meta-llama/Llama-3.3-70B-Instruct":
            # Using Goodfire HF path instead of sae_lens; this value is unused
            return "goodfire-hf"
        # Qwen 3 family (transcoders via sae_lens releases)
        elif self.model_name == "Qwen/Qwen3-0.6B":
            # Use HF repo ids directly so sae_lens can fetch if local release registry lacks these entries
            return "mwhanna/qwen3-0.6b-transcoders-lowl0"
        elif self.model_name == "Qwen/Qwen3-1.7B":
            return "mwhanna/qwen3-1.7b-transcoders-lowl0"
        elif self.model_name == "Qwen/Qwen3-4B":
            return "mwhanna/qwen3-4b-transcoders"
        elif self.model_name == "Qwen/Qwen3-8B":
            return "mwhanna/qwen3-8b-transcoders"
        elif self.model_name == "Qwen/Qwen3-14B":
            return "mwhanna/qwen3-14b-transcoders"
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def _encode_activations(self, activations: np.ndarray) -> np.ndarray:
        """Encode aggregated activations (N, 1, D) through the SAE using configurable batch size."""
        print(f"[DEBUG] Input activations shape: {activations.shape}")

        # Accept (N, 1, D) from ActivationManager aggregated outputs, or (N, D) if caller pre-squeezed
        if activations.ndim == 3:
            if activations.shape[1] == 1:
                activations = activations[:, 0, :]  # (N, D)
        elif activations.ndim == 2:
            # Already (N, D)
            pass
        else:
            raise ValueError(f"Expected activations with 2 or 3 dims; got {activations.shape}")
        print(f"[DEBUG] Using pre-aggregated inputs: {activations.shape}")

        sae = self._load_sae()

        # Encode in batches using configurable batch size
        N = activations.shape[0]
        encoded_list = []
        total_batches = (N + self.encoding_batch_size - 1) // self.encoding_batch_size
        print(f"[DEBUG] Processing {total_batches} batches of size {self.encoding_batch_size}")

        for start in tqdm(range(0, N, self.encoding_batch_size), desc="Encoding activations"):
            end = min(start + self.encoding_batch_size, N)
            batch = activations[start:end]
            batch_tensor = torch.tensor(batch, dtype=torch.bfloat16, device=self.device)
            encoded_batch = sae.encode(batch_tensor).float().cpu().detach().numpy()  # (B, F)
            encoded_list.append(encoded_batch)

        encoded = np.concatenate(encoded_list, axis=0)
        print(f"[DEBUG] Final encoded shape: {encoded.shape}")
        return encoded  # (N, F)

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

        # X_train is already SAE-encoded aggregated features (N, F)
        X_agg = X_train
        print(f"[DEBUG] Feature matrix for selection shape: {X_agg.shape}")
        print(f"[DEBUG] Memory after feature matrix build: {get_memory_usage()}")

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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        masks: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "SAEProbe":
        """Train the SAE probe using sklearn on SAE-encoded aggregated activations.
        
        Args:
            X: Aggregated activations, shape (N, 1, d_model)
            y: Labels, shape (N,)
            masks: Ignored (kept for compatibility)
        """
        print(f"\n=== SAE PROBE TRAINING START (sklearn) ===")
        print(f"Model: {self.model_name}, Layer: {self.layer}")
        print(f"SAE source: {'Goodfire HF' if self.model_name=='meta-llama/Llama-3.3-70B-Instruct' else 'sae_lens'}")
        print(f"Aggregation: {self.aggregation}")
        print(f"Top-k features: {self.top_k_features}")
        print(f"SAE encoding batch size: {self.encoding_batch_size}")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")

        # Step 1: SAE-encode aggregated activations -> (N, F)
        print("Encoding activations through SAE...")
        print(f"[DEBUG] Before encoding. Memory: {get_memory_usage()}")
        X_encoded = self._encode_activations(X)
        print(f"Encoded feature matrix shape: {X_encoded.shape}")
        print(f"[DEBUG] After encoding. Memory: {get_memory_usage()}")

        # Step 2: Feature selection (top-k by difference of means)
        print("Selecting top features...")
        self.feature_indices = self._select_top_features(X_encoded, y)
        print(f"Selected {len(self.feature_indices)} features")

        # Step 3: Select and scale features
        X_selected = X_encoded[:, self.feature_indices]
        X_scaled = self.scaler.fit_transform(X_selected)
        print(f"[DEBUG] X_selected shape: {X_selected.shape}")

        # Step 4: Train sklearn logistic regression
        self.sklearn_model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray, masks: Optional[np.ndarray] = None, batch_size: int = None) -> np.ndarray:
        """Predict using the SAE probe."""
        if self.feature_indices is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        X_encoded = self._encode_activations(X)  # (N, F)
        X_selected = X_encoded[:, self.feature_indices]
        # Upcast to float32 before scaling to avoid float16 overflow during in-place operations
        X_scaled = self.scaler.transform(X_selected.astype(np.float32, copy=False))
        return self.sklearn_model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray, masks: Optional[np.ndarray] = None, batch_size: int = None) -> np.ndarray:
        """Predict probabilities using the SAE probe."""
        if self.feature_indices is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        X_encoded = self._encode_activations(X)
        X_selected = X_encoded[:, self.feature_indices]
        # Upcast to float32 before scaling to avoid float16 overflow during in-place operations
        X_scaled = self.scaler.transform(X_selected.astype(np.float32, copy=False))
        probas = self.sklearn_model.predict_proba(X_scaled)
        return probas[:, 1]

    def predict_logits(self, X: np.ndarray, masks: Optional[np.ndarray] = None, batch_size: int = None) -> np.ndarray:
        """Predict logits using the SAE probe."""
        if self.feature_indices is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        X_encoded = self._encode_activations(X)
        X_selected = X_encoded[:, self.feature_indices]
        # Upcast to float32 before scaling to avoid float16 overflow during in-place operations
        X_scaled = self.scaler.transform(X_selected.astype(np.float32, copy=False))
        return self.sklearn_model.decision_function(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray, masks: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate performance metrics for aggregated inputs."""
        preds = self.predict(X, masks)
        y_prob = self.predict_proba(X, masks)
        auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0.5
        acc = accuracy_score(y, preds)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return {
            "acc": float(acc), "auc": float(auc), "precision": float(precision), "recall": float(recall), "fpr":
            float(fpr)
        }

    def save_state(
        self,
        path: Path,
    ):
        """Save the probe state including SAE info and feature indices."""
        # Save sklearn probe state in .npz format, along with SAE metadata
        if self.feature_indices is None:
            raise ValueError("No feature indices to save; train the probe first.")

        coef = getattr(self.sklearn_model, "coef_", None)
        intercept = getattr(self.sklearn_model, "intercept_", None)
        if coef is None or intercept is None:
            raise ValueError("Sklearn model is not fitted; cannot save state.")

        np.savez_compressed(
            path,
            # sklearn model
            coef=coef,
            intercept=intercept,
            # scaler
            scaler_mean=self.scaler.mean_,
            scaler_scale=self.scaler.scale_,
            scaler_var=self.scaler.var_,
            # sae/probe metadata
            d_model=self.d_model,
            task_type=self.task_type,
            aggregation=self.aggregation,
            model_name=self.model_name,
            layer=self.layer,
            sae_id=self.sae_id if self.sae_id is not None else "",
            hf_sae_repo=self.hf_sae_repo if self.hf_sae_repo is not None else "",
            top_k_features=self.top_k_features,
            feature_indices=np.asarray(self.feature_indices, dtype=np.int64),
            encoding_batch_size=self.encoding_batch_size,
            training_batch_size=self.training_batch_size,
            solver=self.solver,
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

    def load_state(
        self,
        path: Path,
    ):
        """Load the probe state."""
        state = np.load(path, allow_pickle=True)

        # Restore metadata
        self.d_model = int(state['d_model'])
        self.task_type = str(state['task_type'])
        self.aggregation = str(state['aggregation'])
        self.model_name = str(state['model_name'])
        self.layer = int(state['layer'])
        self.sae_id = str(state['sae_id']) if 'sae_id' in state and state['sae_id'].item() != "" else None
        self.hf_sae_repo = str(state['hf_sae_repo']
                               ) if 'hf_sae_repo' in state and state['hf_sae_repo'].item() != "" else self.hf_sae_repo
        self.top_k_features = int(state['top_k_features'])
        self.feature_indices = state['feature_indices'].astype(np.int64)
        self.encoding_batch_size = int(state.get('encoding_batch_size', self.encoding_batch_size))
        self.training_batch_size = int(state.get('training_batch_size', self.training_batch_size))

        # Restore sklearn components
        self.solver = str(state.get('solver', self.solver))
        self.C = float(state.get('C', self.C))
        self.max_iter = int(state.get('max_iter', self.max_iter))
        self.class_weight = str(state.get('class_weight', self.class_weight))
        self.random_state = int(state.get('random_state', self.random_state))

        # Recreate sklearn model with hyperparams
        self.sklearn_model = LogisticRegression(
            solver=self.solver,
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.sklearn_model.coef_ = state['coef']
        self.sklearn_model.intercept_ = state['intercept']
        self.sklearn_model.classes_ = np.array([0, 1])

        # Restore scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = state['scaler_mean']
        self.scaler.scale_ = state['scaler_scale']
        self.scaler.var_ = state['scaler_var']

        # Recreate dummy torch model for BaseProbe compatibility
        self._init_model()
