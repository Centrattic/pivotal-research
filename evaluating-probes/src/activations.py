from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple, Iterable, Dict, Optional
import json

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer


class ActivationManager:
    """Individual activation storage with per-activation lengths.

    For each (layer, component) we store:
    1. **`<hook>.json`** - Metadata file with hash-to-index mapping and individual lengths
    2. **`<hook>.npz`** - Compressed numpy file storing individual activations with their actual lengths

    This approach eliminates padding and allows each activation to be stored at its natural size.
    """

    @classmethod
    def create_readonly(cls, model_name: str, d_model: int, cache_dir: Path):
        """
        Create a read-only ActivationManager that can load existing caches without the full model.
        This is useful for evaluation when activations are already cached.
        """
        instance = cls.__new__(cls)
        instance.model = None  # No model needed for reading
        instance.tokenizer = None  # No tokenizer needed for reading
        instance.device = "cpu"  # Default device for reading
        instance.d_model = d_model
        instance.cache_dir = cache_dir
        instance.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for metadata to avoid reloading
        instance._metadata_cache = {}
        instance._metadata_mtimes = {}
        
        print(f"[INFO] Created read-only ActivationManager for {model_name} (no tokenizer needed)")
        return instance

    def __init__(
        self,
        model: HookedTransformer,
        device: str,
        d_model: int,
        cache_dir: Path,
    ):
        self.model = model
        self.tokenizer = model.tokenizer
        assert self.tokenizer is not None, "Model must carry tokenizer"
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        self.device = device
        self.d_model = d_model
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for metadata to avoid reloading
        self._metadata_cache = {}
        self._metadata_mtimes = {}

    # public entrypoint #
    def get_activations_for_texts(
        self,
        texts: Iterable[str],
        layer: int,
        component: str,
        activation_type: str = "full"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return activations based on activation_type.
        
        Args:
            texts: List of texts to get activations for
            layer: Layer number
            component: Component name (e.g., 'resid_post')
            activation_type: Type of activations to return:
                - "full": Full activations (N, S, D) with masks
                - "linear_mean": Pre-aggregated mean activations (N, D)
                - "linear_max": Pre-aggregated max activations (N, D)
                - "linear_last": Pre-aggregated last token activations (N, D)
                - "linear_softmax": Pre-aggregated softmax activations (N, D)
                - "attention": Full activations for attention probes
                - "sae_mean": Pre-aggregated mean activations for SAE
                - etc.
        
        Returns:
            Tuple of (activations, masks) where masks might be None for pre-aggregated activations
        """
        texts = list(texts)
        metadata_path, activations_path = self._paths(layer, component)

        # Handle pre-aggregated activation types
        if activation_type.startswith("linear_") or activation_type.startswith("sae_"):
            aggregation = activation_type.split("_", 1)[1]  # Extract "mean", "max", "last", "softmax"
            aggregated_acts = self.get_pre_aggregated_activations(texts, layer, component, aggregation)
            return aggregated_acts, None  # No masks for pre-aggregated activations
        
        # Handle attention probes (need full activations)
        elif activation_type == "attention":
            return self._get_full_activations_with_masks(texts, layer, component, activations_path)
        
        # Default: full activations with masks
        elif activation_type == "full":
            return self._get_full_activations_with_masks(texts, layer, component, activations_path)
        
        else:
            raise ValueError(f"Unknown activation_type: {activation_type}. Supported types: full, linear_mean, linear_max, linear_last, linear_softmax, attention, sae_mean, etc.")

    def _get_full_activations_with_masks(self, texts: List[str], layer: int, component: str, activations_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Helper method to get full activations with masks."""
        # Simple approach: just load all activations and return them
        if not activations_path.exists():
            if self.model is None:
                raise ValueError(f"Activations file {activations_path} does not exist and no model available to extract them")
            else:
                # Extract activations for all texts
                self._extract_activations_for_texts(texts, layer, component, activations_path)

        # Load all activations
        print(f"[DEBUG] Loading activations from {activations_path}")
        all_activations = dict(np.load(activations_path, allow_pickle=True))
        print(f"[DEBUG] Loaded {len(all_activations)} activations from file")
        
        # Convert to list in the order of texts
        print(f"[DEBUG] Converting {len(texts)} texts to activations list")
        activations_list = []
        masks_list = []
        
        for text in texts:
            text_hash = self._hash(text)
            if text_hash not in all_activations:
                raise ValueError(f"Text not found in activations: {text[:50]}...")
            
            activation = all_activations[text_hash]
            activations_list.append(activation)
            
            # Create mask (all True for actual tokens)
            mask = np.ones(activation.shape[0], dtype=bool)
            masks_list.append(mask)
        
        print(f"[DEBUG] Converted to lists, now padding to max length")
        # Pad to max length
        max_len = max(act.shape[0] for act in activations_list)
        print(f"[DEBUG] Max length is {max_len}")
        padded_activations = []
        padded_masks = []
        
        for activation, mask in tqdm(zip(activations_list, masks_list)):
            if activation.shape[0] < max_len:
                # Pad activation
                padding_size = max_len - activation.shape[0]
                padded_act = np.pad(activation, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
                # Pad mask
                padded_mask = np.pad(mask, (0, padding_size), mode='constant', constant_values=False)
            else:
                padded_act = activation
                padded_mask = mask
            
            padded_activations.append(padded_act)
            padded_masks.append(padded_mask)
        
        print(f"[DEBUG] Padding complete, now stacking into arrays")
        # Stack into arrays
        activations_array = np.stack(padded_activations)
        print(f"[DEBUG] Stacked activations: {activations_array.shape}")
        masks_array = np.stack(padded_masks)
        print(f"[DEBUG] Stacked masks: {masks_array.shape}")
        
        return activations_array, masks_array

    def get_pre_aggregated_activations(
        self,
        texts: Iterable[str],
        layer: int,
        component: str,
        aggregation: str = "mean"
    ) -> np.ndarray:
        """Return pre-aggregated activations (N, D) for faster loading."""
        texts = list(texts)
        metadata_path, activations_path = self._paths(layer, component)
        
        # Check if pre-aggregated file exists
        aggregated_path = activations_path.parent / f"{activations_path.stem}_aggregated_{aggregation}.npz"
        
        if aggregated_path.exists():
            print(f"[DEBUG] Loading pre-aggregated {aggregation} activations from {aggregated_path}")
            aggregated_data = np.load(aggregated_path)
            
            # Check if all requested texts are in the aggregated file
            missing_texts = []
            for text in texts:
                text_hash = self._hash(text)
                if text_hash not in aggregated_data:
                    missing_texts.append(text[:50] + "...")
            
            if missing_texts:
                print(f"[DEBUG] Found {len(missing_texts)} missing texts in aggregated file, recomputing...")
                print(f"[DEBUG] Missing texts: {missing_texts[:3]}...")
                # Remove the incomplete aggregated file and recompute
                aggregated_path.unlink()
            else:
                # All texts found, load in the order of texts
                aggregated_list = []
                for text in texts:
                    text_hash = self._hash(text)
                    aggregated_list.append(aggregated_data[text_hash])
                
                result = np.stack(aggregated_list)
                print(f"[DEBUG] Loaded pre-aggregated activations: {result.shape}")
                return result
        
        # If not pre-aggregated or missing texts, compute and save them
        print(f"[DEBUG] Computing pre-aggregated {aggregation} activations...")
        if not activations_path.exists():
            if self.model is None:
                raise ValueError(f"Activations file {activations_path} does not exist and no model available to extract them")
            else:
                # Extract activations for all texts
                self._extract_activations_for_texts(texts, layer, component, activations_path)
        
        # Load full activations and aggregate
        print(f"[DEBUG] Loading full activations for aggregation...")
        all_activations = dict(np.load(activations_path, allow_pickle=True))
        print(f"[DEBUG] Loaded {len(all_activations)} full activations")
        
        # Aggregate each activation
        print(f"[DEBUG] Aggregating {len(texts)} texts...")
        aggregated_activations = {}
        for i, text in enumerate(texts):
            text_hash = self._hash(text)
            if text_hash not in all_activations:
                raise ValueError(f"Text not found in activations: {text[:50]}...")
            
            activation = all_activations[text_hash]  # Shape: (seq_len, d_model)
            
            # Aggregate based on method
            if aggregation == "mean":
                aggregated = np.mean(activation, axis=0)
            elif aggregation == "max":
                aggregated = np.max(activation, axis=0)
            elif aggregation == "last":
                aggregated = activation[-1]  # Last token
            elif aggregation == "softmax":
                from scipy.special import softmax
                weights = softmax(activation, axis=0)
                aggregated = np.sum(weights * activation, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            aggregated_activations[text_hash] = aggregated
            
            if (i + 1) % 1000 == 0:
                print(f"[DEBUG] Aggregated {i + 1}/{len(texts)} texts")
        
        # Save aggregated activations
        print(f"[DEBUG] Saving aggregated activations to {aggregated_path}")
        np.savez_compressed(aggregated_path, **aggregated_activations)
        print(f"[DEBUG] Saved pre-aggregated {aggregation} activations to {aggregated_path}")
        
        # Return in the order of texts
        result = np.stack([aggregated_activations[self._hash(text)] for text in texts])
        print(f"[DEBUG] Computed pre-aggregated activations: {result.shape}")
        return result

    def get_actual_max_len(self, layer: int, component: str) -> Optional[int]:
        """Get the actual maximum token length from activations if available."""
        metadata_path, activations_path = self._paths(layer, component)
        
        if not activations_path.exists():
            return None
        
        # Load activations and find max length
        all_activations = dict(np.load(activations_path, allow_pickle=True))
        if not all_activations:
            return None
        
        max_len = max(activation.shape[0] for activation in all_activations.values())
        return max_len

    # internals #
    def _paths(self, layer: int, component: str) -> Tuple[Path, Path]:
        hook = f"blocks.{layer}.hook_{component}".replace(".", "‑")
        metadata = self.cache_dir / f"{hook}.json"
        activations = self.cache_dir / f"{hook}.npz"
        return metadata, activations

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha1(prompt.encode()).hexdigest()

    def clear_metadata_cache(self):
        """Clear activation cache to free memory."""
        import gc
        gc.collect()
