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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return activations (N, S, D) and attention masks (N, S) in the exact order of `texts`."""
        texts = list(texts)
        metadata_path, activations_path = self._paths(layer, component)

        # Load existing metadata
        metadata = self._get_cached_metadata(metadata_path)
        hash_to_idx = metadata.get('hash_to_idx', {})
        lengths = metadata.get('lengths', {})

        hashes = [self._hash(p) for p in texts]
        missing = [(h, p) for h, p in zip(hashes, texts) if h not in hash_to_idx]

        if missing:
            self._append_new(layer, component, missing, metadata, metadata_path, activations_path)
            # Clear cache since we modified the metadata
            self._metadata_cache.pop(metadata_path, None)
            # Reload metadata
            metadata = self._get_cached_metadata(metadata_path)
            hash_to_idx = metadata.get('hash_to_idx', {})
            lengths = metadata.get('lengths', {})

        # Get indices for requested texts
        indices = [hash_to_idx[h] for h in hashes]
        
        # Load activations and create masks
        return self._get_activations_with_masks(activations_path, indices, lengths, hash_to_idx)

    def get_actual_max_len(self, layer: int, component: str) -> Optional[int]:
        """Get the actual maximum token length from the metadata for this layer/component."""
        metadata_path, _ = self._paths(layer, component)
        
        if not metadata_path.exists():
            return None  # No activations cached yet
        
        metadata = self._get_cached_metadata(metadata_path)
        lengths = metadata.get('lengths', {})
        
        if not lengths:
            return None
        
        return max(lengths.values())

    # internals #
    def _paths(self, layer: int, component: str) -> Tuple[Path, Path]:
        hook = f"blocks.{layer}.hook_{component}".replace(".", "‑")
        metadata = self.cache_dir / f"{hook}.json"
        activations = self.cache_dir / f"{hook}.npz"
        return metadata, activations

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha1(prompt.encode()).hexdigest()

    def _get_cached_metadata(self, metadata_path: Path) -> Dict:
        """Get cached metadata, reloading only if file has changed."""
        if not metadata_path.exists():
            return {}
        
        # Check if file has been modified since last cache
        current_mtime = metadata_path.stat().st_mtime
        cached_mtime = self._metadata_mtimes.get(metadata_path)
        
        if metadata_path in self._metadata_cache and cached_mtime == current_mtime:
            print(f"[DEBUG] Using cached metadata for {metadata_path.name}")
            return self._metadata_cache[metadata_path]
        
        # Load metadata
        print(f"[DEBUG] Loading metadata from {metadata_path.name}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Cache the result
        self._metadata_cache[metadata_path] = metadata
        self._metadata_mtimes[metadata_path] = current_mtime
        
        return metadata

    def _append_new(
        self,
        layer: int,
        component: str,
        missing: List[Tuple[str, str]],  # (hash, prompt)
        metadata: Dict,
        metadata_path: Path,
        activations_path: Path,
        bs: int = 1,
    ):
        if not missing:
            return

        new_hashes, new_prompts = zip(*missing)
        N_new = len(new_prompts)

        print(f"[DEBUG] Appending {N_new} new activations")

        # Load existing activations if they exist
        existing_activations = {}
        if activations_path.exists():
            try:
                existing_activations = dict(np.load(activations_path, allow_pickle=True))
                print(f"[DEBUG] Loaded {len(existing_activations)} existing activations")
            except Exception as e:
                print(f"[WARNING] Failed to load existing activations: {e}")
                print("[WARNING] Starting with empty activation cache")
                existing_activations = {}

        # Extract activations for the new prompts
        hook = f"blocks.{layer}.hook_{component}"
        
        for s in tqdm(range(0, N_new, bs), desc=f"Extract L{layer} {component}"):
            batch_prompts = list(new_prompts[s : s + bs])
            
            # First, get actual lengths without padding for metadata
            actual_lengths = []
            for prompt in batch_prompts:
                toks = self.tokenizer([prompt], return_tensors="pt", padding=False, truncation=False)
                actual_lengths.append(toks["input_ids"].shape[1])
            
            # Tokenize with padding for efficient batch processing
            max_length = max(actual_lengths)
            toks = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            
            with torch.no_grad():
                _, cache = self.model.run_with_cache(toks["input_ids"], names_filter=[hook], device=self.device)
            acts = cache[hook].cpu().to(torch.float16).numpy()
            
            # Store each activation with its actual length (remove padding)
            for i, (act, prompt_hash, actual_length) in enumerate(zip(acts, new_hashes[s:s+bs], actual_lengths)):
                # Trim to actual length (remove padding)
                actual_activation = act[:actual_length]
                
                # Store the activation with its actual length (no padding)
                existing_activations[prompt_hash] = actual_activation
                
                # Update metadata
                metadata.setdefault('hash_to_idx', {})[prompt_hash] = len(metadata.get('hash_to_idx', {}))
                metadata.setdefault('lengths', {})[prompt_hash] = actual_length
            
            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save updated metadata and activations
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        np.savez_compressed(activations_path, **existing_activations)

    def _get_activations_with_masks(self, activations_path: Path, indices: List[int], lengths: Dict[str, int], hash_to_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return activations and attention masks for the given indices."""
        if not indices:
            raise ValueError("No indices to get activations for")
        
        # Load all activations
        if not activations_path.exists():
            raise ValueError(f"Activations file {activations_path} does not exist")
        
        all_activations = dict(np.load(activations_path, allow_pickle=True))
        
        # Get activations for requested indices
        requested_activations = []
        requested_lengths = []
        
        # Create reverse mapping from index to hash
        idx_to_hash = {idx: hash_val for hash_val, idx in hash_to_idx.items()}
        
        for idx in indices:
            if idx >= len(idx_to_hash):
                raise ValueError(f"Index {idx} out of bounds (max: {len(idx_to_hash)-1})")
            
            hash_val = idx_to_hash[idx]
            activation = all_activations[hash_val]
            length = lengths[hash_val]
            
            requested_activations.append(activation)
            requested_lengths.append(length)
        
        # Find the maximum length in this batch
        max_len = max(requested_lengths)
        
        # Pad activations to max_len and create masks
        padded_activations = []
        attention_masks = []
        
        for activation, length in zip(requested_activations, requested_lengths):
            # Pad activation to max_len
            if activation.shape[0] < max_len:
                padding_size = max_len - activation.shape[0]
                padded_act = np.pad(activation, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
            else:
                padded_act = activation
            
            padded_activations.append(padded_act)
            
            # Create attention mask
            mask = np.zeros(max_len, dtype=bool)
            mask[:length] = True
            attention_masks.append(mask)
        
        # Stack into arrays
        activations_array = np.stack(padded_activations)
        masks_array = np.stack(attention_masks)
        
        return activations_array, masks_array

    def clear_metadata_cache(self):
        """Clear the cached metadata. Useful if memory usage becomes an issue."""
        self._metadata_cache.clear()
        self._metadata_mtimes.clear()

    def clear_all_caches(self):
        """Clear all caches."""
        self.clear_metadata_cache()

    def clear_activation_cache(self):
        """Clear activation cache to free memory."""
        import gc
        gc.collect()
