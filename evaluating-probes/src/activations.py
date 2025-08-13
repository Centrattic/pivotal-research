from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple, Iterable, Dict, Optional, Any
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def get_aggregation_methods() -> List[str]:
    """Get all supported aggregation methods."""
    return ["mean", "max", "last", "softmax"]


def get_linear_activation_types() -> List[str]:
    """Get all supported linear activation types."""
    return [f"linear_{method}" for method in get_aggregation_methods()]


def get_sae_activation_types() -> List[str]:
    """Get all supported SAE activation types."""
    return [f"sae_{method}" for method in get_aggregation_methods()]


def get_act_sim_activation_types() -> List[str]:
    """Get all supported activation similarity activation types."""
    return [f"act_sim_{method}" for method in get_aggregation_methods()]


def get_all_activation_types() -> List[str]:
    """Get all supported activation types across all probe types."""
    return ["full"] + get_linear_activation_types() + get_sae_activation_types() + get_act_sim_activation_types()


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
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        d_model: int,
        cache_dir: Path,
    ):
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        assert self.tokenizer is not None, "Tokenizer must be provided with the model"
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
        This method ONLY retrieves existing activations - no extraction.
        
        Args:
            texts: List of texts to get activations for
            layer: Layer number
            component: Component name (e.g., 'resid_post')
            activation_type: Type of activations to return:
                - "full": Full activations (N, S, D) with masks
                - "linear_mean", "linear_max", "linear_last", "linear_softmax": Linear probe aggregations
                - "sae_mean", "sae_max", "sae_last", "sae_softmax": SAE probe aggregations
                - "act_sim_mean", "act_sim_max", "act_sim_last", "act_sim_softmax": Activation similarity aggregations
        
        Returns:
            Tuple of (activations, masks) where masks might be None for pre-aggregated activations
        """
        texts = list(texts)
        metadata_path, activations_path = self._paths(layer, component)

        # For pre-aggregated types, load from separate files
        if activation_type in get_all_activation_types() and activation_type != "full":
            # Extract the aggregation method from the activation_type
            # Use the suffix after the last underscore so that
            #   - "linear_mean" -> "mean"
            #   - "sae_max" -> "max"
            #   - "act_sim_mean" -> "mean" (not "sim_mean")
            aggregation = activation_type.rsplit("_", 1)[1]
            aggregated_path = activations_path.parent / f"{activations_path.stem}_aggregated_{aggregation}.npz"
            
            if not aggregated_path.exists():
                raise ValueError(f"Aggregated activations file {aggregated_path} does not exist. Run extract_activations_for_dataset first.")
            
            print(f"[DEBUG] Loading {activation_type} aggregated activations from {aggregated_path}")
            aggregated_data = np.load(aggregated_path, mmap_mode='r')
            
            # Load in the order of texts
            aggregated_list = []
            for text in texts:
                text_hash = self._hash(text)
                if text_hash not in aggregated_data:
                    raise ValueError(f"Text not found in aggregated activations: {text[:50]}...")
                aggregated_list.append(aggregated_data[text_hash])
            
            result = np.stack(aggregated_list)
            print(f"[DEBUG] Loaded {activation_type} aggregated activations: {result.shape}")
            return result, None
        
        # For full activations, use the original method
        else:
            return self._get_full_activations_with_masks(texts, layer, component, activations_path)

    def ensure_texts_cached(
        self,
        texts: Iterable[str],
        layer: int,
        component: str,
    ) -> int:
        """
        Ensure that full activations for the provided texts are present in the cache.
        This will compute and append any missing activations to the full activations file.

        Returns the number of newly computed activations.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("ActivationManager is read-only. A full model is required to compute activations.")

        texts = list(texts)
        _, activations_path = self._paths(layer, component)

        # Load existing activations (if any)
        existing: Dict[str, np.ndarray] = {}
        if activations_path.exists():
            existing = dict(np.load(activations_path, allow_pickle=True))

        # Determine which texts are missing
        missing_texts: List[str] = []
        for text in texts:
            text_hash = self._hash(text)
            if text_hash not in existing:
                missing_texts.append(text)

        if not missing_texts:
            return 0

        hook_name = f"blocks.{layer}.hook_{component}"

        # Compute activations for missing texts in small batches to control memory
        new_entries: Dict[str, np.ndarray] = {}
        batch_size = 1
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for i in tqdm(range(0, len(missing_texts), batch_size)):
                batch_texts = missing_texts[i:i+batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=True,
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                acts_tensor = self._run_and_capture(layer=layer, component=component, input_ids=input_ids, attention_mask=attention_mask)
                acts_tensor = acts_tensor.detach().to("cpu")  # [B, S, D]

                for j, text in enumerate(batch_texts):
                    text_hash = self._hash(text)
                    if text_hash in existing or text_hash in new_entries:
                        continue
                    activation = acts_tensor[j].numpy()  # [S, D]
                    new_entries[text_hash] = activation

        # Merge and save
        if new_entries:
            to_save = {**existing, **new_entries}
            np.savez_compressed(activations_path, **to_save)

        return len(new_entries)

    def _get_full_activations_with_masks(self, texts: List[str], layer: int, component: str, activations_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Helper method to get full activations with masks. ONLY retrieves existing activations."""
        # Check if activations exist
        if not activations_path.exists():
            raise ValueError(f"Activations file {activations_path} does not exist. Run extract_activations_for_dataset first.")

        # Load all activations
        print(f"[DEBUG] Loading activations from {activations_path}")
        all_activations = dict(np.load(activations_path, allow_pickle=True, mmap_mode='r'))
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

    def compute_and_save_all_aggregations(self, layer: int, component: str, force_recompute: bool = False):
        """Compute and save aggregated activations from the full activations file.
        Incrementally updates existing aggregated files with any newly added texts.
        """
        _, activations_path = self._paths(layer, component)
        
        # If full activations don't exist, we can't compute aggregations
        if not activations_path.exists():
            raise ValueError(f"Full activations file {activations_path} does not exist. Extract activations first.")
        
        # Load all full activations once
        all_activations: Dict[str, np.ndarray] = dict(np.load(activations_path, allow_pickle=True, mmap_mode='r'))
        print(f"[DEBUG] Loaded {len(all_activations)} full activations for aggregation")
        
        from scipy.special import softmax
        
        for aggregation in get_aggregation_methods():
            aggregated_path = activations_path.parent / f"{activations_path.stem}_aggregated_{aggregation}.npz"
            # Load existing aggregated if present
            if aggregated_path.exists() and not force_recompute:
                aggregated_dict: Dict[str, np.ndarray] = dict(np.load(aggregated_path, allow_pickle=True))
                print(f"[DEBUG] Existing {aggregation} aggregated entries: {len(aggregated_dict)}")
            else:
                aggregated_dict = {}
                if force_recompute:
                    print(f"[DEBUG] Force recompute enabled for {aggregation}; computing from scratch")
                else:
                    print(f"[DEBUG] No existing {aggregation} aggregated file, computing from scratch")
            
            # Compute for missing hashes only
            missing_keys = [k for k in all_activations.keys() if k not in aggregated_dict]
            if not missing_keys and not force_recompute:
                print(f"[DEBUG] {aggregation}: no missing entries; skipping")
                continue
            if force_recompute:
                print(f"[DEBUG] {aggregation}: force recompute over all {len(all_activations)} entries")
                missing_keys = list(all_activations.keys())
            else:
                print(f"[DEBUG] {aggregation}: computing {len(missing_keys)} missing entries")
            
            for text_hash in missing_keys:
                activation = all_activations[text_hash]
                if aggregation == "mean":
                    aggregated = np.mean(activation, axis=0)
                elif aggregation == "max":
                    aggregated = np.max(activation, axis=0)
                elif aggregation == "last":
                    aggregated = activation[-1]
                elif aggregation == "softmax":
                    weights = softmax(activation, axis=0)
                    aggregated = np.sum(weights * activation, axis=0)
                aggregated_dict[text_hash] = aggregated
            
            # Save updated aggregated dict
            np.savez_compressed(aggregated_path, **aggregated_dict)
            print(f"[DEBUG] Saved updated {aggregation} aggregated activations to {aggregated_path} ({len(aggregated_dict)} entries)")
        
        print(f"[DEBUG] Aggregation update complete")

    def get_actual_max_len(self, layer: int, component: str) -> Optional[int]:
        """Get the actual maximum token length from activations if available."""
        metadata_path, activations_path = self._paths(layer, component)
        
        if not activations_path.exists():
            return None
        
        # Load activations and find max length
        all_activations = dict(np.load(activations_path, allow_pickle=True, mmap_mode='r'))
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

    # Backwards-compatible alias used by Dataset
    def clear_activation_cache(self):
        self.clear_metadata_cache()

    # HF hooking utilities #
    def _get_block_module(self, layer_index: int) -> Any:
        """
        Try to resolve a transformer block module across popular architectures.
        Returns the module whose forward output is the per-token hidden states at that layer.
        """
        m = self.model
        # Common: Llama/Gemma: model.model.layers
        return m.model.layers[layer_index]
        raise ValueError(f"Could not locate transformer block module for layer {layer_index}")

    def _run_and_capture(self, layer: int, component: str, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Run the model and capture per-token activations at the specified layer/component.
        Only 'resid_post' is currently supported and maps to the block output hidden states.
        Returns tensor of shape [B, S, D].
        """
        if component != "resid_post":
            raise NotImplementedError(f"Component '{component}' is not supported without TransformerLens. Supported: 'resid_post'.")

        block_module = self._get_block_module(layer)
        captured: Dict[str, torch.Tensor] = {"x": None}

        def hook_fn(module, inputs, output):  # type: ignore[override]
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured["x"] = hidden

        handle = block_module.register_forward_hook(hook_fn)
        try:
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=False)
        finally:
            handle.remove()

        if captured["x"] is None:
            raise RuntimeError("Failed to capture activations; hook did not run.")
        
        print(f"[DEBUG] Captured activation shape: {captured['x'].shape}")
        return captured["x"]
