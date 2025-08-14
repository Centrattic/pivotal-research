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
    def create_readonly(
        cls,
        model_name: str,
        d_model: int,
        cache_dir: Path,
    ):
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
        activation_type: str = "full",
        response_texts: Optional[Iterable[str]] = None,
        format_type: str = "r-no-it"
    ) -> np.ndarray:
        """
        Get activations for texts, with caching and aggregation support.
        
        Args:
            texts: Input texts (prompts)
            layer: Layer number
            component: Component name (e.g., 'resid_post')
            activation_type: Type of activation to return:
                - "full": Full activations (padded to max length)
                - "linear_mean", "linear_max", "linear_last": Linear aggregations
                - "sae_mean", "sae_max", "sae_last": SAE aggregations
                - "act_sim_mean", "act_sim_max", "act_sim_last", "act_sim_softmax": Activation similarity aggregations
            response_texts: Optional list of response/question texts (for on-policy experiments)
            format_type: Format for activation extraction:
                - "qr": On-policy - format as prompt+question using chat template, extract question tokens
                - "r": Off-policy instruct - format prompt as user prompt using chat template, extract prompt tokens
                - "r-no-it": Off-policy non-instruct - use prompt text directly without chat template
        
        Returns:
            Activations array (no masks returned)
        """
        texts = list(texts)
        metadata_path, activations_path = self._paths(layer, component)

        # Check which texts need extraction
        existing: Dict[str, np.ndarray] = {}
        if activations_path.exists():
            existing = dict(np.load(activations_path, allow_pickle=True))

        missing_texts: List[str] = []
        missing_response_texts: List[str] = []
        for i, text in enumerate(texts):
            text_hash = self._hash(text)
            if text_hash not in existing:
                missing_texts.append(text)
                if response_texts is not None:
                    missing_response_texts.append(response_texts[i])

        # Extract missing activations
        if missing_texts:
            print(f"[DEBUG] Extracting activations for {len(missing_texts)} missing texts")
            new_activations = self._extract_activations(
                missing_texts,
                layer,
                component,
                format_type,
                missing_response_texts if response_texts is not None else None
            )

            # Save new activations
            all_activations = {**existing}
            for i, text in enumerate(missing_texts):
                text_hash = self._hash(text)
                all_activations[text_hash] = new_activations[i]

            np.savez(activations_path, **all_activations)

        # Load all activations in the order of texts
        activations_list = []
        for text in texts:
            text_hash = self._hash(text)
            if text_hash not in existing and activations_path.exists():
                # Reload to get newly saved activations
                existing = dict(np.load(activations_path, allow_pickle=True))
            activations_list.append(existing[text_hash])

        # Return appropriate activations based on type
        return self._get_activations_for_return(activations_list, activation_type, layer, component, texts)

    def get_activations_for_responses(
        self,
        responses: Iterable[str],
        layer: int,
        component: str,
        activation_type: str = "full"
    ) -> Tuple[np.ndarray,
               np.ndarray]:
        """
        Return activations for response texts that were computed using ensure_conversation_texts_cached.
        This method ONLY retrieves existing activations - no extraction.
        
        Args:
            responses: List of response texts to get activations for
            layer: Layer number
            component: Component name (e.g., 'resid_post')
            activation_type: Type of activations to return (same as get_activations_for_texts)
        
        Returns:
            Tuple of (activations, masks) where masks might be None for pre-aggregated activations
        """
        # This method is essentially the same as get_activations_for_texts
        # since we store response activations using the response text as the key
        return self.get_activations_for_texts(responses, layer, component, activation_type)

    def ensure_texts_cached(
        self,
        texts: Iterable[str],
        layer: int,
        component: str,
        response_texts: Optional[Iterable[str]] = None,
        format_type: str = "r-no-it"
    ) -> int:
        """
        Ensure texts are cached. Returns number of newly computed activations.
        
        Args:
            texts: Input texts
            layer: Layer number
            component: Component name
            response_texts: Optional response texts for on-policy
            format_type: Format type for extraction
            
        Returns:
            Number of newly computed activations
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("ActivationManager is read-only. A full model is required to compute activations.")

        texts = list(texts)
        _, activations_path = self._paths(layer, component)

        # Check which texts are missing
        existing: Dict[str, np.ndarray] = {}
        if activations_path.exists():
            existing = dict(np.load(activations_path, allow_pickle=True))

        missing_texts: List[str] = []
        missing_response_texts: List[str] = []
        for i, text in enumerate(texts):
            text_hash = self._hash(text)
            if text_hash not in existing:
                missing_texts.append(text)
                if response_texts is not None:
                    missing_response_texts.append(response_texts[i])

        if not missing_texts:
            return 0

        # Extract missing activations
        print(f"[DEBUG] Extracting activations for {len(missing_texts)} missing texts")
        new_activations = self._extract_activations(
            missing_texts,
            layer,
            component,
            format_type,
            missing_response_texts if response_texts is not None else None
        )

        # Save new activations
        all_activations = {**existing}
        for i, text in enumerate(missing_texts):
            text_hash = self._hash(text)
            all_activations[text_hash] = new_activations[i]

        np.savez(activations_path, **all_activations)

        return len(missing_texts)

    def ensure_conversation_texts_cached(
        self,
        conversations: Iterable[Tuple[str, str]],  # (full_conversation, response_only)
        layer: int,
        component: str,
    ) -> int:
        """
        Ensure that activations for response tokens in conversations are present in the cache.
        This runs the model over the full conversation but extracts activations only for the response tokens.
        
        Args:
            conversations: Iterable of (full_conversation, response_only) tuples
            layer: Layer number
            component: Component name (e.g., 'resid_post')
            
        Returns:
            Number of newly computed activations
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("ActivationManager is read-only. A full model is required to compute activations.")

        conversations = list(conversations)
        _, activations_path = self._paths(layer, component)

        # Load existing activations (if any)
        existing: Dict[str, np.ndarray] = {}
        if activations_path.exists():
            existing = dict(np.load(activations_path, allow_pickle=True))

        # Determine which responses are missing
        missing_conversations: List[Tuple[str, str]] = []
        for full_conv, response_only in conversations:
            response_hash = self._hash(response_only)
            if response_hash not in existing:
                missing_conversations.append((full_conv, response_only))

        if not missing_conversations:
            return 0

        # Compute activations for missing conversations in small batches to control memory
        new_entries: Dict[str, np.ndarray] = {}
        batch_size = 1
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for i in tqdm(range(0, len(missing_conversations), batch_size)):
                batch_conversations = missing_conversations[i:i + batch_size]
                full_convs = [conv[0] for conv in batch_conversations]
                responses = [conv[1] for conv in batch_conversations]

                # Tokenize the full conversations
                enc_full = self.tokenizer(
                    full_convs,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=True,
                )
                input_ids_full = enc_full["input_ids"].to(self.device)
                attention_mask_full = enc_full.get("attention_mask")
                if attention_mask_full is not None:
                    attention_mask_full = attention_mask_full.to(self.device)

                # Tokenize just the responses to get their token positions
                enc_responses = self.tokenizer(
                    responses,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=True,
                )
                response_token_counts = [len(ids) for ids in enc_responses["input_ids"]]

                # Run model on full conversations
                acts_tensor = self._run_and_capture(
                    layer=layer,
                    component=component,
                    input_ids=input_ids_full,
                    attention_mask=attention_mask_full
                )
                acts_tensor = acts_tensor.detach().to("cpu")  # [B, S, D]

                for j, (full_conv, response_only) in enumerate(batch_conversations):
                    response_hash = self._hash(response_only)
                    if response_hash in existing or response_hash in new_entries:
                        continue

                    # Extract activations for response tokens only
                    full_activation = acts_tensor[j].numpy()  # [S, D]
                    response_token_count = response_token_counts[j]

                    # Get the last response_token_count tokens as the response activations
                    response_activation = full_activation[-response_token_count:]
                    new_entries[response_hash] = response_activation

        # Merge and save
        if new_entries:
            to_save = {**existing, **new_entries}
            np.savez_compressed(activations_path, **to_save)

        return len(new_entries)

    def _extract_activations(
        self,
        texts: List[str],
        layer: int,
        component: str,
        format_type: str,
        response_texts: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Core activation extraction logic based on format type.
        
        Args:
            texts: Input texts (prompts)
            layer: Layer number
            component: Component name
            format_type: Format type ("qr", "r", "r-no-it")
            response_texts: Optional response/question texts for on-policy
            
        Returns:
            Extracted activations array
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("ActivationManager is read-only. A full model is required to compute activations.")

        hook_name = f"blocks.{layer}.hook_{component}"
        activations_list = []

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for i in tqdm(range(len(texts)), desc=f"Extracting activations for {format_type}"):
                text = texts[i]

                if format_type == "qr" and response_texts is not None:
                    # On-policy: format as prompt+question using chat template, extract question tokens
                    question = response_texts[i]
                    if hasattr(self.tokenizer, 'apply_chat_template'):
                        formatted = self.tokenizer.apply_chat_template(
                            [{
                                "role": "user",
                                "content": text
                            },
                             {
                                 "role": "assistant",
                                 "content": question
                             }],
                            tokenize=False
                        )
                    else:
                        formatted = f"{text}\n\n{question}"

                    # Tokenize and get activations
                    enc = self.tokenizer(
                        formatted,
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                        add_special_tokens=True,
                    )
                    input_ids = enc["input_ids"].to(self.device)

                    # Get activations and extract only question tokens
                    question_enc = self.tokenizer(
                        question,
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                        add_special_tokens=True,
                    )
                    question_ids = question_enc["input_ids"][0]

                    # Find question token positions in full sequence
                    full_ids = input_ids[0]
                    question_start = None
                    for j in range(len(full_ids) - len(question_ids) + 1):
                        if torch.equal(full_ids[j:j + len(question_ids)], question_ids):
                            question_start = j
                            break

                    if question_start is None:
                        # Fallback: use all tokens
                        activations = self._get_activations_for_input_ids(input_ids, hook_name)
                    else:
                        activations = self._get_activations_for_input_ids(input_ids, hook_name)
                        activations = activations[:, question_start:question_start + len(question_ids), :]

                elif format_type == "r":
                    # Off-policy instruct: format prompt as user prompt using chat template, extract prompt tokens
                    if hasattr(self.tokenizer, 'apply_chat_template'):
                        formatted = self.tokenizer.apply_chat_template(
                            [{
                                "role": "user",
                                "content": text
                            }],
                            tokenize=False
                        )
                    else:
                        formatted = text

                    enc = self.tokenizer(
                        formatted,
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                        add_special_tokens=True,
                    )
                    input_ids = enc["input_ids"].to(self.device)
                    activations = self._get_activations_for_input_ids(input_ids, hook_name)

                elif format_type == "r-no-it":
                    # Off-policy non-instruct: use prompt text directly without chat template
                    enc = self.tokenizer(
                        text,
                        return_tensors="pt",
                        padding=False,
                        truncation=False,
                        add_special_tokens=True,
                    )
                    input_ids = enc["input_ids"].to(self.device)
                    activations = self._get_activations_for_input_ids(input_ids, hook_name)

                else:
                    raise ValueError(f"Invalid format_type: {format_type}")

                activations_list.append(activations.cpu().numpy())

        return np.concatenate(activations_list, axis=0)

    def _get_activations_for_input_ids(self, input_ids: torch.Tensor, hook_name: str) -> torch.Tensor:
        """Helper to get activations for input_ids with hook."""
        activations = None

        def hook_fn(
            module,
            input,
            output,
        ):
            nonlocal activations
            activations = output.detach()

        handle = self.model.register_forward_hook(hook_fn)
        try:
            self.model(input_ids)
            return activations
        finally:
            handle.remove()

    def _get_activations_for_return(
        self,
        activations_list: List[np.ndarray],
        activation_type: str,
        layer: int,
        component: str,
        texts: List[str]
    ) -> np.ndarray:
        """
        Determine what activations to return based on activation_type.
        
        Args:
            activations_list: List of activation arrays
            activation_type: Type of activation to return
            layer: Layer number
            component: Component name
            
        Returns:
            Appropriate activations array
        """
        if activation_type == "full":
            # Pad to max length and stack
            max_len = max(act.shape[1] for act in activations_list)
            padded_activations = []

            for act in activations_list:
                if act.shape[1] < max_len:
                    # Pad with zeros
                    padding = np.zeros((act.shape[0], max_len - act.shape[1], act.shape[2]))
                    act = np.concatenate([act, padding], axis=1)
                padded_activations.append(act)

            return np.concatenate(padded_activations, axis=0)

        else:
            # For aggregated types, load from saved aggregation files
            # Extract the aggregation method from the activation_type
            # Use the suffix after the last underscore so that
            #   - "linear_mean" -> "mean"
            #   - "sae_max" -> "max"
            #   - "act_sim_mean" -> "mean" (not "sim_mean")
            aggregation = activation_type.rsplit("_", 1)[1]
            aggregated_path = activations_path.parent / f"{activations_path.stem}_aggregated_{aggregation}.npz"

            if not aggregated_path.exists():
                raise ValueError(
                    f"Aggregated activations file {aggregated_path} does not exist. Run extract_activations_for_dataset first."
                )

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
            return result

    def _aggregate_activations(self, activations: np.ndarray, method: str) -> np.ndarray:
        """Helper to aggregate activations using specified method."""
        if method == "mean":
            return np.mean(activations, axis=1)
        elif method == "max":
            return np.max(activations, axis=1)
        elif method == "last":
            return activations[:, -1, :]
        elif method == "softmax":
            # For activation similarity
            return activations[:, -1, :]  # Use last token for softmax
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def compute_and_save_all_aggregations(
        self,
        layer: int,
        component: str,
        force_recompute: bool = False,
    ):
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
            print(
                f"[DEBUG] Saved updated {aggregation} aggregated activations to {aggregated_path} ({len(aggregated_dict)} entries)"
            )

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

    def clear_metadata_cache(
        self,
    ):
        """Clear activation cache to free memory."""
        import gc
        gc.collect()

    # Backwards-compatible alias used by Dataset
    def clear_activation_cache(
        self,
    ):
        self.clear_metadata_cache()

    # HF hooking utilities #
    def _get_block_module(self, layer_index: int) -> Any:
        """
        Try to resolve a transformer block module across popular architectures.
        Returns the module whose forward output is the per-token hidden states at that layer.
        """
        m = self.model
        # For llama/gemma: model.model.layers
        return m.model.layers[layer_index]
        raise ValueError(f"Could not locate transformer block module for layer {layer_index}")

    def _run_and_capture(
        self,
        layer: int,
        component: str,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Run the model and capture per-token activations at the specified layer/component.
        Only 'resid_post' is currently supported and maps to the block output hidden states.
        Returns tensor of shape [B, S, D].
        """
        if component != "resid_post":
            raise NotImplementedError(
                f"Component '{component}' is not supported without TransformerLens. Supported: 'resid_post'."
            )

        block_module = self._get_block_module(layer)
        captured: Dict[str, torch.Tensor] = {"x": None}

        def hook_fn(
            module,
            inputs,
            output,
        ):  # type: ignore[override]
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured["x"] = hidden

        handle = block_module.register_forward_hook(hook_fn)
        try:
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False
            )
        finally:
            handle.remove()

        if captured["x"] is None:
            raise RuntimeError("Failed to capture activations; hook did not run.")

        print(f"[DEBUG] Captured activation shape: {captured['x'].shape}")
        return captured["x"]
