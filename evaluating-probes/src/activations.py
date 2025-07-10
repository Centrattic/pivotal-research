# src/activations.py
import torch
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm
from pathlib import Path
from typing import List

class ActivationManager:
    def __init__(self, model_name: str, device: str, d_model: int):
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.tokenizer = self.model.tokenizer
        
        # FIX: Add an assertion to handle the optional type
        assert self.tokenizer is not None, "Tokenizer not found on the model."

        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"
        self.device = device
        self.d_model = d_model
        self.max_len = self.model.cfg.n_ctx

    def _get_hook_name(self, layer: int, component: str) -> str:
        if component in {"resid_post", "resid_mid", "resid_pre"}:
            return f"blocks.{layer}.hook_{component}"
        raise ValueError(f"Unknown component: {component}")

    def _get_cache_path(self, cache_dir: Path, layer: int, component: str) -> Path:
        hook_safe_name = self._get_hook_name(layer, component).replace(".", "-")
        return cache_dir / f"{hook_safe_name}.mmap"

    def get_activations(
        self,
        texts: List[str],
        layer: int,
        component: str,
        use_cache: bool,
        cache_dir: Path,
        batch_size: int = 32,
    ) -> np.ndarray:
        
        mmap_path = self._get_cache_path(cache_dir, layer, component)
        shape = (len(texts), self.max_len, self.d_model)
        
        if use_cache and mmap_path.exists():
            print(f"  - Loading activations from cache: {mmap_path}")
            return np.memmap(mmap_path, dtype=np.float16, mode='r', shape=shape)

        print("  - Generating activations...")
        N = len(texts)
        hook_name = self._get_hook_name(layer, component)

        mmap_file = []
        
        if use_cache:
            mmap_file = np.memmap(mmap_path, dtype=np.float16, mode='w+', shape=shape)

        all_acts = []
        
        for i in tqdm(range(0, N, batch_size), desc="  - Activation Extraction"):
            batch_texts = texts[i:i + batch_size]

            assert self.tokenizer is not None, "Tokenizer not found on the model."

            tokens = self.tokenizer(
                batch_texts, return_tensors="pt", padding="max_length",
                truncation=True, max_length=self.max_len
            ).to(self.device)

            _, cache = self.model.run_with_cache(tokens.input_ids, names_filter=[hook_name])
            chunk = cache[hook_name].cpu().to(torch.float16).numpy()
            
            if use_cache:
                mmap_file[i:i + len(batch_texts)] = chunk
            else:
                all_acts.append(chunk)
            
            del cache
            torch.cuda.empty_cache()

        if use_cache:
            mmap_file.flush() # Todo; Runtime logic is fine. Fix type issue later
            return mmap_file
        else:
            return np.concatenate(all_acts, axis=0)