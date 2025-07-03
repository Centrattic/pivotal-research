# extract.py
"""Streaming activation extractor for TransformerLens models.

Highlights
==========
* **Single forward‑pass** per batch using `run_with_cache` – no huge .pt dumps.
* Supports **any TransformerLens‑compatible model** (default GPT‑2 medium).
* Flexibly choose *layer*, *component*, *aggregation* strategy.
* Returns `(N, d)` NumPy array ready for `utils_probe`.

Example
-------
```python
from extract import Extractor
from utils_probe import LogisticRegressionProbe

ext = Extractor(model_name="gpt2-medium", device="cuda")
X_train = ext.features(text_batch, layer=8, component="resid_post", agg="mean")
probe = LogisticRegressionProbe().fit(X_train, labels)
```
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Literal, Sequence

import numpy as np
import torch
from transformer_lens import HookedTransformer, ActivationCache
from configs import *
from tqdm import tqdm

# Enumerations of components exposed

# For probing, we only use resid_pre. But may extract others for visualization.
class Comp(str, enum.Enum):
    """User‑facing component keys."""

    resid_pre = "resid_pre"     # before MLP+attn block
    resid_mid = "resid_mid"   # after block
    attn_q = "attn_q"           # queries  (batch, seq, n_heads, d_head)
    attn_k = "attn_k"
    attn_v = "attn_v"
    attn_out = "attn_out"       # result of V*P
    attn_pattern = "attn_pattern"  # attention weights (batch, heads, seq, seq)

# Helper to map (layer, component) -> hook name(s)
def hook_name(layer: int, comp: Comp) -> str:
    if comp in {Comp.resid_pre, Comp.resid_mid}:
        return f"blocks.{layer}.hook_{comp}"
    elif comp == Comp.attn_q:
        return f"blocks.{layer}.attn.hook_q"
    elif comp == Comp.attn_k:
        return f"blocks.{layer}.attn.hook_k"
    elif comp == Comp.attn_v:
        return f"blocks.{layer}.attn.hook_v"
    elif comp == Comp.attn_out:
        return f"blocks.{layer}.attn.hook_result"
    elif comp == Comp.attn_pattern:
        return f"blocks.{layer}.attn.hook_attn_probs"
    else:
        raise ValueError(f"Unknown component {comp}")

# Aggregation helpers

def aggregate(t: torch.Tensor, mode: Literal["mean", "first", "last", "max", "flatten"] = "mean") -> np.ndarray:  # noqa: D401
    """Convert `(B, seq, *d)` to `(B, D)`.

    * If `flatten`, flattens all dims after batch.
    """
    if mode == "flatten":
        return t.flatten(start_dim=1).cpu().numpy()
    if t.ndim < 3:
        raise ValueError("Need at least (B, seq, feat) for mean/first/last/max")
    if mode == "mean":
        out = t.mean(1)
    elif mode == "first":
        out = t[:, 0, ...]
    elif mode == "last":
        out = t[:, -1, ...]
    elif mode == "max":
        out = t.max(1).values
    else:
        raise ValueError(mode)
    return out.flatten(start_dim=1).cpu().numpy()

# Extractor class

@dataclass
class Extractor:
    model_name: str = "gpt2-medium"
    device: str = DEFAULT_DEVICE
    max_len: int = 64 # figures for now, historical figure names r short

    def __post_init__(self):
        self.model: HookedTransformer = HookedTransformer.from_pretrained(
            self.model_name, device=self.device)
        tok = self.model.tokenizer
        assert tok is not None, "TransformerLens returned None tokenizer"
        self.tokenizer: PreTrainedTokenizerBase = tok  # type: ignore
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"

    def _batch_tokens(self, texts: Sequence[str]) -> torch.Tensor:
        tok_out = self.tokenizer(
            list(texts), return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_len)
        return tok_out["input_ids"].to(self.device)

    def features(
        self,
        texts: Sequence[str],
        layer: int,
        component: Comp | str = Comp.resid_pre,
        agg: str = "mean",
        batch_size: int = 300,
    ) -> np.ndarray:
        comp = Comp(component) if not isinstance(component, Comp) else component
        hk   = hook_name(layer, comp)
    
        out_list: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size)):
            toks = self._batch_tokens(texts[start : start + batch_size])
            _, cache = self.model.run_with_cache(toks, names_filter=[hk])
            act = cache[hk]
    
            if comp == Comp.attn_pattern:
                act = act.flatten(start_dim=1, end_dim=2)
            if comp in {Comp.attn_q, Comp.attn_k, Comp.attn_v}:
                act = act.flatten(start_dim=2)
    
            out_list.append(aggregate(act, mode=agg))
            # Free GPU memory
            del cache
            torch.cuda.empty_cache()
    
        return np.concatenate(out_list, axis=0)


