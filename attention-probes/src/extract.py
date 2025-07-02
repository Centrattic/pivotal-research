from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from datasets import load_dataset, DatasetDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import transformer_lens as tl
import yaml

from params import *

def bucket_attention(vec: torch.Tensor, max_len: int, num_bins: int) -> torch.Tensor:
    """Compress variable‑length attention vector *vec* (T,) into *num_bins* bins.
    Uses simple average pooling."""
    if vec.shape[-1] < max_len:
        # pad to max_len so we can reshape evenly
        pad_len = max_len - vec.shape[-1]
        vec = torch.nn.functional.pad(vec, (0, pad_len))
    step = max_len // num_bins
    return vec.unfold(-1, step, step).mean(-1)

class AttentionFeatureExtractor:
    def __init__(self, model_name: str = HF_MODEL_NAME, max_seq_len: int = DEFAULT_MAX_LENGTH, bins: int = 16):
        self.model = tl.HookedTransformer.from_pretrained(model_name, device=DEVICE)
        self.tokenizer = self.model.tokenizer  # Huggingface GPT‑2 tokenizer
        self.max_seq_len = max_seq_len
        self.bins = bins
        self.last_layer = self.model.cfg.n_layers - 1

    def _get_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return last‑token attentions from final layer; shape (B, H, T)."""
        _, cache = self.model.run_with_cache(tokens, names_filter=[f'attn_pattern_{self.last_layer}'])
        pattern = cache[f'attn_pattern_{self.last_layer}']  # (B, H, T, T)
        last_tok = pattern[:, :, -1, :]  # (B, H, T)
        return last_tok

    def encode(self, texts: List[str]) -> torch.Tensor:
        # Tokenise with truncation
        tok = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_seq_len)
        tokens = tok['input_ids'].to(DEVICE)
        with torch.no_grad():
            attn = self._get_attention(tokens)  # (B, H, T)
            B, H, T = attn.shape
            feats = []
            for i in range(B):
                # bucket each head separately then concatenate
                head_feats = [bucket_attention(attn[i, h], self.max_seq_len, self.bins) for h in range(H)]
                feats.append(torch.cat(head_feats, dim=0))
            return torch.stack(feats).cpu()  # (B, H*bins)

