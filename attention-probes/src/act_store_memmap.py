# act_store_memmap.py
import os, re, numpy as np

def file_name(dataset: str, layer: int, component: str,
              cache_dir: str = "data/act_cache") -> str:
    """
    component  ∈ {resid_post, attn_q, attn_k, attn_v}
    """
    if component == "resid_post":
        hook = f"blocks-{layer}-hook_resid_post"
    elif component in {"attn_q", "attn_k", "attn_v"}:
        hook = f"blocks-{layer}-attn-hook_{component.split('_')[-1]}"
    elif component == "embed":
        hook = "hook_embed"
    else:
        raise ValueError(component)

    pattern = f"{dataset}__{hook}.mmp"
    path    = os.path.join(cache_dir, pattern)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def load_memmap(path: str, max_len: int, d_last: int,
                dtype=np.float16) -> np.memmap:
    """
    Opens *.mmp written in FP16 with shape (N, S, D_last...) .
    You supply max_len and trailing feature dimension.
    """
    numel = os.path.getsize(path) // np.dtype(dtype).itemsize
    N = numel // (max_len * d_last)
    shape = (N, max_len, d_last)
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)
