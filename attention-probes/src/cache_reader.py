import torch

class ActStore:
    def __init__(self, path):          # e.g. data/act_cache/7_hist_fig.pt
        self._data: dict[str, torch.Tensor] = torch.load(path, mmap=True)

    def get(self, layer: int, comp: str) -> torch.Tensor:
        """
        comp in {"resid_post", "attn_q", "attn_pattern", ...}
        Returns tensor shape (N, seq, …) on CPU; convert/reshape as needed.
        """
        if comp == "resid_post":
            return self._data[f"blocks.{layer}.hook_resid_post"]
        if comp == "attn_q":
            return self._data[f"blocks.{layer}.attn.hook_q"]
        if comp == "pattern":
            return self._data[f"blocks.{layer}.attn.hook_attn_probs"]
        raise KeyError(comp)
