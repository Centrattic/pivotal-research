from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple, Iterable, Dict

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer


class ActivationManager:
    """Single‑cache design with a *binary hash index* memmap.

    For each (dataset, layer, component) we store two memmaps side by side:

    1. **`<hook>.mmap`**   – float16 activations  (rows × max_len × d_model)
    2. **`<hook>.hash.mmap`** – `S40` (40‑byte hex) prompt hashes (rows,)

    The hash memmap lets us rebuild a dict ⟨hash → row⟩ in O(rows) once, then
    we do constant‑time membership checks to detect which prompts still need
    extraction.
    """

    _HASH_DTYPE = np.dtype("S40")  # 40‑byte ascii hex sha1

    # ------------------------------- init ------------------------------ #
    def __init__(
        self,
        model: HookedTransformer,
        device: str,
        d_model: int,
        max_len: int,
        cache_dir: Path,
        *,
        hash_dtype=_HASH_DTYPE,
    ):
        self.model = model
        self.tokenizer = model.tokenizer
        assert self.tokenizer is not None, "Model must carry tokenizer"
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        self.device = device
        self.d_model = d_model
        self.max_len = int(max_len)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._row_bytes = self.max_len * self.d_model * np.dtype(np.float16).itemsize
        self.hash_dtype = hash_dtype

    # ------------------------ public entrypoint ------------------------ #
    def get_activations_for_texts(
        self,
        texts: Iterable[str],
        layer: int,
        component: str,
    ) -> np.ndarray:
        """Return activations (N, S, D) in the exact order of `texts`."""
        texts = list(texts)
        act_path, hash_path, shape_path = self._paths(layer, component)

        # load existing index into memory – {hash:str → row:int}
        index = self._build_index(hash_path)

        hashes = [self._hash(p) for p in texts]
        missing = [(h, p) for h, p in zip(hashes, texts) if h not in index]

        if missing:
            self._append_new(layer, component, missing, index, act_path, hash_path, shape_path)

        # read activations in requested order
        act_mm = self._open_act(act_path, shape_path, writable=False)
        rows = [index[h] for h in hashes]
        return act_mm[rows]

    # --------------------------- internals ----------------------------- #
    # ~~~ path helpers ~~~
    def _paths(self, layer: int, component: str) -> Tuple[Path, Path, Path]:
        hook = f"blocks.{layer}.hook_{component}".replace(".", "‑")
        act = self.cache_dir / f"{hook}.mmap"
        return act, act.with_suffix(".hash.mmap"), act.with_suffix(".shape.json")

    # ~~~ hashing & index ~~~
    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha1(prompt.encode()).hexdigest()

    def _build_index(self, hash_path: Path) -> Dict[str, int]:
        if not hash_path.exists():
            return {}
        hashes = np.memmap(hash_path, dtype=self.hash_dtype, mode="r")
        return {h.decode(): i for i, h in enumerate(hashes)}

    # ~~~ mmap open helpers ~~~
    def _rows_from_shape(self, shape_path: Path, act_path: Path) -> int:
        if shape_path.exists():
            import json
            return json.loads(shape_path.read_text())["rows"]
        # infer
        rows = act_path.stat().st_size // self._row_bytes if act_path.exists() else 0
        shape_path.write_text(f"{{\"rows\": {rows}}}")
        return rows

    def _open_act(self, act_path: Path, shape_path: Path, *, writable: bool):
        rows = self._rows_from_shape(shape_path, act_path)
        if rows == 0 and not writable:
            return np.empty((0, self.max_len, self.d_model), dtype=np.float16)
        mode = "r+" if writable else "r"
        return np.memmap(act_path, dtype=np.float16, mode=mode, shape=(rows, self.max_len, self.d_model))

    def _open_hash(self, hash_path: Path, rows: int, *, writable: bool):
        if rows == 0 and not writable:
            return np.empty((0,), dtype=self.hash_dtype)
        mode = "r+" if writable else "r"
        return np.memmap(hash_path, dtype=self.hash_dtype, mode=mode, shape=(rows,))

    # ~~~ extraction & append ~~~
    def _append_new(
        self,
        layer: int,
        component: str,
        missing: List[Tuple[str, str]],  # (hash, prompt)
        index: Dict[str, int],
        act_path: Path,
        hash_path: Path,
        shape_path: Path,
    ):
        if not missing:
            return
        new_hashes, new_prompts = zip(*missing)
        N_new = len(new_prompts)

        old_rows = self._rows_from_shape(shape_path, act_path)
        new_rows = old_rows + N_new

        # --- resize / create activation mmap ---
        act_mm = self._grow_act(act_path, old_rows, new_rows)
        hash_mm = self._grow_hash(hash_path, old_rows, new_rows)

        # --- extract activations for the new prompts ---
        hook = f"blocks.{layer}.hook_{component}"
        bs = 8
        for s in tqdm(range(0, N_new, bs), desc=f"Extract L{layer} {component}"):
            batch_prompts = list(new_prompts[s : s + bs])
            toks = self.tokenizer(batch_prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                _, cache = self.model.run_with_cache(toks["input_ids"], names_filter=[hook], device=self.device)
            acts = cache[hook].cpu().to(torch.float16).numpy()
            row_slice = slice(old_rows + s, old_rows + s + len(batch_prompts))
            act_mm[row_slice] = acts
            hash_mm[row_slice] = np.array(list(new_hashes[s : s + len(batch_prompts)]), dtype=self.hash_dtype)
            del cache; torch.cuda.empty_cache()

        # flush & close
        act_mm.flush(); hash_mm.flush()
        import json
        shape_path.write_text(json.dumps({"rows": new_rows}))

        # update in‑memory index
        for off, h in enumerate(new_hashes):
            index[h] = old_rows + off

    # ---- grow helpers ----
    def _grow_act(self, act_path: Path, old_rows: int, new_rows: int):
        shape = (new_rows, self.max_len, self.d_model)
        if not act_path.exists():
            return np.memmap(act_path, dtype=np.float16, mode="w+", shape=shape)
        old_mm = np.memmap(act_path, dtype=np.float16, mode="r", shape=(old_rows, self.max_len, self.d_model))
        tmp = np.memmap(act_path.with_suffix(".tmp"), dtype=np.float16, mode="w+", shape=shape)
        tmp[:old_rows] = old_mm[:]
        act_path.unlink(); tmp.flush(); tmp._mmap.close()
        act_path.with_suffix(".tmp").rename(act_path)
        return np.memmap(act_path, dtype=np.float16, mode="r+", shape=shape)

    def _grow_hash(self, hash_path: Path, old_rows: int, new_rows: int):
        shape = (new_rows,)
        if not hash_path.exists():
            return np.memmap(hash_path, dtype=self.hash_dtype, mode="w+", shape=shape)
        old_h = np.memmap(hash_path, dtype=self.hash_dtype, mode="r", shape=(old_rows,))
        tmp = np.memmap(hash_path.with_suffix(".tmp"), dtype=self.hash_dtype, mode="w+", shape=shape)
        tmp[:old_rows] = old_h[:]
        hash_path.unlink(); tmp.flush(); tmp._mmap.close()
        hash_path.with_suffix(".tmp").rename(hash_path)
        return np.memmap(hash_path, dtype=self.hash_dtype, mode="r+", shape=shape)
