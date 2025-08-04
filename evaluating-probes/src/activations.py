from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple, Iterable, Dict
import time

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
import json


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
        self._update_row_bytes()
        self.hash_dtype = hash_dtype
        
        # Cache for hash indices to avoid rebuilding them every time
        self._index_cache = {}
        self._hash_file_mtimes = {}

    def _update_row_bytes(self):
        """Recalculate _row_bytes when max_len changes."""
        self._row_bytes = self.max_len * self.d_model * np.dtype(np.float16).itemsize

    @property
    def max_len(self):
        return self._max_len

    @max_len.setter
    def max_len(self, value):
        self._max_len = int(value)
        self._update_row_bytes()

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
        index = self._get_cached_index(hash_path)

        hashes = [self._hash(p) for p in texts]
        missing = [(h, p) for h, p in zip(hashes, texts) if h not in index]

        if missing:
            act_mm = self._append_new(layer, component, missing, index, act_path, hash_path, shape_path)
            # Clear cache since we modified the hash file
            self._index_cache.pop(hash_path, None)
            rows = [index[h] for h in hashes]
            return act_mm[rows]
        else:
            # No new activations needed, use existing memmap
            act_mm = self._open_act(act_path, shape_path, writable=False)
            rows = [index[h] for h in hashes]
            return act_mm[rows]

        # # read activations in requested order
        # act_mm = self._open_act(act_path, shape_path, writable=False)
        # rows = [index[h] for h in hashes]
        # return act_mm[rows]

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

    def _get_cached_index(self, hash_path: Path) -> Dict[str, int]:
        """Get cached index, rebuilding only if hash file has changed."""
        if not hash_path.exists():
            return {}
        
        # Check if file has been modified since last cache
        current_mtime = hash_path.stat().st_mtime
        cached_mtime = self._hash_file_mtimes.get(hash_path)
        
        if hash_path in self._index_cache and cached_mtime == current_mtime:
            print(f"[DEBUG] Using cached index for {hash_path.name} ({len(self._index_cache[hash_path])} entries)")
            return self._index_cache[hash_path]
        
        # Rebuild index
        print(f"[DEBUG] Rebuilding index for {hash_path.name}")
        hashes = np.memmap(hash_path, dtype=self.hash_dtype, mode="r")
        index = {h.decode(): i for i, h in enumerate(hashes)}
        
        # Cache the result
        self._index_cache[hash_path] = index
        self._hash_file_mtimes[hash_path] = current_mtime
        print(f"[DEBUG] Cached index for {hash_path.name} ({len(index)} entries)")
        
        return index

    def _build_index(self, hash_path: Path) -> Dict[str, int]:
        """Deprecated: use _get_cached_index instead."""
        return self._get_cached_index(hash_path)

    def clear_index_cache(self):
        """Clear the cached hash indices. Useful if memory usage becomes an issue."""
        self._index_cache.clear()
        self._hash_file_mtimes.clear()

    def clear_all_caches(self):
        """Clear all caches (index and lazy loader caches)."""
        self.clear_index_cache()
        # Note: LazyActivationLoader instances are not stored in ActivationManager
        # They are created per-request and should be garbage collected automatically

    def _selective_flush(self, act_mm, hash_mm, old_rows, new_rows, max_length):
        """Fast flush that skips explicit flush for small additions."""
        start_time = time.time()
        
        print(f"[DEBUG] Fast flush: handling {new_rows - old_rows} new rows")
        
        # For small additions, skip explicit flush and let OS handle it
        if new_rows - old_rows <= 100:  # Small addition, arbitrary
            print(f"[DEBUG] Skipping explicit flush for small addition ({new_rows - old_rows} rows)")
            print(f"[DEBUG] OS will handle flush when memmap is closed")
        else:
            # For larger additions, use the original flush method
            print(f"[DEBUG] Using full flush for larger addition ({new_rows - old_rows} rows)")
            act_mm.flush()
            hash_mm.flush()
        
        elapsed = time.time() - start_time
        print(f"[DEBUG] Fast flush took {elapsed:.2f} seconds")
    
    # ~~~ mmap open helpers ~~~
    def _rows_from_shape(self, shape_path: Path, act_path: Path, max_length: int = None) -> int:
        if max_length is None:
            max_length = min(self.max_len, 4096)  # Use truncated length by default
        if shape_path.exists():
            return json.loads(shape_path.read_text())["rows"]
        # infer using the actual max_length
        row_bytes = max_length * self.d_model * np.dtype(np.float16).itemsize
        rows = act_path.stat().st_size // row_bytes if act_path.exists() else 0
        shape_path.write_text(json.dumps({"rows": rows}))
        return rows

    def _open_act(self, act_path: Path, shape_path: Path, *, writable: bool, max_length: int = None):
        if max_length is None:
            max_length = min(self.max_len, 4096)  # Use truncated length by default
        rows = self._rows_from_shape(shape_path, act_path, max_length)
        if rows == 0 and not writable:
            return np.empty((0, max_length, self.d_model), dtype=np.float16)
        
        # For read-only access, use lazy loading for large files
        if not writable and rows > 10000:  # Threshold for "large" files
            print(f"[DEBUG] Using lazy loading for large activation file ({rows} rows)")
            return LazyActivationLoader(act_path, rows, max_length, self.d_model)
        
        mode = "r+" if writable else "r"
        return np.memmap(act_path, dtype=np.float16, mode=mode, shape=(rows, max_length, self.d_model))

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
        bs: int = 1,
    ):
        if not missing:
            return
        new_hashes, new_prompts = zip(*missing)
        N_new = len(new_prompts)

        # Define max_length at the beginning to avoid UnboundLocalError
        max_length = min(self.max_len, 4096)
        
        old_rows = self._rows_from_shape(shape_path, act_path, max_length)
        new_rows = old_rows + N_new
        # --- resize / create activation mmap ---
        # Use truncated max_length for memmap to match actual activation shapes
        act_mm = self._grow_act(act_path, old_rows, new_rows, max_length)
        hash_mm = self._grow_hash(hash_path, old_rows, new_rows)

        # --- extract activations for the new prompts ---
        hook = f"blocks.{layer}.hook_{component}"
        
        for s in tqdm(range(0, N_new, bs), desc=f"Extract L{layer} {component}"):
            batch_prompts = list(new_prompts[s : s + bs])
            toks = self.tokenizer(batch_prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
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
        # self._selective_flush(act_mm, hash_mm, old_rows, new_rows, max_length)
        shape_path.write_text(json.dumps({"rows": new_rows}))

        # update in‑memory index
        for off, h in enumerate(new_hashes):
            index[h] = old_rows + off

        return act_mm # avoid recreating after appending

    # ---- grow helpers ----
    def _grow_act(self, act_path: Path, old_rows: int, new_rows: int, max_length: int = None):
        if max_length is None:
            max_length = min(self.max_len, 4096)  # Use truncated length by default
        shape = (new_rows, max_length, self.d_model)
        
        if not act_path.exists():
            return np.memmap(act_path, dtype=np.float16, mode="w+", shape=shape)
        
        # For small additions (< 10% of file size), use append-only growth
        growth_ratio = (new_rows - old_rows) / old_rows
        start_time = time.time()
        
        if growth_ratio < 0.1:  # Less than 10% growth
            print(f"[DEBUG] Using append-only growth for activation file (adding {new_rows - old_rows} rows)")
            result = self._append_grow_act(act_path, old_rows, new_rows, max_length)
        else:
            print(f"[DEBUG] Using full copy growth for activation file (adding {new_rows - old_rows} rows)")
            result = self._copy_grow_act(act_path, old_rows, new_rows, max_length)
        
        elapsed = time.time() - start_time
        print(f"[DEBUG] Activation file growth took {elapsed:.2f} seconds")
        return result

    def _append_grow_act(self, act_path: Path, old_rows: int, new_rows: int, max_length: int):
        """Append-only growth: extend the file without copying existing data."""
        # Open existing file in append mode
        existing_mm = np.memmap(act_path, dtype=np.float16, mode="r+", shape=(old_rows, max_length, self.d_model))
        
        # Calculate the new file size needed
        row_size = max_length * self.d_model * 2  # float16 = 2 bytes
        new_file_size = new_rows * row_size
        
        # Extend the file by writing zeros to the end
        with open(act_path, 'ab') as f:
            bytes_to_add = (new_rows - old_rows) * row_size
            f.write(b'\x00' * bytes_to_add)
        
        # Reopen with new shape
        return np.memmap(act_path, dtype=np.float16, mode="r+", shape=(new_rows, max_length, self.d_model))

    def _copy_grow_act(self, act_path: Path, old_rows: int, new_rows: int, max_length: int):
        """Full copy growth: copy entire file (used for large additions)."""
        old_mm = np.memmap(act_path, dtype=np.float16, mode="r", shape=(old_rows, max_length, self.d_model))
        tmp = np.memmap(act_path.with_suffix(".tmp"), dtype=np.float16, mode="w+", shape=(new_rows, max_length, self.d_model))
        tmp[:old_rows] = old_mm[:]
        act_path.unlink(); tmp.flush(); tmp._mmap.close()
        act_path.with_suffix(".tmp").rename(act_path)
        return np.memmap(act_path, dtype=np.float16, mode="r+", shape=(new_rows, max_length, self.d_model))

    def _grow_hash(self, hash_path: Path, old_rows: int, new_rows: int):
        shape = (new_rows,)
        if not hash_path.exists():
            return np.memmap(hash_path, dtype=self.hash_dtype, mode="w+", shape=shape)
        
        # For small additions, use append-only growth
        growth_ratio = (new_rows - old_rows) / old_rows
        start_time = time.time()
        
        if growth_ratio < 0.9:  # Less than 90% growth
            print(f"[DEBUG] Using append-only growth for hash file (adding {new_rows - old_rows} rows)")
            result = self._append_grow_hash(hash_path, old_rows, new_rows)
        else:
            print(f"[DEBUG] Using full copy growth for hash file (adding {new_rows - old_rows} rows)")
            result = self._copy_grow_hash(hash_path, old_rows, new_rows)
        
        elapsed = time.time() - start_time
        print(f"[DEBUG] Hash file growth took {elapsed:.2f} seconds")
        return result

    def _append_grow_hash(self, hash_path: Path, old_rows: int, new_rows: int):
        """Append-only growth for hash file."""
        # Open existing file in append mode
        existing_mm = np.memmap(hash_path, dtype=self.hash_dtype, mode="r+", shape=(old_rows,))
        
        # Calculate the new file size needed
        hash_size = 40  # S40 dtype = 40 bytes
        bytes_to_add = (new_rows - old_rows) * hash_size
        
        # Extend the file by writing zeros to the end
        with open(hash_path, 'ab') as f:
            f.write(b'\x00' * bytes_to_add)
        
        # Reopen with new shape
        return np.memmap(hash_path, dtype=self.hash_dtype, mode="r+", shape=(new_rows,))

    def _copy_grow_hash(self, hash_path: Path, old_rows: int, new_rows: int):
        """Full copy growth for hash file."""
        old_h = np.memmap(hash_path, dtype=self.hash_dtype, mode="r", shape=(old_rows,))
        tmp = np.memmap(hash_path.with_suffix(".tmp"), dtype=self.hash_dtype, mode="w+", shape=(new_rows,))
        tmp[:old_rows] = old_h[:]
        hash_path.unlink(); tmp.flush(); tmp._mmap.close()
        hash_path.with_suffix(".tmp").rename(hash_path)
        return np.memmap(hash_path, dtype=self.hash_dtype, mode="r+", shape=(new_rows,))

class LazyActivationLoader:
    """Lazy loader for large activation files that only loads requested rows."""
    
    def __init__(self, act_path: Path, total_rows: int, max_length: int, d_model: int):
        self.act_path = act_path
        self.total_rows = total_rows
        self.max_length = max_length
        self.d_model = d_model
        self.row_size = max_length * d_model * 2  # float16 = 2 bytes
        self._cache = {}
    
    def __getitem__(self, key):
        if isinstance(key, int):
            # Single row
            return self._load_row(key)
        elif isinstance(key, list):
            # List of row indices
            return np.stack([self._load_row(i) for i in key])
        elif isinstance(key, slice):
            # Slice of rows
            start, stop, step = key.indices(self.total_rows)
            indices = list(range(start, stop, step))
            return np.stack([self._load_row(i) for i in indices])
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")
    
    def _load_row(self, row_idx: int):
        """Load a single row from the file."""
        if row_idx in self._cache:
            return self._cache[row_idx]
        
        if row_idx >= self.total_rows:
            raise IndexError(f"Row index {row_idx} out of bounds (max: {self.total_rows-1})")
        
        # Calculate file offset for this row
        offset = row_idx * self.row_size
        
        # Read the row directly from file
        with open(self.act_path, 'rb') as f:
            f.seek(offset)
            data = f.read(self.row_size)
        
        # Convert to numpy array
        row_data = np.frombuffer(data, dtype=np.float16).reshape(self.max_length, self.d_model)
        
        # Cache the result
        self._cache[row_idx] = row_data
        return row_data
    
    def clear_cache(self):
        """Clear the row cache to free memory."""
        self._cache.clear()
