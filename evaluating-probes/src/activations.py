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
    """Single‑cache design with a *binary hash index* memmap and token lengths.

    For each (dataset, layer, component) we store three memmaps side by side:

    1. **`<hook>.mmap`**   – float16 activations  (rows × max_len × d_model)
    2. **`<hook>.hash.mmap`** – `S40` (40‑byte hex) prompt hashes (rows,)
    3. **`<hook>.length.mmap`** – `uint16` token lengths (rows,)

    The hash memmap lets us rebuild a dict ⟨hash → row⟩ in O(rows) once, then
    we do constant‑time membership checks to detect which prompts still need
    extraction. The length memmap stores the actual token count for each prompt.
    """

    _HASH_DTYPE = np.dtype("S40")  # 40‑byte ascii hex sha1
    _LENGTH_DTYPE = np.dtype("uint16")  # 16-bit unsigned integer for token lengths

    # init #
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

    # public entrypoint #
    def get_activations_for_texts(
        self,
        texts: Iterable[str],
        layer: int,
        component: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return activations (N, S, D) and attention masks (N, S) in the exact order of `texts`."""
        texts = list(texts)
        act_path, hash_path, length_path, shape_path = self._paths(layer, component)

        # load existing index into memory – {hash:str → row:int}
        index = self._get_cached_index(hash_path)

        hashes = [self._hash(p) for p in texts]
        missing = [(h, p) for h, p in zip(hashes, texts) if h not in index]

        if missing:
            act_mm = self._append_new(layer, component, missing, index, act_path, hash_path, length_path, shape_path)
            # Clear cache since we modified the hash file
            self._index_cache.pop(hash_path, None)
            rows = [index[h] for h in hashes]
            return self._get_activations_with_masks(act_mm, length_path, rows)
        else:
            # No new activations needed, use existing memmap
            print(f"[DEBUG] No new activations needed, using existing memmap")
            act_mm = self._open_act(act_path, shape_path, writable=False)
            rows = [index[h] for h in hashes]
            return self._get_activations_with_masks(act_mm, length_path, rows)

    def get_actual_max_len(self, layer: int, component: str) -> int:
        """Get the actual maximum token length from the length mmap for this layer/component."""
        _, _, length_path, _ = self._paths(layer, component)
        
        if not length_path.exists():
            return None  # No activations cached yet
        
        # Load the length data and find the maximum
        length_mm = np.memmap(length_path, dtype=self._LENGTH_DTYPE, mode="r")
        if len(length_mm) == 0:
            return None
        
        # The lengths stored are the actual token counts (no padding)
        actual_max_len = int(length_mm.max())
        return actual_max_len

    # internals #
    # ~~~ path helpers ~~~
    def _paths(self, layer: int, component: str) -> Tuple[Path, Path, Path, Path]:
        hook = f"blocks.{layer}.hook_{component}".replace(".", "‑")
        act = self.cache_dir / f"{hook}.mmap"
        return act, act.with_suffix(".hash.mmap"), act.with_suffix(".length.mmap"), act.with_suffix(".shape.json")

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
    
    def clear_activation_cache(self):
        """Clear activation cache to free memory. Call this when done with activations."""
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        # Clear any cached lazy loaders
        if hasattr(self, '_lazy_loaders'):
            for loader in self._lazy_loaders.values():
                if hasattr(loader, 'clear_cache'):
                    loader.clear_cache()
            self._lazy_loaders.clear()

    def _selective_flush(self, act_mm, hash_mm, old_rows, new_rows, max_length):
        """Fast flush that skips explicit flush for small additions."""
        start_time = time.time()
        
        print(f"[DEBUG] Fast flush: handling {new_rows - old_rows} new rows")
        
        # For small additions, skip explicit flush and let OS handle it
        if new_rows - old_rows <= 100:  # Small addition, arbitrary
            print(f"[DEBUG] Skipping explicit flush for small addition ({new_rows - old_rows} rows)")
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

    def _open_length(self, length_path: Path, rows: int, *, writable: bool):
        if rows == 0 and not writable:
            return np.empty((0,), dtype=self._LENGTH_DTYPE)
        mode = "r+" if writable else "r"
        return np.memmap(length_path, dtype=self._LENGTH_DTYPE, mode=mode, shape=(rows,))

    # ~~~ extraction & append ~~~
    def _append_new(
        self,
        layer: int,
        component: str,
        missing: List[Tuple[str, str]],  # (hash, prompt)
        index: Dict[str, int],
        act_path: Path,
        hash_path: Path,
        length_path: Path,
        shape_path: Path,
        bs: int = 10,
    ):
        if not missing:
            return
        new_hashes, new_prompts = zip(*missing)
        N_new = len(new_prompts)

        print(f"[DEBUG] Appending {N_new} new activations")

        # Define max_length at the beginning to avoid UnboundLocalError
        max_length = min(self.max_len, 4096)
        
        old_rows = self._rows_from_shape(shape_path, act_path, max_length)
        new_rows = old_rows + N_new
        # --- resize / create activation mmap ---
        # Use truncated max_length for memmap to match actual activation shapes
        act_mm = self._grow_act(act_path, old_rows, new_rows, max_length)
        hash_mm = self._grow_hash(hash_path, old_rows, new_rows)
        length_mm = self._grow_length(length_path, old_rows, new_rows)

        # --- extract activations for the new prompts ---
        hook = f"blocks.{layer}.hook_{component}"
        
        for s in tqdm(range(0, N_new, bs), desc=f"Extract L{layer} {component}"):
            batch_prompts = list(new_prompts[s : s + bs])
            toks = self.tokenizer(batch_prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                _, cache = self.model.run_with_cache(toks["input_ids"], names_filter=[hook], device=self.device)
            acts = cache[hook].cpu().to(torch.float16).numpy()
            
            # Calculate actual token lengths (sum of attention mask)
            # This works correctly for left padding since attention mask has 1s for actual tokens
            batch_lengths = toks["attention_mask"].sum(dim=1).cpu().numpy()
            
            # Store only the actual token activations (no padding)
            for i, (act, length) in enumerate(zip(acts, batch_lengths)):
                actual_act = act[:length]  # Only keep actual tokens
                row_idx = old_rows + s + i
                
                # Store the actual activation (padded to max_length for mmap compatibility)
                if actual_act.shape[0] < max_length:
                    # Pad with zeros to max_length
                    padded_act = np.pad(actual_act, ((0, max_length - actual_act.shape[0]), (0, 0)), 
                                      mode='constant', constant_values=0)
                else:
                    padded_act = actual_act[:max_length]
                
                act_mm[row_idx] = padded_act
                hash_mm[row_idx] = new_hashes[s + i].encode()
                length_mm[row_idx] = min(length, max_length)  # Store actual length (capped at max_length)
            
            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # flush & close
        act_mm.flush(); hash_mm.flush(); length_mm.flush()
        # self._selective_flush(act_mm, hash_mm, old_rows, new_rows, max_length)
        shape_path.write_text(json.dumps({"rows": new_rows}))

        # update in‑memory index
        for off, h in enumerate(new_hashes):
            index[h] = old_rows + off

        return act_mm # avoid recreating after appending

    def _get_activations_with_masks(self, act_mm, length_path: Path, rows: list) -> Tuple[np.ndarray, np.ndarray]:
        """Return activations padded to actual max_length and attention masks."""
        if not rows:
            raise ValueError("No rows to get activations for")
        
        # Load the length data
        length_mm = np.memmap(length_path, dtype=self._LENGTH_DTYPE, mode="r")
        
        # Get the activations for the requested rows
        activations = act_mm[rows]  # Shape: (N, max_len, d_model)
        lengths = length_mm[rows]   # Shape: (N,)
        
        # Find the actual maximum length in this batch
        actual_max_len = int(lengths.max())
        
        # Trim activations to actual lengths and create attention masks
        trimmed_activations = []
        attention_masks = []
        
        for i, (act, length) in enumerate(zip(activations, lengths)):
            # Trim activation to actual length (remove padding)
            actual_act = act[:length]  # Shape: (actual_length, d_model)
            
            # Pad to the batch's max_length
            if actual_act.shape[0] < actual_max_len:
                # Pad with zeros
                padding_size = actual_max_len - actual_act.shape[0]
                padded_act = np.pad(actual_act, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
            else:
                # Already at max length
                padded_act = actual_act
            
            trimmed_activations.append(padded_act)
            
            # Create attention mask: 1 for actual tokens, 0 for padding
            mask = np.zeros(actual_max_len, dtype=bool)
            mask[:length] = True  # First 'length' tokens are actual tokens
            attention_masks.append(mask)
        
        # Stack into arrays
        activations_array = np.stack(trimmed_activations) # Shape: (N, actual_max_len, d_model)
        masks_array = np.stack(attention_masks)           # Shape: (N, actual_max_len)
        
        return activations_array, masks_array

    # ---- grow helpers ----
    def _grow_act(self, act_path: Path, old_rows: int, new_rows: int, max_length: int = None):
        if max_length is None:
            max_length = min(self.max_len, 4096)  # Use truncated length by default
        shape = (new_rows, max_length, self.d_model)
        
        if not act_path.exists():
            return np.memmap(act_path, dtype=np.float16, mode="w+", shape=shape)
        
        # Always use append-only growth for simplicity
        print(f"[DEBUG] Using append-only growth for activation file (adding {new_rows - old_rows} rows)")
        return self._append_grow_act(act_path, old_rows, new_rows, max_length)

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

    def _grow_hash(self, hash_path: Path, old_rows: int, new_rows: int):
        shape = (new_rows,)
        if not hash_path.exists():
            return np.memmap(hash_path, dtype=self.hash_dtype, mode="w+", shape=shape)
        # Always use append-only growth for simplicity
        print(f"[DEBUG] Using append-only growth for hash file (adding {new_rows - old_rows} rows)")
        return self._append_grow_hash(hash_path, old_rows, new_rows)

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

    def _grow_length(self, length_path: Path, old_rows: int, new_rows: int):
        shape = (new_rows,)
        if not length_path.exists():
            return np.memmap(length_path, dtype=self._LENGTH_DTYPE, mode="w+", shape=shape)  
        print(f"[DEBUG] Using append-only growth for length file (adding {new_rows - old_rows} rows)")
        return self._append_grow_length(length_path, old_rows, new_rows)

    def _append_grow_length(self, length_path: Path, old_rows: int, new_rows: int):
        """Append-only growth for length file."""
        # Open existing file in append mode
        existing_mm = np.memmap(length_path, dtype=self._LENGTH_DTYPE, mode="r+", shape=(old_rows,))
        
        # Calculate the new file size needed
        length_size = 2 # uint16 = 2 bytes
        bytes_to_add = (new_rows - old_rows) * length_size
        
        # Extend the file by writing zeros to the end
        with open(length_path, 'ab') as f:
            f.write(b'\x00' * bytes_to_add)
        
        # Reopen with new shape
        return np.memmap(length_path, dtype=self._LENGTH_DTYPE, mode="r+", shape=(new_rows,))

class LazyActivationLoader:
    """Efficient GPU-optimized loader for large activation files."""
    
    def __init__(self, act_path: Path, total_rows: int, max_length: int, d_model: int, use_cache: bool = False):
        self.act_path = act_path
        self.total_rows = total_rows
        self.max_length = max_length
        self.d_model = d_model
        self.row_size = max_length * d_model * 2  # float16 = 2 bytes
        self.use_cache = use_cache
        
        # Use memory mapping for efficient access
        self._mmap = np.memmap(act_path, dtype=np.float16, mode='r', 
                              shape=(total_rows, max_length, d_model))
        
        # Cache for frequently accessed rows (small cache to avoid memory explosion)
        if self.use_cache:
            self._cache = {}
            self._cache_size_limit = 1000  # Limit cache size
        
    def __getitem__(self, key):
        if isinstance(key, int):
            # Single row
            return self._load_row(key)
        elif isinstance(key, list):
            # List of row indices - load in batch for efficiency
            return self._load_rows_batch(key)
        elif isinstance(key, slice):
            # Slice of rows - use direct memmap slicing
            start, stop, step = key.indices(self.total_rows)
            if step == 1:  # Contiguous slice - most efficient
                return self._mmap[start:stop].copy()  # Copy to avoid memmap issues
            else:
                indices = list(range(start, stop, step))
                return self._load_rows_batch(indices)
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")
    
    def _load_row(self, row_idx: int):
        """Load a single row with caching."""
        if self.use_cache and hasattr(self, '_cache') and row_idx in self._cache:
            return self._cache[row_idx]
        
        if row_idx >= self.total_rows:
            raise IndexError(f"Row index {row_idx} out of bounds (max: {self.total_rows-1})")
        
        # Use memmap for efficient access
        row_data = self._mmap[row_idx].copy()  # Copy to avoid memmap issues
        
        # Cache with size limit
        if self.use_cache and hasattr(self, '_cache') and len(self._cache) < self._cache_size_limit:
            self._cache[row_idx] = row_data
        
        return row_data
    
    def _load_rows_batch(self, indices: list):
        """Load multiple rows efficiently in batch."""
        if not indices:
            return np.empty((0, self.max_length, self.d_model), dtype=np.float16)
        
        # Check cache first
        cached_rows = {}
        uncached_indices = []
        
        if self.use_cache and hasattr(self, '_cache'):
            for idx in indices:
                if idx in self._cache:
                    cached_rows[idx] = self._cache[idx]
                else:
                    uncached_indices.append(idx)
        else:
            uncached_indices = indices
        
        # Load uncached rows in batch using memmap
        if uncached_indices:
            # Use advanced indexing for batch loading
            batch_data = self._mmap[uncached_indices].copy()
            
            # Cache the new rows
            if self.use_cache and hasattr(self, '_cache'):
                for i, idx in enumerate(uncached_indices):
                    if len(self._cache) < self._cache_size_limit:
                        self._cache[idx] = batch_data[i]
                    cached_rows[idx] = batch_data[i]
            else:
                for i, idx in enumerate(uncached_indices):
                    cached_rows[idx] = batch_data[i]
        
        # Return in the original order
        result = np.stack([cached_rows[idx] for idx in indices])
        return result
    
    def clear_cache(self):
        """Clear the row cache to free memory."""
        if self.use_cache and hasattr(self, '_cache'):
            self._cache.clear()
    
    def __del__(self):
        """Clean up memmap when object is destroyed."""
        if hasattr(self, '_mmap'):
            del self._mmap
