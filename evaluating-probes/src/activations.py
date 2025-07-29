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
            # If we have missing texts and the cache appears corrupted, try a clean reset
            if self._is_cache_corrupted(act_path, hash_path, shape_path):
                print(f"Warning: Cache appears corrupted, performing clean reset...")
                self._reset_cache(act_path, hash_path, shape_path)
                index = {}  # Start with empty index
                # All texts are now missing since we reset
                missing = [(h, p) for h, p in zip(hashes, texts)]
            
            self._append_new(layer, component, missing, index, act_path, hash_path, shape_path)
            # After appending, rebuild the index to ensure consistency
            index = self._build_index(hash_path)

        # read activations in requested order
        act_mm = self._open_act(act_path, shape_path, writable=False)
        
        # Validate that all requested indices are within bounds
        rows = []
        invalid_hashes = []
        for h in hashes:
            if h not in index:
                invalid_hashes.append(h)
                continue
            row_idx = index[h]
            if row_idx >= act_mm.shape[0]:
                invalid_hashes.append(h)
                continue
            rows.append(row_idx)
        
        # If we have invalid hashes, try to rebuild the index from the actual files
        if invalid_hashes:
            print(f"Warning: Found {len(invalid_hashes)} invalid hashes, rebuilding index from files...")
            index = self._rebuild_index_from_files(hash_path, act_path, shape_path)
            
            # Try again with the rebuilt index
            rows = []
            still_invalid = []
            for h in hashes:
                if h not in index:
                    still_invalid.append(h)
                    continue
                row_idx = index[h]
                if row_idx >= act_mm.shape[0]:
                    still_invalid.append(h)
                    continue
                rows.append(row_idx)
            
            if still_invalid:
                raise ValueError(f"After rebuilding index, still have {len(still_invalid)} invalid hashes. "
                               f"This suggests the requested texts are not in the cache.")
        
        return act_mm[rows]

    def _is_cache_corrupted(self, act_path: Path, hash_path: Path, shape_path: Path) -> bool:
        """Check if the cache appears to be corrupted."""
        if not act_path.exists() or not hash_path.exists():
            return False
        
        try:
            # Check if shape metadata matches actual file sizes
            if shape_path.exists():
                import json
                shape_data = json.loads(shape_path.read_text())
                recorded_rows = shape_data["rows"]
                
                actual_act_size = act_path.stat().st_size
                expected_act_size = recorded_rows * self._row_bytes
                
                actual_hash_size = hash_path.stat().st_size
                expected_hash_size = recorded_rows * self.hash_dtype.itemsize
                
                # If either file size is significantly off, consider it corrupted
                if (abs(actual_act_size - expected_act_size) > 1024 or 
                    abs(actual_hash_size - expected_hash_size) > 1024):
                    return True
            
            return False
        except Exception:
            return True

    def _reset_cache(self, act_path: Path, hash_path: Path, shape_path: Path):
        """Completely reset the cache by deleting all files."""
        print(f"  Resetting cache: deleting {act_path}, {hash_path}, {shape_path}")
        act_path.unlink(missing_ok=True)
        hash_path.unlink(missing_ok=True)
        shape_path.unlink(missing_ok=True)
        # Also clean up any temporary files
        act_path.with_suffix(".tmp").unlink(missing_ok=True)
        hash_path.with_suffix(".tmp").unlink(missing_ok=True)

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
        try:
            hashes = np.memmap(hash_path, dtype=self.hash_dtype, mode="r")
            return {h.decode(): i for i, h in enumerate(hashes)}
        except Exception as e:
            print(f"Warning: Error reading hash file {hash_path}: {e}")
            print(f"  Returning empty index")
            return {}

    def _rebuild_index_from_files(self, hash_path: Path, act_path: Path, shape_path: Path) -> Dict[str, int]:
        """Rebuild the index from the actual files, handling any corruption."""
        if not hash_path.exists() or not act_path.exists():
            return {}
        
        try:
            # Get the actual number of rows from the activation file
            actual_file_size = act_path.stat().st_size
            actual_rows = actual_file_size // self._row_bytes
            
            # Read the hash file with the correct size
            hashes = np.memmap(hash_path, dtype=self.hash_dtype, mode="r", shape=(actual_rows,))
            index = {h.decode(): i for i, h in enumerate(hashes)}
            
            # Update the shape file to match
            import json
            shape_path.write_text(json.dumps({"rows": actual_rows}))
            
            print(f"  Rebuilt index with {len(index)} entries from actual file size")
            return index
        except Exception as e:
            print(f"Error rebuilding index: {e}")
            return {}

    # ~~~ mmap open helpers ~~~
    def _rows_from_shape(self, shape_path: Path, act_path: Path) -> int:
        if shape_path.exists():
            import json
            try:
                shape_data = json.loads(shape_path.read_text())
                recorded_rows = shape_data["rows"]
                
                # Validate that the recorded rows match the actual file size
                if act_path.exists():
                    actual_file_size = act_path.stat().st_size
                    expected_file_size = recorded_rows * self._row_bytes
                    
                    # If file size doesn't match, the metadata is corrupted
                    if abs(actual_file_size - expected_file_size) > 1024:  # Allow 1KB tolerance
                        print(f"Warning: Shape metadata mismatch for {act_path}")
                        print(f"  Recorded rows: {recorded_rows}, expected size: {expected_file_size}")
                        print(f"  Actual file size: {actual_file_size}")
                        print(f"  Recalculating from file size...")
                        
                        # Recalculate from actual file size
                        actual_rows = actual_file_size // self._row_bytes
                        # Update the shape file with correct data
                        shape_path.write_text(json.dumps({"rows": actual_rows}))
                        return actual_rows
                
                return recorded_rows
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Corrupted shape file {shape_path}: {e}")
                # Fall through to file size calculation
        
        # Calculate from file size (or 0 if file doesn't exist)
        rows = act_path.stat().st_size // self._row_bytes if act_path.exists() else 0
        # Create/update shape file
        shape_path.write_text(json.dumps({"rows": rows}))
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
        bs: int = 64,
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

        # update in‑memory index - but only if the hash file wasn't reset
        # If the hash file was reset (old_rows == 0), we need to rebuild the entire index
        if old_rows == 0:
            # Hash file was reset, so we need to rebuild the index from scratch
            # The new hashes start at index 0
            for off, h in enumerate(new_hashes):
                index[h] = off
        else:
            # Normal case: append to existing index
            for off, h in enumerate(new_hashes):
                index[h] = old_rows + off

    # ---- grow helpers ----
    def _grow_act(self, act_path: Path, old_rows: int, new_rows: int):
        shape = (new_rows, self.max_len, self.d_model)
        if not act_path.exists():
            return np.memmap(act_path, dtype=np.float16, mode="w+", shape=shape)
        
        # Safety check: verify the file size matches expected size
        actual_file_size = act_path.stat().st_size
        expected_file_size = old_rows * self._row_bytes
        
        if abs(actual_file_size - expected_file_size) > 1024:  # Allow 1KB tolerance
            print(f"Warning: File size mismatch in _grow_act for {act_path}")
            print(f"  Expected size for {old_rows} rows: {expected_file_size}")
            print(f"  Actual file size: {actual_file_size}")
            print(f"  Recalculating actual rows...")
            
            # Recalculate actual rows from file size
            actual_old_rows = actual_file_size // self._row_bytes
            if actual_old_rows == 0:
                # File is empty or corrupted, start fresh
                print(f"  File appears empty/corrupted, starting fresh")
                act_path.unlink(missing_ok=True)
                return np.memmap(act_path, dtype=np.float16, mode="w+", shape=shape)
            
            # Use actual file size for copying
            old_rows = actual_old_rows
        
        try:
            old_mm = np.memmap(act_path, dtype=np.float16, mode="r", shape=(old_rows, self.max_len, self.d_model))
            tmp = np.memmap(act_path.with_suffix(".tmp"), dtype=np.float16, mode="w+", shape=shape)
            tmp[:old_rows] = old_mm[:]
            act_path.unlink(); tmp.flush(); tmp._mmap.close()
            act_path.with_suffix(".tmp").rename(act_path)
            return np.memmap(act_path, dtype=np.float16, mode="r+", shape=shape)
        except Exception as e:
            print(f"Error in _grow_act: {e}")
            print(f"  Attempting to recover by starting fresh...")
            # If anything goes wrong, start fresh
            act_path.unlink(missing_ok=True)
            act_path.with_suffix(".tmp").unlink(missing_ok=True)
            return np.memmap(act_path, dtype=np.float16, mode="w+", shape=shape)

    def _grow_hash(self, hash_path: Path, old_rows: int, new_rows: int):
        shape = (new_rows,)
        if not hash_path.exists():
            return np.memmap(hash_path, dtype=self.hash_dtype, mode="w+", shape=shape)
        
        # Safety check: verify the file size matches expected size
        actual_file_size = hash_path.stat().st_size
        expected_file_size = old_rows * self.hash_dtype.itemsize
        
        if abs(actual_file_size - expected_file_size) > 1024:  # Allow 1KB tolerance
            print(f"Warning: Hash file size mismatch in _grow_hash for {hash_path}")
            print(f"  Expected size for {old_rows} rows: {expected_file_size}")
            print(f"  Actual file size: {actual_file_size}")
            print(f"  Recalculating actual rows...")
            
            # Recalculate actual rows from file size
            actual_old_rows = actual_file_size // self.hash_dtype.itemsize
            if actual_old_rows == 0:
                # File is empty or corrupted, start fresh
                print(f"  Hash file appears empty/corrupted, starting fresh")
                hash_path.unlink(missing_ok=True)
                return np.memmap(hash_path, dtype=self.hash_dtype, mode="w+", shape=shape)
            
            # Use actual file size for copying, but don't reset the file
            # This preserves existing hashes
            old_rows = actual_old_rows
        
        try:
            old_h = np.memmap(hash_path, dtype=self.hash_dtype, mode="r", shape=(old_rows,))
            tmp = np.memmap(hash_path.with_suffix(".tmp"), dtype=self.hash_dtype, mode="w+", shape=shape)
            tmp[:old_rows] = old_h[:]
            hash_path.unlink(); tmp.flush(); tmp._mmap.close()
            hash_path.with_suffix(".tmp").rename(hash_path)
            return np.memmap(hash_path, dtype=self.hash_dtype, mode="r+", shape=shape)
        except Exception as e:
            print(f"Error in _grow_hash: {e}")
            print(f"  Attempting to recover by starting fresh...")
            # If anything goes wrong, start fresh
            hash_path.unlink(missing_ok=True)
            hash_path.with_suffix(".tmp").unlink(missing_ok=True)
            return np.memmap(hash_path, dtype=self.hash_dtype, mode="w+", shape=shape)
