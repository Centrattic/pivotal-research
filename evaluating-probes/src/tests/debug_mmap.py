# src/dump_mmap.py
from pathlib import Path
import numpy as np
import csv
import sys


def dump_mmap_to_csv(
    mmap_path: str | Path,
    csv_out: str | Path,
    *,
    dtype: str = "float16",
    chunk_size: int = 10_000_000,   # 10 M values → ≈ 20 MB per chunk for float16
    preview_rows: int = 50,
) -> None:
    """Stream an activation .mmap file to CSV without knowing its shape."""
    mmap_path, csv_out = Path(mmap_path), Path(csv_out)
    if not mmap_path.exists():
        print(f"❌  File not found: {mmap_path}", file=sys.stdout, flush=True)
        return

    # Open as flat mem-map
    item_sz = np.dtype(dtype).itemsize
    total_vals = mmap_path.stat().st_size // item_sz
    flat = np.memmap(
        mmap_path,
        dtype=dtype,
        mode="r",
        shape=(total_vals,
               )
    )

    print(
        f"{mmap_path} : {total_vals:,} {dtype} values "
        f"({mmap_path.stat().st_size/1e6:.1f} MB)",
        file=sys.stdout,
        flush=True,
    )

    # # --- stream to CSV -------------------------------------------------- #
    # with open(csv_out, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     written = 0
    #     for start in range(0, total_vals, chunk_size):
    #         end = min(start + chunk_size, total_vals)
    #         writer.writerows([[v] for v in flat[start:end]])
    #         written += end - start
    #         print(f"  wrote {written:,}/{total_vals:,} values",
    #               file=sys.stdout, flush=True)

    # print(f" Full CSV written → {csv_out}", file=sys.stdout, flush=True)

    # --- preview -------------------------------------------------------- #
    preview_path = csv_out.with_suffix(".preview.txt")
    np.savetxt(preview_path, flat[:preview_rows], fmt="%.6f")
    print(f"🔍  Preview (first {preview_rows} rows) → {preview_path}", file=sys.stdout, flush=True)


# ----------------------------------------------------------------------- #
# Edit only these two paths, then run:  python -m src.dump_mmap
if __name__ == "__main__":
    mmap_file = "activation_cache/gemma-2-9b/33_truthqa_tf/blocks-20-hook_resid_post.mmap"
    csv_file = "blocks20_raw.csv"

    dump_mmap_to_csv(mmap_file, csv_file)
