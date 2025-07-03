# ── build_cache.py ───────────────────────────────────────────────────────────
import os, torch, argparse, glob, random, numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm
import pandas as pd


HOOKS = [
    "hook_embed",
    *[f"blocks.{l}.hook_resid_post" for l in range(12)],
    *[f"blocks.{l}.attn.hook_q"     for l in range(12)],
    *[f"blocks.{l}.attn.hook_k"     for l in range(12)],
    # *[f"blocks.{l}.attn.hook_v"     for l in range(12)],
]

DTYPE   = np.float16          # disk dtype
MAX_LEN = 32                 # must match tokenizer truncation

def memmap_path(out_dir, dataset_name, hook):
    safe = hook.replace(".", "-")
    return os.path.join(out_dir, f"{dataset_name}__{safe}.mmp")

# ---------------------------------------------------------------------------

def cache_dataset(model, tokenizer, csv_path, out_dir,
                  batch_size=200, max_len=MAX_LEN, device="cuda"):
    dataset = os.path.splitext(os.path.basename(csv_path))[0]
    text    = pd.read_csv(csv_path)["prompt"].tolist()
    random.shuffle(text)
    N = len(text)

    # ­­­ create or open mem-maps ------------------------------------------------
    maps = {}
    for hk in HOOKS:
        fn   = memmap_path(out_dir, dataset, hk)
        if os.path.exists(fn):
            raise RuntimeError(f"{fn} already exists – delete first")
        # shape depends on hook:
        if "hook_resid_post" in hk or hk == "hook_embed":
            shape = (N, MAX_LEN, model.cfg.d_model)          # (N,S,d)
        elif "attn.hook_" in hk:                             # q/k/v
            shape = (N, MAX_LEN, model.cfg.n_heads, model.cfg.d_head)
        else:
            raise ValueError(hk)
        maps[hk] = np.memmap(fn, mode="w+", dtype=DTYPE, shape=shape)

    # ­­­ stream batches --------------------------------------------------------
    row = 0
    for i in tqdm(range(0, N, batch_size)):
        toks = tokenizer(text[i:i+batch_size],
                         return_tensors="pt",
                         padding="max_length",
                         truncation=True,
                         max_length=max_len).to(device)

        _, cache = model.run_with_cache(toks.input_ids, names_filter=HOOKS)

        B = toks.input_ids.shape[0]
        for hk in HOOKS:
            chunk = cache[hk].cpu().to(torch.float16).numpy()     # FP16
            maps[hk][row : row+B] = chunk                         # direct write
        row += B

        del cache; torch.cuda.empty_cache()

    # flush & close
    for mm in maps.values():
        mm.flush()
    print(f"{dataset}: wrote mem-maps to {out_dir}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--data_glob", default="data/cleaned_data/*.csv")
    p.add_argument("--out_dir",  default="data/act_cache")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--max_len",  type=int, default=MAX_LEN)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    tok   = model.tokenizer
    tok.padding_side = "right"; tok.truncation_side = "left"

    for csv in glob.glob(args.data_glob):
        cache_dataset(model, tok, csv, args.out_dir,
                      batch_size=200, max_len=args.max_len, device=args.device)
