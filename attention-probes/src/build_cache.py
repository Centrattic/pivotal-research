# build_cache.py
import os, torch, argparse, glob, random
from transformer_lens import HookedTransformer
from tqdm import tqdm

HOOKS = [
    "hook_embed",
    *[f"blocks.{l}.hook_resid_post"   for l in range(12)],   # GPT-2-medium
    # *[f"blocks.{l}.attn.hook_q"       for l in range(12)],
    # *[f"blocks.{l}.attn.hook_k"       for l in range(12)],
    # *[f"blocks.{l}.attn.hook_v"       for l in range(12)],
]
# can just compute attention patterns from these

# build_cache.py
def cache_dataset(model, tokenizer, csv_path, out_path,
                  batch_size=128, max_len=256, device="cuda"):
    import pandas as pd, torch, random
    text = pd.read_csv(csv_path)["prompt"].tolist()
    random.shuffle(text)

    store: dict[str, torch.Tensor] = {}         # will hold running cat

    for i in tqdm(range(0, len(text), batch_size)):
        toks = tokenizer(
            text[i:i + batch_size],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        ).to(device)

        _, cache = model.run_with_cache(toks.input_ids, names_filter=HOOKS)
        for hk in HOOKS:
            chunk = cache[hk].cpu().half()      # (B, …)

            if hk in store:
                store[hk] = torch.cat((store[hk], chunk), dim=0)
            else:                               # first time
                store[hk] = chunk

        del cache; torch.cuda.empty_cache()

    torch.save(store, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2-medium")
    p.add_argument("--data_glob", default="data/cleaned_data/*.csv")
    p.add_argument("--out_dir", default="data/act_cache")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    tok   = model.tokenizer
    max_len = 32

    for csv in glob.glob(args.data_glob):
        name = os.path.splitext(os.path.basename(csv))[0]
        out  = os.path.join(args.out_dir, f"{name}.pt")
        if os.path.exists(out):
            print("skip", name)
            continue
        cache_dataset(model, tok, csv, out, max_len=max_len, device="cuda")
