from pathlib import Path
import json, pandas as pd, numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import Dataset  # ← your refactored class

DATASET = "33_truthqa_tf"  # change to whichever set you want
LAYER = 20
COMP = "resid_post"
MODEL = "gemma-2-9b"

# ---------- load dataset (no activations yet) ----------
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="cpu",
    torch_dtype=torch.float32,
)  # CPU is fine for counting
tokenizer = AutoTokenizer.from_pretrained(MODEL)
ds = Dataset(
    DATASET,
    model=model,
    tokenizer=tokenizer,
    model_name=MODEL,
    device="cpu",
)

# ---- counts straight from Dataset object ----
n_train, n_test = len(ds.X_train_text), len(ds.X_test_text)
u_train = len(set(ds.X_train_text))
u_test = len(set(ds.X_test_text))
u_total = len(set(ds.X_train_text + ds.X_test_text))
overlap = len(set(ds.X_train_text) & set(ds.X_test_text))

print(f"{DATASET}:")
print(f"{n_train},{n_test},{u_train},{u_test},{u_total},{overlap}")
