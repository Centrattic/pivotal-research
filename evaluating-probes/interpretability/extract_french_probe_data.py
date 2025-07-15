#!/usr/bin/env python
"""
Collects the 12 French-experiment probes (layers 13 & 22, linear vs
attention, three datasets) and writes french_probe_data.json:

{
  "layers": {
    "13": {
      "vectors": { "eng-french_linear": [...], ... },
      "auroc":   { "eng-french_linear": 0.74, ... },
      "cosine":  [ { "p1": "...", "p2": "...", "sim": 0.83 }, ... ],
      "mean_vec": [...]
    },
    "22": { ... }
  }
}
"""

import json, re
from pathlib import Path
from itertools import combinations
import numpy as np

RESULTS_ROOT = Path(__file__).parent.parent / "results" / "french_probing"
OUT_PATH     = Path(__file__).parent / "french_probe_data.json"

# ---- helpers ----------------------------------------------------------


def extract_vector(state_npz: Path) -> np.ndarray | None:
    data = np.load(state_npz)
    if "theta" in data:
        return data["theta"].reshape(-1)
    if "theta_q" in data:          # attention probe
        return data["theta_q"].reshape(-1)
    return None


def infer_probe_id(state_file: Path) -> str:
    """
    From   .../train_<dataset>/train_on_<dataset>_<arch>_L13_resid_post_state.npz
    →      "<dataset>_<arch>"
    """
    m = re.search(r"train_on_(.+?)_(linear|attention)_L\d+", state_file.name)
    if not m:
        raise ValueError(f"Cannot parse probe id from {state_file}")
    dataset, arch = m.groups()
    return f"{dataset}_{arch}"


def load_results_json(state_file: Path) -> dict[str, float]:
    """
    Finds the matching *results.json* for this state & returns the AUROC.
    """
    base = state_file.name.replace("_state.npz", "")
    res_file = state_file.parent / f"{base}_mean_results.json"
    if not res_file.exists():
        # fallback: any _results.json with same prefix
        matches = list(state_file.parent.glob(f"{base}*_results.json"))
        res_file = matches[0] if matches else None
    if not res_file:
        return {}
    with open(res_file) as f:
        js = json.load(f)
    return { "auc": js["metrics"]["auc"] }


# ---- main -------------------------------------------------------------


layers_data: dict[str, dict] = { "13": {}, "22": {} }

for state_npz in RESULTS_ROOT.glob("**/*_state.npz"):
    layer = "13" if "_L13_" in state_npz.name else "22" if "_L22_" in state_npz.name else None
    if layer not in layers_data:
        continue

    probe_id  = infer_probe_id(state_npz)
    vec       = extract_vector(state_npz)
    if vec is None:
        continue

    # save vector
    layers_data[layer].setdefault("vectors", {})[probe_id] = vec.tolist()

    # save AUROC
    score = load_results_json(state_npz)
    if score:
        layers_data[layer].setdefault("auroc", {})[probe_id] = score["auc"]

# --- cosine similarities & mean probe ------------------
for layer, blob in layers_data.items():
    vecs = {k: np.array(v) for k, v in blob["vectors"].items()}
    if not vecs:
        continue
    mean_vec = np.mean(list(vecs.values()), axis=0)
    blob["mean_vec"] = mean_vec.tolist()

    sims = []
    # pairwise
    for p1, p2 in combinations(vecs.keys(), 2):
        v1, v2 = vecs[p1], vecs[p2]
        sims.append({
            "p1": p1, "p2": p2,
            "sim": float(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12))
        })
    # each vs mean
    for p, v in vecs.items():
        sims.append({
            "p1": p, "p2": "mean", "sim":
            float(np.dot(v, mean_vec) / (np.linalg.norm(v)*np.linalg.norm(mean_vec)+1e-12))
        })
    blob["cosine"] = sims

with open(OUT_PATH, "w") as f:
    json.dump({ "layers": layers_data }, f, indent=2)

print("✅ wrote", OUT_PATH)
