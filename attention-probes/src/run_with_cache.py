from act_store_memmap import file_name, load_memmap
from utils_data import DataLoader
from utils_probe import LogisticRegressionProbe, MassMeanProbe, aggregate_sequence
import torch

TAG        = "4_hist_fig_ismale"
LAYER      = 8
COMPONENT  = "resid_pre"          # "attn_q", "attn_k", ...
HOOK       = f"blocks.{LAYER}.hook_{COMPONENT}" if COMPONENT.startswith("resid") \
             else f"blocks.{LAYER}.attn.hook_{COMPONENT.split('_')[-1]}"

path = file_name(TAG, LAYER, COMPONENT, cache_dir="../results/act_cache")
d_model = 768                   # GPT-2-medium
# d_last = 12 * 64
acts = load_memmap(path, max_len=32, d_last=d_model)   # (N, 256, 768)

# collapse sequence → feature matrix
X = aggregate_sequence(torch.tensor(acts), how="mean")   # (N, d)

# labels & split
data = DataLoader(TAG)
idx_tr, idx_te = data.split(test_size=0.2)
y_tr, y_te = data.labels[idx_tr], data.labels[idx_te]

# probes
probe_lr = LogisticRegressionProbe().fit(X[idx_tr], y_tr)
# probe_mm = MassMeanProbe().fit(X[idx_tr], y_tr)

probe_lr.save_results(X[idx_te], y_te,
    model_name="gpt2-medium", dataset=TAG, layer=LAYER, component=COMPONENT)

# probe_mm.save_results(X[idx_te], y_te,
#     model_name="gpt2-medium", dataset=TAG, layer=LAYER, component=COMPONENT)
