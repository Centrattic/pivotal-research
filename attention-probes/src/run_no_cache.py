from extract import Extractor, Comp
from utils_data import DataLoader
from utils_probe import LogisticRegressionProbe, MassMeanProbe

print("Starting")
ext  = Extractor("gpt2-medium")
print("Extraction initialized")
data = DataLoader("4_hist_fig_ismale")
print("Data loaded")

# Parameters to play with: layer, component, aggregation method
Xtr, ytr, Xte, yte = data.features(ext, layer=8, component=Comp.resid_pre, agg="mean")
print("Activations retrieved")
probe_lg = LogisticRegressionProbe().fit(Xtr, ytr)
probe_mm = MassMeanProbe().fit(Xtr, ytr)
print("Probes trained")

print(probe_lg.score(Xte, yte))
print(probe_mm.score(Xte, yte))
print("Done")

# To Do: account for the three data types - multiclass classification, boolean class, and regression.
# Some probes only work for some of those
# To Do: manage length of prompt