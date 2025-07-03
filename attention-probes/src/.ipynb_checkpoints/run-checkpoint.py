from extract import Extractor, Comp
from utils_data import DataLoader
from utils_probe import LogisticRegressionProbe, MassMeanProbe

print("Starting")
ext  = Extractor("gpt2-medium")
print("Extractor initialized")
data = DataLoader("7_nyc_lat")
print("Data loaded")

Xtr, ytr, Xte, yte = data.features(ext, layer=8, component=Comp.resid_post, agg="mean")
print("Retrieved activations")
probe_lg = LogisticRegressionProbe().fit(Xtr, ytr)
probe_mm = MassMeanProbe().fit(Xtr, ytr)
print("Probes trained")
print(probe_lg.score(Xte, yte))
print(probe_mm.score(Xte, yte))
print("Done")