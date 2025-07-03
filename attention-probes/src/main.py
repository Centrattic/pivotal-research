from extract import Extractor, Comp
from utils_data import DataLoader
from utils_probe import LogisticRegressionProbe, MassMeanProbe

ext  = Extractor("gpt2-medium")
data = DataLoader("7_hist_fig_ispolitician")

Xtr, ytr, Xte, yte = data.features(ext, layer=8, component=Comp.resid_post, agg="mean")
probe_lg = LogisticRegressionProbe().fit(Xtr, ytr)
probe_mm = MassMeanProbe().fit(Xtr, ytr)

print(probe_lg.score(Xte, yte))
print(probe_mm.score(Xte, yte))
