End‑to‑end pipeline for training *attention probes* on GPT‑2‑medium using
[Transformer Lens](https://github.com/neelnanda‑io/TransformerLens).

The implementation follows two recent works:

* **Sparse‑Autoencoder (SAE) Probe Benchmarks** – we restrict training to
  the binary‑classification datasets listed in Table 3 of *Are Sparse
  Autoencoders Useful? A Case Study in Sparse Probing* and (optionally)
  filter to those where a *linear last‑token probe* achieves **< 0.8 AUC**
  under standard conditionsfileciteturn1file11L17-L24.

* **Sentinel (2025)** – we extract the **last‑token, last‑layer decoder
  attention** pattern A ∈ ℝ<sup>H×T</sup> and build fixed‑length features by
  averaging attention weights over the input tokens, exactly as described
  in §2.2 of Sentinel (“We extract the attention tensor … from the final
  decoder token” and average over tokens)fileciteturn2file4L17-L34.

Overview
--------
1.  **Data loader** (`DatasetManager`) – pulls HF datasets listed in
    `datasets.yml`, shuffles & splits into train/val/test.
2.  **Feature extractor** (`AttentionFeatureExtractor`) – runs GPT‑2‑medium
    with Transformer Lens, returning a `torch.Tensor` of shape
    *(N, H·B)* where *H* is #heads and *B* the max‑sequence buckets.
3.  **Probe** (`LinearAttentionProbe`) – L1‑regularised logistic
    regression from scikit‑learn.
4.  **Trainer** – orchestrates feature caching, cross‑validation & test
    evaluation, then logs metrics to `results.csv`.

Usage (CLI)
-----------
```bash
# install deps
pip install transformer_lens==1.* transformers datasets scikit-learn pyyaml tqdm

# run a single dataset
python attention_probe_project.py --dataset "squad_v2" \
       --cache_dir ~/.cache/attn_probes

# full sweep on all datasets <80 AUC
python attention_probe_project.py --all_datasets
```