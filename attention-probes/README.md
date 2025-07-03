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

## Datasets

- Each dataset is listed in the CSV `dataset\main.csv`. Adding new datasets is quite simple. The datasets schema matches that defined in [SAE probes](https://github.com/JoshEngels/SAE-Probes/tree/main)

1. Add your raw CSV file to the source folder.
2. Add a row to the index CSV with:
   - New number,
   - `source` (filename),
   - `Probe from` and `Probe to` (the column names)
   - Extraction methods: `col` for direct select of col as a Pandas Series, or Series methods like `col == 1`, `col.str.contains(...)` which will be evaled directly.
3. Run the script to generate cleaned output in the target folder.


## Probing

This module is **model‑agnostic** – you only pass in numpy / torch
matrices that you have already extracted with a TransformerLens hook.
It currently implements two probe families:

1. *Logistic Regression Probe* (linear‑in‑features classifier)
2. *Mass‑Mean Probe* (mean‑difference direction with optional LDA tilt)

Both probes share the same public interface:

```
probe = SomeProbe(**hyperparams)
probe.fit(X_train, y_train)          # mandatory
scores = probe.score(X_test, y_test) # returns dict of metrics
preds  = probe.predict(X_test)       # logits or class probs
```

`X_*` should be 2‑D: **(samples, features)**. If you start with
(sequence, hidden_size) tensors you must flatten/aggregate first – the
helper in `extract.py` does this for you.

You may freely modify / extend the hyperparameters.  See the README for
examples.