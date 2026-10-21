# PEREGRINE v4 — Validation evaluation, first stage

This script evaluates the v2 Experiment 3 checkpoint on the **exact 400 validation object crops selected in v2** (20 per DIOR class, seed 42 + 10000). It reproduces v2's validation preprocessing and loads weights with `weights_only=True`, `strict=True`.

From repository root:

```bash
PYTHONPATH=src python -m unittest -v test_evaluate_dior_v4.py
PYTHONPATH=src python evaluate_dior_v4.py --output-dir experiments/resnet_v4_validation
```

Artifacts: `metrics.json`, `per_class.csv`, `predictions.csv`, `confusion_matrix.csv`.

**Caution:** The v2 checkpoint was selected using validation macro-F1. The 400-crop metrics are for replication and diagnostics, **not independent generalization performance**. The official DIOR test set is never opened by this script. Do not publish validation performance as test accuracy.

The DIOR loader currently catches image-reading exceptions and substitutes black images. Before a final held-out benchmark, audit that behavior and ensure data-read failures cannot silently bias metrics. The existing dataset also enumerates annotations unsorted; deterministic sampling uses stable object identifiers via `choose_indices`.
