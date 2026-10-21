# PEREGRINE v3 — resumable CPU training pilot

Copy `train_dior_v3.py` into the **root** of your existing `~/PEREGRINE` repo. It does not replace v2.

## What it does
- Uses the existing, approved `dior_split_manifest_v1.json`.
- Trains ResNetBaseline on source-image-disjoint, equal-per-class pilot subsets.
- Writes `best_resnet_v3_state_dict.pth` (model with highest validation macro-F1), `latest_checkpoint.pth` (full state after each epoch), and `results.json`.
- Full checkpoints hold model, optimizer, completed epoch, Python/NumPy/Torch/shuffle RNG states, history, and checked configuration.
- Saves last-state and results **after every completed epoch** with temporary-file replacement; interrupted epochs are repeated.
- Refuses to create a new experiment over an existing output directory and refuses incompatible resumes.

## Tiny test on CPU (one epoch, then continue to two total)
```bash
cd ~/PEREGRINE
source peregrine-env/bin/activate
PYTHONPATH=src python train_dior_v3.py --train-per-class 2 --val-per-class 2 --epochs 1 --batch-size 8 --threads 2 --seed 42 --output-dir experiments/resnet_v3_resume_smoke
PYTHONPATH=src python train_dior_v3.py --train-per-class 2 --val-per-class 2 --epochs 2 --batch-size 8 --threads 2 --seed 42 --output-dir experiments/resnet_v3_resume_smoke --resume
```

## Important limitations
- `--epochs 2` means two epochs in **total**; when resumed from epoch 1, only epoch 2 runs.
- Resume is permitted only with identical manifest bytes (SHA256), resolved data root, model, seed, subset sizes, batch size, and threads. Increase only `--epochs`.
- Do not resume v2 `best_resnet_v2_state_dict.pth`: it contains no optimizer/RNG state.
- `torch.load(..., weights_only=False)` loads pickle objects to recover NumPy RNG state. **Only resume files you generated locally and trust.** Never load unknown checkpoints.
- This is intended as a local CPU reproducibility pilot, not a production training service or a guarantee of bitwise reproducibility on all platforms.
- The official DIOR test split is not used. Metrics here are validation, not independent test results.
- Checkpoint + JSON saving uses separate atomic replacements; power failure between them can leave JSON one epoch behind the checkpoint, in which case a future resume uses the checkpoint as source of truth.
