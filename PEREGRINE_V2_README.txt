PEREGRINE V2 — RESNET CPU PILOT

Purpose: Test reproducible image-disjoint training/validation, NOT claim high accuracy.

1. Place train_dior_v2.py in ~/PEREGRINE/ (beside original scripts).
2. Keep existing dior_split_manifest_v1.json in the root.
3. Activate existing peregrine-env.
4. First run (40 train objects / 40 validation objects, tiny smoke test):
   cd ~/PEREGRINE
   PYTHONPATH=src python train_dior_v2.py --smoke
5. If it passes, run a small pilot with a unique output folder:
   PYTHONPATH=src python train_dior_v2.py --output-dir experiments/resnet_v2_cpu_pilot_01

SAFETY:
- Does not edit training scripts, annotations, datasets, or original checkpoints.
- Refuses to overwrite an existing output folder.
- Does not read official DIOR test split.
- Splits by source image ID from the preverified manifest.
- Selects an equal number of object crops per class for this pilot only.
- Tracks validation accuracy and macro-F1; selects checkpoint by validation macro-F1.
- Checkpoint is separate and must not be represented as a model validated on test data.

LIMITATIONS:
- This is a tiny CPU sanity check, not sufficient for reliable model quality metrics.
- Existing dataset loader silently substitutes black images for some reading/crop failures.
- Selection is object-balanced; large images can contribute more than one object.
- CPU runtime depends on machine and may be slow. Stop with Ctrl+C if necessary.
- Before larger experiments, add invalid-image detection and a more rigorous end-to-end test evaluation.
