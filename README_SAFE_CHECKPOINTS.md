# PEREGRINE v3 checkpoint safety patch

This replaces only `train_dior_v3.py` (currently uncommitted) and adds `test_checkpoint_format.py`. The script stores NumPy RNG data as primitive Python types instead of NumPy ndarray/scalar objects, then restores it as MT19937 state. The full checkpoint now uses `torch.load(..., weights_only=True)` and schema version **2**.

**Important:** Existing v3 schema-1 `latest_checkpoint.pth` files are **not migrated** or modified. They will not resume in the patched script. Keep those prior experiments as historical evidence; run new tests in new output directories. Do not delete any existing checkpoint.

The source code compile/test checks are not proof that end-to-end DIOR training/resume matches until the user repeats the small two-epoch comparison. Do not commit until that passes, including comparing **latest** (not best) model states.

To test safe serialization alone from repo root: `PYTHONPATH=src python -m unittest -v test_checkpoint_format.py`.

To test full integration: perform one-epoch smoke in a **new** output directory, resume that directory to two total epochs, then independently train two uninterrupted epochs in another **new** directory. Both latest checkpoints should load with `weights_only=True`. Compare `['model_state']` entries with `torch.equal`.
