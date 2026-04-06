# notebooks/README.md

Primary notebook:

- `SSD_AIMO3_Thesis_Validation_Engine.ipynb`

This notebook is the intended first engine for Colab-based experimentation.

Default behavior:

- run it without edits for a self-contained fixture-backed starter pass
- switch `EXPERIMENT_MODE` to `real` when you want to point it at a real model and corpus

It is structured to:

- bootstrap the repo into a fresh Colab runtime when needed
- define the thesis and falsification criteria up front
- normalize real data into manifests if needed
- materialize a Colab config bundle from one parameter cell
- run A0, A1, and A5 in sequence
- emit paired comparisons and `decision_summary.json`

Use it together with:

- `docs/COLAB_DEPLOYMENT.md`
- `scripts/materialize_colab_bundle.py`
- `configs/colab_gpu_*.yaml`
