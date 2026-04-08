# notebooks/README.md

Primary notebook:

- `SSD_AIMO3_Thesis_Validation_Engine.ipynb`

This notebook is the intended first engine for Colab-based experimentation.

Default behavior:

- leave it in the default `EXPERIMENT_MODE = "auto"` setting for a thesis-facing first run
- in `auto`, the notebook now bootstraps a public GSM8K-based real benchmark and a small public math model by default
- switch `EXPERIMENT_MODE` to `starter` when you want the fast fixture-backed harness check instead
- switch `EXPERIMENT_MODE` to `real` with `REAL_MODE_PRESET = ""` only when you want to bring your own model and manifests
- provide a Colab secret named `WANDB_API_KEY` so the notebook can authenticate W&B automatically

It is structured to:

- bootstrap the repo into a fresh Colab runtime when needed
- authenticate W&B and propagate grouped logging env vars to subprocess scripts
- define the thesis and falsification criteria up front
- bootstrap a public benchmark via `scripts/prepare_public_math_benchmark.py` when using the default real preset
- normalize real data into manifests if needed
- materialize a Colab config bundle from one parameter cell
- run A0, A1, and A5 in sequence
- emit paired comparisons and `decision_summary.json`

Use it together with:

- `docs/COLAB_DEPLOYMENT.md`
- `scripts/materialize_colab_bundle.py`
- `configs/colab_gpu_*.yaml`
