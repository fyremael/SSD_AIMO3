# COLAB_DEPLOYMENT.md
## Initial Colab Deployment Plan

The recommended first deployment target is **Colab GPU**, not TPU.

The recommended first operating surface is the notebook:

- [`../notebooks/SSD_AIMO3_Thesis_Validation_Engine.ipynb`](../notebooks/SSD_AIMO3_Thesis_Validation_Engine.ipynb)

That notebook is meant to be the primary initial experimentation engine because it keeps the full thesis-validation ladder in one place:

- define the hypothesis and falsification posture up front
- normalize raw problem data into manifests if needed
- materialize a clean config bundle for A0, A1, A1 student eval, and A5
- run the same extraction and comparison contract across the full ladder
- emit a decision-ready summary before another round of prompt or budget changes

It now defaults to a **zero-edit starter preset**:

- leave the notebook in its default `EXPERIMENT_MODE = "auto"` setting for a zero-edit first run
- in `auto`, it uses your real model and manifests only when they are fully configured and otherwise falls back to the built-in fixture ladder
- switch `EXPERIMENT_MODE` to `real` only when you want to supply your own model and manifests and fail fast if they are incomplete
- authenticate W&B automatically from the `WANDB_API_KEY` Colab secret and group child runs under the notebook session

Reason:

- the repo now has a working Colab GPU generation path via `scripts/colab_hf_generate.py`
- it also has a LoRA training path via `scripts/colab_train_lora.py`
- both are designed to plug into the repo's existing JSONL command hooks
- the TPU lane is still best treated as exploratory because the first-class training path here depends on the GPU-oriented PEFT + bitsandbytes stack

## 1. Start the Colab runtime

Choose `Runtime -> Change runtime type -> T4 / L4 / A100 GPU` if available.

If you are starting from the notebook, open [`../notebooks/SSD_AIMO3_Thesis_Validation_Engine.ipynb`](../notebooks/SSD_AIMO3_Thesis_Validation_Engine.ipynb) in Colab first. It bootstraps the repo into `/content/SSD_AIMO3` and then runs the same steps documented below.

If you just want a no-configuration smoke pass, leave the notebook in its default `auto` mode and run it top-to-bottom.

Probe the runtime first:

```bash
python scripts/colab_runtime_probe.py --output-json runs/colab_runtime.json
```

If `recommended_lane` is not `gpu`, do not start with training.

## 2. Install dependencies

Colab already includes PyTorch in most GPU runtimes, so the repo dependency file only adds the missing stack:

```bash
pip install -r requirements/colab-gpu.txt
```

## 2a. Configure W&B secret

For the notebook-driven path, add a Colab secret named `WANDB_API_KEY`.

The notebook now:

- reads that secret directly from Colab secrets
- runs `wandb.login(...)` automatically
- exports grouped `SSD_AIMO3_WANDB_*` env vars so subprocess scripts log under the same session family
- keeps bulky JSONL traces local while uploading compact metrics/manifests/artifacts

## 3. Normalize your raw problem source

If your source data is still in CSV or generic JSON, normalize it first:

```bash
python scripts/build_problem_manifests.py \
  --input-path data/raw/problems.csv \
  --prompt-output-jsonl data/real_prompts.jsonl \
  --eval-output-jsonl data/real_eval_prompts.jsonl \
  --metadata-output-jsonl data/real_problem_metadata.jsonl \
  --summary-json data/real_manifest_summary.json \
  --problem-id-field id \
  --prompt-field problem \
  --answer-field answer \
  --topic-field topic \
  --difficulty-field difficulty \
  --tags-field tags \
  --source-name real_corpus
```

Use this on **problem-level rows**, not sample-level generations.

## 4. Configure the Colab stage files

If you are using the notebook path, prefer `scripts/materialize_colab_bundle.py` over hand-editing multiple stage files.

Start from:

- `configs/colab_gpu_a0.yaml`
- `configs/colab_gpu_a1.yaml`
- `configs/colab_gpu_a1_student_eval.yaml`
- `configs/colab_gpu_a5.yaml`

Replace:

- `REPLACE_WITH_HF_MODEL_ID`
- `REPLACE_WITH_UNLABELED_PROMPTS_JSONL`
- `REPLACE_WITH_EVAL_PROMPTS_JSONL`
- `REPLACE_WITH_EVAL_GENERATIONS_JSONL`
- `REPLACE_WITH_PROBLEM_METADATA_JSONL`

For a quick debug pass on Colab with a tiny public model, use:

- `configs/colab_gpu_fixture_debug.yaml`

For the notebook-driven path, the bundle materializer keeps these synchronized from one parameter cell:

```bash
python scripts/materialize_colab_bundle.py \
  --output-dir runs/colab_bundle \
  --model-id your/model \
  --prompt-manifest-jsonl data/real_prompts.jsonl \
  --eval-prompt-manifest-jsonl data/real_eval_prompts.jsonl \
  --problem-metadata-jsonl data/real_problem_metadata.jsonl
```

## 5. Generate A1 self-samples on GPU

```bash
python scripts/generate_self_samples.py \
  --config configs/colab_gpu_a1.yaml \
  --input-jsonl data/real_prompts.jsonl \
  --output-dir runs/colab_a1_generation
```

This calls `scripts/colab_hf_generate.py` through the repo's `command_jsonl` interface.
With notebook-driven W&B enabled, both the higher-level generation script and the backend generator emit grouped telemetry.

## 6. Package and launch A1 LoRA training

```bash
python scripts/train_ssd_math.py \
  --config configs/colab_gpu_a1.yaml \
  --input-jsonl runs/colab_a1_generation/generations.jsonl \
  --output-dir runs/colab_a1_train \
  --launch
```

This creates:

- `train_dataset.jsonl`
- `training_audit.jsonl`
- `training_plan.json`
- `adapter/` with the LoRA adapter when training succeeds

With notebook-driven W&B enabled, the dataset-packaging script and backend trainer both log compact metrics and output artifacts.

## 7. Produce eval generations and score them

The repo still separates generation from evaluation on purpose.

For A0 or A1 eval:

1. generate eval samples with `scripts/generate_self_samples.py` using your eval prompt manifest
2. score them with `scripts/run_eval_math.py`

For the trained A1 student on Colab, use `configs/colab_gpu_a1_student_eval.yaml` or the notebook-generated `a1_student_eval.yaml` so the adapter is evaluated through the same extraction surface as A0.

Example:

```bash
python scripts/generate_self_samples.py \
  --config configs/colab_gpu_a0.yaml \
  --input-jsonl data/real_eval_prompts.jsonl \
  --num-samples 8 \
  --output-dir runs/colab_a0_eval_generation

python scripts/run_eval_math.py \
  --config configs/colab_gpu_a0.yaml \
  --input-jsonl runs/colab_a0_eval_generation/generations.jsonl \
  --output-dir runs/colab_a0_eval
```

Then compare runs with `scripts/compare_eval_runs.py`.

## 8. TPU guidance

TPU is not the recommended first deployment target for this repo yet.

Current honest posture:

- manifest building works anywhere
- extraction, aggregation, and paired comparison work anywhere
- the Colab GPU generation path is first-class
- the Colab GPU LoRA training path is first-class
- TPU training would need a separate `torch_xla` or JAX trainer, not just a config toggle

If you want, the next TPU-focused milestone after this should be:

1. add a small-model eval-only TPU path
2. add a dedicated `torch_xla` trainer script
3. only then promote TPU to a first-class training lane
