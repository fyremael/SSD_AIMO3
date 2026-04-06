# COLAB_DEPLOYMENT.md
## Initial Colab Deployment Plan

The recommended first deployment target is **Colab GPU**, not TPU.

Reason:

- the repo now has a working Colab GPU generation path via `scripts/colab_hf_generate.py`
- it also has a LoRA training path via `scripts/colab_train_lora.py`
- both are designed to plug into the repo's existing JSONL command hooks
- the TPU lane is still best treated as exploratory because the first-class training path here depends on the GPU-oriented PEFT + bitsandbytes stack

## 1. Start the Colab runtime

Choose `Runtime -> Change runtime type -> T4 / L4 / A100 GPU` if available.

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

Start from:

- `configs/colab_gpu_a0.yaml`
- `configs/colab_gpu_a1.yaml`
- `configs/colab_gpu_a5.yaml`

Replace:

- `REPLACE_WITH_HF_MODEL_ID`
- `REPLACE_WITH_UNLABELED_PROMPTS_JSONL`
- `REPLACE_WITH_EVAL_PROMPTS_JSONL`
- `REPLACE_WITH_EVAL_GENERATIONS_JSONL`
- `REPLACE_WITH_PROBLEM_METADATA_JSONL`

For a quick debug pass on Colab with a tiny public model, use:

- `configs/colab_gpu_fixture_debug.yaml`

## 5. Generate A1 self-samples on GPU

```bash
python scripts/generate_self_samples.py \
  --config configs/colab_gpu_a1.yaml \
  --input-jsonl data/real_prompts.jsonl \
  --output-dir runs/colab_a1_generation
```

This calls `scripts/colab_hf_generate.py` through the repo's `command_jsonl` interface.

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

## 7. Produce eval generations and score them

The repo still separates generation from evaluation on purpose.

For A0 or A1 eval:

1. generate eval samples with `scripts/generate_self_samples.py` using your eval prompt manifest
2. score them with `scripts/run_eval_math.py`

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
