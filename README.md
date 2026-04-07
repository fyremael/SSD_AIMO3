# SSD_AIMO3

Validation harness for testing SSD-style self-distillation on AIMO-style exact-integer math evaluation.

## What this repo is for

- builds prompt, eval, and metadata manifests from raw source data
- canonicalizes final-answer extraction
- supports majority-vote and tropical reranking evaluation
- packages A1-style self-sample training datasets
- compares A0, A1, and A5 runs with paired reports
- includes a replayable fixture ladder for regression testing
- includes an initial Colab GPU deployment path for generation and LoRA training
- includes a notebook-first Colab experimentation engine aimed at thesis validation, not demo optimization
- includes grouped Weights & Biases logging for Colab instrumentation, authenticated from Colab secrets

## Main workflows

### 1. Local fixture ladder

Use this to verify the full repo spine before touching real data:

```bash
python scripts/run_validation_ladder.py --output-dir runs/fixture_ladder
```

### 2. Real data normalization

Turn raw problem sources into the JSONL manifests used everywhere else:

```bash
python scripts/build_problem_manifests.py \
  --input-path data/raw/problems.csv \
  --prompt-output-jsonl data/real_prompts.jsonl \
  --eval-output-jsonl data/real_eval_prompts.jsonl \
  --metadata-output-jsonl data/real_problem_metadata.jsonl \
  --summary-json data/real_manifest_summary.json
```

### 3. Notebook-first Colab validation

Use the notebook when the goal is to methodically validate or invalidate the thesis:

- open [`notebooks/SSD_AIMO3_Thesis_Validation_Engine.ipynb`](notebooks/SSD_AIMO3_Thesis_Validation_Engine.ipynb) in Colab
- run it as-is for a guaranteed self-contained fixture-backed starter pass
- switch `EXPERIMENT_MODE` from `\"starter\"` to `\"real\"` when you are ready to plug in a real model and corpus
- let it clone the repo, install the GPU stack, and normalize manifests if needed
- let it authenticate W&B from the `WANDB_API_KEY` Colab secret and group script runs under one notebook session
- use the paired summaries before changing prompts or budgets

### 4. Colab GPU deployment

Use the Colab GPU path for the first real deployment lane:

- install `requirements/colab-gpu.txt`
- start with `docs/COLAB_DEPLOYMENT.md`
- use `scripts/materialize_colab_bundle.py` or configure `configs/colab_gpu_a0.yaml`, `configs/colab_gpu_a1.yaml`, `configs/colab_gpu_a1_student_eval.yaml`, and `configs/colab_gpu_a5.yaml`

## Repo map

- `configs/` stage configs, shared defaults, fixture configs, and Colab configs
- `data/` fixture prompt packs, eval samples, and raw-bank examples
- `notebooks/` notebook-first Colab experimentation engine
- `scripts/` manifest builders, generation/training entry points, evaluation, comparison, and doc automation
- `tests/` unit coverage for the validation spine and deployment helpers
- `docs/` runbooks, deployment guides, generated indexes, and architecture notes

## Key docs

- `docs/RUNBOOK.md`
- [`notebooks/SSD_AIMO3_Thesis_Validation_Engine.ipynb`](notebooks/SSD_AIMO3_Thesis_Validation_Engine.ipynb)
- `docs/ARCHITECTURE.md`
- `docs/REAL_RUNS.md`
- `docs/COLAB_DEPLOYMENT.md`
- `docs/FIXTURE_LADDER.md`
- `docs/AUTOMATION.md`
- `docs/INDEX.md`
- `docs/STATUS.md`

## Documentation automation

Generated documentation can be refreshed locally with:

```bash
python scripts/update_docs.py
```

The repo also includes a GitHub Actions workflow that refreshes generated doc indexes and status pages automatically.
