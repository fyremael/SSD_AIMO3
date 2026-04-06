# SSD_AIMO3

Validation harness for testing SSD-style self-distillation on AIMO-style exact-integer math evaluation.

## What this repo does

- builds prompt, eval, and metadata manifests from raw source data
- canonicalizes final-answer extraction
- supports majority-vote and tropical reranking evaluation
- packages A1-style self-sample training datasets
- compares A0, A1, and A5 runs with paired reports
- includes a replayable fixture ladder for regression testing
- includes an initial Colab GPU deployment path for generation and LoRA training

## Quick start

Run the local fixture ladder:

```bash
python scripts/run_validation_ladder.py --output-dir runs/fixture_ladder
```

Prepare real problem manifests:

```bash
python scripts/build_problem_manifests.py \
  --input-path data/raw/problems.csv \
  --prompt-output-jsonl data/real_prompts.jsonl \
  --eval-output-jsonl data/real_eval_prompts.jsonl \
  --metadata-output-jsonl data/real_problem_metadata.jsonl \
  --summary-json data/real_manifest_summary.json
```

## Colab deployment

For the initial deployment lane, use Colab GPU:

- install `requirements/colab-gpu.txt`
- start with `docs/COLAB_DEPLOYMENT.md`
- configure `configs/colab_gpu_a0.yaml`, `configs/colab_gpu_a1.yaml`, and `configs/colab_gpu_a5.yaml`

## Key docs

- `docs/RUNBOOK.md`
- `docs/REAL_RUNS.md`
- `docs/COLAB_DEPLOYMENT.md`
- `docs/FIXTURE_LADDER.md`
