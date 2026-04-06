# ARCHITECTURE.md
## Validation Harness Architecture

This repo is organized around one question:

> Does SSD-style self-distillation improve exact final-answer accuracy for AIMO-style math under a fair, auditable evaluation stack?

## Core flow

```text
raw problem source
  -> manifest normalization
  -> prompt rendering
  -> generation
  -> answer extraction
  -> aggregation / reranking
  -> paired comparison
  -> recommendation
```

## Main surfaces

### 1. Data and manifests

- `scripts/build_problem_manifests.py`
- `data/*.jsonl`

These files convert raw problem sources into:

- prompt manifests
- eval manifests
- problem metadata manifests

Those manifests are the stable contract for the rest of the repo.

### 2. Generation

- `scripts/generate_self_samples.py`
- `scripts/colab_hf_generate.py`

Generation can run in three modes:

- replay from provided candidate generations
- synthetic dry-run generation for smoke tests
- external command-backed generation for real model execution

### 3. Training dataset packaging

- `scripts/train_ssd_math.py`
- `scripts/colab_train_lora.py`

The training stage packages self-generated traces into a conservative SFT dataset. It can stop at dataset export or launch an external trainer through a configured command template.

### 4. Evaluation

- `scripts/extract_answer.py`
- `scripts/run_eval_math.py`
- `scripts/aggregate_votes.py`
- `scripts/tropical_rerank.py`

Evaluation is intentionally modular:

- extraction is standalone and auditable
- majority vote and tropical reranking are separated
- metrics and manifests are always written to disk

### 5. Comparison and decisions

- `scripts/compare_eval_runs.py`
- `scripts/run_validation_ladder.py`

Comparison is paired by `problem_id`, with slice summaries and discordant-pair statistics. The ladder runner wires A0, A1, and A5 into a single reproducible benchmark sequence.

## Configuration model

Configs are layered and composable:

- `configs/global.yaml` holds shared defaults
- `configs/compute_profiles.yaml` holds reusable compute envelopes
- stage configs define A0 through A5
- fixture configs wire the local regression ladder
- Colab configs wire the initial GPU deployment lane

Config inheritance supports fragment references such as `compute_profiles.yaml#small`.

## Deployment posture

The repo currently has two maturity levels:

- first-class: local fixture ladder, external command hooks, Colab GPU generation, Colab GPU LoRA training
- exploratory: TPU execution, which still needs its own dedicated training surface

## Documentation posture

The documentation is split into:

- narrative guides in `docs/*.md`
- generated navigational/status pages from `scripts/update_docs.py`
- automation in `.github/workflows/docs-sync.yml`
