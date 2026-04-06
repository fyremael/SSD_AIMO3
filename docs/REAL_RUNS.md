# REAL_RUNS.md
## Moving From Fixtures To Real Runs

The repo now supports three practical entry points for real execution:

1. normalize raw source data into manifests,
2. call an external generation backend,
3. call an external training backend.

## Build manifests from raw data

If your source data is CSV, JSON, or JSONL, normalize it first:

```bash
python scripts/build_problem_manifests.py ^
  --input-path data/raw/problems.csv ^
  --prompt-output-jsonl data/real_prompts.jsonl ^
  --eval-output-jsonl data/real_eval_manifest.jsonl ^
  --metadata-output-jsonl data/real_problem_metadata.jsonl ^
  --summary-json data/real_manifest_summary.json ^
  --problem-id-field id ^
  --prompt-field problem ^
  --answer-field answer ^
  --topic-field topic ^
  --difficulty-field difficulty ^
  --tags-field tags ^
  --source-name real_corpus
```

Use this on **problem-level rows**, not sample-level generation logs. If multiple rows share a `problem_id`, the builder now deduplicates exact duplicates and raises on conflicting prompt or answer data.

## External generation backend

Set `generation.backend: command_jsonl` and provide a command that:

- reads `{input_jsonl}`,
- writes `{output_jsonl}`,
- preserves `problem_id` and `sample_index`,
- and emits a generation field such as `generation_text`.

Available template variables:

- `{python_executable}`
- `{workspace_root}`
- `{output_dir}`
- `{input_jsonl}`
- `{output_jsonl}`
- `{model_id}`
- `{generation_field}`
- `{num_rows}`

Example config: `configs/command_backend_example.yaml`
Colab GPU-specific configs live in `configs/colab_gpu_*.yaml`.

## External training backend

Set `training.launcher.command` and run:

```bash
python scripts/train_ssd_math.py --config <your_config> --input-jsonl <generations.jsonl> --output-dir <run_dir> --launch
```

Available template variables:

- `{python_executable}`
- `{workspace_root}`
- `{output_dir}`
- `{dataset_path}`
- `{model_id}`
- `{learning_rate}`
- `{num_train_epochs}`
- `{max_seq_length}`
- `{train_rows}`

## Smoke-test the external hooks

The repo includes mock backends so the command path can be verified locally:

```bash
python scripts/generate_self_samples.py --config configs/command_backend_example.yaml --input-jsonl data/fixture_unlabeled_prompts.jsonl --output-dir runs/command_backend_generation

python scripts/train_ssd_math.py --config configs/command_backend_example.yaml --input-jsonl runs/command_backend_generation/generations.jsonl --output-dir runs/command_backend_train --launch
```

Those commands exercise the same external hook surfaces a real vLLM, transformers, or custom inference/training wrapper would use.

For the initial Colab deployment path, see `docs/COLAB_DEPLOYMENT.md`.
