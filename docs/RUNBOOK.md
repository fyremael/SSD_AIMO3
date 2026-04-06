# RUNBOOK.md
## SSD-Style Validation for AIMO3

This runbook turns the validation pack into a disciplined first-execution workflow for Grand Challenge research agents, coder agents, evaluation agents, and infra agents.

It assumes the program posture already fixed in `TASKS.md`:

> We are validating whether the described SSD-style method family is useful for olympiad-style math, not attempting exact reproduction of Apple’s code-generation tables.

---

## 1. Operating principle

The SSD paper reports that a model can improve code generation by sampling its own raw outputs, fine-tuning on those samples with ordinary next-token loss, and then decoding with a separately chosen evaluation temperature. The paper reports large LiveCodeBench gains, including Qwen3-30B-Instruct moving from 42.4% to 55.3% pass@1 on LiveCodeBench v6, and frames the mechanism as reshaping the precision–exploration tradeoff in decoding. citeturn938483search0turn938483search4

AIMO3 is a different target. Kaggle’s overview describes olympiad-style math problems with integer answers, while the competition data page states there are 110 problems and each answer is a non-negative integer in `[0, 99999]`. That means our primary metric is exact final-answer accuracy, not broad reasoning plausibility. citeturn938483search1turn938483search3

Because of that mismatch, we treat SSD as a **domain-adaptation primitive**. The method is useful only if it improves exact integer answers after fair tuning of prompting, answer extraction, and inference-time selection.

---

## 2. Directory layout

Recommended project layout after first wiring:

```text
SSD_AIMO3_Validation_Pack/
  README.md
  TASKS.md
  docs/
    RUNBOOK.md
    validation_notes.md
  configs/
    global.yaml
    compute_profiles.yaml
    prompt_templates.yaml
    a0_frozen_baseline.yaml
    a1_naive_ssd.yaml
    a2_ssd_plus_selection.yaml
    a3_ssd_plus_weak_filtering.yaml
    a4_ssd_plus_light_verifier.yaml
  prompts/
    unlabeled_prompt_record.schema.json
  scripts/
    common.py
    generate_self_samples.py
    train_ssd_math.py
    extract_answer.py
    aggregate_votes.py
    run_eval_math.py
  tests/
    test_extract_answer.py
  runs/
    <run_name>/
      config_resolved.yaml
      run_manifest.json
      generations.jsonl
      prepared_samples.jsonl
      extracted.jsonl
      aggregates.jsonl
      metrics.json
      samples/
```

---

## 3. Phases and gates

### Phase A — Environment and manifests

Complete these before any training:

1. Pin the Python environment and transformer stack.
2. Produce a stable evaluation manifest.
3. Produce a stable unlabeled prompt corpus manifest.
4. Freeze prompt templates for the first ladder pass.
5. Validate answer extraction on hand-built edge cases.

**Gate A:** no training begins until extraction and evaluation work on known-answer fixtures.

### Phase B — A0 frozen baseline

Goal: establish the strongest fair frozen baseline.

1. Select 1–2 candidate base models.
2. Run prompt-template sweep.
3. Run decoding sweep over temperature / top-p / top-k / return-count.
4. Extract answers and aggregate votes.
5. Record exact-answer accuracy, extraction success, invalid-rate, average reasoning length, and vote entropy.

**Gate B:** choose the single strongest frozen baseline family. No SSD tier is allowed to compare against a weakly tuned baseline.

### Phase C — A1 naïve SSD

Goal: test the core claim in the AIMO3 regime.

1. Generate one self-sampled reasoning trace per unlabeled prompt.
2. Apply only minimal filtering.
3. Fine-tune with plain next-token loss.
4. Re-evaluate under a matched inference budget.
5. Compare to A0 with paired analysis.

**Gate C:** proceed only if A1 shows a directional gain that survives two seeds or two independent decode batches.

### Phase D — A2 / A3 / A4

These tiers exist to test whether the method becomes materially more useful with small, disciplined additions.

- **A2:** selection only
- **A3:** weak filtering of obviously broken self-samples
- **A4:** lightweight verifier signals

The point is not to make the method fancy. The point is to learn whether a paper-faithful core becomes practically useful with modest, auditable additions.

---

## 4. Recommended success thresholds

For the first validation pass:

- **Positive signal:** student beats best frozen baseline on exact final-answer accuracy under matched or stricter inference budget.
- **Convincing signal:** same direction under at least two seeds or decode batches, and under at least two prompt templates.
- **Strong signal:** gain persists while extraction failure rate stays flat or improves, and vote concentration improves.

Recommended red flags:

- longer traces with unchanged exact-answer accuracy
- gains only under one template
- extraction failures drive apparent gains
- student only wins because baseline was under-tuned

---

## 5. What each script is for

### `scripts/generate_self_samples.py`

Builds the SSD training corpus by replaying provided candidate generations or, in `--dry-run`, synthesizing deterministic smoke-test traces. Saves JSONL with prompt metadata, rendered prompt text, raw generation, extraction outcome, and filtering decision.

### `scripts/train_ssd_math.py`

Builds an SFT dataset package from the generated corpus and writes a machine-readable training plan. This is intentionally conservative: plain next-token loss, no RL, no verifier reward, no teacher model.

### `scripts/extract_answer.py`

Canonicalizes final-answer extraction into a standalone audited surface. This script should be stable and versioned early, because extraction bugs can invalidate the whole experiment.

### `scripts/aggregate_votes.py`

Computes majority-vote style aggregation, vote entropy, answer counts, and deterministic tie-breaking.

### `scripts/run_eval_math.py`

Runs end-to-end evaluation on a manifest, canonicalizes answer extraction from raw traces when requested, applies aggregation, and writes machine-readable metrics. It now supports a selectable aggregation backend so the same entry point can compare majority-vote A2/A4-style runs against tropical A5 reranking.

---

## 6. First-pass command sequence

Example sequence for a dry-run validation:

```bash
python scripts/generate_self_samples.py   --config configs/a1_naive_ssd.yaml   --output-dir runs/a1_seed17_dryrun   --dry-run

python scripts/train_ssd_math.py   --config configs/a1_naive_ssd.yaml   --input-jsonl runs/a1_seed17_dryrun/generations.jsonl   --output-dir runs/a1_seed17_train   --dry-run

python scripts/run_eval_math.py   --config configs/a0_frozen_baseline.yaml   --input-jsonl runs/a1_seed17_dryrun/generations.jsonl   --output-dir runs/a0_eval_seed17   --dry-run
```

Once wiring is verified, replace dry-run mode with real model IDs, real paths, and locked environments.

Config files now support inheritance fragments such as `compute_profiles.yaml#small`, which lets stage configs share a common base while pinning profile-specific overrides in one place.

---

## 7. Required record fields

Every generation record should contain at least:

```json
{
  "problem_id": "...",
  "source": "...",
  "topic": "...",
  "prompt_hash": "...",
  "prompt_text": "...",
  "template_name": "template_a",
  "model_id": "...",
  "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 50},
  "generation_text": "...",
  "extracted_answer": 1234,
  "extraction_status": "ok",
  "filter_status": "kept"
}
```

For evaluation runs, add gold answer when available and whether the prediction was correct.

---

## 8. Decision rules for the agent swarm

### Research Agents
- Do not call A1 positive unless the baseline was tuned fairly.
- Inspect whether SSD helps answer correctness or just reasoning style.
- Maintain a failure-mode ledger by topic.

### Coder Agents
- Treat the current scripts as the auditable spine, not the final production stack.
- Preserve JSONL and metrics schemas even if the internal training engine changes.

### Evaluation Agents
- Own the final word on extraction correctness and metric integrity.
- Require paired comparisons wherever possible.

### Infra Agents
- Pin package versions and capture environment snapshots for every reportable run.
- Fail loudly on missing config keys or placeholder model IDs.

---

## 9. Minimal extensions that are allowed

Allowed without changing the scientific question:

- self-consistency voting
- weak syntactic filtering
- lightweight verifier features that are easy to audit
- prompt-template sweeps
- decode-budget sweeps

Not allowed in the first validation claim:

- RLHF / RLAIF / verifier-reward optimization
- using labeled training solutions from the eval distribution
- silently switching to a different objective or dataset without new manifests

---

## 10. Recommended first coding priorities

1. Make `extract_answer.py` boring and correct.
2. Make `run_eval_math.py` emit metrics the research team can trust.
3. Make `generate_self_samples.py` preserve complete raw artifacts.
4. Only then wire real training into `train_ssd_math.py`.

That order is deliberate. In AIMO3-style evaluation, bad extraction or bad bookkeeping can fake progress faster than any actual model improvement.

## Paired comparison workflow

Use `scripts/compare_eval_runs.py` to compare two eval directories and emit a paired report.

Example:

```bash
python scripts/compare_eval_runs.py   --run-a-dir runs/eval_a2   --run-b-dir runs/eval_a5   --run-a-label majority_vote   --run-b-label tropical_rerank   --metadata-jsonl data/problem_metadata.jsonl   --output-dir runs/compare_a2_vs_a5
```

Outputs:
- `comparison_summary.json`
- `paired_problem_comparison.jsonl`
- `paired_problem_comparison.csv`
- `comparison_report.md`
- `comparison_manifest.json`

The optional metadata file should be keyed by `problem_id` and may include `topic`, `difficulty`, and `tags`.

## Fixture ladder

For a replayable local benchmark of the whole A0 -> A1 -> A5 path, run:

```bash
python scripts/run_validation_ladder.py --output-dir runs/fixture_ladder
```

This uses the fixture configs and data pack in `configs/fixture_*.yaml` and `data/fixture_*.jsonl`. See `docs/FIXTURE_LADDER.md` for what the fixture is and is not intended to prove.

## Real-run plumbing

For real source data and real backend execution, see `docs/REAL_RUNS.md`.
The repo now supports:

- `scripts/build_problem_manifests.py` for normalizing raw datasets into JSONL manifests
- `generation.backend: command_jsonl` for external inference backends
- `training.launcher.command` for external trainer launchers
