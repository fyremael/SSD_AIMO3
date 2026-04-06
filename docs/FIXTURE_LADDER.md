# FIXTURE_LADDER.md
## Local Validation Fixture Pack

This fixture pack is a **replayable benchmark harness**, not a scientific result.

Its purpose is to make the A0 -> A1 -> A5 workflow executable end-to-end on a fresh checkout so that:

- config inheritance is exercised,
- canonical extraction is exercised,
- majority-vote and tropical aggregation are both exercised,
- paired comparison reports are emitted,
- and future code changes can be regression-tested against a stable local ladder.

## Included assets

- `data/fixture_unlabeled_prompts.jsonl`
- `data/fixture_eval_samples_a0.jsonl`
- `data/fixture_eval_samples_a1.jsonl`
- `data/fixture_problem_metadata.jsonl`
- `configs/fixture_a0.yaml`
- `configs/fixture_a1.yaml`
- `configs/fixture_a5.yaml`

## One-command run

```bash
python scripts/run_validation_ladder.py --output-dir runs/fixture_ladder
```

This produces:

- `runs/fixture_ladder/a1_generation/`
- `runs/fixture_ladder/a1_train/`
- `runs/fixture_ladder/a0_eval/`
- `runs/fixture_ladder/a1_eval/`
- `runs/fixture_ladder/a5_eval/`
- `runs/fixture_ladder/compare_a0_a1/`
- `runs/fixture_ladder/compare_a1_a5/`
- `runs/fixture_ladder/ladder_summary.json`
- `runs/fixture_ladder/ladder_report.md`

## What the fixture is meant to prove

The fixture is meant to prove that the repo can distinguish:

- a stronger student from a weaker frozen baseline,
- a tropical reranker from plain majority vote,
- and a directional gain from a no-gain comparison.

It is **not** meant to claim that those exact deltas transfer to real AIMO3 data.

## Expected headline behavior

The current fixture pack is intentionally shaped so that:

- A1 beats A0 on exact-answer accuracy,
- A5 beats A1 on one hard combinatorics case,
- paired comparison outputs include topic and difficulty slices,
- and the sign-test fields remain honest about low sample count.
