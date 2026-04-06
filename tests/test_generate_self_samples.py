from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_self_samples import build_generation_rows, choose_template, load_prompt_records  # noqa: E402


def test_load_prompt_records_falls_back_to_built_in_dry_run_prompts() -> None:
    rows = load_prompt_records({}, None, dry_run=True)
    assert len(rows) >= 3
    assert rows[0]["problem_id"].startswith("dryrun_")


def test_build_generation_rows_adds_extraction_and_filtering() -> None:
    config = {
        "model": {"base_model_id": "stub-model"},
        "generation": {"eval_temperature": 0.7, "eval_top_p": 0.95, "eval_top_k": 50},
        "filtering": {"policy": "weak_broken_trace_filter"},
    }
    prompts = [{"problem_id": "p1", "prompt_text": "Compute 2+3.", "gold_answer": 5}]

    rows = build_generation_rows(
        prompts,
        config=config,
        template_name="template_a",
        template_text="Problem:\n{prompt_text}",
        num_samples=4,
        max_answer=99999,
        dry_run=True,
    )

    assert len(rows) == 4
    assert rows[0]["extraction_status"] == "ok"
    assert rows[-1]["filter_status"] == "rejected"


def test_choose_template_uses_configured_templates() -> None:
    config = {
        "prompting": {
            "template_name": "template_b",
            "templates": {"template_b": "Solve:\n{prompt_text}"},
        }
    }
    name, template = choose_template(config, None)
    assert name == "template_b"
    assert "{prompt_text}" in template
