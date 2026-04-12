from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_quality_gate import compute_gate_report  # noqa: E402


def test_compute_gate_report_passes_when_metrics_meet_thresholds() -> None:
    report = compute_gate_report(
        label="a0_eval",
        metrics={"valid_answer_rate": 0.7, "sample_extraction_success_rate": 0.8, "num_problems": 64},
        sample_rows=[],
        min_valid_answer_rate=0.5,
        min_extraction_success_rate=0.5,
        max_failure_examples=2,
        text_field="generation_text",
    )
    assert report["passed"] is True
    assert report["failures"] == []


def test_compute_gate_report_collects_failure_examples() -> None:
    report = compute_gate_report(
        label="a1_self_samples",
        metrics={"extraction_success_rate": 0.25, "num_generations": 4},
        sample_rows=[
            {
                "problem_id": "p1",
                "sample_index": 0,
                "extraction_status": "missing",
                "prompt_text": "Find x.",
                "generation_text": "I am not sure.",
            },
            {
                "problem_id": "p2",
                "sample_index": 1,
                "extraction_status": "ok",
                "prompt_text": "Find y.",
                "generation_text": "Final Answer: 4",
            },
        ],
        min_valid_answer_rate=None,
        min_extraction_success_rate=0.5,
        max_failure_examples=3,
        text_field="generation_text",
    )
    assert report["passed"] is False
    assert report["failures"][0]["metric"] == "extraction_success_rate"
    assert len(report["failure_examples"]) == 1
    assert report["failure_examples"][0]["problem_id"] == "p1"
