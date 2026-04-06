from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_ssd_math import build_training_dataset  # noqa: E402


def test_build_training_dataset_keeps_only_rows_that_pass_filters() -> None:
    records = [
        {
            "problem_id": "p1",
            "sample_index": 0,
            "prompt_text": "Compute 2+3.",
            "rendered_prompt_text": "Problem:\nCompute 2+3.",
            "generation_text": "Final Answer: 5",
            "extracted_answer": 5,
            "extraction_status": "ok",
            "filter_status": "kept",
            "template_name": "template_a",
        },
        {
            "problem_id": "p1",
            "sample_index": 1,
            "prompt_text": "Compute 2+3.",
            "generation_text": "No final answer",
            "extracted_answer": None,
            "extraction_status": "missing",
            "filter_status": "kept",
            "template_name": "template_a",
        },
    ]

    dataset_rows, audit_rows = build_training_dataset(
        records,
        generation_field="generation_text",
        max_answer=99999,
        filtering_cfg={"require_extracted_answer": True},
    )

    assert len(dataset_rows) == 1
    assert dataset_rows[0]["problem_id"] == "p1"
    assert len(audit_rows) == 2
    assert audit_rows[1]["keep_for_training"] is False
