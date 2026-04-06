from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_self_samples import merge_external_generation_responses  # noqa: E402


def test_merge_external_generation_responses_aligns_by_problem_and_sample() -> None:
    request_rows = [
        {"problem_id": "p1", "sample_index": 0, "rendered_prompt_text": "A"},
        {"problem_id": "p1", "sample_index": 1, "rendered_prompt_text": "B"},
    ]
    response_rows = [
        {"problem_id": "p1", "sample_index": 1, "generation_text": "Final Answer: 2"},
        {"problem_id": "p1", "sample_index": 0, "generation_text": "Final Answer: 1"},
    ]

    merged = merge_external_generation_responses(request_rows, response_rows, generation_field="generation_text")

    assert [row["generation_text"] for row in merged] == ["Final Answer: 1", "Final Answer: 2"]


def test_merge_external_generation_responses_rejects_missing_rows() -> None:
    request_rows = [{"problem_id": "p1", "sample_index": 0}]
    response_rows: list[dict[str, object]] = []

    try:
        merge_external_generation_responses(request_rows, response_rows, generation_field="generation_text")
    except ValueError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("Expected missing response error")
