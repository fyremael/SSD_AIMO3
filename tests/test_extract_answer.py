from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from extract_answer import extract_final_answer, extract_record  # noqa: E402


def test_extracts_boxed_answer() -> None:
    result = extract_final_answer("Reasoning... therefore \\boxed{42}.")
    assert result["extracted_answer"] == 42
    assert result["extraction_status"] == "ok"
    assert result["extraction_method"] == "boxed"


def test_extracts_final_answer_phrase() -> None:
    result = extract_final_answer("We conclude. Final Answer: 17")
    assert result["extracted_answer"] == 17
    assert result["extraction_status"] == "ok"
    assert result["extraction_method"] == "final_answer"


def test_conflicting_markers_fail_closed() -> None:
    result = extract_final_answer("Final Answer: 12. Therefore \\boxed{14}.")
    assert result["extracted_answer"] is None
    assert result["extraction_status"] == "conflict"
    assert result["candidate_answers"] == [12, 14]


def test_out_of_range_values_are_rejected() -> None:
    result = extract_final_answer("Final Answer: 123456")
    assert result["extracted_answer"] is None
    assert result["extraction_status"] == "out_of_range"


def test_extract_record_adds_canonical_fields() -> None:
    row = extract_record({"problem_id": "p1", "generation_text": "Answer = 9"}, text_field="generation_text")
    assert row["problem_id"] == "p1"
    assert row["extracted_answer"] == 9
    assert row["extraction_status"] == "ok"
    assert row["text_field"] == "generation_text"
