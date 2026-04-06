from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_problem_manifests import _load_rows, build_manifests  # noqa: E402


def test_build_manifests_splits_prompt_eval_and_metadata() -> None:
    rows = [
        {
            "problem_id": "p1",
            "prompt_text": "Compute 2+3.",
            "gold_answer": "5",
            "topic": "algebra",
            "difficulty": "easy",
            "tags": "arithmetic,basics",
        },
        {
            "problem_id": "p2",
            "prompt_text": "Open problem prompt.",
            "gold_answer": "",
            "topic": "combinatorics",
            "difficulty": "hard",
            "tags": '["counting"]',
        },
    ]

    prompt_rows, eval_rows, metadata_rows = build_manifests(
        rows,
        problem_id_field="problem_id",
        prompt_field="prompt_text",
        answer_field="gold_answer",
        topic_field="topic",
        difficulty_field="difficulty",
        tags_field="tags",
        source_field=None,
        source_name="source_a",
    )

    assert len(prompt_rows) == 2
    assert len(eval_rows) == 1
    assert eval_rows[0]["gold_answer"] == 5
    assert metadata_rows[1]["tags"] == ["counting"]


def test_load_rows_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "problems.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem_id", "prompt_text", "gold_answer"])
        writer.writeheader()
        writer.writerow({"problem_id": "p1", "prompt_text": "Compute 1+1.", "gold_answer": "2"})

    rows = _load_rows(csv_path, "auto")
    assert rows[0]["problem_id"] == "p1"


def test_build_manifests_dedupes_identical_problem_rows() -> None:
    rows = [
        {"problem_id": "p1", "prompt_text": "Compute 2+3.", "gold_answer": "5", "topic": "algebra"},
        {"problem_id": "p1", "prompt_text": "Compute 2+3.", "gold_answer": "5", "topic": "algebra"},
    ]

    prompt_rows, eval_rows, metadata_rows = build_manifests(
        rows,
        problem_id_field="problem_id",
        prompt_field="prompt_text",
        answer_field="gold_answer",
        topic_field="topic",
        difficulty_field=None,
        tags_field=None,
        source_field=None,
        source_name="source_a",
    )

    assert len(prompt_rows) == 1
    assert len(eval_rows) == 1
    assert len(metadata_rows) == 1
