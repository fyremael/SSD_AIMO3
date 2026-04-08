from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_public_math_benchmark import build_manifest_rows, normalize_answer, select_subset


def test_normalize_answer_accepts_only_non_negative_in_range() -> None:
    assert normalize_answer("12", max_answer=99999) == 12
    assert normalize_answer(-3, max_answer=99999) is None
    assert normalize_answer(100000, max_answer=99999) is None
    assert normalize_answer("not-an-int", max_answer=99999) is None


def test_select_subset_is_deterministic_and_bounded() -> None:
    rows = [(index, {"value": index}, index) for index in range(10)]
    first = select_subset(rows, limit=4, seed=17)
    second = select_subset(rows, limit=4, seed=17)
    assert first == second
    assert len(first) == 4
    assert [item[0] for item in first] == sorted(item[0] for item in first)


def test_build_manifest_rows_filters_negative_answers_and_applies_limits() -> None:
    train_rows = [
        {"question": "q1", "answer": 5},
        {"question": "q2", "answer": -2},
        {"question": "q3", "answer": 7},
    ]
    eval_rows = [
        {"question": "e1", "answer": 11},
        {"question": "e2", "answer": -1},
        {"question": "e3", "answer": 13},
    ]

    prompt_manifest, eval_manifest, metadata_manifest, summary = build_manifest_rows(
        preset_name="public_gsm8k_qwen25_math_1p5b",
        train_rows=train_rows,
        eval_rows=eval_rows,
        train_limit=1,
        eval_limit=2,
        seed=17,
        max_answer=99999,
    )

    assert len(prompt_manifest) == 1
    assert len(eval_manifest) == 2
    assert len(metadata_manifest) == 2
    assert all(row["gold_answer"] >= 0 for row in eval_manifest)
    assert summary["eligible_train_rows"] == 2
    assert summary["eligible_eval_rows"] == 2
    assert summary["recommended_model_id"] == "Qwen/Qwen2.5-Math-1.5B-Instruct"
