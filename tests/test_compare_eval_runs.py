from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_eval_runs import compare_runs  # noqa: E402


def test_compare_runs_counts_flips_and_slice_deltas() -> None:
    run_a = [
        {"problem_id": "p1", "predicted_answer": 5, "gold_answer": 7, "is_correct": False, "vote_margin": 1},
        {"problem_id": "p2", "predicted_answer": 9, "gold_answer": 9, "is_correct": True, "vote_margin": 2},
        {"problem_id": "p3", "predicted_answer": 4, "gold_answer": 4, "is_correct": True, "vote_margin": 3},
    ]
    run_b = [
        {"problem_id": "p1", "predicted_answer": 7, "gold_answer": 7, "is_correct": True, "best_trace_penalty": 1.0},
        {"problem_id": "p2", "predicted_answer": 8, "gold_answer": 9, "is_correct": False, "best_trace_penalty": 2.0},
        {"problem_id": "p3", "predicted_answer": 4, "gold_answer": 4, "is_correct": True, "best_trace_penalty": 0.5},
    ]
    metadata = {
        "p1": {"problem_id": "p1", "topic": "number_theory", "difficulty": "hard", "tags": ["modular"]},
        "p2": {"problem_id": "p2", "topic": "algebra", "difficulty": "medium", "tags": ["equations"]},
        "p3": {"problem_id": "p3", "topic": "number_theory", "difficulty": "medium", "tags": ["modular"]},
    }

    summary, comparisons = compare_runs(run_a, run_b, run_a_label="majority", run_b_label="tropical", metadata_index=metadata)

    assert len(comparisons) == 3
    assert summary["run_a_only_correct"] == 1
    assert summary["run_b_only_correct"] == 1
    assert summary["discordant_pairs"] == 2
    assert summary["net_gain_b_minus_a"] == 0
    assert summary["win_rate_b_given_discordant"] == 0.5
    assert summary["paired_sign_test_pvalue_two_sided"] == 1.0
    assert summary["both_correct"] == 1
    assert summary["answer_flips"] == 2
    assert summary["topic_slices"]["number_theory"]["net_gain_b_minus_a"] == 1
    assert summary["difficulty_slices"]["medium"]["net_gain_b_minus_a"] == -1


def test_compare_runs_handles_missing_problem_in_one_run() -> None:
    run_a = [
        {"problem_id": "p1", "predicted_answer": 2, "gold_answer": 2, "is_correct": True},
    ]
    run_b = [
        {"problem_id": "p1", "predicted_answer": 2, "gold_answer": 2, "is_correct": True},
        {"problem_id": "p2", "predicted_answer": 3, "gold_answer": 3, "is_correct": True},
    ]
    summary, comparisons = compare_runs(run_a, run_b, run_a_label="a", run_b_label="b", metadata_index={})
    assert summary["missing_in_run_a"] == 1
    assert summary["missing_in_run_b"] == 0
    assert any(row["verdict"] == "missing_in_a" for row in comparisons)
