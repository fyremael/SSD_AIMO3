from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_eval_math import aggregate_majority_vote, compute_eval_metrics, prepare_records_for_eval  # noqa: E402
from tropical_rerank import aggregate_answers as tropical_aggregate_answers  # noqa: E402


def test_majority_vote_prefers_most_supported_answer() -> None:
    rows = [
        {"problem_id": "p1", "extracted_answer": 11, "gold_answer": 11},
        {"problem_id": "p1", "extracted_answer": 11, "gold_answer": 11},
        {"problem_id": "p1", "extracted_answer": 13, "gold_answer": 11},
    ]
    aggregates = aggregate_majority_vote(rows)
    assert len(aggregates) == 1
    assert aggregates[0]["predicted_answer"] == 11
    assert aggregates[0]["is_correct"] is True


def test_tropical_metrics_flow_through_common_eval_summary() -> None:
    aggregates = [
        {
            "problem_id": "p1",
            "predicted_answer": 7,
            "gold_answer": 7,
            "is_correct": True,
            "num_samples": 4,
            "num_valid_answers": 4,
            "num_invalid_answers": 0,
            "best_trace_penalty": 1.5,
        }
    ]
    sample_records = [{"problem_id": "p1", "extraction_status": "ok"} for _ in range(4)]
    metrics = compute_eval_metrics(aggregates, backend="tropical_rerank", dry_run=True, sample_records=sample_records)
    assert metrics["aggregation_backend"] == "tropical_rerank"
    assert metrics["exact_final_answer_accuracy"] == 1.0
    assert metrics["mean_best_trace_penalty"] == 1.5
    assert metrics["sample_extraction_success_rate"] == 1.0


def test_tropical_answer_aggregation_prefers_lower_penalty_answer() -> None:
    proof_states = [
        {
            "problem_id": "p1",
            "extracted_answer": 12,
            "tropical_score": {"total_penalty": 9.0},
            "gold_answer": 12,
        },
        {
            "problem_id": "p1",
            "extracted_answer": 12,
            "tropical_score": {"total_penalty": 6.0},
            "gold_answer": 12,
        },
        {
            "problem_id": "p1",
            "extracted_answer": 13,
            "tropical_score": {"total_penalty": 1.0},
            "gold_answer": 12,
        },
    ]
    cfg = {
        "answer_score": {
            "support_bonus_weight": 0.75,
            "mean_penalty_weight": 0.20,
            "min_penalty_weight": 1.0,
        }
    }
    aggregates = tropical_aggregate_answers(proof_states, cfg)
    assert aggregates[0]["predicted_answer"] == 13
    assert aggregates[0]["is_correct"] is False


def test_prepare_records_for_eval_extracts_from_raw_text() -> None:
    rows = [
        {"problem_id": "p1", "generation_text": "Reasoning. Final Answer: 21", "gold_answer": 21},
        {"problem_id": "p1", "generation_text": "Reasoning. Therefore \\boxed{21}.", "gold_answer": 21},
        {"problem_id": "p1", "generation_text": "No final integer here", "gold_answer": 21},
    ]

    prepared = prepare_records_for_eval(
        rows,
        text_field="generation_text",
        answer_field="extracted_answer",
        extraction_policy="overwrite_all",
        max_answer=99999,
    )

    assert [row["extracted_answer"] for row in prepared] == [21, 21, None]
    assert [row["extraction_status"] for row in prepared] == ["ok", "ok", "missing"]
    aggregates = aggregate_majority_vote(prepared)
    assert aggregates[0]["predicted_answer"] == 21
