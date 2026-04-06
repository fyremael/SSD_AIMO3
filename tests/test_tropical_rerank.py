from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from constraint_library import (  # noqa: E402
    check_explicit_integer_arithmetic,
    compute_tropical_penalty,
    detect_multiple_final_answers,
    detect_parity_conflicts,
    evaluate_trace_constraints,
)


def test_arithmetic_checks_detect_success_and_failure() -> None:
    text = "We compute 2+3=5 and also 7*8=55."
    checks = check_explicit_integer_arithmetic(text)
    assert len(checks) == 2
    assert checks[0].passed is True
    assert checks[1].passed is False


def test_detect_conflicting_final_answers() -> None:
    text = "Final Answer: 12. Therefore \\boxed{14}."
    result = detect_multiple_final_answers(
        text,
        [r"(?i)final\s+answer\s*[:=]\s*([0-9]{1,10})", r"\\boxed\{\s*([0-9]{1,10})\s*\}"],
    )
    assert result["has_conflict"] is True
    assert result["answers"] == [12, 14]


def test_detect_parity_conflict() -> None:
    text = "Let n is even. Later, n is odd."
    result = detect_parity_conflicts(text)
    assert result["has_conflict"] is True
    assert result["conflicted_vars"] == ["n"]


def test_penalty_increases_for_bad_trace() -> None:
    cfg = {
        "contradiction_patterns": [r"(?i)final\s+answer\s*[:=]\s*([0-9]{1,10})", r"\\boxed\{\s*([0-9]{1,10})\s*\}"],
        "enabled_checks": {
            "arithmetic_equalities": True,
            "modular_claims": True,
            "parity_conflicts": True,
            "repeated_final_answer_conflicts": True,
        },
        "extraction_failure_penalty": 100.0,
        "contradiction_penalty": 8.0,
        "arithmetic_failure_penalty": 4.0,
        "modular_failure_penalty": 4.0,
        "missing_structure_penalty": 1.5,
        "reward_boxed_answer": -0.5,
        "reward_final_answer_phrase": -0.5,
        "overlong_threshold_chars": 1600,
        "complexity_penalty_per_200_chars": 0.5,
    }

    good_text = "We compute 2+3=5. Final Answer: 5."
    bad_text = "We compute 2+3=6. Final Answer: 5. Therefore \\boxed{7}. Also n is even and n is odd."

    good_eval = evaluate_trace_constraints(good_text, extracted_answer=5, extraction_status="ok", config=cfg)
    bad_eval = evaluate_trace_constraints(bad_text, extracted_answer=5, extraction_status="ok", config=cfg)

    good_penalty = compute_tropical_penalty(good_eval["features"], good_eval["constraint_checks"], cfg)
    bad_penalty = compute_tropical_penalty(bad_eval["features"], bad_eval["constraint_checks"], cfg)

    assert bad_penalty["total_penalty"] > good_penalty["total_penalty"]
