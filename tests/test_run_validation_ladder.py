from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_validation_ladder import build_ladder_summary, render_ladder_report  # noqa: E402


def test_build_ladder_summary_flags_directional_gain() -> None:
    summary = build_ladder_summary(
        {"exact_final_answer_accuracy": 0.33},
        {"exact_final_answer_accuracy": 0.66},
        {"exact_final_answer_accuracy": 0.83},
        {"net_gain_b_minus_a": 2, "discordant_pairs": 2, "paired_sign_test_pvalue_two_sided": 0.5},
        {"net_gain_b_minus_a": 1, "discordant_pairs": 1, "paired_sign_test_pvalue_two_sided": 1.0},
    )

    assert summary["a1_directional_gain"] is True
    assert summary["a5_directional_gain"] is True
    assert round(float(summary["a1_minus_a0_accuracy_delta"]), 2) == 0.33


def test_render_ladder_report_mentions_pairwise_sections() -> None:
    text = render_ladder_report(
        {
            "a0_accuracy": 0.33,
            "a1_accuracy": 0.66,
            "a5_accuracy": 0.83,
            "a1_minus_a0_accuracy_delta": 0.33,
            "a5_minus_a1_accuracy_delta": 0.17,
            "a1_directional_gain": True,
            "a5_directional_gain": True,
            "a0_vs_a1": {"net_gain_b_minus_a": 2, "discordant_pairs": 2, "paired_sign_test_pvalue_two_sided": 0.5},
            "a1_vs_a5": {"net_gain_b_minus_a": 1, "discordant_pairs": 1, "paired_sign_test_pvalue_two_sided": 1.0},
        }
    )

    assert "Paired A0 vs A1" in text
    assert "Net gain for A1: 2" in text
    assert "Net gain for A5: 1" in text
