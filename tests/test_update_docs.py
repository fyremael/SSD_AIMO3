from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from update_docs import build_summary, render_index, render_status  # noqa: E402


def test_build_summary_reports_core_counts() -> None:
    summary = build_summary()
    assert summary["config_count"] >= 1
    assert summary["script_count"] >= 1
    assert summary["test_count"] >= 1
    assert summary["notebook_count"] >= 1


def test_render_index_mentions_colab_runbook_and_notebook() -> None:
    text = render_index(build_summary())
    assert "docs/COLAB_DEPLOYMENT.md" in text
    assert "docs/RUNBOOK.md" in text
    assert "notebooks/SSD_AIMO3_Thesis_Validation_Engine.ipynb" in text


def test_render_status_mentions_automation() -> None:
    text = render_status(build_summary())
    assert "Automated upkeep" in text
    assert "Notebook-first Colab experimentation engine" in text
    assert "Weights & Biases telemetry" in text
