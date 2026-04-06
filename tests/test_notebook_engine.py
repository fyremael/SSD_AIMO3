from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "SSD_AIMO3_Thesis_Validation_Engine.ipynb"


def _load_notebook() -> dict:
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_notebook_is_valid_ipynb_json() -> None:
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    assert isinstance(notebook.get("cells"), list)
    assert notebook["cells"]


def test_notebook_covers_bootstrap_bundle_and_decision_flow() -> None:
    notebook = _load_notebook()
    cell_ids = {cell.get("metadata", {}).get("id") for cell in notebook["cells"]}
    assert {"setup", "params", "bootstrap", "runtime", "bundle", "a1eval", "a5compare", "decision", "starter"} <= cell_ids

    notebook_text = json.dumps(notebook)
    assert "github.com/fyremael/SSD_AIMO3.git" in notebook_text
    assert "decision_summary.json" in notebook_text
    assert "EXPERIMENT_MODE" in notebook_text
    assert "starter_fixture_complete" in notebook_text
    assert "run_validation_ladder.py" in notebook_text
