from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import resolve_config_path  # noqa: E402


def test_resolve_config_path_supports_fragment_inheritance(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "\n".join(
            [
                "model:",
                "  base_model_id: base-model",
                "training:",
                "  learning_rate: 2.0e-5",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "profiles.yaml").write_text(
        "\n".join(
            [
                "small:",
                "  generation:",
                "    num_return_sequences_eval: 4",
                "  training:",
                "    gradient_accumulation_steps: 16",
                "large:",
                "  generation:",
                "    num_return_sequences_eval: 16",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "experiment.yaml").write_text(
        "\n".join(
            [
                "inherits_from:",
                "  - base.yaml",
                "  - profiles.yaml#small",
                "experiment:",
                "  name: smoke",
            ]
        ),
        encoding="utf-8",
    )

    resolved = resolve_config_path(str(tmp_path / "experiment.yaml"))

    assert resolved["model"]["base_model_id"] == "base-model"
    assert resolved["training"]["learning_rate"] == 2.0e-5
    assert resolved["training"]["gradient_accumulation_steps"] == 16
    assert resolved["generation"]["num_return_sequences_eval"] == 4
    assert resolved["experiment"]["name"] == "smoke"


def test_resolve_config_path_rejects_inheritance_cycles(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("inherits_from:\n  - b.yaml\nalpha: 1\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("inherits_from:\n  - a.yaml\nbeta: 2\n", encoding="utf-8")

    try:
        resolve_config_path(str(tmp_path / "a.yaml"))
    except ValueError as exc:
        assert "cycle" in str(exc).lower()
    else:
        raise AssertionError("Expected config cycle detection to raise ValueError")
