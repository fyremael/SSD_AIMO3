from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_ssd_math import build_training_plan  # noqa: E402


def test_build_training_plan_includes_launcher_config() -> None:
    output_dir = Path("F:/_codex/SSD_AIMO3/runs/test_plan_placeholder")
    plan = build_training_plan(
        {
            "training": {
                "mode": "sft",
                "save_dataset_only": False,
                "launcher": {"enabled": True, "command": "train_here", "workdir": "F:/tmp"},
            }
        },
        output_dir=output_dir,
        num_training_rows=12,
    )

    assert plan["launcher_enabled"] is True
    assert plan["launcher_command"] == "train_here"
    assert plan["train_rows"] == 12
