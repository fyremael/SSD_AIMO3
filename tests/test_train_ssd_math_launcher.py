from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_ssd_math import build_training_plan, maybe_launch_training  # noqa: E402


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


def test_maybe_launch_training_renders_batch_size_fields(tmp_path: Path, monkeypatch) -> None:
    observed: Dict[str, Any] = {}

    def fake_run(command: str, cwd: str, shell: bool, check: bool) -> None:
        observed["command"] = command
        observed["cwd"] = cwd
        observed["shell"] = shell
        observed["check"] = check

    monkeypatch.setattr("train_ssd_math.subprocess.run", fake_run)

    output_dir = tmp_path / "train"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_dataset.jsonl").write_text("", encoding="utf-8")
    training_plan = {"launch_status": "planned_dataset_only"}

    launch_record = maybe_launch_training(
        config={
            "model": {"base_model_id": "Qwen/Qwen2.5-Math-1.5B"},
            "training": {
                "learning_rate": 2.0e-5,
                "num_train_epochs": 1,
                "max_seq_length": 2048,
                "per_device_train_batch_size": 3,
                "gradient_accumulation_steps": 7,
                "launcher": {
                    "enabled": True,
                    "command": "train --batch {per_device_train_batch_size} --grad-accum {gradient_accumulation_steps}",
                    "workdir": str(tmp_path),
                },
            },
        },
        output_dir=output_dir,
        training_plan=training_plan,
        num_training_rows=12,
        requested_launch=True,
    )

    assert observed["command"] == "train --batch 3 --grad-accum 7"
    assert observed["cwd"] == str(tmp_path.resolve())
    assert observed["shell"] is True
    assert observed["check"] is True
    assert training_plan["launch_status"] == "completed"
    assert launch_record is not None
    assert launch_record["status"] == "completed"
