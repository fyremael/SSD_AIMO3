from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from materialize_colab_bundle import build_bundle_manifest, build_overrides, materialize_config  # noqa: E402


def test_build_overrides_sets_model_and_manifest_paths() -> None:
    overrides = build_overrides(
        model_id="test-model",
        prompt_manifest_jsonl="data/prompts.jsonl",
        eval_prompt_manifest_jsonl="data/eval.jsonl",
        problem_metadata_jsonl="data/meta.jsonl",
        student_adapter_path="runs/adapter",
    )

    assert overrides["model"]["base_model_id"] == "test-model"
    assert overrides["model"]["adapter_path"] == "runs/adapter"
    assert overrides["paths"]["problem_metadata_jsonl"] == "data/meta.jsonl"


def test_materialize_config_writes_resolved_yaml(tmp_path: Path) -> None:
    output_path = tmp_path / "a1.yaml"
    materialize_config(
        base_config_path=str(ROOT / "configs" / "colab_gpu_a1.yaml"),
        output_path=output_path,
        overrides=build_overrides(
            model_id="test-model",
            prompt_manifest_jsonl="data/prompts.jsonl",
            eval_prompt_manifest_jsonl="data/eval.jsonl",
            problem_metadata_jsonl="data/meta.jsonl",
            student_adapter_path=None,
        ),
    )

    with output_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["model"]["base_model_id"] == "test-model"
    assert data["paths"]["unlabeled_prompt_manifest_jsonl"] == "data/prompts.jsonl"


def test_build_bundle_manifest_reports_materialized_paths(tmp_path: Path) -> None:
    manifest = build_bundle_manifest(
        output_dir=tmp_path,
        model_id="test-model",
        prompt_manifest_jsonl="data/prompts.jsonl",
        eval_prompt_manifest_jsonl="data/eval.jsonl",
        problem_metadata_jsonl="data/meta.jsonl",
        student_adapter_path="runs/adapter",
    )

    assert manifest["model_id"] == "test-model"
    assert manifest["student_adapter_path"].endswith("runs\\adapter") or manifest["student_adapter_path"].endswith("runs/adapter")
    assert manifest["materialized_configs"]["a1_student_eval"].endswith("a1_student_eval.yaml")
    assert manifest["recommended_run_dirs"]["compare_a1_a5"].endswith("compare_a1_a5")
