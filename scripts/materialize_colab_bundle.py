from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from common import deep_merge, resolve_config_path, write_json
from wandb_support import wandb_run_context


JsonDict = Dict[str, Any]


def _is_placeholder(value: Optional[str]) -> bool:
    return isinstance(value, str) and value.startswith("REPLACE_WITH_")


def validate_inputs(
    *,
    model_id: str,
    prompt_manifest_jsonl: str,
    eval_prompt_manifest_jsonl: str,
    problem_metadata_jsonl: str,
    student_adapter_path: Optional[str],
) -> None:
    errors = []
    if not str(model_id).strip() or _is_placeholder(model_id):
        errors.append("model_id must be set to a real model identifier before materializing the Colab bundle")

    for label, raw_path in (
        ("prompt_manifest_jsonl", prompt_manifest_jsonl),
        ("eval_prompt_manifest_jsonl", eval_prompt_manifest_jsonl),
        ("problem_metadata_jsonl", problem_metadata_jsonl),
    ):
        if _is_placeholder(raw_path):
            errors.append(f"{label} is still a placeholder: {raw_path}")
            continue
        path = Path(raw_path).resolve()
        if not path.exists():
            errors.append(f"{label} does not exist: {path}")

    if student_adapter_path:
        if _is_placeholder(student_adapter_path):
            errors.append(f"student_adapter_path is still a placeholder: {student_adapter_path}")
        else:
            adapter_path = Path(student_adapter_path).resolve()
            if not adapter_path.exists():
                errors.append(f"student_adapter_path does not exist: {adapter_path}")

    if errors:
        raise ValueError("Invalid Colab bundle inputs:\n- " + "\n- ".join(errors))


def materialize_config(
    *,
    base_config_path: str,
    output_path: Path,
    overrides: Mapping[str, Any],
) -> JsonDict:
    config = resolve_config_path(base_config_path)
    merged = deep_merge(dict(config), dict(overrides))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return merged


def build_overrides(
    *,
    model_id: str,
    prompt_manifest_jsonl: str,
    eval_prompt_manifest_jsonl: str,
    problem_metadata_jsonl: str,
    student_adapter_path: Optional[str],
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "model": {
            "base_model_id": model_id,
        },
        "paths": {
            "unlabeled_prompt_manifest_jsonl": prompt_manifest_jsonl,
            "eval_manifest_jsonl": eval_prompt_manifest_jsonl,
            "problem_metadata_jsonl": problem_metadata_jsonl,
        },
        "inputs": {
            "prompt_input_jsonl": prompt_manifest_jsonl,
            "eval_input_jsonl": eval_prompt_manifest_jsonl,
            "tropical_input_jsonl": eval_prompt_manifest_jsonl,
        },
    }
    if student_adapter_path:
        overrides.setdefault("model", {})["adapter_path"] = student_adapter_path
    return overrides


def build_bundle_manifest(
    *,
    output_dir: Path,
    model_id: str,
    prompt_manifest_jsonl: str,
    eval_prompt_manifest_jsonl: str,
    problem_metadata_jsonl: str,
    student_adapter_path: Optional[str],
) -> JsonDict:
    run_root = output_dir / "runs"
    return {
        "model_id": model_id,
        "prompt_manifest_jsonl": str(Path(prompt_manifest_jsonl).resolve()),
        "eval_prompt_manifest_jsonl": str(Path(eval_prompt_manifest_jsonl).resolve()),
        "problem_metadata_jsonl": str(Path(problem_metadata_jsonl).resolve()),
        "student_adapter_path": str(Path(student_adapter_path).resolve()) if student_adapter_path else None,
        "materialized_configs": {
            "a0": str((output_dir / "a0.yaml").resolve()),
            "a1": str((output_dir / "a1.yaml").resolve()),
            "a1_student_eval": str((output_dir / "a1_student_eval.yaml").resolve()),
            "a5": str((output_dir / "a5.yaml").resolve()),
        },
        "recommended_run_dirs": {
            "a0_eval_generation": str((run_root / "a0_eval_generation").resolve()),
            "a0_eval": str((run_root / "a0_eval").resolve()),
            "a1_self_samples": str((run_root / "a1_self_samples").resolve()),
            "a1_train": str((run_root / "a1_train").resolve()),
            "a1_eval_generation": str((run_root / "a1_eval_generation").resolve()),
            "a1_eval": str((run_root / "a1_eval").resolve()),
            "a5_eval": str((run_root / "a5_eval").resolve()),
            "compare_a0_a1": str((run_root / "compare_a0_a1").resolve()),
            "compare_a1_a5": str((run_root / "compare_a1_a5").resolve()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize a Colab experiment config bundle from notebook parameters")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-manifest-jsonl", required=True)
    parser.add_argument("--eval-prompt-manifest-jsonl", required=True)
    parser.add_argument("--problem-metadata-jsonl", required=True)
    parser.add_argument("--student-adapter-path", default=None)
    parser.add_argument("--a0-base-config", default="configs/colab_gpu_a0.yaml")
    parser.add_argument("--a1-base-config", default="configs/colab_gpu_a1.yaml")
    parser.add_argument("--a1-student-base-config", default="configs/colab_gpu_a1_student_eval.yaml")
    parser.add_argument("--a5-base-config", default="configs/colab_gpu_a5.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with wandb_run_context(
        config=None,
        output_dir=output_dir,
        script_name="materialize_colab_bundle.py",
        job_type="bundle_materialization",
        extra_config={
            "model_id": args.model_id,
            "prompt_manifest_jsonl": args.prompt_manifest_jsonl,
            "eval_prompt_manifest_jsonl": args.eval_prompt_manifest_jsonl,
            "problem_metadata_jsonl": args.problem_metadata_jsonl,
            "student_adapter_path": args.student_adapter_path,
        },
    ) as wandb_session:
        validate_inputs(
            model_id=args.model_id,
            prompt_manifest_jsonl=args.prompt_manifest_jsonl,
            eval_prompt_manifest_jsonl=args.eval_prompt_manifest_jsonl,
            problem_metadata_jsonl=args.problem_metadata_jsonl,
            student_adapter_path=args.student_adapter_path,
        )

        overrides = build_overrides(
            model_id=args.model_id,
            prompt_manifest_jsonl=args.prompt_manifest_jsonl,
            eval_prompt_manifest_jsonl=args.eval_prompt_manifest_jsonl,
            problem_metadata_jsonl=args.problem_metadata_jsonl,
            student_adapter_path=args.student_adapter_path,
        )

        materialize_config(base_config_path=args.a0_base_config, output_path=output_dir / "a0.yaml", overrides=overrides)
        materialize_config(base_config_path=args.a1_base_config, output_path=output_dir / "a1.yaml", overrides=overrides)
        materialize_config(
            base_config_path=args.a1_student_base_config,
            output_path=output_dir / "a1_student_eval.yaml",
            overrides=overrides,
        )
        materialize_config(base_config_path=args.a5_base_config, output_path=output_dir / "a5.yaml", overrides=overrides)

        manifest = build_bundle_manifest(
            output_dir=output_dir,
            model_id=args.model_id,
            prompt_manifest_jsonl=args.prompt_manifest_jsonl,
            eval_prompt_manifest_jsonl=args.eval_prompt_manifest_jsonl,
            problem_metadata_jsonl=args.problem_metadata_jsonl,
            student_adapter_path=args.student_adapter_path,
        )
        write_json(output_dir / "bundle_manifest.json", manifest)

        wandb_session.log_metrics(
            {
                "num_materialized_configs": len(manifest["materialized_configs"]),
                "num_recommended_run_dirs": len(manifest["recommended_run_dirs"]),
            },
            prefix="bundle",
        )
        wandb_session.update_summary(
            {
                "model_id": args.model_id,
                "student_adapter_path": args.student_adapter_path,
                "num_materialized_configs": len(manifest["materialized_configs"]),
            },
            prefix="bundle",
        )
        wandb_session.log_output_artifact(
            output_dir=output_dir,
            candidate_files=[
                "bundle_manifest.json",
                "a0.yaml",
                "a1.yaml",
                "a1_student_eval.yaml",
                "a5.yaml",
            ],
            artifact_type="bundle_outputs",
            metadata={
                "model_id": args.model_id,
                "student_adapter_path": args.student_adapter_path,
            },
        )


if __name__ == "__main__":
    main()
