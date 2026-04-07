from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from common import (
    build_arg_parser,
    read_jsonl,
    render_string_template,
    resolve_config_from_args,
    save_resolved_config,
    save_run_manifest,
    write_json,
    write_jsonl,
)
from extract_answer import DEFAULT_MAX_ANSWER, extract_record
from wandb_support import wandb_run_context


JsonDict = Dict[str, Any]


def _count_values(rows: Iterable[Mapping[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "missing")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _normalize_generation_record(record: Mapping[str, Any], *, generation_field: str, max_answer: int) -> JsonDict:
    normalized = dict(record)
    if normalized.get("extracted_answer") is None and str(normalized.get(generation_field, "")).strip():
        normalized = extract_record(normalized, text_field=generation_field, max_answer=max_answer)
    return normalized


def _should_keep_for_training(record: Mapping[str, Any], filtering_cfg: Mapping[str, Any]) -> tuple[bool, Optional[str]]:
    if str(record.get("filter_status") or "kept") != "kept":
        return False, str(record.get("filter_reason") or "filtered_upstream")

    require_extracted = bool(filtering_cfg.get("require_extracted_answer", False))
    extraction_status = str(record.get("extraction_status") or "missing")
    if require_extracted and extraction_status != "ok":
        return False, f"extraction_status={extraction_status}"
    if not str(record.get("generation_text", "")).strip():
        return False, "empty_generation"
    if not str(record.get("prompt_text", "")).strip():
        return False, "empty_prompt"
    return True, None


def build_training_dataset(
    records: List[JsonDict],
    *,
    generation_field: str,
    max_answer: int,
    filtering_cfg: Mapping[str, Any],
) -> tuple[List[JsonDict], List[JsonDict]]:
    dataset_rows: List[JsonDict] = []
    audit_rows: List[JsonDict] = []

    for record in records:
        normalized = _normalize_generation_record(record, generation_field=generation_field, max_answer=max_answer)
        keep, reason = _should_keep_for_training(normalized, filtering_cfg)
        audit_row = {
            "problem_id": normalized.get("problem_id"),
            "sample_index": normalized.get("sample_index"),
            "keep_for_training": keep,
            "audit_reason": reason or "kept",
            "extraction_status": normalized.get("extraction_status"),
            "filter_status": normalized.get("filter_status"),
        }
        audit_rows.append(audit_row)
        if not keep:
            continue

        prompt_text = str(normalized.get("rendered_prompt_text") or normalized.get("prompt_text") or "")
        generation_text = str(normalized.get(generation_field, "")).strip()
        dataset_rows.append(
            {
                "problem_id": normalized.get("problem_id"),
                "sample_index": normalized.get("sample_index"),
                "prompt_text": prompt_text,
                "completion_text": generation_text,
                "training_text": f"{prompt_text}\n\n{generation_text}",
                "template_name": normalized.get("template_name"),
                "extracted_answer": normalized.get("extracted_answer"),
                "extraction_status": normalized.get("extraction_status"),
            }
        )

    return dataset_rows, audit_rows


def build_training_plan(config: Mapping[str, Any], *, output_dir: Path, num_training_rows: int) -> JsonDict:
    training_cfg = config.get("training", {})
    launcher_cfg = training_cfg.get("launcher", {}) if isinstance(training_cfg.get("launcher"), Mapping) else {}
    return {
        "mode": training_cfg.get("mode", "sft"),
        "save_dataset_only": bool(training_cfg.get("save_dataset_only", True)),
        "max_seq_length": training_cfg.get("max_seq_length"),
        "learning_rate": training_cfg.get("learning_rate"),
        "num_train_epochs": training_cfg.get("num_train_epochs"),
        "per_device_train_batch_size": training_cfg.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps"),
        "launcher_enabled": bool(launcher_cfg.get("enabled", False)),
        "launcher_command": launcher_cfg.get("command"),
        "launcher_workdir": launcher_cfg.get("workdir"),
        "dataset_path": str((output_dir / "train_dataset.jsonl").resolve()),
        "train_rows": num_training_rows,
        "launch_status": "planned_dataset_only",
    }


def maybe_launch_training(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    training_plan: JsonDict,
    num_training_rows: int,
    requested_launch: bool,
) -> Optional[JsonDict]:
    training_cfg = config.get("training", {})
    launcher_cfg = training_cfg.get("launcher", {}) if isinstance(training_cfg.get("launcher"), Mapping) else {}
    enabled = bool(launcher_cfg.get("enabled", False))
    command_template = launcher_cfg.get("command")
    if not (requested_launch or enabled):
        training_plan["launch_status"] = "skipped"
        return None
    if not isinstance(command_template, str) or not command_template.strip():
        raise ValueError("training.launcher.command must be configured when launching training")

    dataset_path = (output_dir / "train_dataset.jsonl").resolve()
    values = {
        "python_executable": sys.executable,
        "workspace_root": str(Path.cwd().resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset_path": str(dataset_path),
        "model_id": str(config.get("model", {}).get("base_model_id") or ""),
        "learning_rate": training_cfg.get("learning_rate"),
        "num_train_epochs": training_cfg.get("num_train_epochs"),
        "max_seq_length": training_cfg.get("max_seq_length"),
        "train_rows": num_training_rows,
    }
    command = render_string_template(str(command_template), values)
    workdir = Path(str(launcher_cfg.get("workdir") or Path.cwd())).resolve()
    subprocess.run(command, cwd=str(workdir), shell=True, check=True)

    launch_record = {
        "command": command,
        "workdir": str(workdir),
        "dataset_path": str(dataset_path),
        "status": "completed",
    }
    training_plan["launch_status"] = "completed"
    return launch_record


def main() -> None:
    parser = build_arg_parser("Build an SSD math SFT dataset package from self-generated traces")
    parser.add_argument("--input-jsonl", required=True, help="JSONL with generated self-samples")
    parser.add_argument("--generation-field", default="generation_text")
    parser.add_argument("--launch", action="store_true", help="Execute configured external training launcher")
    args = parser.parse_args()

    config = resolve_config_from_args(args)
    output_dir = Path(args.output_dir)
    with wandb_run_context(
        config=config,
        output_dir=output_dir,
        script_name="train_ssd_math.py",
        job_type="training_dataset",
        extra_config={
            "input_jsonl": args.input_jsonl,
            "generation_field": args.generation_field,
            "requested_launch": bool(args.launch),
            "dry_run": bool(args.dry_run),
        },
    ) as wandb_session:
        save_resolved_config(output_dir, config)

        records = read_jsonl(args.input_jsonl)
        filtering_cfg = config.get("filtering", {})
        max_answer = int(config.get("extraction", {}).get("max_answer", DEFAULT_MAX_ANSWER))
        dataset_rows, audit_rows = build_training_dataset(
            records,
            generation_field=args.generation_field,
            max_answer=max_answer,
            filtering_cfg=filtering_cfg,
        )

        metrics = {
            "num_input_rows": len(records),
            "num_training_rows": len(dataset_rows),
            "num_rejected_rows": len(audit_rows) - len(dataset_rows),
            "keep_for_training_counts": _count_values(audit_rows, "keep_for_training"),
            "rejection_reason_counts": _count_values(
                [row for row in audit_rows if not row.get("keep_for_training")],
                "audit_reason",
            ),
            "template_name_counts": _count_values(dataset_rows, "template_name"),
            "dry_run": bool(args.dry_run),
        }
        training_plan = build_training_plan(config, output_dir=output_dir, num_training_rows=len(dataset_rows))

        write_jsonl(output_dir / "train_dataset.jsonl", dataset_rows)
        write_jsonl(output_dir / "training_audit.jsonl", audit_rows)
        write_json(output_dir / "training_metrics.json", metrics)

        launch_record = maybe_launch_training(
            config=config,
            output_dir=output_dir,
            training_plan=training_plan,
            num_training_rows=len(dataset_rows),
            requested_launch=bool(args.launch),
        )

        write_json(output_dir / "training_plan.json", training_plan)
        if launch_record is not None:
            write_json(output_dir / "training_launch.json", launch_record)

        save_run_manifest(
            output_dir,
            {
                "script": "train_ssd_math.py",
                "input_jsonl": str(args.input_jsonl),
                "num_input_rows": len(records),
                "num_training_rows": len(dataset_rows),
                "launch_status": training_plan.get("launch_status"),
                "dry_run": bool(args.dry_run),
            },
        )

        wandb_session.log_metrics(metrics, prefix="training")
        wandb_session.update_summary(
            {
                "num_input_rows": len(records),
                "num_training_rows": len(dataset_rows),
                "launch_status": training_plan.get("launch_status"),
                "save_dataset_only": training_plan.get("save_dataset_only"),
                "dry_run": bool(args.dry_run),
            },
            prefix="training",
        )
        wandb_session.log_output_artifact(
            output_dir=output_dir,
            candidate_files=[
                "config_resolved.yaml",
                "training_metrics.json",
                "training_plan.json",
                "training_launch.json",
                "run_manifest.json",
                "adapter_summary.json",
            ],
            artifact_type="training_outputs",
            metadata={
                "num_training_rows": len(dataset_rows),
                "launch_status": training_plan.get("launch_status"),
            },
        )


if __name__ == "__main__":
    main()
