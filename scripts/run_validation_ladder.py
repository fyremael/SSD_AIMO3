from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from common import read_json, resolve_config_path, write_json


JsonDict = Dict[str, Any]
ROOT = Path(__file__).resolve().parents[1]


def _nested_get(mapping: Mapping[str, Any], path: Sequence[str]) -> Optional[Any]:
    cursor: Any = mapping
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _resolve_config_input(config: Mapping[str, Any], *candidate_paths: Sequence[str]) -> str:
    for candidate in candidate_paths:
        value = _nested_get(config, candidate)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(f"Could not find configured input path in candidates: {candidate_paths!r}")


def run_python_step(args: List[str], *, cwd: Path) -> None:
    subprocess.run([sys.executable, *args], cwd=str(cwd), check=True)


def build_ladder_summary(
    a0_metrics: Mapping[str, Any],
    a1_metrics: Mapping[str, Any],
    a5_metrics: Mapping[str, Any],
    a0_vs_a1: Mapping[str, Any],
    a1_vs_a5: Mapping[str, Any],
) -> JsonDict:
    a0_acc = a0_metrics.get("exact_final_answer_accuracy")
    a1_acc = a1_metrics.get("exact_final_answer_accuracy")
    a5_acc = a5_metrics.get("exact_final_answer_accuracy")
    return {
        "a0_accuracy": a0_acc,
        "a1_accuracy": a1_acc,
        "a5_accuracy": a5_acc,
        "a1_minus_a0_accuracy_delta": (a1_acc - a0_acc) if isinstance(a0_acc, (int, float)) and isinstance(a1_acc, (int, float)) else None,
        "a5_minus_a1_accuracy_delta": (a5_acc - a1_acc) if isinstance(a1_acc, (int, float)) and isinstance(a5_acc, (int, float)) else None,
        "a1_directional_gain": bool((a0_vs_a1.get("net_gain_b_minus_a") or 0) > 0 and isinstance(a1_acc, (int, float)) and isinstance(a0_acc, (int, float)) and a1_acc >= a0_acc),
        "a5_directional_gain": bool((a1_vs_a5.get("net_gain_b_minus_a") or 0) > 0 and isinstance(a5_acc, (int, float)) and isinstance(a1_acc, (int, float)) and a5_acc >= a1_acc),
        "a0_vs_a1": dict(a0_vs_a1),
        "a1_vs_a5": dict(a1_vs_a5),
    }


def render_ladder_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Validation ladder summary",
        "",
        "## Headline",
        f"- A0 exact final-answer accuracy: {summary.get('a0_accuracy')}",
        f"- A1 exact final-answer accuracy: {summary.get('a1_accuracy')}",
        f"- A5 exact final-answer accuracy: {summary.get('a5_accuracy')}",
        f"- A1 - A0 delta: {summary.get('a1_minus_a0_accuracy_delta')}",
        f"- A5 - A1 delta: {summary.get('a5_minus_a1_accuracy_delta')}",
        f"- A1 directional gain: {summary.get('a1_directional_gain')}",
        f"- A5 directional gain: {summary.get('a5_directional_gain')}",
        "",
        "## Paired A0 vs A1",
        f"- Net gain for A1: {summary['a0_vs_a1'].get('net_gain_b_minus_a')}",
        f"- Discordant pairs: {summary['a0_vs_a1'].get('discordant_pairs')}",
        f"- Sign-test p-value: {summary['a0_vs_a1'].get('paired_sign_test_pvalue_two_sided')}",
        "",
        "## Paired A1 vs A5",
        f"- Net gain for A5: {summary['a1_vs_a5'].get('net_gain_b_minus_a')}",
        f"- Discordant pairs: {summary['a1_vs_a5'].get('discordant_pairs')}",
        f"- Sign-test p-value: {summary['a1_vs_a5'].get('paired_sign_test_pvalue_two_sided')}",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the A0/A1/A5 validation ladder and emit paired summaries")
    parser.add_argument("--a0-config", default="configs/fixture_a0.yaml")
    parser.add_argument("--a1-config", default="configs/fixture_a1.yaml")
    parser.add_argument("--a5-config", default="configs/fixture_a5.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-jsonl", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    a0_cfg = resolve_config_path(args.a0_config)
    a1_cfg = resolve_config_path(args.a1_config)
    a5_cfg = resolve_config_path(args.a5_config)
    metadata_jsonl = args.metadata_jsonl or _resolve_config_input(
        a0_cfg,
        ("paths", "problem_metadata_jsonl"),
    )

    a1_generation_dir = output_dir / "a1_generation"
    a1_train_dir = output_dir / "a1_train"
    a0_eval_dir = output_dir / "a0_eval"
    a1_eval_dir = output_dir / "a1_eval"
    a5_eval_dir = output_dir / "a5_eval"
    compare_a0_a1_dir = output_dir / "compare_a0_a1"
    compare_a1_a5_dir = output_dir / "compare_a1_a5"

    a1_prompt_input = _resolve_config_input(a1_cfg, ("inputs", "prompt_input_jsonl"), ("paths", "unlabeled_prompt_manifest_jsonl"))
    a0_eval_input = _resolve_config_input(a0_cfg, ("inputs", "eval_input_jsonl"), ("paths", "eval_manifest_jsonl"))
    a1_eval_input = _resolve_config_input(a1_cfg, ("inputs", "eval_input_jsonl"), ("paths", "eval_manifest_jsonl"))
    a5_eval_input = _resolve_config_input(a5_cfg, ("inputs", "eval_input_jsonl"), ("inputs", "tropical_input_jsonl"), ("paths", "eval_manifest_jsonl"))

    commands = [
        [
            "scripts/generate_self_samples.py",
            "--config",
            args.a1_config,
            "--input-jsonl",
            a1_prompt_input,
            "--output-dir",
            str(a1_generation_dir),
        ],
        [
            "scripts/train_ssd_math.py",
            "--config",
            args.a1_config,
            "--input-jsonl",
            str(a1_generation_dir / "generations.jsonl"),
            "--output-dir",
            str(a1_train_dir),
        ],
        [
            "scripts/run_eval_math.py",
            "--config",
            args.a0_config,
            "--input-jsonl",
            a0_eval_input,
            "--output-dir",
            str(a0_eval_dir),
        ],
        [
            "scripts/run_eval_math.py",
            "--config",
            args.a1_config,
            "--input-jsonl",
            a1_eval_input,
            "--output-dir",
            str(a1_eval_dir),
        ],
        [
            "scripts/run_eval_math.py",
            "--config",
            args.a5_config,
            "--input-jsonl",
            a5_eval_input,
            "--output-dir",
            str(a5_eval_dir),
        ],
        [
            "scripts/compare_eval_runs.py",
            "--run-a-dir",
            str(a0_eval_dir),
            "--run-b-dir",
            str(a1_eval_dir),
            "--run-a-label",
            "a0_majority_vote",
            "--run-b-label",
            "a1_majority_vote",
            "--metadata-jsonl",
            metadata_jsonl,
            "--output-dir",
            str(compare_a0_a1_dir),
        ],
        [
            "scripts/compare_eval_runs.py",
            "--run-a-dir",
            str(a1_eval_dir),
            "--run-b-dir",
            str(a5_eval_dir),
            "--run-a-label",
            "a1_majority_vote",
            "--run-b-label",
            "a5_tropical_rerank",
            "--metadata-jsonl",
            metadata_jsonl,
            "--output-dir",
            str(compare_a1_a5_dir),
        ],
    ]

    if args.dry_run:
        commands = [command + ["--dry-run"] if command[0].endswith(".py") and "compare_eval_runs.py" not in command[0] else command for command in commands]

    for command in commands:
        run_python_step(command, cwd=ROOT)

    a0_metrics = read_json(a0_eval_dir / "metrics.json")
    a1_metrics = read_json(a1_eval_dir / "metrics.json")
    a5_metrics = read_json(a5_eval_dir / "metrics.json")
    a0_vs_a1 = read_json(compare_a0_a1_dir / "comparison_summary.json")
    a1_vs_a5 = read_json(compare_a1_a5_dir / "comparison_summary.json")

    summary = build_ladder_summary(a0_metrics, a1_metrics, a5_metrics, a0_vs_a1, a1_vs_a5)
    write_json(output_dir / "ladder_summary.json", summary)
    (output_dir / "ladder_report.md").write_text(render_ladder_report(summary), encoding="utf-8")
    write_json(
        output_dir / "ladder_manifest.json",
        {
            "a0_config": str(Path(args.a0_config).resolve()),
            "a1_config": str(Path(args.a1_config).resolve()),
            "a5_config": str(Path(args.a5_config).resolve()),
            "metadata_jsonl": str(Path(metadata_jsonl).resolve()),
            "commands": commands,
            "dry_run": bool(args.dry_run),
        },
    )


if __name__ == "__main__":
    main()
