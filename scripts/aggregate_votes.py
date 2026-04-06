from __future__ import annotations

from pathlib import Path

from common import build_arg_parser, read_jsonl, resolve_config_from_args, save_resolved_config, save_run_manifest, write_json, write_jsonl
from run_eval_math import aggregate_majority_vote, compute_eval_metrics


def main() -> None:
    parser = build_arg_parser("Aggregate extracted answers with majority vote")
    parser.add_argument("--input-jsonl", required=True, help="JSONL with extracted answers")
    args = parser.parse_args()

    config = resolve_config_from_args(args)
    output_dir = Path(args.output_dir)
    save_resolved_config(output_dir, config)

    records = read_jsonl(args.input_jsonl)
    aggregates = aggregate_majority_vote(records)
    metrics = compute_eval_metrics(aggregates, backend="majority_vote", dry_run=bool(args.dry_run))

    write_jsonl(output_dir / "aggregates.jsonl", aggregates)
    write_json(output_dir / "metrics.json", metrics)
    save_run_manifest(
        output_dir,
        {
            "script": "aggregate_votes.py",
            "input_jsonl": str(args.input_jsonl),
            "num_records": len(records),
            "num_problems": len(aggregates),
            "dry_run": bool(args.dry_run),
        },
    )


if __name__ == "__main__":
    main()
