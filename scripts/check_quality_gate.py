from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from common import add_verbosity_args, log_event, read_json, read_jsonl, write_json


JsonDict = Dict[str, Any]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _snippet(value: Any, *, limit: int = 240) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def compute_gate_report(
    *,
    label: str,
    metrics: Mapping[str, Any],
    sample_rows: Optional[Iterable[Mapping[str, Any]]] = None,
    min_valid_answer_rate: Optional[float],
    min_extraction_success_rate: Optional[float],
    max_failure_examples: int,
    text_field: str,
) -> JsonDict:
    sample_rows = list(sample_rows or [])
    valid_answer_rate = _safe_float(metrics.get("valid_answer_rate"))
    extraction_success_rate = _safe_float(
        metrics.get("sample_extraction_success_rate", metrics.get("extraction_success_rate"))
    )

    failures: List[JsonDict] = []
    if min_valid_answer_rate is not None:
        if valid_answer_rate is None or valid_answer_rate < float(min_valid_answer_rate):
            failures.append(
                {
                    "metric": "valid_answer_rate",
                    "value": valid_answer_rate,
                    "threshold": float(min_valid_answer_rate),
                }
            )
    if min_extraction_success_rate is not None:
        if extraction_success_rate is None or extraction_success_rate < float(min_extraction_success_rate):
            failures.append(
                {
                    "metric": "extraction_success_rate",
                    "value": extraction_success_rate,
                    "threshold": float(min_extraction_success_rate),
                }
            )

    failure_examples: List[JsonDict] = []
    if sample_rows:
        for row in sample_rows:
            status = str(row.get("extraction_status") or "missing")
            if status == "ok":
                continue
            failure_examples.append(
                {
                    "problem_id": row.get("problem_id"),
                    "sample_index": row.get("sample_index"),
                    "extraction_status": status,
                    "prompt_text": _snippet(row.get("prompt_text")),
                    text_field: _snippet(row.get(text_field)),
                }
            )
            if len(failure_examples) >= max(0, int(max_failure_examples)):
                break

    return {
        "label": label,
        "passed": not failures,
        "failures": failures,
        "metrics_summary": {
            "valid_answer_rate": valid_answer_rate,
            "extraction_success_rate": extraction_success_rate,
            "num_problems": metrics.get("num_problems"),
            "num_sample_records": metrics.get("num_sample_records", metrics.get("num_generations")),
        },
        "failure_examples": failure_examples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check quality gates for SSD_AIMO3 stage outputs")
    parser.add_argument("--label", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--samples-jsonl", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-valid-answer-rate", type=float, default=None)
    parser.add_argument("--min-extraction-success-rate", type=float, default=None)
    parser.add_argument("--max-failure-examples", type=int, default=3)
    parser.add_argument("--text-field", default="generation_text")
    parser.add_argument("--enforce", action="store_true")
    return add_verbosity_args(parser)


def main() -> None:
    args = build_parser().parse_args()
    metrics = read_json(args.metrics_json)
    sample_rows = read_jsonl(args.samples_jsonl) if args.samples_jsonl else []
    verbose = not bool(args.quiet)

    log_event(
        "Checking quality gate",
        payload={
            "label": args.label,
            "metrics_json": str(Path(args.metrics_json).resolve()),
            "samples_jsonl": str(Path(args.samples_jsonl).resolve()) if args.samples_jsonl else None,
            "min_valid_answer_rate": args.min_valid_answer_rate,
            "min_extraction_success_rate": args.min_extraction_success_rate,
            "enforce": bool(args.enforce),
        },
        verbose=verbose,
    )

    report = compute_gate_report(
        label=args.label,
        metrics=metrics,
        sample_rows=sample_rows,
        min_valid_answer_rate=args.min_valid_answer_rate,
        min_extraction_success_rate=args.min_extraction_success_rate,
        max_failure_examples=args.max_failure_examples,
        text_field=args.text_field,
    )
    write_json(Path(args.output_json), report)
    log_event("Quality gate report", payload=report, verbose=verbose)

    if bool(args.enforce) and not bool(report["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
