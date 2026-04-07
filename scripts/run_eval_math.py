from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from common import (
    build_arg_parser,
    read_jsonl,
    resolve_config_from_args,
    save_resolved_config,
    save_run_manifest,
    write_json,
    write_jsonl,
)
from extract_answer import extract_record
from tropical_rerank import aggregate_answers as tropical_aggregate_answers
from tropical_rerank import build_proof_state
from wandb_support import wandb_run_context


JsonDict = Dict[str, Any]
DEFAULT_EXTRACTION_POLICY = "overwrite_all"


def _vote_entropy(counts: Iterable[int]) -> float:
    values = [int(x) for x in counts if int(x) > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in values:
        p = count / total
        entropy -= p * math.log(p)
    return entropy


def aggregate_majority_vote(records: List[JsonDict]) -> List[JsonDict]:
    grouped: Dict[str, List[JsonDict]] = collections.defaultdict(list)
    for record in records:
        grouped[str(record.get("problem_id", ""))].append(record)

    aggregates: List[JsonDict] = []
    for problem_id, items in sorted(grouped.items()):
        by_answer: Dict[int, List[JsonDict]] = collections.defaultdict(list)
        invalid = 0
        for item in items:
            answer = item.get("extracted_answer")
            if answer is None:
                invalid += 1
                continue
            by_answer[int(answer)].append(item)

        ranked_answers: List[JsonDict] = []
        for answer, answer_items in by_answer.items():
            ranked_answers.append(
                {
                    "answer": answer,
                    "support_count": len(answer_items),
                    "mean_penalty": None,
                    "min_penalty": None,
                    "answer_score": -len(answer_items),
                }
            )
        ranked_answers.sort(key=lambda x: (-x["support_count"], x["answer"]))
        predicted_answer = ranked_answers[0]["answer"] if ranked_answers else None
        gold_answer = items[0].get("gold_answer")
        is_correct = bool(predicted_answer is not None and gold_answer is not None and int(predicted_answer) == int(gold_answer))
        support_counts = [x["support_count"] for x in ranked_answers]
        max_support = max(support_counts) if support_counts else 0
        vote_margin = None
        if len(support_counts) >= 2:
            ordered = sorted(support_counts, reverse=True)
            vote_margin = ordered[0] - ordered[1]
        elif len(support_counts) == 1:
            vote_margin = support_counts[0]

        aggregates.append(
            {
                "problem_id": problem_id,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "is_correct": is_correct if gold_answer is not None else None,
                "num_samples": len(items),
                "num_valid_answers": sum(support_counts),
                "num_invalid_answers": invalid,
                "ranked_answers": ranked_answers,
                "vote_entropy": _vote_entropy(support_counts),
                "vote_margin": vote_margin,
                "max_support": max_support,
            }
        )
    return aggregates


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_precomputed_record(record: Mapping[str, Any], *, answer_field: str, max_answer: int) -> JsonDict:
    normalized = dict(record)
    raw_answer = _coerce_int(record.get(answer_field))
    extracted_answer = raw_answer if raw_answer is not None and 0 <= raw_answer <= max_answer else None
    if raw_answer is not None and extracted_answer is None:
        extraction_status = "out_of_range"
        candidate_answers = [raw_answer]
    elif extracted_answer is not None:
        extraction_status = str(record.get("extraction_status") or record.get("extraction", {}).get("status") or "ok")
        candidate_answers = [extracted_answer]
    else:
        extraction_status = str(record.get("extraction_status") or record.get("extraction", {}).get("status") or "missing")
        candidate_answers = []

    normalized["extracted_answer"] = extracted_answer
    normalized["extraction_status"] = extraction_status
    normalized["extraction_method"] = record.get("extraction_method") or ("precomputed" if extracted_answer is not None else "none")
    normalized["candidate_answers"] = record.get("candidate_answers") or candidate_answers
    return normalized


def prepare_records_for_eval(
    records: List[JsonDict],
    *,
    text_field: str,
    answer_field: str,
    extraction_policy: str,
    max_answer: int,
) -> List[JsonDict]:
    prepared: List[JsonDict] = []
    for record in records:
        normalized = _normalize_precomputed_record(record, answer_field=answer_field, max_answer=max_answer)
        has_text = bool(str(record.get(text_field, "")).strip())
        should_extract = extraction_policy == "overwrite_all" or (
            extraction_policy == "overwrite_missing" and normalized.get("extracted_answer") is None
        )

        if should_extract and has_text:
            normalized = extract_record(record, text_field=text_field, max_answer=max_answer)
        prepared.append(normalized)
    return prepared


def _count_extraction_statuses(records: List[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in records:
        status = str(row.get("extraction_status") or "missing")
        counts[status] = counts.get(status, 0) + 1
    return counts


def compute_eval_metrics(
    aggregates: List[JsonDict],
    *,
    backend: str,
    dry_run: bool,
    sample_records: Optional[List[JsonDict]] = None,
) -> JsonDict:
    with_gold = [x for x in aggregates if x.get("gold_answer") is not None]
    exact_accuracy = None
    if with_gold:
        exact_accuracy = sum(1 for x in with_gold if x.get("is_correct")) / len(with_gold)

    valid_rate = None
    if aggregates:
        valid_rate = sum(int(x.get("num_valid_answers", 0)) for x in aggregates) / max(
            1, sum(int(x.get("num_samples", 0)) for x in aggregates)
        )

    extraction_status_counts = _count_extraction_statuses(sample_records or [])
    sample_ok_count = extraction_status_counts.get("ok", 0)

    metrics: JsonDict = {
        "aggregation_backend": backend,
        "num_problems": len(aggregates),
        "num_with_gold": len(with_gold),
        "exact_final_answer_accuracy": exact_accuracy,
        "mean_num_samples": (sum(int(x.get("num_samples", 0)) for x in aggregates) / len(aggregates)) if aggregates else None,
        "mean_num_invalid_answers": (sum(int(x.get("num_invalid_answers", 0)) for x in aggregates) / len(aggregates)) if aggregates else None,
        "valid_answer_rate": valid_rate,
        "mean_vote_entropy": (
            sum(float(x.get("vote_entropy", 0.0)) for x in aggregates) / len(aggregates)
            if aggregates and "vote_entropy" in aggregates[0]
            else None
        ),
        "mean_vote_margin": (
            sum(float(x.get("vote_margin", 0.0)) for x in aggregates if x.get("vote_margin") is not None)
            / max(1, sum(1 for x in aggregates if x.get("vote_margin") is not None))
            if aggregates and any(x.get("vote_margin") is not None for x in aggregates)
            else None
        ),
        "mean_best_trace_penalty": (
            sum(float(x.get("best_trace_penalty", 0.0)) for x in aggregates if x.get("best_trace_penalty") is not None)
            / max(1, sum(1 for x in aggregates if x.get("best_trace_penalty") is not None))
            if aggregates and any(x.get("best_trace_penalty") is not None for x in aggregates)
            else None
        ),
        "num_sample_records": len(sample_records or []),
        "sample_extraction_success_rate": (sample_ok_count / len(sample_records)) if sample_records else None,
        "sample_extraction_status_counts": extraction_status_counts,
        "dry_run": dry_run,
    }
    return metrics


def main() -> None:
    parser = build_arg_parser("Run math evaluation with optional tropical reranking backend")
    parser.add_argument("--input-jsonl", required=True, help="JSONL with raw generations or per-sample extracted answers")
    parser.add_argument("--text-field", default="generation_text")
    parser.add_argument("--answer-field", default="extracted_answer")
    parser.add_argument(
        "--aggregation-backend",
        choices=["auto", "majority_vote", "tropical_rerank"],
        default="auto",
        help="Aggregation backend to apply",
    )
    parser.add_argument(
        "--extraction-policy",
        choices=["reuse", "overwrite_missing", "overwrite_all"],
        default=None,
        help="Whether to trust pre-extracted answers or canonicalize from raw text",
    )
    parser.add_argument("--max-answer", type=int, default=None, help="Maximum allowed extracted integer answer")
    args = parser.parse_args()

    config = resolve_config_from_args(args)
    output_dir = Path(args.output_dir)
    with wandb_run_context(
        config=config,
        output_dir=output_dir,
        script_name="run_eval_math.py",
        job_type="evaluation",
        extra_config={
            "input_jsonl": args.input_jsonl,
            "aggregation_backend": args.aggregation_backend,
            "text_field": args.text_field,
            "answer_field": args.answer_field,
            "dry_run": bool(args.dry_run),
        },
    ) as wandb_session:
        save_resolved_config(output_dir, config)

        requested_backend = args.aggregation_backend
        config_backend = str(config.get("aggregation", {}).get("strategy", "majority_vote"))
        backend = config_backend if requested_backend == "auto" else requested_backend
        extraction_cfg = config.get("extraction", {})
        extraction_policy = args.extraction_policy or str(extraction_cfg.get("policy", DEFAULT_EXTRACTION_POLICY))
        max_answer = int(args.max_answer if args.max_answer is not None else extraction_cfg.get("max_answer", 99999))

        records = read_jsonl(args.input_jsonl)
        prepared_records = prepare_records_for_eval(
            records,
            text_field=args.text_field,
            answer_field=args.answer_field,
            extraction_policy=extraction_policy,
            max_answer=max_answer,
        )
        write_jsonl(output_dir / "prepared_samples.jsonl", prepared_records)

        proof_states: Optional[List[JsonDict]] = None
        if backend == "tropical_rerank":
            constraint_cfg = config.get("constraints", {})
            proof_states = [build_proof_state(record, constraint_cfg, args.text_field, "extracted_answer") for record in prepared_records]
            aggregates = tropical_aggregate_answers(proof_states, config.get("aggregation", {}))
            write_jsonl(output_dir / "proof_states.jsonl", proof_states)
            write_jsonl(output_dir / "tropical_aggregates.jsonl", aggregates)
        elif backend == "majority_vote":
            aggregates = aggregate_majority_vote(prepared_records)
            write_jsonl(output_dir / "aggregates.jsonl", aggregates)
        else:
            raise ValueError(f"Unsupported aggregation backend: {backend}")

        metrics = compute_eval_metrics(aggregates, backend=backend, dry_run=bool(args.dry_run), sample_records=prepared_records)
        write_json(output_dir / "metrics.json", metrics)
        save_run_manifest(
            output_dir,
            {
                "script": "run_eval_math.py",
                "input_jsonl": str(args.input_jsonl),
                "aggregation_backend": backend,
                "num_records": len(records),
                "num_prepared_records": len(prepared_records),
                "num_problems": len(aggregates),
                "extraction_policy": extraction_policy,
                "max_answer": max_answer,
                "dry_run": bool(args.dry_run),
            },
        )

        wandb_session.log_metrics(metrics, prefix="evaluation")
        wandb_session.update_summary(
            {
                "aggregation_backend": backend,
                "num_records": len(records),
                "num_prepared_records": len(prepared_records),
                "num_problems": len(aggregates),
                "extraction_policy": extraction_policy,
                "max_answer": max_answer,
                "dry_run": bool(args.dry_run),
            },
            prefix="evaluation",
        )
        wandb_session.log_output_artifact(
            output_dir=output_dir,
            candidate_files=[
                "config_resolved.yaml",
                "metrics.json",
                "run_manifest.json",
                "aggregates.jsonl",
                "tropical_aggregates.jsonl",
                "proof_states.jsonl",
            ],
            artifact_type="evaluation_outputs",
            metadata={
                "aggregation_backend": backend,
                "num_records": len(records),
                "num_prepared_records": len(prepared_records),
            },
        )


if __name__ == "__main__":
    main()
