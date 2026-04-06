from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from common import build_arg_parser, read_jsonl, resolve_config_from_args, save_resolved_config, save_run_manifest, write_json, write_jsonl
from constraint_library import compute_tropical_penalty, evaluate_trace_constraints


JsonDict = Dict[str, Any]


def build_proof_state(record: Mapping[str, Any], config: Mapping[str, Any], text_field: str, answer_field: str) -> JsonDict:
    text = str(record.get(text_field, ""))
    extracted_answer = record.get(answer_field)
    extraction_status = str(record.get("extraction_status") or record.get("extraction", {}).get("status") or "unknown")

    evaluated = evaluate_trace_constraints(
        text,
        extracted_answer=int(extracted_answer) if extracted_answer is not None else None,
        extraction_status=extraction_status,
        config=config,
    )
    features = evaluated["features"]
    checks = evaluated["constraint_checks"]
    tropical_score = compute_tropical_penalty(features, checks, config)

    return {
        "problem_id": str(record.get("problem_id", "")),
        "sample_index": int(record.get("sample_index", 0) or 0),
        "template_name": record.get("template_name"),
        "trace_text": text,
        "extracted_answer": extracted_answer,
        "steps": [],
        "features": features,
        "constraint_checks": checks,
        "tropical_score": tropical_score,
        "gold_answer": record.get("gold_answer"),
    }


def aggregate_answers(records: List[JsonDict], cfg: Mapping[str, Any]) -> List[JsonDict]:
    grouped: Dict[str, List[JsonDict]] = collections.defaultdict(list)
    for record in records:
        grouped[str(record["problem_id"])].append(record)

    support_bonus_weight = float(cfg.get("answer_score", {}).get("support_bonus_weight", 0.75))
    mean_penalty_weight = float(cfg.get("answer_score", {}).get("mean_penalty_weight", 0.20))
    min_penalty_weight = float(cfg.get("answer_score", {}).get("min_penalty_weight", 1.0))

    aggregates: List[JsonDict] = []
    for problem_id, items in sorted(grouped.items()):
        answer_groups: Dict[int, List[JsonDict]] = collections.defaultdict(list)
        invalid_items: List[JsonDict] = []
        for item in items:
            answer = item.get("extracted_answer")
            if answer is None:
                invalid_items.append(item)
                continue
            answer_groups[int(answer)].append(item)

        ranked_answers: List[JsonDict] = []
        for answer, answer_items in answer_groups.items():
            penalties = [float(x["tropical_score"]["total_penalty"]) for x in answer_items]
            min_penalty = min(penalties)
            mean_penalty = sum(penalties) / len(penalties)
            support_count = len(answer_items)
            answer_score = (
                min_penalty_weight * min_penalty
                + mean_penalty_weight * mean_penalty
                - support_bonus_weight * math.log1p(support_count)
            )
            ranked_answers.append(
                {
                    "answer": answer,
                    "support_count": support_count,
                    "min_penalty": min_penalty,
                    "mean_penalty": mean_penalty,
                    "answer_score": answer_score,
                }
            )

        ranked_answers.sort(key=lambda x: (x["answer_score"], x["min_penalty"], -x["support_count"], x["answer"]))
        predicted_answer = ranked_answers[0]["answer"] if ranked_answers else None
        gold_answer = items[0].get("gold_answer")
        is_correct = bool(predicted_answer is not None and gold_answer is not None and int(predicted_answer) == int(gold_answer))

        aggregates.append(
            {
                "problem_id": problem_id,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "is_correct": is_correct if gold_answer is not None else None,
                "num_samples": len(items),
                "num_valid_answers": sum(x["support_count"] for x in ranked_answers),
                "num_invalid_answers": len(invalid_items),
                "ranked_answers": ranked_answers,
                "best_trace_penalty": ranked_answers[0]["min_penalty"] if ranked_answers else None,
            }
        )
    return aggregates


def main() -> None:
    parser = build_arg_parser("Apply tropical reranking to extracted math traces")
    parser.add_argument("--input-jsonl", required=True, help="JSONL with generation text and extracted answers")
    parser.add_argument("--text-field", default="generation_text")
    parser.add_argument("--answer-field", default="extracted_answer")
    args = parser.parse_args()

    config = resolve_config_from_args(args)
    output_dir = Path(args.output_dir)
    save_resolved_config(output_dir, config)

    constraint_cfg = config.get("constraints", {})
    records = read_jsonl(args.input_jsonl)
    proof_states = [build_proof_state(record, constraint_cfg, args.text_field, args.answer_field) for record in records]
    aggregates = aggregate_answers(proof_states, config.get("aggregation", {}))

    with_gold = [x for x in aggregates if x.get("gold_answer") is not None]
    metrics = {
        "num_records": len(records),
        "num_problems": len(aggregates),
        "exact_final_answer_accuracy": (sum(1 for x in with_gold if x["is_correct"]) / len(with_gold)) if with_gold else None,
        "mean_best_trace_penalty": (sum(float(x["best_trace_penalty"] or 0.0) for x in aggregates) / len(aggregates)) if aggregates else None,
        "num_with_gold": len(with_gold),
        "dry_run": bool(args.dry_run),
    }

    write_jsonl(output_dir / "proof_states.jsonl", proof_states)
    write_jsonl(output_dir / "tropical_aggregates.jsonl", aggregates)
    write_json(output_dir / "tropical_metrics.json", metrics)
    save_run_manifest(
        output_dir,
        {
            "script": "tropical_rerank.py",
            "input_jsonl": str(args.input_jsonl),
            "num_records": len(records),
            "num_problems": len(aggregates),
            "dry_run": bool(args.dry_run),
        },
    )


if __name__ == "__main__":
    main()
