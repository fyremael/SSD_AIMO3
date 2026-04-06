from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from common import (
    build_arg_parser,
    read_jsonl,
    resolve_config_from_args,
    save_resolved_config,
    save_run_manifest,
    write_json,
    write_jsonl,
)


JsonDict = Dict[str, Any]


DEFAULT_MAX_ANSWER = 99999

_BOXED_PATTERN = re.compile(r"\\boxed\{\s*([-+]?\d{1,10})\s*\}")
_FINAL_ANSWER_PATTERN = re.compile(r"(?i)\bfinal\s+answer\b\s*[:=]\s*(?:\\boxed\{\s*)?([-+]?\d{1,10})(?:\s*\})?")
_ANSWER_PATTERN = re.compile(r"(?i)\banswer\b\s*[:=]\s*(?:\\boxed\{\s*)?([-+]?\d{1,10})(?:\s*\})?")
_INTEGER_PATTERN = re.compile(r"(?<!\d)([-+]?\d{1,10})(?!\d)")
_PATTERN_PRIORITY = {"boxed": 0, "final_answer": 1, "answer": 2}


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _in_range(value: Optional[int], *, max_answer: int) -> bool:
    return value is not None and 0 <= value <= max_answer


def _extract_candidates(text: str) -> List[Tuple[str, int, int]]:
    candidates: List[Tuple[str, int, int]] = []
    for label, pattern in (
        ("boxed", _BOXED_PATTERN),
        ("final_answer", _FINAL_ANSWER_PATTERN),
        ("answer", _ANSWER_PATTERN),
    ):
        for match in pattern.finditer(text):
            value = _safe_int(match.group(1))
            if value is not None:
                candidates.append((label, value, match.start()))
    return candidates


def _last_integer_after_marker(text: str) -> Optional[Tuple[int, str]]:
    marker = re.search(r"(?i)\b(final\s+answer|answer)\b\s*[:=]?", text)
    if marker is None:
        return None
    tail = text[marker.end() :]
    integers = [_safe_int(match.group(1)) for match in _INTEGER_PATTERN.finditer(tail)]
    integers = [value for value in integers if value is not None]
    if not integers:
        return None
    return integers[-1], "tail_integer"


def extract_final_answer(text: str, *, max_answer: int = DEFAULT_MAX_ANSWER) -> JsonDict:
    normalized = str(text or "")
    candidates = _extract_candidates(normalized)
    valid_candidates = [(label, value, pos) for label, value, pos in candidates if _in_range(value, max_answer=max_answer)]
    invalid_candidates = [(label, value, pos) for label, value, pos in candidates if not _in_range(value, max_answer=max_answer)]

    unique_values = sorted({value for _, value, _ in valid_candidates})
    if len(unique_values) > 1:
        return {
            "extracted_answer": None,
            "extraction_status": "conflict",
            "extraction_method": "multiple_candidates",
            "candidate_answers": unique_values,
            "matched_text": None,
        }

    if len(unique_values) == 1:
        value = unique_values[0]
        winning = [item for item in valid_candidates if item[1] == value]
        winning.sort(key=lambda item: (_PATTERN_PRIORITY.get(item[0], 99), -item[2]))
        label, _, pos = winning[0]
        return {
            "extracted_answer": value,
            "extraction_status": "ok",
            "extraction_method": label,
            "candidate_answers": unique_values,
            "matched_text": normalized[pos : pos + 64].splitlines()[0],
        }

    fallback = _last_integer_after_marker(normalized)
    if fallback is not None:
        value, method = fallback
        if _in_range(value, max_answer=max_answer):
            return {
                "extracted_answer": value,
                "extraction_status": "ok",
                "extraction_method": method,
                "candidate_answers": [value],
                "matched_text": None,
            }

    if invalid_candidates:
        return {
            "extracted_answer": None,
            "extraction_status": "out_of_range",
            "extraction_method": "pattern_match",
            "candidate_answers": sorted({value for _, value, _ in invalid_candidates}),
            "matched_text": None,
        }

    return {
        "extracted_answer": None,
        "extraction_status": "missing",
        "extraction_method": "none",
        "candidate_answers": [],
        "matched_text": None,
    }


def extract_answer(text: str, *, max_answer: int = DEFAULT_MAX_ANSWER) -> JsonDict:
    return extract_final_answer(text, max_answer=max_answer)


def extract_record(record: Mapping[str, Any], *, text_field: str = "generation_text", max_answer: int = DEFAULT_MAX_ANSWER) -> JsonDict:
    text = str(record.get(text_field, ""))
    extracted = extract_final_answer(text, max_answer=max_answer)
    result = dict(record)
    result.update(extracted)
    result["text_field"] = text_field
    return result


def _count_status(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("extraction_status") or "missing")
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> None:
    parser = build_arg_parser("Extract a canonical final integer answer from math traces")
    parser.add_argument("--input-jsonl", required=True, help="JSONL with generated traces")
    parser.add_argument("--text-field", default="generation_text")
    parser.add_argument("--max-answer", type=int, default=DEFAULT_MAX_ANSWER)
    args = parser.parse_args()

    config = resolve_config_from_args(args)
    output_dir = Path(args.output_dir)
    save_resolved_config(output_dir, config)

    records = read_jsonl(args.input_jsonl)
    extracted_rows = [extract_record(record, text_field=args.text_field, max_answer=args.max_answer) for record in records]
    status_counts = _count_status(extracted_rows)
    ok_count = status_counts.get("ok", 0)

    metrics = {
        "num_records": len(extracted_rows),
        "num_ok": ok_count,
        "extraction_success_rate": (ok_count / len(extracted_rows)) if extracted_rows else None,
        "status_counts": status_counts,
        "max_answer": args.max_answer,
        "dry_run": bool(args.dry_run),
    }

    write_jsonl(output_dir / "extracted.jsonl", extracted_rows)
    write_json(output_dir / "extraction_metrics.json", metrics)
    save_run_manifest(
        output_dir,
        {
            "script": "extract_answer.py",
            "input_jsonl": str(args.input_jsonl),
            "num_records": len(records),
            "num_extracted": len(extracted_rows),
            "dry_run": bool(args.dry_run),
            "text_field": args.text_field,
            "max_answer": args.max_answer,
        },
    )


if __name__ == "__main__":
    main()
