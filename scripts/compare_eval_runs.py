from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from common import ensure_parent, read_json, read_jsonl, write_json, write_jsonl


JsonDict = Dict[str, Any]


def _load_optional_json(path: Optional[str]) -> Optional[JsonDict]:
    if not path:
        return None
    return read_json(path)


def _find_first_existing(run_dir: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def load_run_artifacts(run_dir: str | Path) -> Tuple[JsonDict, List[JsonDict], str]:
    root = Path(run_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")

    metrics_path = _find_first_existing(root, ["metrics.json", "tropical_metrics.json"])
    aggregates_path = _find_first_existing(root, ["aggregates.jsonl", "tropical_aggregates.jsonl"])
    if metrics_path is None:
        raise FileNotFoundError(f"Could not find metrics file in {root}")
    if aggregates_path is None:
        raise FileNotFoundError(f"Could not find aggregates file in {root}")

    metrics = read_json(metrics_path)
    aggregates = read_jsonl(aggregates_path)
    backend = str(metrics.get("aggregation_backend") or metrics.get("backend") or aggregates_path.stem)
    return metrics, aggregates, backend


def index_by_problem(records: List[JsonDict]) -> Dict[str, JsonDict]:
    indexed: Dict[str, JsonDict] = {}
    for row in records:
        indexed[str(row.get("problem_id", ""))] = row
    return indexed


def load_metadata_index(path: Optional[str]) -> Dict[str, JsonDict]:
    if not path:
        return {}
    rows = read_jsonl(path)
    return {str(row.get("problem_id", "")): row for row in rows}


def _normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value)]


def _slice_keys(problem_row: Mapping[str, Any], metadata_row: Mapping[str, Any]) -> Dict[str, List[str]]:
    topic = metadata_row.get("topic", problem_row.get("topic"))
    difficulty = metadata_row.get("difficulty", problem_row.get("difficulty"))
    tags = metadata_row.get("tags", problem_row.get("tags"))
    return {
        "topic": _normalize_tags(topic),
        "difficulty": _normalize_tags(difficulty),
        "tags": _normalize_tags(tags),
    }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _support_proxy(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("vote_margin", "max_support", "num_valid_answers"):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _exact_sign_test_pvalue_two_sided(run_a_only_correct: int, run_b_only_correct: int) -> Optional[float]:
    discordant = int(run_a_only_correct) + int(run_b_only_correct)
    if discordant <= 0:
        return None
    smaller_tail = min(int(run_a_only_correct), int(run_b_only_correct))
    tail_mass = sum(math.comb(discordant, k) for k in range(0, smaller_tail + 1)) / float(2**discordant)
    return min(1.0, 2.0 * tail_mass)


def compare_runs(
    run_a_records: List[JsonDict],
    run_b_records: List[JsonDict],
    *,
    run_a_label: str,
    run_b_label: str,
    metadata_index: Optional[Mapping[str, JsonDict]] = None,
) -> Tuple[JsonDict, List[JsonDict]]:
    metadata_index = metadata_index or {}
    a_by_problem = index_by_problem(run_a_records)
    b_by_problem = index_by_problem(run_b_records)
    all_problem_ids = sorted(set(a_by_problem) | set(b_by_problem))

    slice_buckets: Dict[str, Dict[str, List[JsonDict]]] = defaultdict(lambda: defaultdict(list))
    comparisons: List[JsonDict] = []

    summary: JsonDict = {
        "run_a_label": run_a_label,
        "run_b_label": run_b_label,
        "num_problems_union": len(all_problem_ids),
        "num_problems_intersection": len(set(a_by_problem) & set(b_by_problem)),
        "run_a_only_correct": 0,
        "run_b_only_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
        "answer_flips": 0,
        "missing_in_run_a": 0,
        "missing_in_run_b": 0,
        "problems_with_gold": 0,
        "discordant_pairs": 0,
        "net_gain_b_minus_a": 0,
        "win_rate_b_given_discordant": None,
        "paired_sign_test_pvalue_two_sided": None,
        "mean_accuracy_delta_b_minus_a": None,
        "mean_vote_margin_delta_b_minus_a": None,
        "mean_best_trace_penalty_delta_b_minus_a": None,
        "topic_slices": {},
        "difficulty_slices": {},
        "tag_slices": {},
    }

    accuracy_deltas: List[float] = []
    vote_margin_deltas: List[float] = []
    penalty_deltas: List[float] = []

    for problem_id in all_problem_ids:
        a = a_by_problem.get(problem_id)
        b = b_by_problem.get(problem_id)
        meta = metadata_index.get(problem_id, {})

        if a is None:
            summary["missing_in_run_a"] += 1
        if b is None:
            summary["missing_in_run_b"] += 1

        a_pred = a.get("predicted_answer") if a else None
        b_pred = b.get("predicted_answer") if b else None
        gold = None
        if a and a.get("gold_answer") is not None:
            gold = a.get("gold_answer")
        elif b and b.get("gold_answer") is not None:
            gold = b.get("gold_answer")
        elif meta.get("gold_answer") is not None:
            gold = meta.get("gold_answer")

        a_correct = a.get("is_correct") if a else None
        b_correct = b.get("is_correct") if b else None
        if gold is not None:
            summary["problems_with_gold"] += 1
            if a_correct is None and a_pred is not None:
                a_correct = int(a_pred) == int(gold)
            if b_correct is None and b_pred is not None:
                b_correct = int(b_pred) == int(gold)
            if a_correct is True and b_correct is True:
                summary["both_correct"] += 1
            elif a_correct is False and b_correct is False:
                summary["both_wrong"] += 1
            elif a_correct is True and b_correct is False:
                summary["run_a_only_correct"] += 1
            elif a_correct is False and b_correct is True:
                summary["run_b_only_correct"] += 1

            if a_correct is not None and b_correct is not None:
                accuracy_deltas.append(float(bool(b_correct)) - float(bool(a_correct)))

        if a_pred != b_pred:
            summary["answer_flips"] += 1

        a_margin = _support_proxy(a or {})
        b_margin = _support_proxy(b or {})
        if a_margin is not None and b_margin is not None:
            vote_margin_deltas.append(b_margin - a_margin)

        a_penalty = _safe_float((a or {}).get("best_trace_penalty"))
        b_penalty = _safe_float((b or {}).get("best_trace_penalty"))
        if a_penalty is not None and b_penalty is not None:
            penalty_deltas.append(b_penalty - a_penalty)

        verdict = "unchanged"
        if a_correct is False and b_correct is True:
            verdict = f"flip_to_{run_b_label}"
        elif a_correct is True and b_correct is False:
            verdict = f"flip_to_{run_a_label}"
        elif a_correct is True and b_correct is True:
            verdict = "both_correct"
        elif a_correct is False and b_correct is False:
            verdict = "both_wrong"
        elif a is None:
            verdict = f"missing_in_{run_a_label}"
        elif b is None:
            verdict = f"missing_in_{run_b_label}"
        elif a_pred != b_pred:
            verdict = "answer_changed_without_gold_flip"

        row: JsonDict = {
            "problem_id": problem_id,
            "gold_answer": gold,
            f"{run_a_label}_predicted_answer": a_pred,
            f"{run_b_label}_predicted_answer": b_pred,
            f"{run_a_label}_is_correct": a_correct,
            f"{run_b_label}_is_correct": b_correct,
            f"{run_a_label}_vote_proxy": a_margin,
            f"{run_b_label}_vote_proxy": b_margin,
            f"{run_a_label}_best_trace_penalty": a_penalty,
            f"{run_b_label}_best_trace_penalty": b_penalty,
            "answer_flip": a_pred != b_pred,
            "verdict": verdict,
        }
        row.update({k: v for k, v in meta.items() if k != "problem_id"})
        comparisons.append(row)

        slices = _slice_keys(a or b or {}, meta)
        for slice_name, keys in slices.items():
            for key in keys:
                slice_buckets[slice_name][key].append(row)

    if accuracy_deltas:
        summary["mean_accuracy_delta_b_minus_a"] = sum(accuracy_deltas) / len(accuracy_deltas)
    if vote_margin_deltas:
        summary["mean_vote_margin_delta_b_minus_a"] = sum(vote_margin_deltas) / len(vote_margin_deltas)
    if penalty_deltas:
        summary["mean_best_trace_penalty_delta_b_minus_a"] = sum(penalty_deltas) / len(penalty_deltas)

    summary["discordant_pairs"] = int(summary["run_a_only_correct"]) + int(summary["run_b_only_correct"])
    summary["net_gain_b_minus_a"] = int(summary["run_b_only_correct"]) - int(summary["run_a_only_correct"])
    if summary["discordant_pairs"] > 0:
        summary["win_rate_b_given_discordant"] = float(summary["run_b_only_correct"]) / float(summary["discordant_pairs"])
    summary["paired_sign_test_pvalue_two_sided"] = _exact_sign_test_pvalue_two_sided(
        int(summary["run_a_only_correct"]),
        int(summary["run_b_only_correct"]),
    )

    def _summarize_slice(rows: List[JsonDict]) -> JsonDict:
        a_only = 0
        b_only = 0
        both_correct = 0
        both_wrong = 0
        flips = 0
        with_gold = 0
        for row in rows:
            a_c = row.get(f"{run_a_label}_is_correct")
            b_c = row.get(f"{run_b_label}_is_correct")
            if row.get("answer_flip"):
                flips += 1
            if a_c is None or b_c is None:
                continue
            with_gold += 1
            if a_c is True and b_c is True:
                both_correct += 1
            elif a_c is False and b_c is False:
                both_wrong += 1
            elif a_c is True and b_c is False:
                a_only += 1
            elif a_c is False and b_c is True:
                b_only += 1
        return {
            "num_problems": len(rows),
            "problems_with_gold": with_gold,
            "run_a_only_correct": a_only,
            "run_b_only_correct": b_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "answer_flips": flips,
            "net_gain_b_minus_a": b_only - a_only,
        }

    summary["topic_slices"] = {k: _summarize_slice(v) for k, v in sorted(slice_buckets["topic"].items())}
    summary["difficulty_slices"] = {k: _summarize_slice(v) for k, v in sorted(slice_buckets["difficulty"].items())}
    summary["tag_slices"] = {k: _summarize_slice(v) for k, v in sorted(slice_buckets["tags"].items())}
    return summary, comparisons


def write_csv(path: Path, rows: List[JsonDict]) -> None:
    ensure_parent(path)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown_report(summary: Mapping[str, Any], comparisons: List[JsonDict]) -> str:
    a = str(summary["run_a_label"])
    b = str(summary["run_b_label"])
    lines = [
        f"# Paired comparison: {a} vs {b}",
        "",
        "## Headline",
        f"- Union problems: {summary['num_problems_union']}",
        f"- Intersection problems: {summary['num_problems_intersection']}",
        f"- Problems with gold: {summary['problems_with_gold']}",
        f"- {a}-only correct: {summary['run_a_only_correct']}",
        f"- {b}-only correct: {summary['run_b_only_correct']}",
        f"- Discordant pairs: {summary['discordant_pairs']}",
        f"- Net gain ({b} - {a}): {summary['net_gain_b_minus_a']}",
        f"- Win rate for {b} on discordant pairs: {summary['win_rate_b_given_discordant']}",
        f"- Paired sign-test p-value: {summary['paired_sign_test_pvalue_two_sided']}",
        f"- Both correct: {summary['both_correct']}",
        f"- Both wrong: {summary['both_wrong']}",
        f"- Answer flips: {summary['answer_flips']}",
        f"- Mean accuracy delta ({b} - {a}): {summary['mean_accuracy_delta_b_minus_a']}",
        f"- Mean vote proxy delta ({b} - {a}): {summary['mean_vote_margin_delta_b_minus_a']}",
        f"- Mean best-trace penalty delta ({b} - {a}): {summary['mean_best_trace_penalty_delta_b_minus_a']}",
        "",
        "## Top flips toward run B",
    ]

    flips_to_b = [row for row in comparisons if row.get("verdict") == f"flip_to_{b}"][:10]
    if flips_to_b:
        for row in flips_to_b:
            lines.append(
                f"- {row['problem_id']}: {a}={row.get(f'{a}_predicted_answer')}, {b}={row.get(f'{b}_predicted_answer')}, gold={row.get('gold_answer')}"
            )
    else:
        lines.append("- None")

    lines.extend(["", f"## Top flips toward run A"])
    flips_to_a = [row for row in comparisons if row.get("verdict") == f"flip_to_{a}"][:10]
    if flips_to_a:
        for row in flips_to_a:
            lines.append(
                f"- {row['problem_id']}: {a}={row.get(f'{a}_predicted_answer')}, {b}={row.get(f'{b}_predicted_answer')}, gold={row.get('gold_answer')}"
            )
    else:
        lines.append("- None")

    for title, key in (("Topic slices", "topic_slices"), ("Difficulty slices", "difficulty_slices"), ("Tag slices", "tag_slices")):
        lines.extend(["", f"## {title}"])
        slices = summary.get(key, {}) or {}
        if not slices:
            lines.append("- No slice metadata available")
            continue
        for slice_name, slice_summary in slices.items():
            lines.append(
                f"- {slice_name}: n={slice_summary['num_problems']}, gold={slice_summary['problems_with_gold']}, net_gain_b_minus_a={slice_summary['net_gain_b_minus_a']}, flips={slice_summary['answer_flips']}"
            )

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two eval runs and emit paired win/loss/flip reports")
    parser.add_argument("--run-a-dir", required=True)
    parser.add_argument("--run-b-dir", required=True)
    parser.add_argument("--run-a-label", default=None)
    parser.add_argument("--run-b-label", default=None)
    parser.add_argument("--metadata-jsonl", default=None, help="Optional per-problem metadata with topic/difficulty/tags")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_a_metrics, run_a_records, run_a_backend = load_run_artifacts(args.run_a_dir)
    run_b_metrics, run_b_records, run_b_backend = load_run_artifacts(args.run_b_dir)
    run_a_label = args.run_a_label or run_a_backend
    run_b_label = args.run_b_label or run_b_backend
    metadata_index = load_metadata_index(args.metadata_jsonl)

    summary, comparisons = compare_runs(
        run_a_records,
        run_b_records,
        run_a_label=run_a_label,
        run_b_label=run_b_label,
        metadata_index=metadata_index,
    )

    manifest = {
        "script": "compare_eval_runs.py",
        "run_a_dir": str(Path(args.run_a_dir).resolve()),
        "run_b_dir": str(Path(args.run_b_dir).resolve()),
        "run_a_label": run_a_label,
        "run_b_label": run_b_label,
        "run_a_metrics": run_a_metrics,
        "run_b_metrics": run_b_metrics,
        "metadata_jsonl": str(Path(args.metadata_jsonl).resolve()) if args.metadata_jsonl else None,
    }

    write_json(output_dir / "comparison_summary.json", summary)
    write_jsonl(output_dir / "paired_problem_comparison.jsonl", comparisons)
    write_csv(output_dir / "paired_problem_comparison.csv", comparisons)
    (output_dir / "comparison_report.md").write_text(render_markdown_report(summary, comparisons), encoding="utf-8")
    write_json(output_dir / "comparison_manifest.json", manifest)


if __name__ == "__main__":
    main()
