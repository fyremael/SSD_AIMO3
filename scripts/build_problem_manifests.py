from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from common import ensure_parent, write_json, write_jsonl
from wandb_support import wandb_run_context


JsonDict = Dict[str, Any]


def _load_rows(path: Path, input_format: str) -> List[JsonDict]:
    fmt = input_format
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            fmt = "jsonl"
        elif suffix == ".json":
            fmt = "json"
        elif suffix == ".csv":
            fmt = "csv"
        else:
            raise ValueError(f"Could not infer input format from suffix: {suffix}")

    if fmt == "jsonl":
        rows: List[JsonDict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped))
        return rows
    if fmt == "json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of objects")
        return [dict(row) for row in data]
    if fmt == "csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    raise ValueError(f"Unsupported input format: {fmt}")


def _coerce_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def _optional_field(row: Mapping[str, Any], field_name: Optional[str]) -> Any:
    if not field_name:
        return None
    return row.get(field_name)


def build_manifests(
    rows: List[JsonDict],
    *,
    problem_id_field: str,
    prompt_field: str,
    answer_field: Optional[str],
    topic_field: Optional[str],
    difficulty_field: Optional[str],
    tags_field: Optional[str],
    source_field: Optional[str],
    source_name: Optional[str],
) -> tuple[List[JsonDict], List[JsonDict], List[JsonDict]]:
    prompt_rows: List[JsonDict] = []
    eval_rows: List[JsonDict] = []
    metadata_rows: List[JsonDict] = []
    seen_prompts: Dict[str, JsonDict] = {}
    seen_evals: Dict[str, JsonDict] = {}
    seen_metadata: Dict[str, JsonDict] = {}

    for row in rows:
        problem_id = str(row.get(problem_id_field) or "").strip()
        prompt_text = str(row.get(prompt_field) or "").strip()
        if not problem_id:
            raise ValueError("Every row must have a non-empty problem_id")
        if not prompt_text:
            raise ValueError(f"Row {problem_id!r} is missing prompt text")

        topic = _optional_field(row, topic_field)
        difficulty = _optional_field(row, difficulty_field)
        tags = _coerce_tags(_optional_field(row, tags_field))
        source = _optional_field(row, source_field) or source_name
        gold_answer = _optional_field(row, answer_field)
        if gold_answer not in (None, ""):
            gold_answer = int(gold_answer)
        else:
            gold_answer = None

        prompt_row = {
            "problem_id": problem_id,
            "prompt_text": prompt_text,
            "topic": topic,
            "difficulty": difficulty,
            "tags": tags,
            "source": source,
        }
        existing_prompt = seen_prompts.get(problem_id)
        if existing_prompt is not None and existing_prompt != prompt_row:
            raise ValueError(f"Conflicting prompt rows for problem_id={problem_id!r}")
        if existing_prompt is None:
            seen_prompts[problem_id] = prompt_row
            prompt_rows.append(prompt_row)

        if gold_answer is not None:
            eval_row = {
                "problem_id": problem_id,
                "prompt_text": prompt_text,
                "gold_answer": gold_answer,
                "topic": topic,
                "difficulty": difficulty,
                "tags": tags,
                "source": source,
            }
            existing_eval = seen_evals.get(problem_id)
            if existing_eval is not None and existing_eval != eval_row:
                raise ValueError(f"Conflicting eval rows for problem_id={problem_id!r}")
            if existing_eval is None:
                seen_evals[problem_id] = eval_row
                eval_rows.append(eval_row)

        metadata_row = {
            "problem_id": problem_id,
            "gold_answer": gold_answer,
            "topic": topic,
            "difficulty": difficulty,
            "tags": tags,
            "source": source,
        }
        existing_metadata = seen_metadata.get(problem_id)
        if existing_metadata is not None and existing_metadata != metadata_row:
            raise ValueError(f"Conflicting metadata rows for problem_id={problem_id!r}")
        if existing_metadata is None:
            seen_metadata[problem_id] = metadata_row
            metadata_rows.append(metadata_row)

    return prompt_rows, eval_rows, metadata_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize raw problem data into prompt/eval/metadata JSONL manifests")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--input-format", choices=["auto", "jsonl", "json", "csv"], default="auto")
    parser.add_argument("--prompt-output-jsonl", required=True)
    parser.add_argument("--eval-output-jsonl", required=True)
    parser.add_argument("--metadata-output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--problem-id-field", default="problem_id")
    parser.add_argument("--prompt-field", default="prompt_text")
    parser.add_argument("--answer-field", default="gold_answer")
    parser.add_argument("--topic-field", default="topic")
    parser.add_argument("--difficulty-field", default="difficulty")
    parser.add_argument("--tags-field", default="tags")
    parser.add_argument("--source-field", default=None)
    parser.add_argument("--source-name", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary_path = Path(args.summary_json).resolve()
    with wandb_run_context(
        config=None,
        output_dir=summary_path.parent,
        script_name="build_problem_manifests.py",
        job_type="manifest_build",
        extra_config={
            "input_path": str(Path(args.input_path).resolve()),
            "input_format": args.input_format,
            "problem_id_field": args.problem_id_field,
            "prompt_field": args.prompt_field,
            "answer_field": args.answer_field,
        },
    ) as wandb_session:
        rows = _load_rows(Path(args.input_path), args.input_format)
        prompt_rows, eval_rows, metadata_rows = build_manifests(
            rows,
            problem_id_field=args.problem_id_field,
            prompt_field=args.prompt_field,
            answer_field=args.answer_field,
            topic_field=args.topic_field,
            difficulty_field=args.difficulty_field,
            tags_field=args.tags_field,
            source_field=args.source_field,
            source_name=args.source_name,
        )

        write_jsonl(Path(args.prompt_output_jsonl), prompt_rows)
        write_jsonl(Path(args.eval_output_jsonl), eval_rows)
        write_jsonl(Path(args.metadata_output_jsonl), metadata_rows)
        summary = {
            "input_path": str(Path(args.input_path).resolve()),
            "num_input_rows": len(rows),
            "num_prompt_rows": len(prompt_rows),
            "num_eval_rows": len(eval_rows),
            "num_metadata_rows": len(metadata_rows),
        }
        write_json(summary_path, summary)

        wandb_session.log_metrics(summary, prefix="manifest")
        wandb_session.update_summary(summary, prefix="manifest")
        wandb_session.log_output_artifact(
            output_dir=summary_path.parent,
            candidate_files=[
                Path(args.summary_json).name,
            ],
            artifact_type="manifest_outputs",
            metadata=summary,
        )


if __name__ == "__main__":
    main()
