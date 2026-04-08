from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from common import write_json, write_jsonl
from wandb_support import wandb_run_context


JsonDict = Dict[str, Any]

PUBLIC_PRESETS: Dict[str, JsonDict] = {
    "public_gsm8k_qwen25_math_1p5b": {
        "dataset_id": "567-labs/gsm8k",
        "dataset_config": None,
        "train_split": "train",
        "eval_split": "test",
        "prompt_field": "question",
        "answer_field": "answer",
        "source_name": "567-labs/gsm8k",
        "recommended_model_id": "Qwen/Qwen2.5-Math-1.5B",
        "topic": "grade_school_math",
        "difficulty": "gsm8k",
        "tags": ["gsm8k", "grade_school_math", "word_problem"],
    }
}


def list_presets() -> List[str]:
    return sorted(PUBLIC_PRESETS)


def get_preset(name: str) -> JsonDict:
    if name not in PUBLIC_PRESETS:
        supported = ", ".join(list_presets())
        raise ValueError(f"Unsupported preset {name!r}. Supported presets: {supported}")
    return dict(PUBLIC_PRESETS[name])


def normalize_answer(value: Any, *, max_answer: int) -> Optional[int]:
    try:
        answer = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= answer <= max_answer:
        return answer
    return None


def select_subset(rows: Sequence[Tuple[int, Mapping[str, Any], int]], *, limit: int, seed: int) -> List[Tuple[int, Mapping[str, Any], int]]:
    indexed = list(rows)
    if limit > 0 and len(indexed) > limit:
        rng = random.Random(seed)
        rng.shuffle(indexed)
        indexed = indexed[:limit]
        indexed.sort(key=lambda item: item[0])
    return indexed


def _problem_id(split_name: str, original_index: int) -> str:
    return f"gsm8k_{split_name}_{original_index:05d}"


def build_manifest_rows(
    *,
    preset_name: str,
    train_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    train_limit: int,
    eval_limit: int,
    seed: int,
    max_answer: int,
) -> Tuple[List[JsonDict], List[JsonDict], List[JsonDict], JsonDict]:
    preset = get_preset(preset_name)
    prompt_field = str(preset["prompt_field"])
    answer_field = str(preset["answer_field"])
    topic = preset.get("topic")
    difficulty = preset.get("difficulty")
    tags = list(preset.get("tags") or [])
    source_name = str(preset["source_name"])

    eligible_train = [
        (index, row, answer)
        for index, row in enumerate(train_rows)
        if (answer := normalize_answer(row.get(answer_field), max_answer=max_answer)) is not None
    ]
    eligible_eval = [
        (index, row, answer)
        for index, row in enumerate(eval_rows)
        if (answer := normalize_answer(row.get(answer_field), max_answer=max_answer)) is not None
    ]

    selected_train = select_subset(eligible_train, limit=train_limit, seed=seed)
    selected_eval = select_subset(eligible_eval, limit=eval_limit, seed=seed + 1)

    prompt_manifest: List[JsonDict] = []
    eval_manifest: List[JsonDict] = []
    metadata_manifest: List[JsonDict] = []

    for original_index, row, _ in selected_train:
        prompt_manifest.append(
            {
                "problem_id": _problem_id("train", original_index),
                "prompt_text": str(row.get(prompt_field, "")).strip(),
                "topic": topic,
                "difficulty": difficulty,
                "tags": tags,
                "source": source_name,
            }
        )

    for original_index, row, gold_answer in selected_eval:
        problem_id = _problem_id("test", original_index)
        prompt_text = str(row.get(prompt_field, "")).strip()
        eval_manifest.append(
            {
                "problem_id": problem_id,
                "prompt_text": prompt_text,
                "gold_answer": gold_answer,
                "topic": topic,
                "difficulty": difficulty,
                "tags": tags,
                "source": source_name,
            }
        )
        metadata_manifest.append(
            {
                "problem_id": problem_id,
                "gold_answer": gold_answer,
                "topic": topic,
                "difficulty": difficulty,
                "tags": tags,
                "source": source_name,
            }
        )

    summary = {
        "preset_name": preset_name,
        "dataset_id": preset["dataset_id"],
        "dataset_config": preset.get("dataset_config"),
        "recommended_model_id": preset["recommended_model_id"],
        "source_name": source_name,
        "seed": seed,
        "max_answer": max_answer,
        "requested_train_limit": train_limit,
        "requested_eval_limit": eval_limit,
        "raw_train_rows": len(train_rows),
        "raw_eval_rows": len(eval_rows),
        "eligible_train_rows": len(eligible_train),
        "eligible_eval_rows": len(eligible_eval),
        "selected_train_rows": len(prompt_manifest),
        "selected_eval_rows": len(eval_manifest),
    }
    return prompt_manifest, eval_manifest, metadata_manifest, summary


def load_preset_splits(preset_name: str) -> Tuple[List[JsonDict], List[JsonDict]]:
    from datasets import load_dataset

    preset = get_preset(preset_name)
    dataset = load_dataset(str(preset["dataset_id"]), preset.get("dataset_config"))
    train_split = str(preset["train_split"])
    eval_split = str(preset["eval_split"])
    return list(dataset[train_split]), list(dataset[eval_split])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and normalize a public math benchmark for SSD_AIMO3 real-mode runs")
    parser.add_argument("--preset", choices=list_presets(), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-answer", type=int, default=99999)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with wandb_run_context(
        config=None,
        output_dir=output_dir,
        script_name="prepare_public_math_benchmark.py",
        job_type="public_benchmark_prep",
        extra_config={
            "preset": args.preset,
            "train_limit": args.train_limit,
            "eval_limit": args.eval_limit,
            "seed": args.seed,
            "max_answer": args.max_answer,
        },
    ) as wandb_session:
        train_rows, eval_rows = load_preset_splits(args.preset)
        prompt_manifest, eval_manifest, metadata_manifest, summary = build_manifest_rows(
            preset_name=args.preset,
            train_rows=train_rows,
            eval_rows=eval_rows,
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
            seed=args.seed,
            max_answer=args.max_answer,
        )

        write_jsonl(output_dir / "prompt_manifest.jsonl", prompt_manifest)
        write_jsonl(output_dir / "eval_manifest.jsonl", eval_manifest)
        write_jsonl(output_dir / "problem_metadata.jsonl", metadata_manifest)
        write_json(output_dir / "benchmark_summary.json", summary)

        wandb_session.log_metrics(summary, prefix="public_benchmark")
        wandb_session.update_summary(summary, prefix="public_benchmark")
        wandb_session.log_output_artifact(
            output_dir=output_dir,
            candidate_files=[
                "benchmark_summary.json",
            ],
            artifact_type="public_benchmark_outputs",
            metadata=summary,
        )


if __name__ == "__main__":
    main()
