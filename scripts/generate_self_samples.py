from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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

DEFAULT_DRYRUN_PROMPTS: List[JsonDict] = [
    {
        "problem_id": "dryrun_p1",
        "prompt_text": "Compute 2 + 3.",
        "gold_answer": 5,
        "topic": "arithmetic",
        "source": "synthetic_dry_run",
    },
    {
        "problem_id": "dryrun_p2",
        "prompt_text": "A rectangle has side lengths 4 and 7. What is its area?",
        "gold_answer": 28,
        "topic": "geometry",
        "source": "synthetic_dry_run",
    },
    {
        "problem_id": "dryrun_p3",
        "prompt_text": "What is the remainder when 29 is divided by 6?",
        "gold_answer": 5,
        "topic": "number_theory",
        "source": "synthetic_dry_run",
    },
]


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("REPLACE_WITH_")


def _count_values(rows: Iterable[Mapping[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "missing")
        counts[key] = counts.get(key, 0) + 1
    return counts


def load_prompt_records(config: Mapping[str, Any], input_jsonl: Optional[str], *, dry_run: bool) -> List[JsonDict]:
    configured_path = (
        input_jsonl
        or config.get("inputs", {}).get("prompt_input_jsonl")
        or config.get("paths", {}).get("unlabeled_prompt_manifest_jsonl")
    )
    if isinstance(configured_path, str) and configured_path.strip() and not _is_placeholder(configured_path):
        return read_jsonl(configured_path)
    if dry_run:
        return list(DEFAULT_DRYRUN_PROMPTS)
    raise FileNotFoundError("No prompt manifest provided. Supply --input-jsonl or set paths.unlabeled_prompt_manifest_jsonl.")


def choose_template(config: Mapping[str, Any], requested_template: Optional[str]) -> tuple[str, str]:
    prompting_cfg = config.get("prompting", {})
    templates = dict(prompting_cfg.get("templates", {}) or {})
    template_name = requested_template or prompting_cfg.get("template_name")
    if not template_name:
        active = prompting_cfg.get("active_template_names", []) or list(templates.keys())
        template_name = active[0] if active else "raw_prompt"

    if template_name == "raw_prompt":
        return str(template_name), "{prompt_text}"
    if template_name not in templates:
        raise KeyError(f"Template {template_name!r} not found in config.prompting.templates")
    return str(template_name), str(templates[template_name])


def render_prompt(record: Mapping[str, Any], template_text: str) -> str:
    fields = _SafeFormatDict({str(key): value for key, value in record.items()})
    return template_text.format_map(fields)


def _stable_int(seed_text: str, modulus: int) -> int:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % max(1, modulus)


def synthesize_generation(record: Mapping[str, Any], *, sample_index: int, max_answer: int) -> str:
    base_answer = record.get("synthetic_answer", record.get("gold_answer"))
    if base_answer is None:
        base_answer = _stable_int(f"{record.get('problem_id', '')}:{sample_index}", max_answer + 1)
    answer = int(base_answer)

    mode = sample_index % 4
    if mode == 0:
        return f"Dry-run reasoning for {record.get('problem_id', '')}. Final Answer: {answer}"
    if mode == 1:
        return f"We verify the arithmetic carefully and conclude \\boxed{{{answer}}}."
    if mode == 2:
        alt = (answer + 1) % (max_answer + 1)
        return f"A noisy dry-run sample explores another branch. Final Answer: {alt}"
    return "This dry-run sample intentionally omits a final integer answer."


def determine_filter_status(record: Mapping[str, Any], filtering_cfg: Mapping[str, Any]) -> tuple[str, Optional[str]]:
    policy = str(filtering_cfg.get("policy", "minimal"))
    status = str(record.get("extraction_status") or "missing")

    if policy == "minimal":
        return "kept", None
    if policy == "weak_broken_trace_filter":
        if status != "ok":
            return "rejected", f"extraction_status={status}"
        return "kept", None
    raise ValueError(f"Unsupported filtering policy: {policy}")


def build_generation_requests(
    prompt_records: List[JsonDict],
    *,
    config: Mapping[str, Any],
    template_name: str,
    template_text: str,
    num_samples: int,
) -> List[JsonDict]:
    generation_cfg = config.get("generation", {})
    model_cfg = config.get("model", {})
    rows: List[JsonDict] = []
    for record in prompt_records:
        rendered_prompt = render_prompt(record, template_text)
        for sample_index in range(num_samples):
            rows.append(
                {
                    "problem_id": str(record.get("problem_id", f"prompt_{len(rows)}")),
                    "sample_index": sample_index,
                    "prompt_text": str(record.get("prompt_text", "")),
                    "rendered_prompt_text": rendered_prompt,
                    "template_name": template_name,
                    "topic": record.get("topic"),
                    "difficulty": record.get("difficulty"),
                    "tags": record.get("tags"),
                    "source": record.get("source"),
                    "gold_answer": record.get("gold_answer"),
                    "model_id": model_cfg.get("base_model_id"),
                    "sampling": {
                        "temperature": generation_cfg.get("eval_temperature"),
                        "top_p": generation_cfg.get("eval_top_p"),
                        "top_k": generation_cfg.get("eval_top_k"),
                    },
                }
            )
    return rows


def _request_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return str(row.get("problem_id", "")), int(row.get("sample_index", 0) or 0)


def merge_external_generation_responses(
    request_rows: List[JsonDict],
    response_rows: List[JsonDict],
    *,
    generation_field: str,
) -> List[JsonDict]:
    by_key: Dict[Tuple[str, int], JsonDict] = {}
    for row in response_rows:
        key = _request_key(row)
        if key in by_key:
            raise ValueError(f"Duplicate external generation response for key={key!r}")
        by_key[key] = dict(row)

    merged: List[JsonDict] = []
    missing: List[Tuple[str, int]] = []
    for request in request_rows:
        key = _request_key(request)
        response = by_key.get(key)
        if response is None:
            missing.append(key)
            continue
        combined = dict(request)
        combined.update(response)
        if not str(combined.get(generation_field, "")).strip():
            raise ValueError(f"External generation response for key={key!r} is missing {generation_field!r}")
        merged.append(combined)

    if missing:
        raise ValueError(f"Missing external generation responses for {len(missing)} request(s), first={missing[0]!r}")
    return merged


def _run_external_generation_backend(
    request_rows: List[JsonDict],
    *,
    config: Mapping[str, Any],
    output_dir: Path,
) -> List[JsonDict]:
    generation_cfg = config.get("generation", {})
    generation_field = str(generation_cfg.get("output_text_field", "generation_text"))
    command_template = generation_cfg.get("command")
    if not isinstance(command_template, str) or not command_template.strip():
        raise ValueError("generation.command must be configured when generation.backend=command_jsonl")

    requests_path = output_dir / "generation_requests.jsonl"
    responses_path = output_dir / "generation_responses.jsonl"
    write_jsonl(requests_path, request_rows)

    values = {
        "python_executable": sys.executable,
        "workspace_root": str(Path.cwd().resolve()),
        "output_dir": str(output_dir.resolve()),
        "input_jsonl": str(requests_path.resolve()),
        "output_jsonl": str(responses_path.resolve()),
        "model_id": str(config.get("model", {}).get("base_model_id") or ""),
        "tokenizer_id": str(config.get("model", {}).get("tokenizer_id") or ""),
        "tokenizer_id_arg": "",
        "adapter_path": str(
            config.get("generation", {}).get("adapter_path")
            or config.get("model", {}).get("adapter_path")
            or ""
        ),
        "adapter_path_arg": "",
        "trust_remote_code": str(bool(config.get("model", {}).get("trust_remote_code", False))).lower(),
        "trust_remote_code_flag": "",
        "generation_field": generation_field,
        "num_rows": len(request_rows),
    }
    if values["tokenizer_id"]:
        values["tokenizer_id_arg"] = f'--tokenizer-id "{values["tokenizer_id"]}"'
    if values["adapter_path"]:
        values["adapter_path_arg"] = f'--adapter-path "{values["adapter_path"]}"'
    if values["trust_remote_code"] == "true":
        values["trust_remote_code_flag"] = "--trust-remote-code"
    command = render_string_template(str(command_template), values)
    workdir = Path(str(generation_cfg.get("command_workdir") or Path.cwd())).resolve()
    subprocess.run(command, cwd=str(workdir), shell=True, check=True)

    if not responses_path.exists():
        raise FileNotFoundError(f"External generation command did not create expected output: {responses_path}")
    response_rows = read_jsonl(responses_path)
    return merge_external_generation_responses(request_rows, response_rows, generation_field=generation_field)


def finalize_generation_rows(
    raw_rows: List[JsonDict],
    *,
    filtering_cfg: Mapping[str, Any],
    max_answer: int,
    generation_field: str,
) -> List[JsonDict]:
    rows: List[JsonDict] = []
    for row in raw_rows:
        finalized = dict(row)
        finalized = extract_record(finalized, text_field=generation_field, max_answer=max_answer)
        filter_status, filter_reason = determine_filter_status(finalized, filtering_cfg)
        finalized["filter_status"] = filter_status
        finalized["filter_reason"] = filter_reason
        rows.append(finalized)
    return rows


def build_generation_rows(
    prompt_records: List[JsonDict],
    *,
    config: Mapping[str, Any],
    template_name: str,
    template_text: str,
    num_samples: int,
    max_answer: int,
    dry_run: bool,
    output_dir: Optional[Path] = None,
) -> List[JsonDict]:
    generation_cfg = config.get("generation", {})
    filtering_cfg = config.get("filtering", {})
    generation_backend = str(generation_cfg.get("backend", "replay_or_stub"))
    generation_field = str(generation_cfg.get("output_text_field", "generation_text"))

    if generation_backend == "command_jsonl":
        if output_dir is None:
            raise ValueError("output_dir is required for generation.backend=command_jsonl")
        request_rows = build_generation_requests(
            prompt_records,
            config=config,
            template_name=template_name,
            template_text=template_text,
            num_samples=num_samples,
        )
        raw_rows = _run_external_generation_backend(request_rows, config=config, output_dir=output_dir)
        return finalize_generation_rows(
            raw_rows,
            filtering_cfg=filtering_cfg,
            max_answer=max_answer,
            generation_field=generation_field,
        )

    rows: List[JsonDict] = []
    for record in prompt_records:
        rendered_prompt = render_prompt(record, template_text)
        predefined_generations = record.get("candidate_generations") or []
        if isinstance(predefined_generations, list):
            candidate_generations = [str(item) for item in predefined_generations]
        else:
            candidate_generations = []

        for sample_index in range(num_samples):
            if sample_index < len(candidate_generations):
                generation_text = candidate_generations[sample_index]
            elif dry_run or generation_backend == "synthetic_dry_run":
                generation_text = synthesize_generation(record, sample_index=sample_index, max_answer=max_answer)
            else:
                raise RuntimeError(
                    f"Prompt {record.get('problem_id', '')!r} has no candidate_generations. "
                    "Provide replay generations, configure generation.backend=command_jsonl, or run with --dry-run."
                )

            row: JsonDict = {
                "problem_id": str(record.get("problem_id", f"prompt_{len(rows)}")),
                "sample_index": sample_index,
                "prompt_text": str(record.get("prompt_text", "")),
                "rendered_prompt_text": rendered_prompt,
                "template_name": template_name,
                "topic": record.get("topic"),
                "difficulty": record.get("difficulty"),
                "tags": record.get("tags"),
                "source": record.get("source"),
                "gold_answer": record.get("gold_answer"),
                "model_id": config.get("model", {}).get("base_model_id"),
                "sampling": {
                    "temperature": generation_cfg.get("eval_temperature"),
                    "top_p": generation_cfg.get("eval_top_p"),
                    "top_k": generation_cfg.get("eval_top_k"),
                },
                generation_field: generation_text,
            }
            rows.append(row)
    return finalize_generation_rows(rows, filtering_cfg=filtering_cfg, max_answer=max_answer, generation_field=generation_field)


def main() -> None:
    parser = build_arg_parser("Generate self-sampled math traces or synthesize dry-run equivalents")
    parser.add_argument("--input-jsonl", default=None, help="Optional prompt-manifest JSONL")
    parser.add_argument("--template-name", default=None, help="Prompt template name to render")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of generations to emit per prompt")
    args = parser.parse_args()

    config = resolve_config_from_args(args)
    output_dir = Path(args.output_dir)
    with wandb_run_context(
        config=config,
        output_dir=output_dir,
        script_name="generate_self_samples.py",
        job_type="generation",
        extra_config={
            "input_jsonl": args.input_jsonl,
            "template_name": args.template_name,
            "requested_num_samples": args.num_samples,
            "dry_run": bool(args.dry_run),
        },
    ) as wandb_session:
        save_resolved_config(output_dir, config)

        prompt_records = load_prompt_records(config, args.input_jsonl, dry_run=bool(args.dry_run))
        template_name, template_text = choose_template(config, args.template_name)
        num_samples = int(
            args.num_samples if args.num_samples is not None else config.get("generation", {}).get("num_return_sequences_train", 1)
        )
        max_answer = int(config.get("extraction", {}).get("max_answer", DEFAULT_MAX_ANSWER))
        generation_rows = build_generation_rows(
            prompt_records,
            config=config,
            template_name=template_name,
            template_text=template_text,
            num_samples=max(1, num_samples),
            max_answer=max_answer,
            dry_run=bool(args.dry_run),
            output_dir=output_dir,
        )

        metrics = {
            "num_prompts": len(prompt_records),
            "num_generations": len(generation_rows),
            "template_name": template_name,
            "num_samples_per_prompt": max(1, num_samples),
            "generation_backend": str(config.get("generation", {}).get("backend", "replay_or_stub")),
            "extraction_status_counts": _count_values(generation_rows, "extraction_status"),
            "filter_status_counts": _count_values(generation_rows, "filter_status"),
            "dry_run": bool(args.dry_run),
        }

        write_jsonl(output_dir / "generations.jsonl", generation_rows)
        write_json(output_dir / "generation_metrics.json", metrics)
        save_run_manifest(
            output_dir,
            {
                "script": "generate_self_samples.py",
                "input_jsonl": args.input_jsonl,
                "num_prompts": len(prompt_records),
                "num_generations": len(generation_rows),
                "template_name": template_name,
                "generation_backend": str(config.get("generation", {}).get("backend", "replay_or_stub")),
                "num_samples_per_prompt": max(1, num_samples),
                "dry_run": bool(args.dry_run),
            },
        )

        wandb_session.log_metrics(metrics, prefix="generation")
        wandb_session.update_summary(
            {
                "num_prompts": len(prompt_records),
                "num_generations": len(generation_rows),
                "template_name": template_name,
                "generation_backend": metrics["generation_backend"],
                "dry_run": bool(args.dry_run),
            },
            prefix="generation",
        )
        wandb_session.log_output_artifact(
            output_dir=output_dir,
            candidate_files=[
                "config_resolved.yaml",
                "generation_metrics.json",
                "run_manifest.json",
                "hf_generate_summary.json",
            ],
            artifact_type="generation_outputs",
            metadata={
                "template_name": template_name,
                "generation_backend": metrics["generation_backend"],
            },
        )


if __name__ == "__main__":
    main()
