from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

from common import read_jsonl, write_json, write_jsonl
from wandb_support import wandb_run_context


JsonDict = Dict[str, Any]


def batched(items: Sequence[JsonDict], batch_size: int) -> Iterator[List[JsonDict]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def parse_dtype_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return normalized


def normalize_generation_outputs(
    request_rows: Sequence[Mapping[str, Any]],
    generations: Sequence[str],
    *,
    output_field: str,
) -> List[JsonDict]:
    if len(request_rows) != len(generations):
        raise ValueError("request_rows and generations must be the same length")
    rows: List[JsonDict] = []
    for request_row, generated_text in zip(request_rows, generations):
        row = {
            "problem_id": request_row.get("problem_id"),
            "sample_index": request_row.get("sample_index"),
            output_field: str(generated_text).strip(),
        }
        rows.append(row)
    return rows


def configure_tokenizer_for_generation(tokenizer: Any) -> Any:
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def render_model_prompts(
    request_rows: Sequence[Mapping[str, Any]],
    *,
    prompt_field: str,
    tokenizer: Any,
    use_chat_template: bool,
    system_prompt: str | None,
) -> List[str]:
    prompt_rows = [str(row.get(prompt_field, "")) for row in request_rows]
    if not use_chat_template:
        return prompt_rows
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support apply_chat_template but --use-chat-template was requested")

    rendered: List[str] = []
    for prompt_text in prompt_rows:
        messages: List[Dict[str, str]] = []
        if str(system_prompt or "").strip():
            messages.append({"role": "system", "content": str(system_prompt).strip()})
        messages.append({"role": "user", "content": prompt_text})
        rendered.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return rendered


def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    return mapping.get(dtype_name)


def run_generation(args: argparse.Namespace) -> JsonDict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = parse_dtype_name(args.dtype)
    request_rows = read_jsonl(args.input_jsonl)

    tokenizer_id = str(args.tokenizer_id or args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=bool(args.trust_remote_code))
    tokenizer = configure_tokenizer_for_generation(tokenizer)

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(args.trust_remote_code),
        "device_map": "auto",
    }
    torch_dtype = _resolve_torch_dtype(torch, dtype_name)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    if args.quantization == "4bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif args.quantization == "8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    generated_rows: List[JsonDict] = []
    for request_batch in batched(request_rows, args.batch_size):
        prompt_batch = render_model_prompts(
            request_batch,
            prompt_field=args.prompt_field,
            tokenizer=tokenizer,
            use_chat_template=bool(args.use_chat_template),
            system_prompt=args.system_prompt,
        )
        tokenized = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        )
        tokenized = {key: value.to(model_device) for key, value in tokenized.items()}

        with torch.inference_mode():
            output_ids = model.generate(
                **tokenized,
                max_new_tokens=args.max_new_tokens,
                do_sample=bool(args.do_sample),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded_generations: List[str] = []
        attention_mask = tokenized.get("attention_mask")
        for row_index in range(output_ids.shape[0]):
            input_length = int(attention_mask[row_index].sum().item()) if attention_mask is not None else tokenized["input_ids"].shape[1]
            continuation_ids = output_ids[row_index][input_length:]
            decoded_generations.append(tokenizer.decode(continuation_ids, skip_special_tokens=True).strip())

        generated_rows.extend(
            normalize_generation_outputs(request_batch, decoded_generations, output_field=args.output_field)
        )

    write_jsonl(Path(args.output_jsonl), generated_rows)
    summary = {
        "model_id": args.model_id,
        "tokenizer_id": tokenizer_id,
        "adapter_path": args.adapter_path,
        "num_requests": len(request_rows),
        "num_outputs": len(generated_rows),
        "batch_size": args.batch_size,
        "quantization": args.quantization,
        "dtype": dtype_name,
        "prompt_field": args.prompt_field,
        "output_field": args.output_field,
        "use_chat_template": bool(args.use_chat_template),
        "system_prompt": args.system_prompt,
        "padding_side": getattr(tokenizer, "padding_side", None),
    }
    if args.summary_json:
        write_json(Path(args.summary_json), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate JSONL completions with a Hugging Face causal LM on Colab GPU")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--prompt-field", default="rendered_prompt_text")
    parser.add_argument("--output-field", default="generation_text")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="none")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--system-prompt", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.summary_json).resolve().parent if args.summary_json else Path(args.output_jsonl).resolve().parent
    with wandb_run_context(
        config=None,
        output_dir=output_dir,
        script_name="colab_hf_generate.py",
        job_type="hf_generation_backend",
        extra_config={
            "model_id": args.model_id,
            "tokenizer_id": args.tokenizer_id,
            "adapter_path": args.adapter_path,
            "batch_size": args.batch_size,
            "quantization": args.quantization,
            "dtype": args.dtype,
        },
    ) as wandb_session:
        summary = run_generation(args)
        wandb_session.log_metrics(summary, prefix="hf_generation")
        wandb_session.update_summary(summary, prefix="hf_generation")
        wandb_session.log_output_artifact(
            output_dir=output_dir,
            candidate_files=[
                Path(args.summary_json).name if args.summary_json else "",
            ],
            artifact_type="hf_generation_outputs",
            metadata=summary,
        )


if __name__ == "__main__":
    main()
