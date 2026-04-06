from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from common import read_jsonl, write_json


JsonDict = Dict[str, Any]
DEFAULT_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_dtype_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return normalized


def parse_target_modules(value: str | None) -> List[str]:
    if value is None:
        return list(DEFAULT_LORA_TARGETS)
    parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    return parsed or list(DEFAULT_LORA_TARGETS)


def build_training_texts(rows: Sequence[Mapping[str, Any]], text_field: str) -> List[str]:
    texts: List[str] = []
    for row in rows:
        text = str(row.get(text_field, "")).strip()
        if not text:
            continue
        texts.append(text)
    return texts


class CausalLMCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    return mapping.get(dtype_name)


def _guess_target_modules(model: Any) -> List[str]:
    discovered = []
    for name, module in model.named_modules():
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name in DEFAULT_LORA_TARGETS and leaf_name not in discovered:
            discovered.append(leaf_name)
    return discovered or list(DEFAULT_LORA_TARGETS)


def run_training(args: argparse.Namespace) -> JsonDict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    dtype_name = parse_dtype_name(args.dtype)
    rows = read_jsonl(args.dataset_path)
    training_texts = build_training_texts(rows, args.text_field)
    if not training_texts:
        raise ValueError(f"No non-empty training texts found in field {args.text_field!r}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": bool(args.trust_remote_code)}
    torch_dtype = _resolve_torch_dtype(torch, dtype_name)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    if args.quantization == "4bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif args.quantization == "8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if args.quantization in {"4bit", "8bit"}:
        model = prepare_model_for_kbit_training(model)

    target_modules = parse_target_modules(args.target_modules)
    if args.target_modules is None:
        target_modules = _guess_target_modules(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = Dataset.from_dict({"text": training_texts})

    def tokenize_batch(batch: Mapping[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            list(batch["text"]),
            truncation=True,
            max_length=args.max_seq_length,
        )

    tokenized_dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    data_collator = CausalLMCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir).resolve()),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
        remove_unused_columns=False,
        bf16=(dtype_name == "bfloat16"),
        fp16=(dtype_name == "float16"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "model_id": args.model_id,
        "num_input_rows": len(rows),
        "num_training_texts": len(training_texts),
        "text_field": args.text_field,
        "dtype": dtype_name,
        "quantization": args.quantization,
        "target_modules": target_modules,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_loss": train_result.metrics.get("train_loss"),
        "output_dir": str(Path(args.output_dir).resolve()),
    }
    if args.summary_json:
        write_json(Path(args.summary_json), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LoRA SFT on Colab GPU from a JSONL training dataset")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--text-field", default="training_text")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="4bit")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
