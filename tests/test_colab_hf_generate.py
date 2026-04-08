from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from colab_hf_generate import (  # noqa: E402
    batched,
    build_parser,
    normalize_generation_outputs,
    parse_dtype_name,
    render_model_prompts,
)


def test_batched_splits_sequence() -> None:
    rows = [{"idx": i} for i in range(5)]
    batches = list(batched(rows, 2))
    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_parse_dtype_name_rejects_unknown_values() -> None:
    assert parse_dtype_name("bfloat16") == "bfloat16"
    try:
        parse_dtype_name("fp8")
    except ValueError as exc:
        assert "unsupported" in str(exc).lower()
    else:
        raise AssertionError("Expected unsupported dtype error")


def test_normalize_generation_outputs_preserves_request_keys() -> None:
    rows = [{"problem_id": "p1", "sample_index": 0}, {"problem_id": "p2", "sample_index": 1}]
    outputs = normalize_generation_outputs(rows, ["a", "b"], output_field="generation_text")
    assert outputs[0]["problem_id"] == "p1"
    assert outputs[1]["generation_text"] == "b"


def test_build_parser_accepts_tokenizer_and_adapter_paths() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--input-jsonl",
            "in.jsonl",
            "--output-jsonl",
            "out.jsonl",
            "--model-id",
            "base-model",
            "--tokenizer-id",
            "tokenizer-model",
            "--adapter-path",
            "runs/adapter",
            "--use-chat-template",
            "--system-prompt",
            "solve carefully",
        ]
    )
    assert args.tokenizer_id == "tokenizer-model"
    assert args.adapter_path == "runs/adapter"
    assert args.use_chat_template is True
    assert args.system_prompt == "solve carefully"


def test_render_model_prompts_uses_chat_template_when_requested() -> None:
    class StubTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            assert add_generation_prompt is True
            return " | ".join(f"{item['role']}={item['content']}" for item in messages)

    rendered = render_model_prompts(
        [{"rendered_prompt_text": "Compute 2+3."}],
        prompt_field="rendered_prompt_text",
        tokenizer=StubTokenizer(),
        use_chat_template=True,
        system_prompt="solve carefully",
    )
    assert rendered == ["system=solve carefully | user=Compute 2+3."]
