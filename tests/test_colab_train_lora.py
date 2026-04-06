from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from colab_train_lora import (  # noqa: E402
    build_training_texts,
    guess_target_modules_from_module_names,
    parse_dtype_name,
    parse_target_modules,
)


def test_build_training_texts_skips_empty_rows() -> None:
    rows = [{"training_text": "a"}, {"training_text": ""}, {"other": "x"}]
    assert build_training_texts(rows, "training_text") == ["a"]


def test_parse_target_modules_defaults_when_empty() -> None:
    modules = parse_target_modules("")
    assert "q_proj" in modules


def test_parse_dtype_name_accepts_float16() -> None:
    assert parse_dtype_name("float16") == "float16"


def test_guess_target_modules_supports_gpt2_style_names() -> None:
    modules = guess_target_modules_from_module_names(
        [
            "transformer.h.0.attn.c_attn",
            "transformer.h.0.attn.c_proj",
            "transformer.h.0.mlp.c_fc",
        ]
    )
    assert "c_attn" in modules
    assert "c_proj" in modules
    assert "c_fc" in modules
