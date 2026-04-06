from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from colab_train_lora import build_training_texts, parse_dtype_name, parse_target_modules  # noqa: E402


def test_build_training_texts_skips_empty_rows() -> None:
    rows = [{"training_text": "a"}, {"training_text": ""}, {"other": "x"}]
    assert build_training_texts(rows, "training_text") == ["a"]


def test_parse_target_modules_defaults_when_empty() -> None:
    modules = parse_target_modules("")
    assert "q_proj" in modules


def test_parse_dtype_name_accepts_float16() -> None:
    assert parse_dtype_name("float16") == "float16"
