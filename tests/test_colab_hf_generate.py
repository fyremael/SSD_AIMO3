from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from colab_hf_generate import batched, normalize_generation_outputs, parse_dtype_name  # noqa: E402


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
