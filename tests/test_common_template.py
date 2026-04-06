from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import render_string_template  # noqa: E402


def test_render_string_template_substitutes_fields() -> None:
    text = render_string_template("hello {name} from {place}", {"name": "jamie", "place": "lab"})
    assert text == "hello jamie from lab"


def test_render_string_template_rejects_missing_fields() -> None:
    try:
        render_string_template("hello {name} from {place}", {"name": "jamie"})
    except KeyError as exc:
        assert "place" in str(exc)
    else:
        raise AssertionError("Expected missing field error")
