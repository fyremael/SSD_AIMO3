from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from colab_runtime_probe import detect_runtime  # noqa: E402


def test_detect_runtime_returns_recommendation() -> None:
    runtime = detect_runtime()
    assert "recommended_lane" in runtime
