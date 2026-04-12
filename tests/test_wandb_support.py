from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from wandb_support import maybe_init_wandb, wandb_run_context  # noqa: E402


class FakeRun:
    def __init__(self) -> None:
        self.logged = []
        self.summary = {}
        self.artifacts = []
        self.finished = False

    def log(self, payload, step=None) -> None:
        self.logged.append((dict(payload), step))

    def log_artifact(self, artifact, aliases=None) -> None:
        self.artifacts.append((artifact, list(aliases or [])))

    def finish(self) -> None:
        self.finished = True


class FakeArtifact:
    def __init__(self, name: str, type: str, metadata=None) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.files = []

    def add_file(self, path: str, name: str | None = None) -> None:
        self.files.append((path, name))


@pytest.fixture()
def fake_wandb(monkeypatch):
    run = FakeRun()
    state = {"init_kwargs": None}

    def fake_init(**kwargs):
        state["init_kwargs"] = kwargs
        return run

    class FakeSettings:
        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)

    module = SimpleNamespace(init=fake_init, Artifact=FakeArtifact, Settings=FakeSettings)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return run, state


def test_maybe_init_wandb_uses_env_group_and_logs_artifact(monkeypatch, fake_wandb, tmp_path: Path) -> None:
    run, state = fake_wandb
    monkeypatch.setenv("SSD_AIMO3_WANDB_ENABLED", "1")
    monkeypatch.setenv("SSD_AIMO3_WANDB_PROJECT", "test-project")
    monkeypatch.setenv("SSD_AIMO3_WANDB_GROUP", "session-group")
    monkeypatch.setenv("SSD_AIMO3_WANDB_RUN_PREFIX", "session-prefix")

    session = maybe_init_wandb(
        config={"experiment": {"name": "a1"}},
        output_dir=tmp_path,
        script_name="run_eval_math.py",
        job_type="evaluation",
        extra_config={"dry_run": True},
    )

    assert session.enabled is True
    assert state["init_kwargs"]["project"] == "test-project"
    assert state["init_kwargs"]["group"] == "session-group"
    assert state["init_kwargs"]["name"].startswith("session-prefix-")
    assert state["init_kwargs"]["reinit"] == "finish_previous"
    assert state["init_kwargs"]["settings"].kwargs["silent"] is False

    session.log_metrics(
        {
            "exact_final_answer_accuracy": 0.5,
            "topic_slices": {"algebra": {"net_gain_b_minus_a": 1}},
        },
        prefix="evaluation",
    )
    assert run.logged[0][0]["evaluation.exact_final_answer_accuracy"] == 0.5
    assert "evaluation.topic_slices.algebra.net_gain_b_minus_a" not in run.logged[0][0]

    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    session.log_output_artifact(
        output_dir=tmp_path,
        candidate_files=["metrics.json"],
        artifact_type="evaluation_outputs",
        metadata={"kind": "metrics"},
    )
    assert len(run.artifacts) == 1
    artifact, aliases = run.artifacts[0]
    assert artifact.type == "evaluation_outputs"
    assert aliases == ["latest"]


def test_wandb_run_context_marks_failures(monkeypatch, fake_wandb, tmp_path: Path) -> None:
    run, _ = fake_wandb
    monkeypatch.setenv("SSD_AIMO3_WANDB_ENABLED", "1")
    monkeypatch.setenv("SSD_AIMO3_WANDB_PROJECT", "test-project")

    with pytest.raises(RuntimeError):
        with wandb_run_context(
            config=None,
            output_dir=tmp_path,
            script_name="compare_eval_runs.py",
            job_type="paired_comparison",
        ):
            raise RuntimeError("boom")

    assert run.summary["wandb.status"] == "failed"
    assert run.summary["wandb.error_type"] == "RuntimeError"
