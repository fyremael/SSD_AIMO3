from __future__ import annotations

import importlib
import json
import os
import platform
import re
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence


JsonDict = Dict[str, Any]
ENV_PREFIX = "SSD_AIMO3_WANDB_"
DEFAULT_PROJECT = "SSD_AIMO3"
DEFAULT_ARTIFACT_MAX_FILE_SIZE_MB = 25
DEFAULT_CONFIG_SECTIONS = [
    "experiment",
    "model",
    "prompting",
    "sampling",
    "generation",
    "aggregation",
    "extraction",
    "filtering",
    "training",
    "paths",
]
DEFAULT_SKIP_METRIC_PREFIXES = ("topic_slices", "difficulty_slices", "tag_slices", "commands")


def _env(name: str) -> Optional[str]:
    value = os.environ.get(f"{ENV_PREFIX}{name}")
    return value.strip() if isinstance(value, str) and value.strip() else value


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _redact_key(key: str) -> bool:
    normalized = key.lower()
    return any(token in normalized for token in ("secret", "token", "api_key", "apikey", "password"))


def sanitize_for_wandb(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        sanitized: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            sanitized[key_text] = "<redacted>" if _redact_key(key_text) else sanitize_for_wandb(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [sanitize_for_wandb(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_artifact_name(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return collapsed.strip("-_.") or "artifact"


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(sanitize_for_wandb(value), sort_keys=True)


def _flatten_scalars(
    mapping: Mapping[str, Any],
    *,
    prefix: Optional[str] = None,
    skip_prefixes: Sequence[str] = (),
) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}

    def visit(current: Any, path: List[str]) -> None:
        if isinstance(current, Mapping):
            if any(str(item) in path for item in skip_prefixes if str(item).strip()):
                return
            for key, value in current.items():
                visit(value, path + [str(key)])
            return
        key = ".".join(path)
        if key:
            flattened[key] = _jsonable_scalar(current)

    visit(mapping, [prefix] if prefix else [])
    return flattened


def _selected_config_sections(config: Optional[Mapping[str, Any]], section_names: Sequence[str]) -> Dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    selected: Dict[str, Any] = {}
    for section_name in section_names:
        value = config.get(section_name)
        if value is None:
            continue
        selected[section_name] = sanitize_for_wandb(value)
    return selected


class WandbSession:
    def __init__(
        self,
        *,
        enabled: bool,
        module: Any = None,
        run: Any = None,
        artifact_max_file_size_mb: int = DEFAULT_ARTIFACT_MAX_FILE_SIZE_MB,
        log_artifacts: bool = True,
        default_artifact_name: Optional[str] = None,
        disabled_reason: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self._module = module
        self._run = run
        self._artifact_max_file_size_mb = max(1, int(artifact_max_file_size_mb))
        self._log_artifacts = bool(log_artifacts)
        self._default_artifact_name = default_artifact_name
        self.disabled_reason = disabled_reason

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        prefix: Optional[str] = None,
        skip_prefixes: Sequence[str] = DEFAULT_SKIP_METRIC_PREFIXES,
        step: Optional[int] = None,
    ) -> None:
        if not self.enabled or not isinstance(metrics, Mapping):
            return
        payload = _flatten_scalars(metrics, prefix=prefix, skip_prefixes=skip_prefixes)
        if not payload:
            return
        if step is not None:
            self._run.log(payload, step=step)
        else:
            self._run.log(payload)

    def update_summary(
        self,
        data: Mapping[str, Any],
        *,
        prefix: Optional[str] = None,
        skip_prefixes: Sequence[str] = DEFAULT_SKIP_METRIC_PREFIXES,
    ) -> None:
        if not self.enabled or not isinstance(data, Mapping):
            return
        payload = _flatten_scalars(data, prefix=prefix, skip_prefixes=skip_prefixes)
        if not payload:
            return
        for key, value in payload.items():
            self._run.summary[key] = value

    def log_output_artifact(
        self,
        *,
        output_dir: Path,
        candidate_files: Sequence[str],
        artifact_type: str,
        artifact_name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        aliases: Optional[Sequence[str]] = None,
    ) -> None:
        if not self.enabled or not self._log_artifacts:
            return
        max_bytes = self._artifact_max_file_size_mb * 1024 * 1024
        selected_files: List[Path] = []
        skipped_files: Dict[str, str] = {}
        for name in candidate_files:
            path = output_dir / name
            if not path.exists() or not path.is_file():
                continue
            size = path.stat().st_size
            if size > max_bytes:
                skipped_files[name] = f"skipped_size_gt_{self._artifact_max_file_size_mb}mb"
                continue
            selected_files.append(path)

        if skipped_files:
            self._run.summary["wandb.skipped_artifact_files"] = json.dumps(skipped_files, sort_keys=True)
        if not selected_files:
            return

        resolved_name = artifact_name or self._default_artifact_name or f"{output_dir.name}-{artifact_type}"
        artifact = self._module.Artifact(
            _safe_artifact_name(resolved_name),
            type=artifact_type,
            metadata=sanitize_for_wandb(metadata or {}),
        )
        for path in selected_files:
            artifact.add_file(str(path), name=path.name)
        self._run.log_artifact(artifact, aliases=list(aliases or ["latest"]))

    def finish(self, *, status: str, error: Optional[BaseException] = None) -> None:
        if not self.enabled:
            return
        self._run.summary["wandb.status"] = status
        if error is not None:
            self._run.summary["wandb.error_type"] = error.__class__.__name__
            self._run.summary["wandb.error_message"] = str(error)
        try:
            self._run.finish()
        except Exception:
            pass


def maybe_init_wandb(
    *,
    config: Optional[Mapping[str, Any]],
    output_dir: Optional[Path],
    script_name: str,
    job_type: str,
    run_name: Optional[str] = None,
    extra_config: Optional[Mapping[str, Any]] = None,
) -> WandbSession:
    wandb_config = config.get("wandb", {}) if isinstance(config, Mapping) and isinstance(config.get("wandb"), Mapping) else {}
    enabled = _coerce_bool(_env("ENABLED"), _coerce_bool(wandb_config.get("enabled"), False))
    if not enabled:
        return WandbSession(enabled=False, disabled_reason="wandb_disabled")

    try:
        wandb = importlib.import_module("wandb")
    except Exception:
        return WandbSession(enabled=False, disabled_reason="wandb_import_failed")

    project = (
        _env("PROJECT")
        or wandb_config.get("project")
        or os.environ.get("WANDB_PROJECT")
        or DEFAULT_PROJECT
    )
    entity = _env("ENTITY") or wandb_config.get("entity") or os.environ.get("WANDB_ENTITY")
    mode = _env("MODE") or wandb_config.get("mode") or os.environ.get("WANDB_MODE") or "online"
    group = _env("GROUP") or wandb_config.get("group")
    if not group and isinstance(config, Mapping):
        experiment = config.get("experiment")
        if isinstance(experiment, Mapping):
            group = str(experiment.get("name") or "").strip() or None

    prefix = _env("RUN_PREFIX") or wandb_config.get("run_prefix")
    if run_name:
        resolved_name = run_name
    else:
        output_slug = output_dir.name if output_dir is not None else Path(script_name).stem
        resolved_name = f"{Path(script_name).stem}-{output_slug}"
    if prefix:
        resolved_name = f"{prefix}-{resolved_name}"

    tags = _coerce_string_list(_env("TAGS")) or _coerce_string_list(wandb_config.get("tags"))
    notes = _env("NOTES") or wandb_config.get("notes")
    wandb_dir = _env("DIR") or wandb_config.get("dir") or os.environ.get("WANDB_DIR")
    log_artifacts = _coerce_bool(_env("LOG_ARTIFACTS"), _coerce_bool(wandb_config.get("log_artifacts"), True))
    artifact_max_file_size_mb = _coerce_int(
        _env("ARTIFACT_MAX_FILE_SIZE_MB"),
        _coerce_int(wandb_config.get("artifact_max_file_size_mb"), DEFAULT_ARTIFACT_MAX_FILE_SIZE_MB),
    )

    config_sections = _coerce_string_list(wandb_config.get("config_sections")) or list(DEFAULT_CONFIG_SECTIONS)
    init_config: JsonDict = {
        "runtime": sanitize_for_wandb(
            {
                "script_name": script_name,
                "job_type": job_type,
                "output_dir": str(output_dir.resolve()) if output_dir is not None else None,
                "cwd": str(Path.cwd().resolve()),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
            }
        ),
    }
    selected_sections = _selected_config_sections(config, config_sections)
    if selected_sections:
        init_config["config_sections"] = selected_sections
    if extra_config:
        init_config["extra"] = sanitize_for_wandb(extra_config)

    init_kwargs: JsonDict = {
        "project": project,
        "job_type": job_type,
        "name": resolved_name,
        "group": group,
        "config": init_config,
        "reinit": True,
        "mode": mode,
        "tags": tags or None,
        "notes": notes,
    }
    if entity:
        init_kwargs["entity"] = entity
    if wandb_dir:
        init_kwargs["dir"] = wandb_dir

    run = wandb.init(**{key: value for key, value in init_kwargs.items() if value is not None})
    if run is None:
        return WandbSession(enabled=False, disabled_reason="wandb_init_returned_none")

    artifact_name = f"{resolved_name}-outputs"
    return WandbSession(
        enabled=True,
        module=wandb,
        run=run,
        artifact_max_file_size_mb=artifact_max_file_size_mb,
        log_artifacts=log_artifacts,
        default_artifact_name=artifact_name,
    )


@contextmanager
def wandb_run_context(
    *,
    config: Optional[Mapping[str, Any]],
    output_dir: Optional[Path],
    script_name: str,
    job_type: str,
    run_name: Optional[str] = None,
    extra_config: Optional[Mapping[str, Any]] = None,
) -> Iterator[WandbSession]:
    session = maybe_init_wandb(
        config=config,
        output_dir=output_dir,
        script_name=script_name,
        job_type=job_type,
        run_name=run_name,
        extra_config=extra_config,
    )
    try:
        yield session
    except Exception as exc:
        session.finish(status="failed", error=exc)
        raise
    else:
        session.finish(status="completed")
