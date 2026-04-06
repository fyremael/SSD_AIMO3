from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml


JsonDict = Dict[str, Any]


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--output-dir", required=True, help="Directory for run outputs")
    parser.add_argument("--dry-run", action="store_true", help="Run lightweight validation path")
    return parser


def _deep_merge(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must deserialize to a mapping")
    return data


def _split_config_reference(ref: str) -> Tuple[str, Optional[str]]:
    config_ref, has_fragment, fragment = ref.partition("#")
    config_ref = config_ref.strip()
    if not config_ref:
        raise ValueError(f"Invalid inherited config reference: {ref!r}")
    selected_fragment = fragment.strip() if has_fragment and fragment.strip() else None
    return config_ref, selected_fragment


def _resolve_inherited_path(root: Path, ref: str) -> Path:
    config_ref, _ = _split_config_reference(ref)
    candidate = Path(config_ref)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _select_config_fragment(data: Mapping[str, Any], fragment: Optional[str], ref: str) -> JsonDict:
    if not fragment:
        return dict(data)

    cursor: Any = data
    for part in [item.strip() for item in fragment.split(".") if item.strip()]:
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise KeyError(f"Could not resolve config fragment {fragment!r} from reference {ref!r}")
        cursor = cursor[part]

    if not isinstance(cursor, Mapping):
        raise ValueError(f"Config fragment {fragment!r} from reference {ref!r} must resolve to a mapping")
    return dict(cursor)


def _normalize_inheritance_refs(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Sequence):
        refs: List[str] = []
        for item in value:
            if item is None:
                continue
            stripped = str(item).strip()
            if stripped:
                refs.append(stripped)
        return refs
    raise ValueError(f"inherits_from must be a string or sequence of strings, got {type(value)!r}")


def _resolve_config_reference(ref: str, *, root: Path, stack: Sequence[str]) -> JsonDict:
    path = _resolve_inherited_path(root, ref)
    _, fragment = _split_config_reference(ref)
    reference_id = f"{path}#{fragment or ''}"
    if reference_id in stack:
        cycle = " -> ".join(list(stack) + [reference_id])
        raise ValueError(f"Config inheritance cycle detected: {cycle}")

    raw = load_yaml(path)
    data = _select_config_fragment(raw, fragment, ref)
    inherited = _normalize_inheritance_refs(data.get("inherits_from"))
    merged: JsonDict = {}
    next_stack = list(stack) + [reference_id]
    for item in inherited:
        _deep_merge(merged, _resolve_config_reference(str(item), root=path.parent, stack=next_stack))
    data = dict(data)
    data.pop("inherits_from", None)
    _deep_merge(merged, data)
    return merged


def resolve_config_path(config_path: Optional[str]) -> JsonDict:
    if not config_path:
        return {}
    path = Path(config_path).resolve()
    return _resolve_config_reference(str(path), root=path.parent, stack=[])


def resolve_config_from_args(args: argparse.Namespace) -> JsonDict:
    cfg = resolve_config_path(getattr(args, "config", None))
    runtime = cfg.setdefault("runtime", {})
    runtime["dry_run"] = bool(getattr(args, "dry_run", False))
    runtime["output_dir"] = str(Path(args.output_dir).resolve())
    return cfg


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=False)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object in {path}, got: {type(obj)!r}")
            rows.append(obj)
    return rows


def save_resolved_config(output_dir: Path, config: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False)


def save_run_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> None:
    write_json(output_dir / "run_manifest.json", manifest)


class SafeTemplateDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_string_template(template: str, values: Mapping[str, Any]) -> str:
    formatter = string.Formatter()
    parsed = list(formatter.parse(template))
    missing: List[str] = []
    for _, field_name, _, _ in parsed:
        if not field_name:
            continue
        base_name = field_name.split(".", 1)[0].split("[", 1)[0]
        if base_name not in values and base_name not in missing:
            missing.append(base_name)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"Template is missing values for: {missing_text}")
    return template.format_map(SafeTemplateDict({str(k): v for k, v in values.items()}))
