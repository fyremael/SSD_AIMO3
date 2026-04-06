from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from common import write_json


JsonDict = Dict[str, Any]
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
CONFIGS_DIR = ROOT / "configs"
SCRIPTS_DIR = ROOT / "scripts"
TESTS_DIR = ROOT / "tests"
DATA_DIR = ROOT / "data"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
SCRIPT_DESCRIPTION_FALLBACKS: Dict[str, str] = {
    "scripts/common.py": "Shared config resolution, JSONL I/O, and utility helpers used across the repo.",
    "scripts/constraint_library.py": "Cheap auditable constraint checks and penalty scoring helpers for tropical reranking.",
    "scripts/update_docs.py": "Generate repo index, status, and summary documentation artifacts.",
}
CONFIG_DESCRIPTION_FALLBACKS: Dict[str, str] = {
    "configs/global.yaml": "Shared defaults for experiments, extraction, generation, and training.",
    "configs/compute_profiles.yaml": "Reusable small, medium, and large compute envelopes.",
    "configs/prompt_templates.yaml": "Reusable prompt template definitions.",
    "configs/command_backend_example.yaml": "Example config showing command-backed generation and training launchers.",
    "configs/fixture_a0.yaml": "Fixture-backed A0 config for local regression testing.",
    "configs/fixture_a1.yaml": "Fixture-backed A1 config for local regression testing.",
    "configs/fixture_a5.yaml": "Fixture-backed A5 config for local regression testing.",
    "configs/colab_gpu_a0.yaml": "Colab GPU baseline evaluation config.",
    "configs/colab_gpu_a1.yaml": "Colab GPU self-distillation generation and training config.",
    "configs/colab_gpu_a5.yaml": "Colab GPU tropical reranking config.",
    "configs/colab_gpu_fixture_debug.yaml": "Small-model Colab debug config for smoke testing the GPU lane.",
}


def list_relative_files(root: Path, pattern: str) -> List[Path]:
    return sorted(path.relative_to(ROOT) for path in root.glob(pattern) if path.is_file())


def read_yaml(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return dict(data)


def extract_script_description(path: Path) -> str:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "build_arg_parser" and node.args:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value.strip()
            if isinstance(func, ast.Attribute) and func.attr == "ArgumentParser":
                for keyword in node.keywords:
                    if keyword.arg == "description" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                        return keyword.value.value.strip()
    return ""


def collect_config_catalog() -> List[JsonDict]:
    rows: List[JsonDict] = []
    for rel_path in list_relative_files(CONFIGS_DIR, "*.yaml"):
        data = read_yaml(ROOT / rel_path)
        experiment = data.get("experiment", {}) if isinstance(data.get("experiment"), Mapping) else {}
        rows.append(
            {
                "path": rel_path.as_posix(),
                "name": str(experiment.get("name") or rel_path.stem),
                "stage": experiment.get("stage"),
                "objective": experiment.get("objective") or CONFIG_DESCRIPTION_FALLBACKS.get(rel_path.as_posix()),
            }
        )
    return rows


def collect_script_catalog() -> List[JsonDict]:
    rows: List[JsonDict] = []
    for rel_path in list_relative_files(SCRIPTS_DIR, "*.py"):
        if rel_path.name == "__init__.py":
            continue
        rows.append(
            {
                "path": rel_path.as_posix(),
                "description": extract_script_description(ROOT / rel_path) or SCRIPT_DESCRIPTION_FALLBACKS.get(rel_path.as_posix(), ""),
            }
        )
    return rows


def collect_workflow_catalog() -> List[JsonDict]:
    if not WORKFLOWS_DIR.exists():
        return []
    rows: List[JsonDict] = []
    for rel_path in list_relative_files(WORKFLOWS_DIR, "*.yml"):
        data = read_yaml(ROOT / rel_path)
        rows.append(
            {
                "path": rel_path.as_posix(),
                "name": str(data.get("name") or rel_path.stem),
            }
        )
    return rows


def build_summary() -> JsonDict:
    docs_files = list_relative_files(DOCS_DIR, "*.md")
    for generated_rel in (Path("docs/INDEX.md"), Path("docs/STATUS.md")):
        if generated_rel not in docs_files:
            docs_files.append(generated_rel)
    docs_files = sorted(docs_files)
    config_rows = collect_config_catalog()
    script_rows = collect_script_catalog()
    workflow_rows = collect_workflow_catalog()
    summary: JsonDict = {
        "docs_count": len(docs_files),
        "config_count": len(config_rows),
        "script_count": len(script_rows),
        "test_count": len(list_relative_files(TESTS_DIR, "test_*.py")),
        "fixture_data_count": len(list_relative_files(DATA_DIR, "*.jsonl")),
        "workflow_count": len(workflow_rows),
        "docs_files": [path.as_posix() for path in docs_files],
        "config_catalog": config_rows,
        "script_catalog": script_rows,
        "workflow_catalog": workflow_rows,
    }
    return summary


def _render_bullet_paths(paths: Iterable[str]) -> List[str]:
    lines: List[str] = []
    for path in paths:
        lines.append(f"- `{path}`")
    return lines


def render_index(summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# INDEX.md",
        "## Documentation Index",
        "",
        "Generated by `scripts/update_docs.py`.",
        "",
        "## Start here",
        "",
        "- `README.md` for the project overview and quickest entry points",
        "- `docs/RUNBOOK.md` for the validation workflow and experiment posture",
        "- `docs/REAL_RUNS.md` for real-data and external-backend plumbing",
        "- `docs/COLAB_DEPLOYMENT.md` for the initial Colab GPU deployment path",
        "- `docs/FIXTURE_LADDER.md` for the replayable local benchmark harness",
        "",
        "## Repo docs",
        "",
    ]
    lines.extend(_render_bullet_paths(summary.get("docs_files", [])))
    lines.extend(
        [
            "",
            "## Script entry points",
            "",
        ]
    )
    for row in summary.get("script_catalog", []):
        description = str(row.get("description") or "No description extracted.")
        lines.append(f"- `{row['path']}`: {description}")
    lines.extend(
        [
            "",
            "## Config files",
            "",
        ]
    )
    for row in summary.get("config_catalog", []):
        objective = str(row.get("objective") or "").strip()
        stage = str(row.get("stage") or "").strip()
        suffix = []
        if stage:
            suffix.append(stage)
        if objective:
            suffix.append(objective)
        suffix_text = " | ".join(suffix) if suffix else "General configuration"
        lines.append(f"- `{row['path']}`: {suffix_text}")
    if summary.get("workflow_catalog"):
        lines.extend(["", "## Automation workflows", ""])
        for row in summary.get("workflow_catalog", []):
            lines.append(f"- `{row['path']}`: {row['name']}")
    return "\n".join(lines) + "\n"


def render_status(summary: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# STATUS.md",
        "## Documentation And Repo Status",
        "",
        "Generated by `scripts/update_docs.py`.",
        "",
        "## Inventory",
        "",
        f"- Docs pages: {summary.get('docs_count')}",
        f"- Config files: {summary.get('config_count')}",
        f"- Script entry points: {summary.get('script_count')}",
        f"- Test files: {summary.get('test_count')}",
        f"- Fixture data files: {summary.get('fixture_data_count')}",
        f"- GitHub workflows: {summary.get('workflow_count')}",
        "",
        "## Automated upkeep",
        "",
    ]
    workflows = summary.get("workflow_catalog", []) or []
    if workflows:
        for row in workflows:
            lines.append(f"- `{row['name']}` via `{row['path']}`")
    else:
        lines.append("- No GitHub workflow files detected")

    lines.extend(
        [
            "",
            "## Current maintained surfaces",
            "",
            "- Canonical answer extraction and evaluation aggregation",
            "- Paired run comparison with discordant-pair statistics",
            "- Fixture ladder for A0 -> A1 -> A5 regression testing",
            "- Real-run manifest normalization and external backend hooks",
            "- Initial Colab GPU generation and LoRA training path",
        ]
    )
    return "\n".join(lines) + "\n"


def update_docs() -> JsonDict:
    summary = build_summary()
    (DOCS_DIR / "INDEX.md").write_text(render_index(summary), encoding="utf-8")
    (DOCS_DIR / "STATUS.md").write_text(render_status(summary), encoding="utf-8")
    write_json(DOCS_DIR / "docs_summary.json", summary)
    return summary


def main() -> None:
    update_docs()


if __name__ == "__main__":
    main()
