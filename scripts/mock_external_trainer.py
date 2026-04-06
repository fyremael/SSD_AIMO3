from __future__ import annotations

import argparse
from pathlib import Path

from common import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock external training backend for smoke tests")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "mock_checkpoint.json",
        {
            "status": "mock_completed",
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "num_rows": len(rows),
        },
    )


if __name__ == "__main__":
    main()
