from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import read_jsonl, write_jsonl


JsonDict = Dict[str, Any]


def generate_rows(rows: List[JsonDict], *, output_field: str) -> List[JsonDict]:
    generated: List[JsonDict] = []
    for row in rows:
        answer = row.get("gold_answer")
        if answer is None:
            answer = (int(row.get("sample_index", 0) or 0) + 1) * 11
        generated.append(
            {
                "problem_id": row.get("problem_id"),
                "sample_index": row.get("sample_index"),
                output_field: f"Mock backend reasoning for {row.get('problem_id')}. Final Answer: {int(answer)}",
            }
        )
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock external JSONL generator backend for smoke tests")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-field", default="generation_text")
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    write_jsonl(Path(args.output_jsonl), generate_rows(rows, output_field=args.output_field))


if __name__ == "__main__":
    main()
