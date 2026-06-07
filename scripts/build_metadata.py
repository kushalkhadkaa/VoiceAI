#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_prompt_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if "|" in stripped:
                item_id, text = stripped.split("|", 1)
            else:
                item_id, text = next(csv.reader([stripped]))
            rows.append((item_id.strip(), text.strip()))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Piper single-speaker metadata.csv.")
    parser.add_argument("prompts", type=Path, help="Prompt file with id|text rows.")
    parser.add_argument("--output", type=Path, default=Path("dataset/metadata.csv"))
    args = parser.parse_args()
    rows = read_prompt_rows(args.prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(f"{item_id}|{text}" for item_id, text in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
