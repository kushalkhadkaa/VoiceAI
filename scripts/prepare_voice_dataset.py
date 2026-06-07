#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from build_metadata import read_prompt_rows


def normalize(input_path: Path, output_path: Path, sample_rate: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required. Install it with `brew install ffmpeg`.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            "s16",
            "-af",
            "loudnorm=I=-23:LRA=7:TP=-2",
            str(output_path),
        ],
        check=True,
    )


def find_recording(recordings_dir: Path, item_id: str) -> Path | None:
    for extension in (".wav", ".webm", ".m4a", ".mp3"):
        candidate = recordings_dir / f"{item_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Piper-compatible dataset/wav and metadata.csv.")
    parser.add_argument("recordings_dir", type=Path)
    parser.add_argument("prompts", type=Path, help="Prompt text file with id|text rows.")
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()

    rows = read_prompt_rows(args.prompts)
    wav_dir = args.output / "wav"
    metadata_rows: list[str] = []
    missing: list[str] = []
    for item_id, text in rows:
        source = find_recording(args.recordings_dir, item_id)
        if source is None:
            missing.append(item_id)
            continue
        normalize(source, wav_dir / f"{item_id}.wav", args.sample_rate)
        metadata_rows.append(f"{item_id}|{text}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metadata.csv").write_text("\n".join(metadata_rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(metadata_rows)} dataset rows to {args.output}")
    if missing:
        print(f"Missing recordings: {', '.join(missing)}")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
