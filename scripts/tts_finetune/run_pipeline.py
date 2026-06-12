"""Run the full TTS fine-tuning pipeline: download -> clean -> dataset -> train.

Usage:
  python scripts/tts_finetune/run_pipeline.py [--skip-download] [--skip-clean]
                                              [--skip-dataset] [--skip-train]

Each stage prints its own summary; this orchestrator adds a dashboard of
files / minutes / segments between stages. Any stage failure stops the run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLEAN_DIR, DATASET_DIR, RAW_DIR, banner, find_ffmpeg, total_minutes

HERE = Path(__file__).resolve().parent


def dashboard() -> None:
    ffmpeg = find_ffmpeg()
    print("\n--- pipeline dashboard " + "-" * 39)
    for label, d in (("raw", RAW_DIR), ("clean", CLEAN_DIR)):
        wavs = sorted(d.glob("*.wav")) if d.exists() else []
        mins = total_minutes(ffmpeg, wavs) if (ffmpeg and wavs) else 0.0
        print(f"  {label:8s}: {len(wavs):4d} wav files, {mins:7.1f} min")
    meta = DATASET_DIR / "metadata.csv"
    segs = len(meta.read_text(encoding="utf-8").splitlines()) if meta.exists() else 0
    print(f"  dataset : {segs:4d} segments in metadata.csv")
    print("-" * 62)


def run_stage(script: str, extra: list[str]) -> bool:
    result = subprocess.run([sys.executable, str(HERE / script), *extra])
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the multilingual Nepali-English Piper fine-tuning pipeline "
                    "(stages 01-04) sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip stage 1")
    parser.add_argument("--skip-clean", action="store_true", help="Skip stage 2")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip stage 3")
    parser.add_argument("--skip-train", action="store_true", help="Skip stage 4")
    parser.add_argument("--urls-file", type=Path, default=None,
                        help="Passed through to stage 1")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Passed through to stage 4 (chitwan .ckpt)")
    args = parser.parse_args()

    banner("TTS fine-tuning pipeline (Nepali + English, one model)")
    if not find_ffmpeg():
        print("[error] ffmpeg not found — run scripts/install_ffmpeg.py first.")
        return 1

    stages = [
        ("download", args.skip_download, "01_download_audio.py",
         ["--urls-file", str(args.urls_file)] if args.urls_file else []),
        ("clean", args.skip_clean, "02_clean_audio.py", []),
        ("dataset", args.skip_dataset, "03_segment_transcribe.py", []),
        ("train", args.skip_train, "04_train_piper.py",
         ["--checkpoint", str(args.checkpoint)] if args.checkpoint else []),
    ]
    for name, skip, script, extra in stages:
        if skip:
            print(f"\n[skip] stage '{name}'")
            continue
        if not run_stage(script, extra):
            print(f"\n[error] stage '{name}' failed — fix the issue and re-run with "
                  f"the earlier stages skipped.")
            dashboard()
            return 1
        dashboard()

    print("\n[done] pipeline finished. See docs/TTS_FINETUNE.md for testing the voice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
