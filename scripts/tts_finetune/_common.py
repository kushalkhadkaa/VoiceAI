"""Shared helpers for the TTS fine-tuning pipeline.

Resolves project paths and the ffmpeg binary the same way the rest of the
project does (see scripts/install_ffmpeg.py): ffmpeg may be on PATH, in
.venv/Scripts/ffmpeg.exe, or in <project>/bin/ffmpeg.exe.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / ".local" / "tts_dataset"
RAW_DIR = DATASET_ROOT / "raw"
CLEAN_DIR = DATASET_ROOT / "clean"
DATASET_DIR = DATASET_ROOT / "dataset"
TRAIN_DIR = DATASET_ROOT / "train"
MODELS_DIR = PROJECT_ROOT / "models" / "piper"

SAMPLE_RATE = 22050


def find_ffmpeg() -> str | None:
    """Locate ffmpeg: PATH first, then project-local install locations."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    for candidate in (
        PROJECT_ROOT / ".venv" / "Scripts" / "ffmpeg.exe",
        PROJECT_ROOT / "bin" / "ffmpeg.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def require_ffmpeg() -> str:
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        print(f"[ok] ffmpeg found: {ffmpeg}")
        return ffmpeg
    print("[error] ffmpeg not found on PATH, .venv/Scripts, or bin/.")
    print("        Install it with:")
    print(f"          {sys.executable} {PROJECT_ROOT / 'scripts' / 'install_ffmpeg.py'}")
    sys.exit(1)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, echoing the command."""
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kwargs)


def ffprobe_duration(ffmpeg: str, wav: Path) -> float:
    """Duration in seconds via ffprobe (sibling of ffmpeg), 0.0 on failure."""
    ffprobe = str(Path(ffmpeg).with_name("ffprobe" + (".exe" if ffmpeg.endswith(".exe") else "")))
    if not Path(ffprobe).exists() and shutil.which("ffprobe"):
        ffprobe = shutil.which("ffprobe")
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(wav)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def total_minutes(ffmpeg: str, wavs: list[Path]) -> float:
    return sum(ffprobe_duration(ffmpeg, w) for w in wavs) / 60.0


def banner(title: str) -> None:
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)
