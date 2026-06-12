"""Stage 2: Clean raw audio — denoise, trim silence, normalize loudness.

Chain (per file):
  [optional] demucs vocal isolation (if `pip install demucs` was done)
  ffmpeg: highpass=f=80, lowpass=f=8000, afftdn (FFT denoiser)
  ffmpeg: silenceremove + loudnorm I=-19

Usage:
  python scripts/tts_finetune/02_clean_audio.py [--in DIR] [--out DIR] [--no-demucs]

Output: .local/tts_dataset/clean/*.wav (22050 Hz mono)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLEAN_DIR, RAW_DIR, SAMPLE_RATE, banner, require_ffmpeg, run, total_minutes

DENOISE_FILTER = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25"
TRIM_NORM_FILTER = (
    "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.3:"
    "stop_periods=-1:stop_threshold=-45dB:stop_silence=0.5,"
    "loudnorm=I=-19:TP=-1.5:LRA=11"
)


def demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def isolate_vocals(wav: Path, tmp: Path) -> Path | None:
    """Run demucs two-stem separation; return path to vocals wav or None."""
    result = run([sys.executable, "-m", "demucs", "--two-stems", "vocals",
                  "-o", str(tmp), str(wav)])
    if result.returncode != 0:
        return None
    hits = list(tmp.rglob("vocals.wav"))
    return hits[0] if hits else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Denoise, silence-trim and loudness-normalize raw TTS audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in", dest="indir", type=Path, default=RAW_DIR, help="Input directory")
    parser.add_argument("--out", type=Path, default=CLEAN_DIR, help="Output directory")
    parser.add_argument("--no-demucs", action="store_true",
                        help="Skip demucs vocal isolation even if installed")
    args = parser.parse_args()

    banner("Stage 2: Clean audio")
    ffmpeg = require_ffmpeg()

    wavs = sorted(args.indir.glob("*.wav"))
    if not wavs:
        print(f"[error] No .wav files in {args.indir}. Run 01_download_audio.py first.")
        return 1

    use_demucs = not args.no_demucs and demucs_available()
    if use_demucs:
        print("[ok] demucs found — vocal isolation enabled (slow but worth it)")
    elif not args.no_demucs:
        print("[info] demucs not installed — skipping vocal isolation.")
        print(f"       Optional, recommended for sources with any music: "
              f"{sys.executable} -m pip install demucs")

    args.out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i, wav in enumerate(wavs, 1):
        print(f"\n[{i}/{len(wavs)}] {wav.name}")
        src = wav
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            if use_demucs:
                vocals = isolate_vocals(wav, tmp)
                if vocals:
                    src = vocals
                else:
                    print("[warn] demucs failed, using original audio")
            denoised = tmp / "denoised.wav"
            r1 = run([ffmpeg, "-y", "-i", src, "-af", DENOISE_FILTER,
                      "-ar", str(SAMPLE_RATE), "-ac", "1", denoised],
                     capture_output=True)
            if r1.returncode != 0:
                print(f"[warn] denoise failed: {r1.stderr.decode(errors='replace')[-300:]}")
                continue
            out = args.out / wav.name
            r2 = run([ffmpeg, "-y", "-i", denoised, "-af", TRIM_NORM_FILTER,
                      "-ar", str(SAMPLE_RATE), "-ac", "1", out],
                     capture_output=True)
            if r2.returncode != 0:
                print(f"[warn] normalize failed: {r2.stderr.decode(errors='replace')[-300:]}")
                continue
            ok += 1

    cleaned = sorted(args.out.glob("*.wav"))
    print(f"\n[summary] cleaned: {ok}/{len(wavs)} files "
          f"| total clean audio: {total_minutes(ffmpeg, cleaned):.1f} min")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
