"""Stage 1: Download curated YouTube speech audio as 22050 Hz mono WAV.

Usage:
  python scripts/tts_finetune/01_download_audio.py [--urls URL ...] [--urls-file FILE]

Requires: yt-dlp (pip install yt-dlp) and ffmpeg.
Output:   .local/tts_dataset/raw/*.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import RAW_DIR, SAMPLE_RATE, banner, require_ffmpeg, run, total_minutes

# ---------------------------------------------------------------------------
# Curated source list — REPLACE THESE PLACEHOLDERS with real videos.
#
# What makes a GOOD source for a multilingual Nepali-English voice:
#   * ONE clean speaker (interviews are fine if you trim other speakers later;
#     monologue podcasts are ideal)
#   * NO background music, jingles, crowd noise, or phone-line audio
#   * Natural Nepali-English code-switching: Nepali tech podcasts, news
#     interviews, university lectures, banking/finance explainers, etc.
#   * Studio or quiet-room recording, consistent mic distance
#   * 10+ minutes per video; aim for 30 min minimum, 2+ hours ideal in total
#
# LICENSING: only use content you have the rights to use (your own
# recordings, explicit permission from the creator, or suitably licensed
# material). Voice cloning a person without consent may be illegal.
# ---------------------------------------------------------------------------
DEFAULT_SOURCES: list[str] = [
    # "https://www.youtube.com/watch?v=REPLACE_ME  # Nepali tech podcast, single host, no music",
    # "https://www.youtube.com/watch?v=REPLACE_ME  # News interview, clean studio audio",
    # "https://www.youtube.com/watch?v=REPLACE_ME  # Lecture with Nepali-English code-switching",
]


def check_ytdlp() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        print("[error] yt-dlp is not installed. Install it with:")
        print(f"          {sys.executable} -m pip install yt-dlp")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download curated YouTube audio for TTS fine-tuning "
                    "(bestaudio -> 22050 Hz mono WAV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--urls", nargs="*", default=None,
                        help="YouTube URLs (overrides DEFAULT_SOURCES)")
    parser.add_argument("--urls-file", type=Path, default=None,
                        help="Text file with one URL per line (# comments allowed)")
    parser.add_argument("--out", type=Path, default=RAW_DIR, help="Output directory")
    args = parser.parse_args()

    banner("Stage 1: Download audio")
    ffmpeg = require_ffmpeg()
    if not check_ytdlp():
        return 1

    urls: list[str] = []
    if args.urls:
        urls += args.urls
    if args.urls_file:
        urls += [ln.split("#")[0].strip() for ln in args.urls_file.read_text(encoding="utf-8").splitlines()
                 if ln.split("#")[0].strip()]
    if not urls:
        urls = [u.split("#")[0].strip() for u in DEFAULT_SOURCES if "REPLACE_ME" not in u]
    if not urls:
        print("[error] No URLs given. Edit DEFAULT_SOURCES in this script (see the")
        print("        comments there for what makes a good source), or pass --urls / --urls-file.")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[info] {len(urls)} source(s) -> {args.out}")

    ok = 0
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        result = run([
            sys.executable, "-m", "yt_dlp",
            "--ffmpeg-location", str(Path(ffmpeg).parent),
            "-f", "bestaudio/best",
            "--extract-audio", "--audio-format", "wav",
            "--postprocessor-args", f"ffmpeg:-ar {SAMPLE_RATE} -ac 1",
            "--restrict-filenames", "--no-playlist",
            "-o", str(args.out / "%(id)s_%(title).60s.%(ext)s"),
            url,
        ])
        if result.returncode == 0:
            ok += 1
        else:
            print(f"[warn] download failed for {url}")

    wavs = sorted(args.out.glob("*.wav"))
    print(f"\n[summary] downloaded ok: {ok}/{len(urls)} | wav files in raw/: {len(wavs)} "
          f"| total audio: {total_minutes(ffmpeg, wavs):.1f} min")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
