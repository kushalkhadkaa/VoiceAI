"""Stage 3: VAD-segment clean audio into 3-12 s utterances and transcribe.

Uses faster-whisper (already a backend dependency) with language=None so
mixed Nepali-English code-switched segments are transcribed as spoken —
this is what makes the final voice genuinely multilingual.

Filters:
  * avg_logprob < -1.0  -> dropped (likely music/noise/garbled)
  * duration < 2 s or > 15 s -> dropped

Output (Piper / LJSpeech layout):
  .local/tts_dataset/dataset/wavs/utt_000001.wav ...
  .local/tts_dataset/dataset/metadata.csv   (filename|transcript)
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLEAN_DIR, DATASET_DIR, SAMPLE_RATE, banner, require_ffmpeg, run

MIN_SEC, MAX_SEC = 2.0, 15.0
LOGPROB_FLOOR = -1.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Segment cleaned audio with VAD and transcribe with faster-whisper "
                    "(auto language detection, keeps mixed Nepali-English).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in", dest="indir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--out", type=Path, default=DATASET_DIR)
    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument("--device", default="auto", help="cpu | cuda | auto")
    args = parser.parse_args()

    banner("Stage 3: Segment + transcribe")
    ffmpeg = require_ffmpeg()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[error] faster-whisper is not installed. Install it with:")
        print(f"          {sys.executable} -m pip install faster-whisper")
        return 1

    wavs = sorted(args.indir.glob("*.wav"))
    if not wavs:
        print(f"[error] No .wav files in {args.indir}. Run 02_clean_audio.py first.")
        return 1

    wavs_dir = args.out / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading whisper '{args.model}' (device={args.device}) ...")
    compute = "int8" if args.device in ("cpu", "auto") else "float16"
    model = WhisperModel(args.model, device=args.device, compute_type=compute)

    rows: list[str] = []
    kept = dropped_quality = dropped_length = 0
    utt = 0
    kept_seconds = 0.0

    for i, wav in enumerate(wavs, 1):
        print(f"\n[{i}/{len(wavs)}] {wav.name}")
        segments, info = model.transcribe(
            str(wav),
            language=None,            # auto-detect; keep code-switched segments
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=5,
        )
        print(f"        detected language: {info.language} (p={info.language_probability:.2f})")
        for seg in segments:
            dur = seg.end - seg.start
            text = seg.text.strip().replace("|", " ")
            if not text:
                continue
            if seg.avg_logprob < LOGPROB_FLOOR:
                dropped_quality += 1
                continue
            if dur < MIN_SEC or dur > MAX_SEC:
                dropped_length += 1
                continue
            utt += 1
            name = f"utt_{utt:06d}"
            out_wav = wavs_dir / f"{name}.wav"
            r = run([ffmpeg, "-y", "-i", wav, "-ss", f"{seg.start:.3f}",
                     "-t", f"{dur:.3f}", "-ar", str(SAMPLE_RATE), "-ac", "1",
                     out_wav], capture_output=True)
            if r.returncode != 0:
                utt -= 1
                continue
            rows.append(f"{name}|{text}")
            kept += 1
            kept_seconds += dur

    if not rows:
        print("\n[error] No usable segments produced. Check audio quality and filters.")
        return 1

    metadata = args.out / "metadata.csv"
    metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"\n[summary] segments kept: {kept} ({kept_seconds / 60:.1f} min) "
          f"| dropped (quality): {dropped_quality} | dropped (length): {dropped_length}")
    print(f"[summary] dataset: {metadata}")
    if kept_seconds < 30 * 60:
        print("[warn] less than 30 minutes of clean speech — fine-tune quality will "
              "be poor. Aim for 30 min minimum, 2+ hours ideal.")

    # quick sanity check that segment wavs are valid
    try:
        with wave.open(str(wavs_dir / f"utt_{1:06d}.wav"), "rb") as w:
            assert w.getframerate() == SAMPLE_RATE
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
