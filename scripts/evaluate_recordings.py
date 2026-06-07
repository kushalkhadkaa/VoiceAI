#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import struct
import wave
from pathlib import Path


def pcm_samples(frames: bytes, sample_width: int) -> list[int]:
    if sample_width == 1:
        return [value - 128 for value in frames]
    if sample_width == 2:
        return list(struct.unpack(f"<{len(frames) // 2}h", frames))
    if sample_width == 3:
        samples = []
        for offset in range(0, len(frames), 3):
            raw = frames[offset : offset + 3]
            value = int.from_bytes(raw + (b"\xff" if raw[2] & 0x80 else b"\x00"), "little", signed=True)
            samples.append(value)
        return samples
    if sample_width == 4:
        return list(struct.unpack(f"<{len(frames) // 4}i", frames))
    raise ValueError(f"Unsupported sample width: {sample_width}")


def evaluate_wav(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        channels = wav.getnchannels()
        frame_count = wav.getnframes()
    samples = pcm_samples(frames, sample_width)
    max_amplitude = float((1 << (8 * sample_width - 1)) - 1)
    peak = max((abs(sample) for sample in samples), default=0) / max_amplitude
    rms = math.sqrt(sum(sample * sample for sample in samples) / max(len(samples), 1)) / max_amplitude
    clipped_ratio = sum(1 for sample in samples if abs(sample) >= max_amplitude * 0.98) / max(len(samples), 1)
    duration = frame_count / float(frame_rate or 1)
    score = 100
    reasons: list[str] = []
    if duration < 1.4:
        score -= 38
        reasons.append("too_short")
    if duration > 12:
        score -= 24
        reasons.append("too_long")
    if rms < 0.015:
        score -= 30
        reasons.append("quiet")
    if clipped_ratio > 0.002:
        score -= 35
        reasons.append("clipped")
    if channels != 1:
        score -= 12
        reasons.append("not_mono")
    score = max(0, min(100, score))
    verdict = "good" if score >= 82 else "review" if score >= 62 else "reject"
    return {
        "file": str(path),
        "duration_seconds": round(duration, 3),
        "sample_rate": frame_rate,
        "channels": channels,
        "peak": round(peak, 4),
        "rms": round(rms, 4),
        "clipped_ratio": round(clipped_ratio, 5),
        "score": score,
        "verdict": verdict,
        "reason": ",".join(reasons) or "clean",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate WAV recordings for Piper dataset quality.")
    parser.add_argument("recordings_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("recording_quality.csv"))
    args = parser.parse_args()

    rows = [evaluate_wav(path) for path in sorted(args.recordings_dir.glob("*.wav"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} evaluations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
