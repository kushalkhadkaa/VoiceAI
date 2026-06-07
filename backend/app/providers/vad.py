from __future__ import annotations

import wave
import math
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VADResult:
    has_speech: bool
    rms: int
    duration_seconds: float
    reason: str


class EnergyVADProvider:
    def __init__(self, rms_threshold: int = 350, min_duration_seconds: float = 0.25) -> None:
        self.rms_threshold = rms_threshold
        self.min_duration_seconds = min_duration_seconds

    def analyze_wav(self, wav_path: Path) -> VADResult:
        with wave.open(str(wav_path), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            channels = wav.getnchannels()
            duration = wav.getnframes() / float(frame_rate or 1)
        rms = self._rms(frames, sample_width) if frames else 0
        if duration < self.min_duration_seconds:
            return VADResult(False, rms, duration, "too short")
        if channels > 2:
            return VADResult(False, rms, duration, "unsupported channel count")
        if rms < self.rms_threshold:
            return VADResult(False, rms, duration, "below speech energy threshold")
        return VADResult(True, rms, duration, "speech-like energy detected")

    @staticmethod
    def _rms(frames: bytes, sample_width: int) -> int:
        if sample_width == 1:
            samples = [value - 128 for value in frames]
        elif sample_width == 2:
            samples = list(struct.unpack(f"<{len(frames) // 2}h", frames))
        elif sample_width == 3:
            samples = []
            for offset in range(0, len(frames), 3):
                raw = frames[offset : offset + 3]
                value = int.from_bytes(raw + (b"\xff" if raw[2] & 0x80 else b"\x00"), "little", signed=True)
                samples.append(value)
        elif sample_width == 4:
            samples = list(struct.unpack(f"<{len(frames) // 4}i", frames))
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        return int(math.sqrt(sum(sample * sample for sample in samples) / max(len(samples), 1)))
