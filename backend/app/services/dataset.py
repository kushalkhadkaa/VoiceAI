from __future__ import annotations

import io
import math
import shutil
import subprocess
import struct
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.audio_validation import AudioValidator, AudioValidationError


PROMPTS = [
    {"id": "000001", "language": "ne", "text": "नमस्ते, आज म मेरो आवाजको नमूना रेकर्ड गर्दैछु।"},
    {"id": "000002", "language": "ne", "text": "कृपया मलाई छोटो र स्पष्ट जवाफ दिनुहोस्।"},
    {"id": "000003", "language": "en", "text": "Hello, this is a clean sample of my speaking voice."},
    {"id": "000004", "language": "en", "text": "Please answer naturally and keep the response concise."},
    {"id": "000005", "language": "ne", "text": "मेरो आवाज शान्त कोठामा रेकर्ड गरिएको हो।"},
    {"id": "000006", "language": "en", "text": "The quick local assistant should work without cloud services."}
]


@dataclass(frozen=True, slots=True)
class QualityScore:
    score: int
    verdict: str
    duration_seconds: float
    peak: float
    rms: float
    reason: str


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    id: str
    text: str
    language: str
    exists: bool
    audio_url: str | None
    quality: QualityScore | None


class DatasetService:
    def __init__(self, dataset_dir: Path, audio_validator: AudioValidator) -> None:
        self.dataset_dir = dataset_dir
        self.wav_dir = dataset_dir / "wav"
        self.metadata_path = dataset_dir / "metadata.csv"
        self.audio_validator = audio_validator

        # Ensure directories exist
        self.wav_dir.mkdir(parents=True, exist_ok=True)

    def list_prompts(self) -> list[dict[str, str]]:
        return PROMPTS

    def list_recordings(self) -> list[DatasetRecord]:
        records = []
        for prompt in PROMPTS:
            prompt_id = prompt["id"]
            wav_path = self.wav_dir / f"{prompt_id}.wav"
            exists = wav_path.exists()
            audio_url = f"/audio/dataset/wav/{prompt_id}.wav" if exists else None
            quality = None
            if exists:
                try:
                    q_dict = self._evaluate_wav(wav_path)
                    quality = QualityScore(**q_dict)
                except Exception:
                    pass
            records.append(
                DatasetRecord(
                    id=prompt_id,
                    text=prompt["text"],
                    language=prompt["language"],
                    exists=exists,
                    audio_url=audio_url,
                    quality=quality,
                )
            )
        return records

    def save_recording(self, prompt_id: str, audio_bytes: bytes, mime_type: str | None) -> DatasetRecord:
        prompt = next((p for p in PROMPTS if p["id"] == prompt_id), None)
        if not prompt:
            raise ValueError(f"Invalid prompt ID: {prompt_id}")

        # Validate file size and MIME type
        normalized_mime = self.audio_validator.validate_upload(audio_bytes, mime_type)
        
        # Save temporary file to work directory for normalization
        temp_dir = self.dataset_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"temp-{prompt_id}.raw"
        temp_path.write_bytes(audio_bytes)

        target_path = self.wav_dir / f"{prompt_id}.wav"
        try:
            self._normalize_wav(temp_path, target_path)
            # Validate duration of resampled WAV
            self.audio_validator.validate_wav_duration(target_path)
        except Exception as e:
            if target_path.exists():
                target_path.unlink()
            raise AudioValidationError(f"Audio processing failed: {e}") from e
        finally:
            if temp_path.exists():
                temp_path.unlink()

        # Update metadata.csv
        self._update_metadata(prompt_id, prompt["text"])

        # Return updated record
        q_dict = self._evaluate_wav(target_path)
        return DatasetRecord(
            id=prompt_id,
            text=prompt["text"],
            language=prompt["language"],
            exists=True,
            audio_url=f"/audio/dataset/wav/{prompt_id}.wav",
            quality=QualityScore(**q_dict),
        )

    def delete_recording(self, prompt_id: str) -> None:
        wav_path = self.wav_dir / f"{prompt_id}.wav"
        if wav_path.exists():
            wav_path.unlink()
        self._delete_from_metadata(prompt_id)

    def get_zip_bytes(self) -> bytes:
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            if self.metadata_path.exists():
                zf.write(self.metadata_path, "metadata.csv")
            if self.wav_dir.exists():
                for wav_path in sorted(self.wav_dir.glob("*.wav")):
                    zf.write(wav_path, f"wav/{wav_path.name}")
        return bio.getvalue()

    def _normalize_wav(self, input_path: Path, output_path: Path, sample_rate: int = 22050) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for audio normalization.")
        command = [
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
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode('utf-8', errors='ignore')}")

    def _evaluate_wav(self, path: Path) -> dict[str, Any]:
        with wave.open(str(path), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            channels = wav.getnchannels()
            frame_count = wav.getnframes()
        
        samples = self._pcm_samples(frames, sample_width)
        max_amplitude = float((1 << (8 * sample_width - 1)) - 1)
        peak = max((abs(sample) for sample in samples), default=0) / max_amplitude
        rms = math.sqrt(sum(sample * sample for sample in samples) / max(len(samples), 1)) / max_amplitude
        clipped_ratio = sum(1 for sample in samples if abs(sample) >= max_amplitude * 0.98) / max(len(samples), 1)
        duration = frame_count / float(frame_rate or 1)
        
        score = 100
        reasons = []
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
            "score": score,
            "verdict": verdict,
            "duration_seconds": round(duration, 3),
            "peak": round(peak, 4),
            "rms": round(rms, 4),
            "reason": ",".join(reasons) or "clean",
        }

    @staticmethod
    def _pcm_samples(frames: bytes, sample_width: int) -> list[int]:
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

    def _update_metadata(self, prompt_id: str, text: str) -> None:
        records = {}
        if self.metadata_path.exists():
            try:
                with self.metadata_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if "|" in line:
                            k, v = line.strip().split("|", 1)
                            records[k.strip()] = v.strip()
            except Exception:
                pass
        records[prompt_id] = text
        with self.metadata_path.open("w", encoding="utf-8") as f:
            for k in sorted(records.keys()):
                f.write(f"{k}|{records[k]}\n")

    def _delete_from_metadata(self, prompt_id: str) -> None:
        records = {}
        if self.metadata_path.exists():
            try:
                with self.metadata_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if "|" in line:
                            k, v = line.strip().split("|", 1)
                            records[k.strip()] = v.strip()
            except Exception:
                pass
        if prompt_id in records:
            del records[prompt_id]
        with self.metadata_path.open("w", encoding="utf-8") as f:
            for k in sorted(records.keys()):
                f.write(f"{k}|{records[k]}\n")
