from __future__ import annotations

import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from app.providers.stt import ProviderUnavailableError


ALLOWED_MIME_TYPES = {
    "audio/webm",
    "audio/webm;codecs=opus",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
}


class AudioValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AudioValidationResult:
    path: Path
    mime_type: str
    duration_seconds: float | None
    size_bytes: int


class AudioValidator:
    def __init__(self, max_bytes: int, max_seconds: float) -> None:
        self.max_bytes = max_bytes
        self.max_seconds = max_seconds

    def validate_upload(self, audio_bytes: bytes, mime_type: str | None) -> str:
        normalized_mime = (mime_type or "").split(",", 1)[0].strip().lower()
        if not audio_bytes:
            raise AudioValidationError("Audio upload is empty.")
        if len(audio_bytes) > self.max_bytes:
            raise AudioValidationError(f"Audio upload is too large. Limit is {self.max_bytes} bytes.")
        if normalized_mime not in ALLOWED_MIME_TYPES:
            raise AudioValidationError(f"Unsupported audio MIME type: {mime_type or 'missing'}.")
        return normalized_mime

    def validate_wav_duration(self, path: Path) -> AudioValidationResult:
        duration = _wav_duration(path)
        if duration > self.max_seconds:
            raise AudioValidationError(f"Audio is too long. Limit is {self.max_seconds:.0f} seconds.")
        return AudioValidationResult(path=path, mime_type="audio/wav", duration_seconds=duration, size_bytes=path.stat().st_size)


class AudioPipeline:
    def __init__(
        self,
        work_dir: Path,
        validator: AudioValidator,
        keep_turn_audio: bool = False,
    ) -> None:
        self.work_dir = work_dir
        self.validator = validator
        self.keep_turn_audio = keep_turn_audio
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def prepare_stt_audio(self, audio_bytes: bytes, mime_type: str | None) -> Path:
        normalized_mime = self.validator.validate_upload(audio_bytes, mime_type)
        input_suffix = _suffix_for_mime(normalized_mime)
        token = int(time.time() * 1000)
        input_path = self.work_dir / f"turn-{token}{input_suffix}"
        wav_path = self.work_dir / f"turn-{token}.16k.wav"
        input_path.write_bytes(audio_bytes)
        try:
            if input_suffix == ".wav":
                self._convert_to_wav(input_path, wav_path)
            else:
                self._convert_to_wav(input_path, wav_path)
            self.validator.validate_wav_duration(wav_path)
        finally:
            if not self.keep_turn_audio:
                input_path.unlink(missing_ok=True)
        return wav_path

    def cleanup_turn_audio(self, path: Path) -> None:
        if not self.keep_turn_audio:
            path.unlink(missing_ok=True)

    def _convert_to_wav(self, input_path: Path, output_path: Path) -> None:
        if shutil.which("ffmpeg") is None:
            raise ProviderUnavailableError("ffmpeg is required to decode browser audio. Run: brew install ffmpeg")
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(output_path),
        ]
        try:
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise AudioValidationError("Unable to decode audio. Use webm/opus, wav, mp3, m4a, or mp4 audio.") from exc


def _suffix_for_mime(mime_type: str | None) -> str:
    if not mime_type:
        return ".webm"
    if "wav" in mime_type:
        return ".wav"
    if "mp4" in mime_type or "m4a" in mime_type:
        return ".m4a"
    if "mpeg" in mime_type or "mp3" in mime_type:
        return ".mp3"
    return ".webm"


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate() or 1)
    except wave.Error as exc:
        raise AudioValidationError("Decoded audio is not a valid WAV file.") from exc
