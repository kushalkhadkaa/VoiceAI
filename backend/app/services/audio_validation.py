from __future__ import annotations

import platform
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


def _find_ffmpeg_binary() -> str | None:
    """
    Locate ffmpeg in this priority order:
      1. System PATH
      2. Common Windows installation directories
      3. static-ffmpeg bundled binary (auto-downloaded once, then cached)
    Returns the full path string, or None if completely unavailable.
    """
    # 1. System PATH
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2. Common Windows paths
    if platform.system() == "Windows":
        candidates = [
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"),
            Path(r"C:\tools\ffmpeg\bin\ffmpeg.exe"),
            Path.home() / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / "ffmpeg.exe",
        ]
        for c in candidates:
            if c.exists():
                return str(c)

    # 3. static-ffmpeg bundled binary (no system install required)
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()  # downloads ~100 MB once, then cached permanently
        found = shutil.which("ffmpeg")
        if found:
            return found
    except Exception:
        pass

    return None


# Cache the resolved path so we only search once per process lifetime
_FFMPEG_PATH: str | None = None
_FFMPEG_RESOLVED = False


def get_ffmpeg_path() -> str | None:
    global _FFMPEG_PATH, _FFMPEG_RESOLVED
    if not _FFMPEG_RESOLVED:
        _FFMPEG_PATH = _find_ffmpeg_binary()
        _FFMPEG_RESOLVED = True
    return _FFMPEG_PATH


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
        ffmpeg = get_ffmpeg_path()
        if ffmpeg is None:
            raise ProviderUnavailableError(
                "ffmpeg is not available. On Windows run: winget install ffmpeg  "
                "OR install via chocolatey: choco install ffmpeg  "
                "OR download from https://github.com/BtbN/FFmpeg-Builds/releases"
            )
        command = [
            ffmpeg, "-y", "-i", str(input_path),
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
            str(output_path),
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=30,
                shell=(platform.system() == "Windows"),
            )
        except subprocess.CalledProcessError as exc:
            raise AudioValidationError("Unable to decode audio. Use webm/opus, wav, mp3, m4a, or mp4 audio.") from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioValidationError("Audio decoding timed out.") from exc


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
