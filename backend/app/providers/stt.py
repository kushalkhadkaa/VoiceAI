from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ProviderUnavailableError(RuntimeError):
    """Raised when a local provider dependency is missing or not configured."""


@dataclass(frozen=True, slots=True)
class STTResult:
    text: str
    language: str | None
    confidence: float | None
    duration_ms: float


class STTProvider(Protocol):
    def transcribe_file(self, audio_path: Path) -> STTResult:
        ...


class FasterWhisperSTTProvider:
    def __init__(self, model_size: str, device: str = "auto", compute_type: str = "auto") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ProviderUnavailableError(
                    "faster-whisper is not installed. Run `make setup` first."
                ) from exc
            kwargs = {}
            if self.device != "auto":
                kwargs["device"] = self.device
            if self.compute_type != "auto":
                kwargs["compute_type"] = self.compute_type
            self._model = WhisperModel(self.model_size, **kwargs)
        return self._model

    def transcribe_file(self, audio_path: Path) -> STTResult:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        started = time.perf_counter()
        segments, info = self.model.transcribe(str(audio_path), vad_filter=True, beam_size=1)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        duration_ms = (time.perf_counter() - started) * 1000
        return STTResult(
            text=text,
            language=getattr(info, "language", None),
            confidence=getattr(info, "language_probability", None),
            duration_ms=duration_ms,
        )
