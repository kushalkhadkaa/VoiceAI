from __future__ import annotations

import base64
import time
from pathlib import Path


class AudioPipeline:
    def __init__(self, recordings_dir: Path) -> None:
        self.recordings_dir = recordings_dir
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, audio_bytes: bytes, suffix: str = ".webm") -> Path:
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        path = self.recordings_dir / f"turn-{int(time.time() * 1000)}{suffix}"
        path.write_bytes(audio_bytes)
        return path

    def save_base64(self, audio_base64: str, mime_type: str | None = None) -> Path:
        suffix = self._suffix_for_mime(mime_type)
        return self.save_bytes(base64.b64decode(audio_base64), suffix=suffix)

    @staticmethod
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
