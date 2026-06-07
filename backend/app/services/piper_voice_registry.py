from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PiperVoiceManifestEntry:
    id: str
    language: str
    quality: str
    model_path: str
    config_path: str
    source_url: str
    license: str
    commercial_review_status: str
    sample_rate: int | None
    installed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PiperVoiceRegistry:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.manifest_path = model_dir / "voice_manifest.json"

    def list_manifest(self) -> list[PiperVoiceManifestEntry]:
        if not self.manifest_path.exists():
            return []
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        entries: list[PiperVoiceManifestEntry] = []
        for item in payload.get("voices", []):
            model_path = Path(str(item.get("model_path", "")))
            config_path = Path(str(item.get("config_path", "")))
            entries.append(
                PiperVoiceManifestEntry(
                    id=str(item.get("id", "")),
                    language=str(item.get("language", "")),
                    quality=str(item.get("quality", "")),
                    model_path=str(model_path),
                    config_path=str(config_path),
                    source_url=str(item.get("source_url", "")),
                    license=str(item.get("license", "unknown")),
                    commercial_review_status=str(item.get("commercial_review_status", "needs_review")),
                    sample_rate=item.get("sample_rate"),
                    installed=model_path.exists() and config_path.exists(),
                )
            )
        return entries
