from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    id: str
    model_path: str
    config_path: str
    model_exists: bool
    config_exists: bool
    language: str
    status: str
    missing_files: list[str]
    license_path: str | None = None
    model_card_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VoiceRegistry:
    def __init__(self, voices_dir: Path) -> None:
        self.voices_dir = voices_dir
        self._ensure_default_manifest()

    def _ensure_default_manifest(self) -> None:
        manifest_path = self.voices_dir / "voice_manifest.json"
        if manifest_path.exists():
            return
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        import json
        default_data = {
            "source_repo": "rhasspy/piper-voices",
            "voices": [
                {
                    "id": "ne_NP-chitwan-medium",
                    "language": "ne",
                    "quality": "medium",
                    "repo_dir": "ne/ne_NP/chitwan/medium",
                    "license": "Creative Commons Attribution 4.0 International",
                    "commercial_review_status": "allowed",
                    "model_path": str(self.voices_dir / "ne_NP-chitwan-medium.onnx"),
                    "config_path": str(self.voices_dir / "ne_NP-chitwan-medium.onnx.json"),
                    "source_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ne/ne_NP/chitwan/medium/ne_NP-chitwan-medium.onnx"
                },
                {
                    "id": "ne_NP-google-medium",
                    "language": "ne",
                    "quality": "medium",
                    "repo_dir": "ne/ne_NP/google/medium",
                    "license": "Creative Commons Attribution 4.0 International",
                    "commercial_review_status": "allowed",
                    "model_path": str(self.voices_dir / "ne_NP-google-medium.onnx"),
                    "config_path": str(self.voices_dir / "ne_NP-google-medium.onnx.json"),
                    "source_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/ne/ne_NP/google/medium/ne_NP-google-medium.onnx"
                },
                {
                    "id": "en_US-lessac-medium",
                    "language": "en",
                    "quality": "medium",
                    "repo_dir": "en/en_US/lessac/medium",
                    "license": "Creative Commons Attribution 4.0 International",
                    "commercial_review_status": "allowed",
                    "model_path": str(self.voices_dir / "en_US-lessac-medium.onnx"),
                    "config_path": str(self.voices_dir / "en_US-lessac-medium.onnx.json"),
                    "source_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
                },
                {
                    "id": "en_US-ryan-medium",
                    "language": "en",
                    "quality": "medium",
                    "repo_dir": "en/en_US/ryan/medium",
                    "license": "Creative Commons Attribution 4.0 International",
                    "commercial_review_status": "allowed",
                    "model_path": str(self.voices_dir / "en_US-ryan-medium.onnx"),
                    "config_path": str(self.voices_dir / "en_US-ryan-medium.onnx.json"),
                    "source_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx"
                }
            ]
        }
        try:
            manifest_path.write_text(json.dumps(default_data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def list_voices(self, configured_paths: list[Path] | None = None) -> list[VoiceInfo]:
        paths: set[Path] = set()
        if self.voices_dir.exists():
            paths.update(self.voices_dir.glob("*.onnx"))
            paths.update(Path(str(path).removesuffix(".json")) for path in self.voices_dir.glob("*.onnx.json"))
        for path in configured_paths or []:
            paths.add(path)
        
        globbed_voices = [self.inspect(path) for path in paths]
        globbed_ids = {voice.id for voice in globbed_voices}
        
        manifest_path = self.voices_dir / "voice_manifest.json"
        manifest_voices = []
        if manifest_path.exists():
            try:
                import json
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                for item in payload.get("voices", []):
                    voice_id = item.get("id")
                    if voice_id not in globbed_ids:
                        model_path = Path(item.get("model_path", str(self.voices_dir / f"{voice_id}.onnx")))
                        manifest_voices.append(
                            VoiceInfo(
                                id=voice_id,
                                model_path=str(model_path),
                                config_path=str(model_path) + ".json",
                                model_exists=False,
                                config_exists=False,
                                language=item.get("language", "unknown"),
                                status="missing_files",
                                missing_files=[str(model_path), str(model_path) + ".json"],
                                license_path=item.get("license"),
                                model_card_path=item.get("source_url"),
                            )
                        )
            except Exception:
                pass
                
        all_voices = list(globbed_voices) + manifest_voices
        return sorted(all_voices, key=lambda voice: (voice.language, voice.id))

    def inspect(self, model_path: Path) -> VoiceInfo:
        model_path = model_path.expanduser()
        config_path = Path(f"{model_path}.json")
        model_exists = model_path.exists()
        config_exists = config_path.exists()
        missing = []
        if not model_exists:
            missing.append(str(model_path))
        if not config_exists:
            missing.append(str(config_path))
        language = self._language_for(model_path.name)
        sibling_files = list(model_path.parent.glob("*")) if model_path.parent.exists() else []
        license_path = self._first_named(sibling_files, ("LICENSE", "LICENSE.txt", "license.txt", "COPYING"))
        model_card_path = self._first_named(sibling_files, ("MODEL_CARD", "MODEL_CARD.md", "README.md"))
        return VoiceInfo(
            id=model_path.stem,
            model_path=str(model_path),
            config_path=str(config_path),
            model_exists=model_exists,
            config_exists=config_exists,
            language=language,
            status="ready" if model_exists and config_exists else "missing_files",
            missing_files=missing,
            license_path=str(license_path) if license_path else None,
            model_card_path=str(model_card_path) if model_card_path else None,
        )

    @staticmethod
    def _language_for(filename: str) -> str:
        lowered = filename.lower()
        if lowered.startswith("ne_") or "ne_np" in lowered or "nepali" in lowered:
            return "ne"
        if lowered.startswith("en_") or "en_us" in lowered or "english" in lowered:
            return "en"
        return "unknown"

    @staticmethod
    def _first_named(paths: list[Path], names: tuple[str, ...]) -> Path | None:
        wanted = {name.lower() for name in names}
        for path in paths:
            if path.name.lower() in wanted:
                return path
        return None
