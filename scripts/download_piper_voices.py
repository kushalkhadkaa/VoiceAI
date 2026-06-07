from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOICE_DIR = Path("models/piper")


@dataclass(frozen=True, slots=True)
class PiperVoiceSpec:
    id: str
    language: str
    quality: str
    repo_dir: str
    license: str = "unknown; review rhasspy/piper-voices model card before commercial use"
    commercial_review_status: str = "needs_review"

    @property
    def model_name(self) -> str:
        return f"{self.id}.onnx"

    @property
    def config_name(self) -> str:
        return f"{self.id}.onnx.json"


DEFAULT_VOICES = [
    PiperVoiceSpec("ne_NP-chitwan-medium", "ne_NP", "medium", "ne/ne_NP/chitwan/medium"),
    PiperVoiceSpec("ne_NP-google-medium", "ne_NP", "medium", "ne/ne_NP/google/medium"),
    PiperVoiceSpec("en_US-lessac-medium", "en_US", "medium", "en/en_US/lessac/medium"),
    PiperVoiceSpec("en_US-ryan-medium", "en_US", "medium", "en/en_US/ryan/medium"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download default Piper voices for SwarLocal.")
    parser.add_argument("--model-dir", type=Path, default=VOICE_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"source_repo": "rhasspy/piper-voices", "voices": []}
    for spec in DEFAULT_VOICES:
        model_path = args.model_dir / spec.model_name
        config_path = args.model_dir / spec.config_name
        model_url = f"{HF_BASE}/{spec.repo_dir}/{spec.model_name}"
        config_url = f"{HF_BASE}/{spec.repo_dir}/{spec.config_name}"
        _download(model_url, model_path, force=args.force)
        _download(config_url, config_path, force=args.force)
        config = _read_json(config_path)
        manifest["voices"].append(
            {
                **asdict(spec),
                "model_path": str(model_path),
                "config_path": str(config_path),
                "source_url": model_url,
                "config_source_url": config_url,
                "sample_rate": config.get("audio", {}).get("sample_rate"),
                "installed": model_path.exists() and config_path.exists(),
            }
        )
    manifest_path = args.model_dir / "voice_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


def _download(url: str, path: Path, *, force: bool) -> None:
    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"present {path}")
        return
    print(f"download {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if len(data) < 128:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {url}")
    path.write_bytes(data)
    print(f"wrote {path} ({len(data) / (1024 * 1024):.1f} MB)")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
