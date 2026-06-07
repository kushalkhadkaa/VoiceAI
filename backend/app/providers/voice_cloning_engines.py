from __future__ import annotations
from dataclasses import dataclass, asdict
import importlib.util
from typing import Any

@dataclass
class CloningProviderInfo:
    installed: bool
    enabled: bool
    engine_name: str
    license: str
    commercial_status: str  # "production_ready", "commercial_ready", "experimental", "restricted"
    supported_languages: list[str]
    supports_zero_shot: bool
    supports_finetuning: bool
    supports_voice_conversion: bool
    minimum_reference_seconds: float
    recommended_reference_seconds: float
    hardware_requirements: str
    nepali_validated: bool
    english_validated: bool
    mixed_language_validated: bool

class VoiceCloningEngineRegistry:
    def __init__(self) -> None:
        chatterbox_installed = importlib.util.find_spec("chatterbox") is not None
        self._providers = {
            "piper": CloningProviderInfo(
                installed=True,  # Built-in default
                enabled=True,
                engine_name="Stable Piper Training",
                license="MIT / CC BY 4.0",
                commercial_status="production_ready",
                supported_languages=["ne", "en"],
                supports_zero_shot=False,
                supports_finetuning=True,
                supports_voice_conversion=False,
                minimum_reference_seconds=1800.0,  # 30 mins
                recommended_reference_seconds=36000.0, # 10 hours
                hardware_requirements="CPU/GPU",
                nepali_validated=True,
                english_validated=True,
                mixed_language_validated=True
            ),
            "elevenlabs": CloningProviderInfo(
                installed=True,  # Cloud provider, always available
                enabled=True,
                engine_name="ElevenLabs Voice Cloning",
                license="Commercial (ElevenLabs)",
                commercial_status="commercial_ready",
                supported_languages=["ne", "en"],
                supports_zero_shot=True,
                supports_finetuning=False,
                supports_voice_conversion=False,
                minimum_reference_seconds=10.0,
                recommended_reference_seconds=60.0,
                hardware_requirements="Cloud API",
                nepali_validated=True,
                english_validated=True,
                mixed_language_validated=True
            ),
            "chatterbox": CloningProviderInfo(
                installed=chatterbox_installed,
                enabled=True,
                engine_name="Chatterbox Local Zero-Shot Clone",
                license="MIT",
                commercial_status="commercial_ready",
                supported_languages=["ne", "en"],
                supports_zero_shot=True,
                supports_finetuning=True,
                supports_voice_conversion=True,
                minimum_reference_seconds=10.0,
                recommended_reference_seconds=60.0,
                hardware_requirements="Apple Silicon MPS, CUDA, or CPU",
                nepali_validated=False,
                english_validated=True,
                mixed_language_validated=False
            ),
            "f5_tts": CloningProviderInfo(
                installed=False,
                enabled=False,
                engine_name="F5-TTS Engine",
                license="MIT",
                commercial_status="experimental",
                supported_languages=["en"],
                supports_zero_shot=True,
                supports_finetuning=True,
                supports_voice_conversion=False,
                minimum_reference_seconds=15.0,
                recommended_reference_seconds=60.0,
                hardware_requirements="GPU (M1/M2/M3 or CUDA)",
                nepali_validated=False,
                english_validated=True,
                mixed_language_validated=False
            ),
            "openvoice": CloningProviderInfo(
                installed=False,
                enabled=False,
                engine_name="OpenVoice Engine",
                license="MIT",
                commercial_status="experimental",
                supported_languages=["en"],
                supports_zero_shot=True,
                supports_finetuning=False,
                supports_voice_conversion=True,
                minimum_reference_seconds=10.0,
                recommended_reference_seconds=30.0,
                hardware_requirements="GPU (M1/M2/M3 or CUDA)",
                nepali_validated=False,
                english_validated=True,
                mixed_language_validated=False
            ),
            "voxcpm": CloningProviderInfo(
                installed=False,
                enabled=False,
                engine_name="VoxCPM Engine",
                license="Creative Commons BY-NC",
                commercial_status="restricted",  # Non-commercial by default
                supported_languages=["ne", "en"],
                supports_zero_shot=True,
                supports_finetuning=True,
                supports_voice_conversion=True,
                minimum_reference_seconds=30.0,
                recommended_reference_seconds=120.0,
                hardware_requirements="GPU (CUDA)",
                nepali_validated=False,
                english_validated=False,
                mixed_language_validated=False
            )
        }

    def list_engines(self) -> dict[str, dict[str, Any]]:
        return {k: asdict(v) for k, v in self._providers.items()}

    def get_engine(self, engine_id: str) -> CloningProviderInfo | None:
        return self._providers.get(engine_id)
