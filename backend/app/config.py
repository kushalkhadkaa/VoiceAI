from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(slots=True)
class Settings:
    app_env: str = "development"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"
    ollama_fallback_models: tuple[str, ...] = ("qwen3:4b", "llama3.2:3b", "llama3.2:1b")
    ollama_temperature: float = 0.35
    ollama_num_predict: int = 180
    ollama_keep_alive: str = "10m"
    ollama_timeout_seconds: float = 60
    ollama_retries: int = 1
    system_prompt: str = (
        "You are SwarLocal, a helpful Nepali-English voice agent. Reply naturally in the user’s language. "
        "If the user mixes Nepali and English, reply naturally in mixed Nepali-English. Keep spoken answers concise. "
        "Use selected knowledge when available. Do not invent facts."
    )
    whisper_model_size: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    piper_binary: str = "piper"
    piper_nepali_voice: Path = Path("./models/piper/ne_NP-chitwan-medium.onnx")
    piper_english_voice: Path = Path("./models/piper/en_US-lessac-medium.onnx")
    piper_audio_cache_dir: Path = Path(".local/audio_cache")
    piper_train_command: str = ""
    audio_work_dir: Path = Path(".local/audio_work")
    runtime_settings_path: Path = Path(".local/settings.json")
    max_upload_bytes: int = 25 * 1024 * 1024
    max_recording_seconds: float = 30
    keep_turn_audio: bool = False
    low_latency_mode: bool = True
    quality_mode: bool = False
    open_webui_base_url: str = "http://127.0.0.1:8080"
    open_webui_api_key: str = ""
    rag_enabled: bool = False
    rag_default_collection: str = ""
    rag_fallback_to_ollama: bool = True
    internet_retrieval_enabled: bool = False
    internet_max_sources: int = 5
    internet_require_citation: bool = True
    internet_fallback_allowed: bool = False
    llm_provider: str = "local"
    local_model: str = "qwen2.5:7b"
    local_fallback_model: str = "gemma3:4b"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    elevenlabs_api_key: str = ""
    cloud_fallback_to_local: bool = True
    cloud_timeout_seconds: float = 30.0
    cloud_temperature: float = 0.35
    cloud_max_tokens: int = 180
    force_selected_voice: bool = False
    fallback_allowed: bool = True
    single_tts_voice_model: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        settings = cls(
            app_env=os.getenv("APP_ENV", defaults.app_env),
            backend_host=os.getenv("BACKEND_HOST", defaults.backend_host),
            backend_port=int(os.getenv("BACKEND_PORT", str(defaults.backend_port))),
            cors_origins=_csv_env("CORS_ORIGINS", defaults.cors_origins),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", defaults.ollama_base_url).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", defaults.ollama_model),
            ollama_fallback_models=_csv_env("OLLAMA_FALLBACK_MODELS", defaults.ollama_fallback_models),
            ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", str(defaults.ollama_temperature))),
            ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", str(defaults.ollama_num_predict))),
            ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", defaults.ollama_keep_alive),
            ollama_timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", str(defaults.ollama_timeout_seconds))),
            ollama_retries=int(os.getenv("OLLAMA_RETRIES", str(defaults.ollama_retries))),
            system_prompt=os.getenv("SYSTEM_PROMPT", defaults.system_prompt),
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", defaults.whisper_model_size),
            whisper_device=os.getenv("WHISPER_DEVICE", defaults.whisper_device),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", defaults.whisper_compute_type),
            piper_binary=os.getenv("PIPER_BINARY", defaults.piper_binary),
            piper_nepali_voice=Path(os.getenv("PIPER_NEPALI_VOICE", str(defaults.piper_nepali_voice))),
            piper_english_voice=Path(os.getenv("PIPER_ENGLISH_VOICE", str(defaults.piper_english_voice))),
            piper_audio_cache_dir=Path(os.getenv("PIPER_AUDIO_CACHE_DIR", str(defaults.piper_audio_cache_dir))),
            piper_train_command=os.getenv("PIPER_TRAIN_COMMAND", defaults.piper_train_command),
            audio_work_dir=Path(os.getenv("AUDIO_WORK_DIR", str(defaults.audio_work_dir))),
            runtime_settings_path=Path(os.getenv("RUNTIME_SETTINGS_PATH", str(defaults.runtime_settings_path))),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(defaults.max_upload_bytes))),
            max_recording_seconds=float(os.getenv("MAX_RECORDING_SECONDS", str(defaults.max_recording_seconds))),
            keep_turn_audio=_bool_env("KEEP_TURN_AUDIO", defaults.keep_turn_audio),
            low_latency_mode=_bool_env("LOW_LATENCY_MODE", defaults.low_latency_mode),
            quality_mode=_bool_env("QUALITY_MODE", defaults.quality_mode),
            open_webui_base_url=os.getenv("OPEN_WEBUI_BASE_URL", defaults.open_webui_base_url).rstrip("/"),
            open_webui_api_key=os.getenv("OPEN_WEBUI_API_KEY", defaults.open_webui_api_key),
            rag_enabled=_bool_env("RAG_ENABLED", defaults.rag_enabled),
            rag_default_collection=os.getenv("RAG_DEFAULT_COLLECTION", defaults.rag_default_collection),
            rag_fallback_to_ollama=_bool_env("RAG_FALLBACK_TO_OLLAMA", defaults.rag_fallback_to_ollama),
            internet_retrieval_enabled=_bool_env("INTERNET_RETRIEVAL_ENABLED", defaults.internet_retrieval_enabled),
            internet_max_sources=int(os.getenv("INTERNET_MAX_SOURCES", str(defaults.internet_max_sources))),
            internet_require_citation=_bool_env("INTERNET_REQUIRE_CITATION", defaults.internet_require_citation),
            internet_fallback_allowed=_bool_env("INTERNET_FALLBACK_ALLOWED", defaults.internet_fallback_allowed),
            llm_provider=os.getenv("LLM_PROVIDER", defaults.llm_provider),
            local_model=os.getenv("LOCAL_MODEL", defaults.local_model),
            local_fallback_model=os.getenv("LOCAL_FALLBACK_MODEL", defaults.local_fallback_model),
            openai_api_key=os.getenv("OPENAI_API_KEY", defaults.openai_api_key),
            openai_model=os.getenv("OPENAI_MODEL", defaults.openai_model),
            gemini_api_key=os.getenv("GEMINI_API_KEY", defaults.gemini_api_key),
            gemini_model=os.getenv("GEMINI_MODEL", defaults.gemini_model),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", defaults.elevenlabs_api_key),
            cloud_fallback_to_local=_bool_env("CLOUD_FALLBACK_TO_LOCAL", defaults.cloud_fallback_to_local),
            cloud_timeout_seconds=float(os.getenv("CLOUD_TIMEOUT_SECONDS", str(defaults.cloud_timeout_seconds))),
            cloud_temperature=float(os.getenv("CLOUD_TEMPERATURE", str(defaults.cloud_temperature))),
            cloud_max_tokens=int(os.getenv("CLOUD_MAX_TOKENS", str(defaults.cloud_max_tokens))),
            force_selected_voice=_bool_env("FORCE_SELECTED_VOICE", defaults.force_selected_voice),
            fallback_allowed=_bool_env("FALLBACK_ALLOWED", defaults.fallback_allowed),
            single_tts_voice_model=_bool_env("SINGLE_TTS_VOICE_MODEL", defaults.single_tts_voice_model),
        )
        settings._load_runtime_overrides()
        settings.ensure_dirs()
        return settings

    def ensure_dirs(self) -> None:
        self.piper_audio_cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_work_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_settings_path.parent.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
            if isinstance(value, tuple):
                data[key] = list(value)
        # mask keys
        for key in ("openai_api_key", "gemini_api_key", "open_webui_api_key", "elevenlabs_api_key"):
            if key in data and data[key]:
                data[key] = self._mask_key(data[key])
        return data

    @staticmethod
    def _mask_key(key: str) -> str:
        if not key:
            return ""
        if len(key) <= 8:
            return "..."
        return f"{key[:4]}...{key[-4:]}"

    def update_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "ollama_model",
            "ollama_base_url",
            "ollama_temperature",
            "ollama_num_predict",
            "ollama_keep_alive",
            "system_prompt",
            "whisper_model_size",
            "piper_nepali_voice",
            "piper_english_voice",
            "piper_train_command",
            "max_recording_seconds",
            "low_latency_mode",
            "quality_mode",
            "open_webui_base_url",
            "open_webui_api_key",
            "rag_enabled",
            "rag_default_collection",
            "rag_fallback_to_ollama",
            "internet_retrieval_enabled",
            "internet_max_sources",
            "internet_require_citation",
            "internet_fallback_allowed",
            "llm_provider",
            "local_model",
            "local_fallback_model",
            "openai_api_key",
            "openai_model",
            "gemini_api_key",
            "gemini_model",
            "cloud_fallback_to_local",
            "cloud_timeout_seconds",
            "cloud_temperature",
            "cloud_max_tokens",
            "force_selected_voice",
            "fallback_allowed",
            "single_tts_voice_model",
            "elevenlabs_api_key",
        }
        changed: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in allowed or value in (None, ""):
                continue
            if key in ("openai_api_key", "gemini_api_key", "open_webui_api_key", "elevenlabs_api_key") and "..." in str(value):
                continue
            if key.startswith("piper_"):
                value = Path(str(value)).expanduser()
            setattr(self, key, value)
            changed[key] = str(value) if isinstance(value, Path) else value
        if changed:
            self._write_runtime_overrides()
        return changed

    def _load_runtime_overrides(self) -> None:
        if not self.runtime_settings_path.exists():
            return
        try:
            payload = json.loads(self.runtime_settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for key, value in payload.items():
            if not hasattr(self, key):
                continue
            if key.startswith("piper_"):
                value = Path(str(value)).expanduser()
            setattr(self, key, value)

    def _write_runtime_overrides(self) -> None:
        payload = {
            "ollama_model": self.ollama_model,
            "ollama_base_url": self.ollama_base_url,
            "ollama_temperature": self.ollama_temperature,
            "ollama_num_predict": self.ollama_num_predict,
            "ollama_keep_alive": self.ollama_keep_alive,
            "system_prompt": self.system_prompt,
            "whisper_model_size": self.whisper_model_size,
            "piper_nepali_voice": str(self.piper_nepali_voice),
            "piper_english_voice": str(self.piper_english_voice),
            "piper_train_command": self.piper_train_command,
            "max_recording_seconds": self.max_recording_seconds,
            "low_latency_mode": self.low_latency_mode,
            "quality_mode": self.quality_mode,
            "open_webui_base_url": self.open_webui_base_url,
            "open_webui_api_key": self.open_webui_api_key,
            "rag_enabled": self.rag_enabled,
            "rag_default_collection": self.rag_default_collection,
            "rag_fallback_to_ollama": self.rag_fallback_to_ollama,
            "internet_retrieval_enabled": self.internet_retrieval_enabled,
            "internet_max_sources": self.internet_max_sources,
            "internet_require_citation": self.internet_require_citation,
            "internet_fallback_allowed": self.internet_fallback_allowed,
            "llm_provider": self.llm_provider,
            "local_model": self.local_model,
            "local_fallback_model": self.local_fallback_model,
            "openai_api_key": self.openai_api_key,
            "openai_model": self.openai_model,
            "gemini_api_key": self.gemini_api_key,
            "gemini_model": self.gemini_model,
            "elevenlabs_api_key": self.elevenlabs_api_key,
            "cloud_fallback_to_local": self.cloud_fallback_to_local,
            "cloud_timeout_seconds": self.cloud_timeout_seconds,
            "cloud_temperature": self.cloud_temperature,
            "cloud_max_tokens": self.cloud_max_tokens,
            "force_selected_voice": self.force_selected_voice,
            "fallback_allowed": self.fallback_allowed,
            "single_tts_voice_model": self.single_tts_voice_model,
        }
        self.runtime_settings_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
