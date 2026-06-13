from __future__ import annotations

import base64
import binascii
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Response, BackgroundTasks, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import Settings
from app.providers.llm_ollama import OllamaLLMProvider
from app.providers.rag_openwebui import OpenWebUIRagProvider
from app.providers.embeddings import get_embedding_provider
from app.providers.stt import FasterWhisperSTTProvider, ProviderUnavailableError
from app.providers.tts import PiperTTSProvider, TTSPart
from app.providers.web_retrieval import WebRetrievalProvider
from app.schemas import (
    ChatTestRequest,
    DoctorResponse,
    HealthResponse,
    ModelStatusResponse,
    ProviderStatus,
    RagTestRequest,
    SettingsResponse,
    SettingsUpdateRequest,
    TtsTestRequest,
    TtsTestResponse,
    VoiceSocketStatusResponse,
    VoicesResponse,
    DatasetQualityResponse,
    DatasetRecordingResponse,
    RatingRequest,
)
from app.services.audio_validation import AudioPipeline, AudioValidationError, AudioValidator
from app.services.conversation import ConversationService
from app.services.environment import run_environment_checks
from app.services.language_router import LanguageRouter
from app.services.voice_socket_status import build_voice_socket_status
from app.services.voice_registry import VoiceRegistry

from app.services.system_monitor import SystemMonitorService
from app.providers.voice_cloning_engines import VoiceCloningEngineRegistry
from app.services.voice_identity import VoiceIdentityService
from app.services.voice_similarity import VoiceSimilarityService
from app.services.speaker_isolation import SpeakerIsolationService
from app.services.turn_detector import TurnDetector
from app.services.vad_controller import VADController
from app.services.audio_enhancement import AudioEnhancementService
from app.services.noise_reduction import NoiseReductionService
from app.services.voice_clone import VoiceCloneError, VoiceCloneService

settings = Settings.from_env()
language_router = LanguageRouter()
from app.providers.stt import STTRouter
stt_provider = STTRouter(settings)
llm_provider = OllamaLLMProvider(
    settings.ollama_base_url,
    settings.ollama_model,
    timeout_seconds=settings.ollama_timeout_seconds,
    retries=settings.ollama_retries,
    temperature=settings.ollama_temperature,
    num_predict=settings.ollama_num_predict,
    keep_alive=settings.ollama_keep_alive,
    system_prompt=settings.system_prompt,
)
from app.providers.tts import ChatterboxTTSProvider, OpenAITTSProvider, ElevenLabsTTSProvider, TTSRouter

piper_tts = PiperTTSProvider(
    settings.piper_binary,
    settings.piper_nepali_voice,
    settings.piper_english_voice,
    settings.piper_audio_cache_dir,
    single_voice_model=settings.single_tts_voice_model,
)
openai_tts = OpenAITTSProvider(
    settings=settings,
    audio_cache_dir=settings.piper_audio_cache_dir,
)
elevenlabs_tts = ElevenLabsTTSProvider(
    settings=settings,
    audio_cache_dir=settings.piper_audio_cache_dir,
)
chatterbox_tts = ChatterboxTTSProvider(
    settings=settings,
    audio_cache_dir=settings.piper_audio_cache_dir,
)
tts_provider = TTSRouter(piper_tts, openai_tts, elevenlabs_tts, chatterbox_tts)
audio_validator = AudioValidator(settings.max_upload_bytes, settings.max_recording_seconds)
audio_pipeline = AudioPipeline(settings.audio_work_dir, audio_validator, keep_turn_audio=settings.keep_turn_audio)
voice_registry = VoiceRegistry(Path("models/piper"))
from app.providers.llm_openwebui import OpenWebUILLMProvider
from app.services.voice_studio import VoiceStudioService

openwebui_llm_provider = OpenWebUILLMProvider(settings, llm_provider)
voice_studio_service = VoiceStudioService(settings.audio_work_dir.parent / "voices", audio_validator)

system_monitor_service = SystemMonitorService()
voice_cloning_registry = VoiceCloningEngineRegistry()

voices_base_dir = settings.audio_work_dir.parent / "voices"
voice_clone_service = VoiceCloneService(voices_base_dir)
voice_identity_service = VoiceIdentityService(voices_base_dir)
voice_similarity_service = VoiceSimilarityService(voice_identity_service)
speaker_isolation_service = SpeakerIsolationService(voice_identity_service)

audio_enhancement_service = AudioEnhancementService()
noise_reduction_service = NoiseReductionService()

# Instantiate turn detector dynamically based on settings
turn_detector = TurnDetector(
    input_mode="auto",
    silence_timeout_ms=700,
    max_duration_seconds=settings.max_recording_seconds,
)
vad_controller = VADController(turn_detector)

web_retrieval_provider = WebRetrievalProvider(
    settings.internet_retrieval_enabled,
    settings.internet_max_sources,
    settings.internet_require_citation,
    settings.internet_fallback_allowed,
)

from app.services.kyc_service import KYCService
from app.services import admin_auth

kyc_service = KYCService(settings)

conversation_service = ConversationService(
    stt_provider,
    llm_provider,
    openwebui_llm_provider,
    tts_provider,
    language_router,
    web_retrieval_provider,
    kb_service=None,  # set after kb_service is created below
)
from app.services.dataset import DatasetService

rag_provider = OpenWebUIRagProvider(settings)

# Local KB (ChromaDB + Ollama/sentence-transformers embeddings)
from app.services.rag_service import RAGService
def _embedding_model_for(provider: str) -> str:
    """Pick the model name to hand the embedding factory for a given provider.

    For "auto" we pass an empty string so the factory's OpenAI-first / local
    fallback policy chooses the correct default model itself.
    """
    if provider == "ollama":
        return settings.kb_embedding_model
    if provider == "openai":
        return "text-embedding-3-small"
    if provider == "sentence-transformers":
        return settings.kb_embedding_model_st
    return ""  # auto


_kb_embedding = get_embedding_provider(
    settings.kb_embedding_provider,
    settings.ollama_base_url,
    _embedding_model_for(settings.kb_embedding_provider),
    openai_api_key=settings.openai_api_key,
    timeout_seconds=settings.cloud_timeout_seconds,
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _collection_meta_has_docs(path: Path) -> bool:
    meta_path = path / "collections_meta.json"
    if not meta_path.exists():
        return False
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return any(bool(item.get("documents")) for item in payload.values()) if isinstance(payload, dict) else False


def _resolve_kb_db_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    legacy_path = Path(__file__).resolve().parents[1] / raw_path
    if (
        not _collection_meta_has_docs(path)
        and legacy_path.exists()
        and _collection_meta_has_docs(legacy_path)
        and path.resolve() != legacy_path.resolve()
    ):
        path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(legacy_path, path, dirs_exist_ok=True)
    return path


_kb_db_path = _resolve_kb_db_path(settings.kb_chromadb_path)
kb_service = RAGService(
    db_path=str(_kb_db_path),
    embedding_provider=_kb_embedding,
    chunk_size=settings.kb_chunk_size,
    chunk_overlap=settings.kb_chunk_overlap,
    max_results=settings.kb_max_results,
    similarity_threshold=settings.kb_similarity_threshold,
    search_mode=settings.kb_search_mode,
    chunk_strategy=settings.kb_chunk_strategy,
    reranking_enabled=settings.kb_reranking_enabled,
    reranking_model=settings.kb_reranking_model,
    query_analytics=settings.kb_query_analytics,
)

conversation_service.kb_service = kb_service  # wire in after kb_service is created

dataset_service = DatasetService(settings.audio_work_dir.parent / "dataset", audio_validator)

from app.database import init_db
init_db()

app = FastAPI(title="SwarLocal Voice Assistant", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
dataset_wav_dir = settings.audio_work_dir.parent / "dataset" / "wav"
dataset_wav_dir.mkdir(parents=True, exist_ok=True)
app.mount("/audio/dataset/wav", StaticFiles(directory=str(dataset_wav_dir)), name="dataset-wav")

voices_dir = settings.audio_work_dir.parent / "voices"
voices_dir.mkdir(parents=True, exist_ok=True)
app.mount("/audio/voices", StaticFiles(directory=str(voices_dir)), name="voices-audio")
app.mount("/audio", StaticFiles(directory=str(settings.piper_audio_cache_dir)), name="audio")


# When a heavy job (cloned-voice synthesis, voice build) is already running, any
# new request to start another is rejected with HTTP 429 and a clear, friendly
# message — handled in one place so it applies to every endpoint.
from app.services.heavy_jobs import HeavyJobBusy as _HeavyJobBusy, running_jobs as _running_jobs


@app.exception_handler(_HeavyJobBusy)
async def _heavy_job_busy_handler(_request, exc: _HeavyJobBusy):
    return JSONResponse(status_code=429, content={"detail": exc.message},
                        headers={"Retry-After": str(exc.retry_after)})


@app.get("/system/busy")
def system_busy():
    """What heavy job (if any) is running right now, so the UI can show it."""
    jobs = _running_jobs()
    return {"busy": bool(jobs), "running": jobs}


import time as _time
_BACKEND_STARTED = _time.time()


@app.get("/system/pulse")
def system_pulse():
    """Fast heartbeat for the UI: uptime + lightweight health checks + any active
    problems with plain-language what/why/how-to-fix. Uses only fast, local checks
    (no network calls) so it stays responsive even when the machine is busy."""
    import datetime
    checks: list[dict] = [{"key": "backend", "label": "Backend API", "ok": True,
                           "detail": "Responding", "severity": "ok"}]
    problems: list[dict] = []

    # Knowledge base (in-memory metadata — fast)
    try:
        kb_status = kb_service.status()
        cols = kb_service.list_collections()
        docs = sum(getattr(c, "document_count", 0) for c in cols)
        chunks = sum(getattr(c, "chunk_count", 0) for c in cols)
        rag_ok = len(cols) > 0
        kb_detail = f"{len(cols)} collection(s) · {docs} docs · {chunks} chunks · {kb_status.get('db_path', 'local KB')}" if rag_ok else (
            f"No local knowledge bases in {kb_status.get('db_path', 'the current KB path')}."
        )
        checks.append({"key": "rag", "label": "Knowledge base", "ok": rag_ok,
                       "detail": kb_detail,
                       "severity": "ok" if rag_ok else "info"})
        if not rag_ok:
            problems.append({
                "what": "No indexed knowledge base is loaded",
                "why": "RAG is not permanently broken; the current backend is looking at an empty local KB path.",
                "fix": "Create or import a knowledge base in the Knowledge page. If you expected old data, confirm you opened /Users/kushalkhadka/VoiceAI and backend port 8001.",
            })
    except Exception as exc:
        checks.append({"key": "rag", "label": "Knowledge base", "ok": False,
                       "detail": f"Could not read knowledge bases: {exc}", "severity": "error"})
        problems.append({"what": "Knowledge base unreadable",
                         "why": "The RAG metadata store could not be opened.",
                         "fix": "Restart the backend; if it persists, check .local/swarlocal.db."})

    # Heavy jobs in progress
    jobs = _running_jobs()
    if jobs:
        j = jobs[0]
        checks.append({"key": "busy", "label": "Heavy job running", "ok": True,
                       "detail": f"{j['label']} — ~{j['remaining_seconds']}s left", "severity": "warn"})

    # Disk + memory (fast, local)
    try:
        total, _used, free = shutil.disk_usage("/")
        free_gb = round(free / (1024 ** 3), 1)
        disk_ok = free_gb >= 8
        checks.append({"key": "disk", "label": "Disk space", "ok": disk_ok,
                       "detail": f"{free_gb} GB free", "severity": "ok" if disk_ok else "warn"})
        if not disk_ok:
            problems.append({"what": f"Low disk space ({free_gb} GB free)",
                             "why": "Cloned-voice models (~6 GB) and audio cache need room; low disk can crash synthesis and the backend.",
                             "fix": "Free up disk to 15 GB+ (empty Trash, clear ~/.cache/huggingface of unused models)."})
    except Exception:
        pass
    try:
        m = system_monitor_service.get_realtime_metrics()
        ram_free = m.get("ram_available_gb")
        if ram_free is not None:
            ram_ok = ram_free >= 1.5
            checks.append({"key": "memory", "label": "Free memory", "ok": ram_ok,
                           "detail": f"{ram_free} GB free", "severity": "ok" if ram_ok else "warn"})
            if not ram_ok:
                problems.append({"what": f"Low free memory ({ram_free} GB)",
                                 "why": "Running a local model + cloned-voice synthesis together can exhaust RAM and the OS kills the backend.",
                                 "fix": "Close other apps, or use OpenAI (cloud) voice/brain instead of local ones."})
    except Exception:
        pass

    uptime = int(_time.time() - _BACKEND_STARTED)
    status = "down" if any(not c["ok"] and c.get("severity") == "error" for c in checks) else \
             ("degraded" if problems else "healthy")
    return {
        "ok": True,
        "status": status,
        "uptime_seconds": uptime,
        "started_at": datetime.datetime.fromtimestamp(_BACKEND_STARTED).isoformat(),
        "now": datetime.datetime.now().isoformat(),
        "checks": checks,
        "problems": problems,
    }


@app.exception_handler(ProviderUnavailableError)
async def provider_unavailable_handler(_, exc: ProviderUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(AudioValidationError)
async def audio_validation_handler(_, exc: AudioValidationError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_, exc: RuntimeError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, app="SwarLocal", version=__version__)


@app.get("/settings", response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    return SettingsResponse(settings=settings.public_dict())


@app.post("/settings", response_model=SettingsResponse)
def update_settings(payload: SettingsUpdateRequest) -> SettingsResponse:
    changed = settings.update_from_payload(payload.model_dump(exclude_none=True))
    if changed:
        try:
            voice_studio_service.log_audit(
                user_id="system",
                event="settings_updated",
                details=f"Updated settings: {', '.join(changed.keys())}"
            )
        except Exception:
            pass
    if {"ollama_model", "ollama_base_url", "ollama_temperature", "ollama_num_predict", "ollama_keep_alive", "system_prompt"} & set(changed):
        llm_provider.base_url = settings.ollama_base_url
        llm_provider.model = settings.ollama_model
        llm_provider.temperature = settings.ollama_temperature
        llm_provider.num_predict = settings.ollama_num_predict
        llm_provider.keep_alive = settings.ollama_keep_alive
        llm_provider.system_prompt = settings.system_prompt
    if {"piper_nepali_voice", "piper_english_voice"} & set(changed):
        piper_tts.nepali_voice = settings.piper_nepali_voice
        piper_tts.english_voice = settings.piper_english_voice
    if "single_tts_voice_model" in changed:
        piper_tts.single_voice_model = settings.single_tts_voice_model
    if {"open_webui_base_url", "open_webui_api_key", "rag_enabled", "rag_default_collection", "rag_fallback_to_ollama"} & set(changed):
        rag_provider.settings = settings
        openwebui_llm_provider.settings = settings
        conversation_service.settings = settings
    if {
        "llm_provider",
        "stt_provider",
        "tts_provider",
        "openai_api_key",
        "openai_model",
        "openai_tts_voice",
        "cloud_fallback_to_local",
        "cloud_timeout_seconds",
        "cloud_temperature",
        "cloud_max_tokens",
        "bank_instruction",
    } & set(changed):
        conversation_service.settings = settings
        openwebui_llm_provider.settings = settings
        stt_provider.settings = settings
        openai_tts.settings = settings
        openai_tts.timeout_seconds = settings.cloud_timeout_seconds
    if {"internet_retrieval_enabled", "internet_max_sources", "internet_require_citation", "internet_fallback_allowed"} & set(changed):
        web_retrieval_provider.enabled = settings.internet_retrieval_enabled
        web_retrieval_provider.max_sources = settings.internet_max_sources
        web_retrieval_provider.require_citation = settings.internet_require_citation
        web_retrieval_provider.fallback_allowed = settings.internet_fallback_allowed
    if {
        "kb_embedding_provider",
        "kb_embedding_model",
        "kb_embedding_model_st",
        "kb_chunk_size",
        "kb_chunk_overlap",
        "kb_max_results",
        "kb_similarity_threshold",
        "openai_api_key",
        "cloud_timeout_seconds",
    } & set(changed):
        new_emb = get_embedding_provider(
            settings.kb_embedding_provider,
            settings.ollama_base_url,
            _embedding_model_for(settings.kb_embedding_provider),
            openai_api_key=settings.openai_api_key,
            timeout_seconds=settings.cloud_timeout_seconds,
        )
        kb_service.embedding_provider = new_emb
        kb_service.chunk_size = settings.kb_chunk_size
        kb_service.chunk_overlap = settings.kb_chunk_overlap
        kb_service.max_results = settings.kb_max_results
        kb_service.similarity_threshold = settings.kb_similarity_threshold
    return SettingsResponse(settings=settings.public_dict())


@app.get("/ai-providers")
def get_ai_providers():
    return {
        "providers": [
            {
                "id": "local",
                "name": "Local AI (Ollama)",
                "description": "Run open-source models locally on your Mac. Keep all transcripts and reasoning completely private.",
                "default_model": "qwen2.5:7b",
                "fallback_model": "gemma3:4b",
                "current_model": settings.local_model,
                "current_fallback_model": settings.local_fallback_model,
                "ollama_base_url": settings.ollama_base_url,
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "description": "Utilize OpenAI cloud API for reasoning. Requires an OpenAI API key.",
                "default_model": "gpt-4o-mini",
                "current_model": settings.openai_model,
                "has_key": bool(settings.openai_api_key),
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "description": "Utilize Google Gemini API for reasoning. Requires a Gemini API key.",
                "default_model": "gemini-1.5-flash",
                "current_model": settings.gemini_model,
                "has_key": bool(settings.gemini_api_key),
            }
        ],
        "active_provider": settings.llm_provider or "local"
    }


@app.get("/ai-providers/status")
def get_ai_providers_status():
    from app.providers.llm_ollama import OllamaLLMProvider
    from app.providers.llm_openai import OpenAILLMProvider
    from app.providers.llm_gemini import GeminiLLMProvider

    local_p = OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.local_model or "qwen2.5:7b",
    )
    openai_p = OpenAILLMProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_model or "gpt-4o-mini",
    )
    gemini_p = GeminiLLMProvider(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model or "gemini-1.5-flash",
    )

    return {
        "local": {
            "available": local_p.available,
            "models": local_p.list_models() if local_p.available else [],
        },
        "openai": {
            "available": openai_p.available,
            "models": openai_p.list_models(),
        },
        "gemini": {
            "available": gemini_p.available,
            "models": gemini_p.list_models(),
        }
    }


@app.post("/ai-providers/test/local")
def test_local_provider():
    from app.providers.llm_ollama import OllamaLLMProvider
    p = OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.local_model or "qwen2.5:7b",
        timeout_seconds=settings.ollama_timeout_seconds,
        retries=settings.ollama_retries,
        temperature=settings.ollama_temperature,
        num_predict=settings.ollama_num_predict,
        keep_alive=settings.ollama_keep_alive,
        system_prompt=settings.system_prompt,
    )
    return p.test_connection()


@app.post("/ai-providers/test/openai")
def test_openai_provider(payload: dict | None = None):
    from app.providers.llm_openai import OpenAILLMProvider
    api_key = payload.get("openai_api_key") if payload else None
    if api_key is None:
        api_key = settings.openai_api_key
    elif "..." in api_key:
        api_key = settings.openai_api_key

    model = payload.get("openai_model") if payload else None
    if not model:
        model = settings.openai_model or "gpt-4o-mini"

    p = OpenAILLMProvider(
        api_key=api_key,
        model=model,
        temperature=settings.cloud_temperature,
        max_tokens=settings.cloud_max_tokens,
        timeout_seconds=settings.cloud_timeout_seconds,
    )
    res = p.test_connection()
    if res.get("ok"):
        from app.services.environment import PROVIDER_TESTS
        PROVIDER_TESTS["openai"] = True
        if payload and api_key and "..." not in str(payload.get("openai_api_key", "")):
            settings.openai_api_key = api_key
            settings.openai_model = model
            settings._write_runtime_overrides()
    return res


@app.post("/ai-providers/test/gemini")
def test_gemini_provider(payload: dict | None = None):
    from app.providers.llm_gemini import GeminiLLMProvider
    api_key = payload.get("gemini_api_key") if payload else None
    if api_key is None:
        api_key = settings.gemini_api_key
    elif "..." in api_key:
        api_key = settings.gemini_api_key

    model = payload.get("gemini_model") if payload else None
    if not model:
        model = settings.gemini_model or "gemini-1.5-flash"

    p = GeminiLLMProvider(
        api_key=api_key,
        model=model,
        temperature=settings.cloud_temperature,
        max_tokens=settings.cloud_max_tokens,
        timeout_seconds=settings.cloud_timeout_seconds,
    )
    res = p.test_connection()
    if res.get("ok"):
        from app.services.environment import PROVIDER_TESTS
        PROVIDER_TESTS["gemini"] = True
        if payload and api_key and "..." not in str(payload.get("gemini_api_key", "")):
            settings.gemini_api_key = api_key
            settings.gemini_model = model
            settings._write_runtime_overrides()
    return res


@app.post("/ai-providers/test/elevenlabs")
def test_elevenlabs_provider_endpoint(payload: dict | None = None):
    return test_elevenlabs_provider(payload)


def test_elevenlabs_provider(payload: dict | None = None) -> dict:
    api_key = payload.get("elevenlabs_api_key") if payload else None
    if api_key is None:
        api_key = settings.elevenlabs_api_key
    elif "..." in api_key:
        api_key = settings.elevenlabs_api_key
        
    if not api_key:
        return {"ok": False, "detail": "ElevenLabs API key is required"}
        
    import urllib.request
    import urllib.error
    import time
    
    url = "https://api.elevenlabs.io/v1/voices"
    req = urllib.request.Request(url, headers={"xi-api-key": api_key}, method="GET")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        latency = (time.perf_counter() - start) * 1000
        
        # Save key if connection succeeded!
        if payload and api_key and "..." not in str(payload.get("elevenlabs_api_key", "")):
            settings.elevenlabs_api_key = api_key
            settings._write_runtime_overrides()
            
        return {"ok": True, "detail": "Successfully connected to ElevenLabs", "latency_ms": latency}
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="ignore")
        return {"ok": False, "detail": f"ElevenLabs API error ({exc.code}): {err_body}"}
    except Exception as exc:
        return {"ok": False, "detail": f"ElevenLabs connection failed: {exc}"}


@app.post("/ai-providers/test")
def test_provider_general(payload: dict):
    provider = payload.get("provider")
    if provider == "local":
        return test_local_provider()
    elif provider == "openai":
        return test_openai_provider(payload)
    elif provider == "gemini":
        return test_gemini_provider(payload)
    elif provider == "elevenlabs":
        return test_elevenlabs_provider(payload)
    else:
        return {"ok": False, "detail": f"Unknown provider: {provider}"}


@app.post("/settings/ai-provider", response_model=SettingsResponse)
def update_settings_ai_provider(payload: SettingsUpdateRequest) -> SettingsResponse:
    return update_settings(payload)


@app.post("/ai-providers/set-active")
def set_active_ai_provider(payload: dict):
    """Set the active LLM provider (local | openai | gemini)."""
    provider = payload.get("provider", "local")
    allowed = {"local", "openai", "gemini"}
    if provider not in allowed:
        return JSONResponse(status_code=400, content={"detail": f"Unknown provider: {provider}"})
    settings.llm_provider = provider
    settings._write_runtime_overrides()
    try:
        voice_studio_service.log_audit(
            user_id="system",
            event="active_provider_updated",
            details=f"Set active LLM provider to: {provider}"
        )
    except Exception:
        pass
    return {"ok": True, "active_provider": provider}


def _resolve_active_llm():
    """Build the active LLM provider instance from settings.llm_provider."""
    provider = settings.llm_provider or "local"
    if provider == "openai":
        from app.providers.llm_openai import OpenAILLMProvider
        return OpenAILLMProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model or "gpt-4o-mini",
            temperature=settings.cloud_temperature,
            max_tokens=settings.cloud_max_tokens,
            timeout_seconds=settings.cloud_timeout_seconds,
        )
    if provider == "gemini":
        from app.providers.llm_gemini import GeminiLLMProvider
        return GeminiLLMProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model or "gemini-1.5-flash",
            temperature=settings.cloud_temperature,
            max_tokens=settings.cloud_max_tokens,
            timeout_seconds=settings.cloud_timeout_seconds,
        )
    return OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.local_model or settings.ollama_model,
        # SQL generation feeds a large schema prompt; CPU-only models need
        # far more than the default chat timeout.
        timeout_seconds=max(settings.ollama_timeout_seconds, 180.0),
        retries=settings.ollama_retries,
        temperature=settings.ollama_temperature,
        num_predict=settings.ollama_num_predict,
        keep_alive=settings.ollama_keep_alive,
        system_prompt=settings.system_prompt,
    )


@app.get("/kyc/status")
def kyc_status():
    return kyc_service.status()


@app.post("/kyc/query")
def kyc_query(payload: dict):
    question = (payload.get("question") or "").strip()
    if not question:
        return JSONResponse(status_code=400, content={"ok": False, "detail": "question is required"})
    llm = _resolve_active_llm()
    result = kyc_service.ask(question, llm)
    if result.get("ok") and result.get("rows") is not None:
        try:
            import json as _json
            preview = _json.dumps(result["rows"][:10], default=str)
            summary = llm.chat(
                preview,
                system_prompt=(
                    "Summarize this banking query result in 1-3 sentences for a bank officer. "
                    f"Question: {question}"
                ),
            )
            result["answer"] = summary.text.strip()
        except Exception:
            pass
    return result


@app.post("/admin/login")
def admin_login(payload: dict):
    session = admin_auth.login(payload.get("username", ""), payload.get("password", ""))
    if session is None:
        return JSONResponse(status_code=401, content={"ok": False, "detail": "Invalid username or password"})
    return {"ok": True, **session}


@app.post("/admin/logout")
def admin_logout(payload: dict):
    admin_auth.logout(payload.get("token", ""))
    return {"ok": True}


@app.get("/admin/verify")
def admin_verify(token: str = ""):
    session = admin_auth.verify(token)
    if session is None:
        return {"ok": False}
    return {"ok": True, **session}


@app.get("/ai-providers/openai/models")
def get_openai_models_live():
    """Fetch available OpenAI models from the API (live)."""
    from app.providers.llm_openai import OpenAILLMProvider
    p = OpenAILLMProvider(api_key=settings.openai_api_key, model=settings.openai_model or "gpt-4o-mini")
    models = p.list_models_live()
    return {"ok": True, "models": models, "current": settings.openai_model}


@app.delete("/settings/openai-key", response_model=SettingsResponse)
def delete_openai_key() -> SettingsResponse:
    settings.openai_api_key = ""
    settings._write_runtime_overrides()
    try:
        voice_studio_service.log_audit(
            user_id="system",
            event="openai_key_deleted",
            details="OpenAI API key was deleted."
        )
    except Exception:
        pass
    from app.services.environment import PROVIDER_TESTS
    PROVIDER_TESTS["openai"] = False
    return SettingsResponse(settings=settings.public_dict())


@app.delete("/settings/gemini-key", response_model=SettingsResponse)
def delete_gemini_key() -> SettingsResponse:
    settings.gemini_api_key = ""
    settings._write_runtime_overrides()
    try:
        voice_studio_service.log_audit(
            user_id="system",
            event="gemini_key_deleted",
            details="Gemini API key was deleted."
        )
    except Exception:
        pass
    from app.services.environment import PROVIDER_TESTS
    PROVIDER_TESTS["gemini"] = False
    return SettingsResponse(settings=settings.public_dict())


@app.delete("/settings/elevenlabs-key", response_model=SettingsResponse)
def delete_elevenlabs_key() -> SettingsResponse:
    settings.elevenlabs_api_key = ""
    settings._write_runtime_overrides()
    try:
        voice_studio_service.log_audit(
            user_id="system",
            event="elevenlabs_key_deleted",
            details="ElevenLabs API key was deleted."
        )
    except Exception:
        pass
    return SettingsResponse(settings=settings.public_dict())


@app.get("/models/status", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    report = run_environment_checks(settings)
    providers = [
        ProviderStatus(
            name=check.name,
            ok=check.ok,
            detail=check.detail,
            critical=check.critical,
            fix=check.fix,
        )
        for check in report.checks
    ]
    providers.insert(0, ProviderStatus(name="python_backend", ok=True, detail="FastAPI is running", critical=True))
    return ModelStatusResponse(ready=report.ready, providers=providers)


@app.get("/doctor", response_model=DoctorResponse)
def doctor() -> DoctorResponse:
    report = run_environment_checks(settings)
    return DoctorResponse(ready=report.ready, checks=report.to_dict()["checks"])


@app.get("/ws/voice/status", response_model=VoiceSocketStatusResponse)
def voice_socket_status() -> VoiceSocketStatusResponse:
    return VoiceSocketStatusResponse(**build_voice_socket_status(settings))


@app.get("/voices", response_model=VoicesResponse)
def voices() -> VoicesResponse:
    selected = {
        "nepali": voice_registry.inspect(settings.piper_nepali_voice).to_dict(),
        "english": voice_registry.inspect(settings.piper_english_voice).to_dict(),
    }
    registry = voice_registry.list_voices([settings.piper_nepali_voice, settings.piper_english_voice])
    voice_list = [voice.to_dict() for voice in registry]
    
    # Add custom voices
    custom_voices = voice_studio_service.get_gallery_voices()
    for cv in custom_voices:
        if cv["publish_status"] == "published":
            voice_dir = voices_base_dir / cv["id"]
            if cv.get("engine") == "elevenlabs":
                model_exists = (voice_dir / "elevenlabs_id.txt").exists()
                missing = [] if model_exists else ["ElevenLabs cloned voice id missing"]
            elif cv.get("engine") == "chatterbox":
                model_exists = (voice_dir / "chatterbox_reference.wav").exists()
                missing = [] if model_exists else ["Chatterbox reference audio missing"]
            else:
                model_exists = bool(cv.get("model_exists"))
                missing = [] if model_exists else ["real custom Piper artifact missing; copied base models do not count"]
            voice_list.append({
                "id": cv["id"],
                "name": cv["name"],
                "owner_name": cv["owner_name"],
                "language": cv["language"],
                "engine": cv["engine"],
                "quality_score": cv["quality_score"],
                "status": cv["status"],
                "publish_status": cv["publish_status"],
                "consent_status": cv["consent_status"],
                "commercial_allowed": bool(cv["commercial_allowed"]),
                "model_exists": model_exists,
                "config_exists": model_exists,
                "missing_files": missing,
                "disabled_reason": None if model_exists else "This voice has recordings but no usable clone artifact yet.",
            })

    # Add OpenAI Cloud voices
    openai_voices = [
        {"id": "openai-alloy", "name": "Alloy", "language": "en", "engine": "openai"},
        {"id": "openai-echo", "name": "Echo", "language": "en", "engine": "openai"},
        {"id": "openai-fable", "name": "Fable", "language": "en", "engine": "openai"},
        {"id": "openai-onyx", "name": "Onyx", "language": "en", "engine": "openai"},
        {"id": "openai-nova", "name": "Nova", "language": "en", "engine": "openai"},
        {"id": "openai-shimmer", "name": "Shimmer", "language": "en", "engine": "openai"},
    ]
    openai_ready = bool(settings.openai_api_key)
    for ov in openai_voices:
        voice_list.append({
            "id": ov["id"],
            "name": f"OpenAI {ov['name']}",
            "owner_name": "OpenAI Cloud",
            "language": ov["language"],
            "engine": ov["engine"],
            "quality_score": 95.0,
            "status": "ready" if openai_ready else "missing_files",
            "publish_status": "published",
            "consent_status": "completed",
            "commercial_allowed": True,
            "model_exists": True,
            "config_exists": True,
            "missing_files": [] if openai_ready else ["OpenAI API key missing in settings"],
            "disabled_reason": None if openai_ready else "OpenAI API key is missing. Add it in Settings tab.",
        })
            
    return VoicesResponse(voices=voice_list, selected=selected)


@app.post("/voices/download")
def download_voices(background_tasks: BackgroundTasks):
    import subprocess
    import sys
    def run_download():
        subprocess.run([sys.executable, "scripts/download_piper_voices.py"])
    background_tasks.add_task(run_download)
    return {"status": "downloading", "detail": "Background download started"}


@app.get("/rag/status")
def rag_status():
    return rag_provider.status()


@app.get("/rag/collections")
def rag_collections():
    return rag_provider.collections()


@app.post("/rag/test")
def rag_test(payload: RagTestRequest):
    return rag_provider.test(payload.query, payload.collection_id)


@app.post("/rag/sync")
def rag_sync():
    return rag_provider.sync()


# ============================================================
# Local Knowledge Base (ChromaDB) routes  — /kb/*
# ============================================================

@app.get("/kb/status")
def kb_status():
    return kb_service.status()


@app.get("/kb/embedding/status")
def kb_embedding_status():
    return kb_service.embedding_provider.status()


@app.get("/kb/collections")
def kb_list_collections():
    cols = kb_service.list_collections()
    return {"collections": [vars(c) for c in cols]}


@app.post("/kb/collections")
def kb_create_collection(payload: dict):
    name = (payload.get("name") or "").strip()
    if not name:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Collection name is required.")
    description = payload.get("description", "")
    col = kb_service.create_collection(name, description)
    return vars(col)


@app.delete("/kb/collections/{collection_id}")
def kb_delete_collection(collection_id: str):
    ok = kb_service.delete_collection(collection_id)
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Collection not found.")
    return {"ok": True}


@app.get("/kb/collections/{collection_id}")
def kb_get_collection(collection_id: str):
    col = kb_service.get_collection(collection_id)
    if not col:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Collection not found.")
    return vars(col)


@app.get("/kb/collections/{collection_id}/documents")
def kb_list_documents(collection_id: str):
    docs = kb_service.list_documents(collection_id)
    return {"documents": [vars(d) for d in docs]}


@app.post("/kb/collections/{collection_id}/ingest")
async def kb_ingest_file(collection_id: str, file: UploadFile = File(...)):
    data = await file.read()
    try:
        doc = kb_service.ingest_file(collection_id, data, file.filename or "upload", file.content_type)
        return {"ok": True, "document": vars(doc)}
    except (ValueError, RuntimeError) as exc:
        return JSONResponse(status_code=400, content={"ok": False, "detail": str(exc)})


@app.post("/kb/collections/{collection_id}/ingest-url")
def kb_ingest_url(collection_id: str, payload: dict):
    url = (payload.get("url") or "").strip()
    if not url:
        return JSONResponse(status_code=400, content={"ok": False, "detail": "URL is required."})
    try:
        doc = kb_service.ingest_url(collection_id, url)
        return {"ok": True, "document": vars(doc)}
    except (ValueError, RuntimeError) as exc:
        return JSONResponse(status_code=400, content={"ok": False, "detail": str(exc)})


@app.post("/kb/collections/{collection_id}/crawl-site")
def kb_crawl_site(collection_id: str, payload: dict):
    """Stream a full-site crawl via Server-Sent Events (NDJSON lines).

    Body: { url, max_pages?, same_domain_only?, delay_ms? }
    Response: text/event-stream — each line is a JSON event.
    """
    from fastapi.responses import StreamingResponse

    url = (payload.get("url") or "").strip()
    if not url:
        return JSONResponse(status_code=400, content={"ok": False, "detail": "url is required"})

    max_pages = int(payload.get("max_pages", 500))
    same_domain_only = bool(payload.get("same_domain_only", True))
    delay_ms = int(payload.get("delay_ms", 150))
    render_js = bool(payload.get("render_js", True))

    def _stream():
        import json as _json
        try:
            for event in kb_service.crawl_site(
                collection_id=collection_id,
                start_url=url,
                max_pages=max_pages,
                same_domain_only=same_domain_only,
                delay_ms=delay_ms,
                render_js=render_js,
            ):
                yield f"data: {_json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'status': 'fatal', 'error': str(exc)})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.delete("/kb/collections/{collection_id}/documents/{doc_id}")
def kb_delete_document(collection_id: str, doc_id: str):
    ok = kb_service.delete_document(collection_id, doc_id)
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"ok": True}


@app.patch("/kb/collections/{collection_id}/documents/{doc_id}")
def kb_rename_document(collection_id: str, doc_id: str, payload: dict):
    """Rename a document (update filename in metadata)."""
    new_name = (payload.get("filename") or "").strip()
    if not new_name:
        return JSONResponse(status_code=400, content={"ok": False, "detail": "filename is required."})
    m = kb_service._meta.get(collection_id, {})
    if doc_id not in m.get("documents", {}):
        return JSONResponse(status_code=404, content={"ok": False, "detail": "Document not found."})
    kb_service._meta[collection_id]["documents"][doc_id]["filename"] = new_name
    kb_service._save_meta()
    return {"ok": True, "filename": new_name}


@app.get("/kb/collections/{collection_id}/documents/{doc_id}/chunks")
def kb_get_document_chunks(collection_id: str, doc_id: str, limit: int = 5):
    """Return chunks of a document (with page numbers) for preview, plus whether
    the original file is stored so the UI can offer an 'Open original file' link."""
    try:
        chunks = kb_service.get_document_chunks(collection_id, doc_id, limit=limit)
        raw_available = kb_service.get_raw_file_path(collection_id, doc_id) is not None
        return {"ok": True, "chunks": chunks, "raw_available": raw_available}
    except Exception as exc:
        return JSONResponse(status_code=400, content={"ok": False, "detail": str(exc)})


@app.get("/kb/collections/{collection_id}/documents/{doc_id}/file")
def kb_get_document_file(collection_id: str, doc_id: str):
    """Serve the original uploaded file (PDF/docx/txt) so the user can open the
    exact source. Only available for files uploaded after raw-file storage was
    added; older documents return 404 (their text is still viewable in-app)."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    path = kb_service.get_raw_file_path(collection_id, doc_id)
    if not path:
        raise HTTPException(status_code=404, detail="Original file not stored for this document.")
    return FileResponse(str(path), filename=path.name)


@app.post("/kb/query")
def kb_query(payload: dict):
    query_text = (payload.get("query") or "").strip()
    if not query_text:
        return JSONResponse(status_code=400, content={"ok": False, "detail": "query is required."})
    collection_ids = payload.get("collection_ids") or None
    n_results = int(payload.get("n_results") or kb_service.max_results)
    results = kb_service.query(query_text, collection_ids, n_results)
    return {
        "ok": True,
        "query": query_text,
        "results": [vars(r) for r in results],
        "context": kb_service._build_context_from_results(results),
    }


@app.post("/kb/query/advanced")
def kb_query_advanced(payload: dict):
    """Advanced RAG query: hybrid/semantic/keyword, reranking, metadata filters."""
    query_text = (payload.get("query") or "").strip()
    if not query_text:
        return JSONResponse(status_code=400, content={"ok": False, "detail": "query is required."})
    return kb_service.query_advanced(
        query_text=query_text,
        collection_ids=payload.get("collection_ids") or None,
        n_results=int(payload.get("n_results") or kb_service.max_results),
        mode=payload.get("mode") or kb_service.search_mode,
        source_type_filter=payload.get("source_type_filter") or None,
        rerank=bool(payload.get("rerank", kb_service.reranking_enabled)),
        min_score=float(payload["min_score"]) if payload.get("min_score") is not None else None,
        doc_id_filter=payload.get("doc_id_filter") or None,
    )


@app.post("/kb/reindex")
def kb_reindex_all():
    """Re-embed ALL collections with the current active embedding provider."""
    return kb_service.reindex_all()


@app.post("/kb/collections/{collection_id}/reindex")
def kb_reindex_collection(collection_id: str):
    """Re-embed a single collection with the current active embedding provider."""
    result = kb_service.reindex_collection(collection_id)
    if not result.get("ok"):
        return JSONResponse(status_code=400, content=result)
    return result


@app.get("/kb/analytics")
def kb_get_analytics(limit: int = 100):
    return kb_service.get_analytics(limit=limit)


@app.delete("/kb/analytics")
def kb_clear_analytics():
    kb_service.clear_analytics()
    return {"ok": True}


@app.get("/kb/collections/{collection_id}/stats")
def kb_collection_stats(collection_id: str):
    return kb_service.get_collection_stats(collection_id)


@app.get("/kb/collections/{collection_id}/export")
def kb_export_collection(collection_id: str):
    return kb_service.export_collection(collection_id)


@app.get("/web-retrieval/status")
def web_retrieval_status():
    return web_retrieval_provider.status().to_dict()


@app.post("/chat/test")
def chat_test(payload: ChatTestRequest):
    return conversation_service.handle_text(
        payload.text,
        voice_id=payload.voice_id,
        knowledge_id=payload.knowledge_id,
        use_internet=payload.use_internet,
        llm_provider_id=payload.llm_provider_id,
        stt_provider_id=payload.stt_provider_id,
        document_id=payload.document_id,
        temperature=payload.temperature,
    )


@app.post("/turn/speak")
def turn_speak(payload: dict):
    """On-demand TTS for a text answer (so chat text turns return instantly and
    audio is only synthesized when the user presses Play)."""
    from fastapi import HTTPException
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required.")
    voice_id = (payload.get("voice_id") or "").strip() or None
    parts = [TTSPart(text=chunk, language=lang) for chunk, lang in language_router.split_for_tts(text)]
    try:
        fallback_allowed = not getattr(settings, "force_selected_voice", False) and getattr(settings, "fallback_allowed", True)
        result = tts_provider.synthesize(parts, voice_id=voice_id, fallback_allowed=fallback_allowed)
    except _HeavyJobBusy:
        raise  # surfaced as a friendly 429 by the global handler
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {exc}")
    return {"ok": True, "audio_url": f"/audio/{result.audio_path.name}",
            "actual_voice_name": result.actual_voice_name, "engine": result.actual_tts_engine}


# ── RAG Evaluation: generate questions from a document, answer with one model,
#    verify the answers with a second model ──────────────────────────────────
def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction — tolerates qwen <think> blocks and code fences."""
    import re
    if not text:
        return {}
    # strip thinking traces / code fences
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def _eval_gen_provider(provider_id: str | None):
    pid = provider_id or ("openai" if settings.openai_api_key else "local")
    return conversation_service._instantiate_provider(pid), pid


def _normalized_question(text: str) -> str:
    import re
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _coerce_qas(text: str, limit: int, source_doc: str) -> list[dict]:
    payload = _extract_json(text)
    rows = payload.get("qas", []) if isinstance(payload, dict) else []
    out: list[dict] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        q = (item.get("q") or item.get("question") or "").strip()
        a = (item.get("a") or item.get("answer") or "").strip()
        if q and a:
            out.append({
                "q": q,
                "a": a,
                "source_doc": (item.get("source_doc") or source_doc or "").strip(),
                "dimension": (item.get("dimension") or "").strip(),
            })
        if len(out) >= limit:
            break
    return out


@app.post("/kb/eval/generate")
def kb_eval_generate(payload: dict):
    from fastapi import HTTPException
    import random
    import time
    cid = (payload.get("collection_id") or "").strip()
    doc_id = (payload.get("document_id") or "").strip() or None
    n = max(1, min(int(payload.get("n", 5)), 100))
    answer_style = (payload.get("answer_style") or "detailed").strip().lower()
    answer_instruction = (
        "Answers must be complete and useful: use 1-3 concise sentences when the fact needs context; "
        "short names/numbers are OK only when the question truly asks for a name, title, amount, date, phone, or limit."
        if answer_style != "short"
        else "Answers must be short and taken directly from the text."
    )
    if not cid:
        raise HTTPException(status_code=400, detail="collection_id is required.")

    docs = [d for d in kb_service.list_documents(cid) if (not doc_id or d.id == doc_id)]
    if not docs:
        raise HTTPException(status_code=400, detail="No documents found in this knowledge base scope.")

    # Spread generation across documents instead of feeding one large first-document-biased blob.
    rng = random.Random(f"{time.time_ns()}-{cid}-{doc_id or 'all'}")
    rng.shuffle(docs)
    doc_payloads: list[dict] = []
    docs_to_use = docs if doc_id else docs[:min(len(docs), max(1, n))]
    for d in docs_to_use:
        per_doc_limit = 80 if doc_id else max(4, min(12, 240 // max(1, len(docs_to_use))))
        chunks = kb_service.get_document_chunks(cid, d.id, limit=per_doc_limit)
        chunks = [c for c in chunks if len((c.get("text") or "").strip()) > 80]
        if not chunks:
            continue
        rng.shuffle(chunks)
        text = "\n\n".join(c["text"] for c in chunks[:8])[:5500]
        doc_payloads.append({"doc_id": d.id, "filename": d.filename, "text": text})

    if not doc_payloads:
        raise HTTPException(status_code=400, detail="No indexed text found to generate questions from.")

    total_docs = len(doc_payloads)
    base = n // total_docs
    extra = n % total_docs
    plan = []
    for idx, item in enumerate(doc_payloads):
        count = base + (1 if idx < extra else 0)
        if count > 0:
            plan.append((item, min(count, 10)))
    # If n is larger than docs*10, run additional rounds with the same docs.
    remaining = n - sum(count for _, count in plan)
    round_idx = 0
    while remaining > 0 and doc_payloads:
        item = doc_payloads[round_idx % len(doc_payloads)]
        count = min(10, remaining)
        plan.append((item, count))
        remaining -= count
        round_idx += 1

    if doc_id:
        # For a selected document, rotate snippets per batch so repeated generation is not identical.
        rng.shuffle(plan)

    provider, used = _eval_gen_provider(payload.get("gen_model"))
    qas: list[dict] = []
    seen: set[str] = set()
    last_err = None
    dimensions = (
        "Mix question dimensions across the available evidence: product/service definitions, eligibility, limits, fees/charges, steps/processes, documents required, security warnings, contacts/branches, roles/people, dates, exceptions, and policy conditions. "
        "Do not generate the same template repeatedly."
    )
    for source_doc, want in plan:
        if len(qas) >= n:
            break
        prompt = (
            f"From this Nabil Bank source document, write exactly {want} factual question/answer pairs. "
            f"Document name: {source_doc['filename']}. {dimensions} "
            "Every question must be self-contained and name its subject; never use only 'it', 'this', or 'the individual'. "
            f"{answer_instruction} "
            'Return ONLY JSON: {"qas":[{"q":"...","a":"...","source_doc":"...","dimension":"..."}]}.\n\n'
            f"TEXT:\n{source_doc['text']}"
        )
        token_budget = min(3500, want * 240 + 500)
        gen_opts = {"max_tokens": token_budget, "num_predict": token_budget, "temperature": 0.45}
        for attempt in range(3):
            try:
                result = provider.chat(prompt, system_prompt="You output only valid minified JSON. No prose.", options=gen_opts)
                batch = _coerce_qas(result.text, want, source_doc["filename"])
                added = 0
                for qa in batch:
                    key = _normalized_question(qa["q"])
                    if key in seen:
                        continue
                    seen.add(key)
                    qas.append(qa)
                    added += 1
                    if len(qas) >= n:
                        break
                if added:
                    break
            except Exception as exc:
                last_err = exc
                if attempt == 1:
                    prompt = prompt.replace(source_doc["text"], source_doc["text"][:2800])
    if not qas:
        detail = (f"Could not generate questions (the generation model returned no usable output"
                  + (f"; last error: {last_err}" if last_err else "")
                  + "). Try again, pick a specific document, or switch the answer model.")
        raise HTTPException(status_code=502, detail=detail)
    return {"ok": True, "count": len(qas), "questions": qas,
            "collection_id": cid, "document_id": doc_id, "gen_model": used,
            "documents_used": sorted({q.get("source_doc", "") for q in qas if q.get("source_doc")})}


@app.post("/kb/eval/run")
def kb_eval_run(payload: dict):
    from fastapi import HTTPException
    import time
    cid = (payload.get("collection_id") or "").strip()
    doc_id = (payload.get("document_id") or "").strip() or None
    questions = payload.get("questions") or []
    answer_model = payload.get("answer_model") or "local"
    verify_model = payload.get("verify_model") or ("openai" if settings.openai_api_key else "local")
    temperature = payload.get("temperature")
    voice_id = (payload.get("voice_id") or "").strip() or None
    if not cid or not questions:
        raise HTTPException(status_code=400, detail="collection_id and questions are required.")

    verifier = conversation_service._instantiate_provider(verify_model)
    rows = []
    counts = {"correct": 0, "partial": 0, "incorrect": 0, "error": 0}
    for item in questions[:100]:
        q = (item.get("q") or "").strip()
        ref = (item.get("a") or "").strip()
        if not q:
            continue
        t0 = time.time()
        try:
            ans_resp = conversation_service.handle_text(
                q, voice_id=voice_id, knowledge_id=cid, document_id=doc_id,
                llm_provider_id=answer_model, temperature=temperature,
                synthesize=bool(voice_id),
            )
            ans = ans_resp.response or ""
            rag_used = bool(getattr(ans_resp, "rag_used", False))
            audio_url = getattr(ans_resp, "audio_url", None)
        except Exception as exc:
            ans, rag_used, audio_url = f"(error: {exc})", False, None
        latency = round(time.time() - t0, 1)

        # Verify with the second model.
        try:
            jprompt = (
                f"QUESTION: {q}\nREFERENCE ANSWER (from the source document): {ref}\n"
                f"SYSTEM ANSWER: {ans[:1200]}\n\n"
                'Grade whether the SYSTEM ANSWER is factually correct and consistent with the reference. '
                '"correct" = right fact; "partial" = on-topic but incomplete; "incorrect" = wrong/refuses/empty. '
                'Also assign score 0-100 where 100 means fully correct and source-grounded. '
                'Return ONLY JSON: {"verdict":"correct|partial|incorrect","score":0-100,"reason":"<short>"}'
            )
            jres = verifier.chat(jprompt, system_prompt="You are a strict grader. Output only JSON.")
            jd = _extract_json(jres.text)
            verdict = jd.get("verdict", "incorrect")
            score = int(float(jd.get("score", 0)))
            reason = jd.get("reason", "")
        except Exception as exc:
            verdict, score, reason = "error", 0, str(exc)[:80]
        if verdict not in counts:
            verdict = "incorrect"
        score = max(0, min(100, score))
        counts[verdict] += 1
        rows.append({"question": q, "reference": ref, "answer": ans, "rag_used": rag_used,
                     "verdict": verdict, "score": score, "reason": reason, "latency_s": latency, "audio_url": audio_url,
                     "source_doc": item.get("source_doc"), "dimension": item.get("dimension")})

    n = len(rows) or 1
    accuracy = round(100 * (counts["correct"] + 0.5 * counts["partial"]) / n, 1)
    return {"ok": True, "answer_model": answer_model, "verify_model": verify_model,
            "total": len(rows), "accuracy": accuracy, "counts": counts, "results": rows}


def _eval_judge_prompt(q: str, ref: str, ans: str) -> str:
    return (
        f"QUESTION: {q}\nREFERENCE ANSWER (from the source document): {ref}\n"
        f"SYSTEM ANSWER: {ans[:1200]}\n\n"
        'Grade whether the SYSTEM ANSWER is factually correct and consistent with the reference. '
        '"correct" = right fact; "partial" = on-topic but incomplete; "incorrect" = wrong/refuses/empty. '
        'Also assign score 0-100 where 100 means fully correct and source-grounded. '
        'Return ONLY JSON: {"verdict":"correct|partial|incorrect","score":0-100,"reason":"<short>"}'
    )


@app.post("/kb/eval/run-stream")
def kb_eval_run_stream(payload: dict):
    """Same as /kb/eval/run but streams one result at a time (SSE) so the UI can
    animate progress and reveal each graded answer as it lands."""
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse
    import time, json as _json
    cid = (payload.get("collection_id") or "").strip()
    doc_id = (payload.get("document_id") or "").strip() or None
    questions = (payload.get("questions") or [])[:100]
    answer_model = payload.get("answer_model") or "local"
    verify_model = payload.get("verify_model") or ("openai" if settings.openai_api_key else "local")
    temperature = payload.get("temperature")
    voice_id = (payload.get("voice_id") or "").strip() or None
    if not cid or not questions:
        raise HTTPException(status_code=400, detail="collection_id and questions are required.")

    def gen():
        verifier = conversation_service._instantiate_provider(verify_model)
        counts = {"correct": 0, "partial": 0, "incorrect": 0, "error": 0}
        valid = [it for it in questions if (it.get("q") or "").strip()]
        total = len(valid)
        yield "data: " + _json.dumps({"type": "start", "total": total,
                                      "answer_model": answer_model, "verify_model": verify_model}) + "\n\n"
        for idx, item in enumerate(valid, start=1):
            q = (item.get("q") or "").strip()
            ref = (item.get("a") or "").strip()
            t0 = time.time()
            try:
                ans_resp = conversation_service.handle_text(
                    q, voice_id=voice_id, knowledge_id=cid, document_id=doc_id,
                    llm_provider_id=answer_model, temperature=temperature,
                    synthesize=bool(voice_id),
                )
                ans = ans_resp.response or ""
                rag_used = bool(getattr(ans_resp, "rag_used", False))
                audio_url = getattr(ans_resp, "audio_url", None)
            except Exception as exc:
                ans, rag_used, audio_url = f"(error: {exc})", False, None
            latency = round(time.time() - t0, 1)
            try:
                jres = verifier.chat(_eval_judge_prompt(q, ref, ans),
                                     system_prompt="You are a strict grader. Output only JSON.")
                jd = _extract_json(jres.text)
                verdict = jd.get("verdict", "incorrect")
                score = int(float(jd.get("score", 0)))
                reason = jd.get("reason", "")
            except Exception as exc:
                verdict, score, reason = "error", 0, str(exc)[:80]
            if verdict not in counts:
                verdict = "incorrect"
            score = max(0, min(100, score))
            counts[verdict] += 1
            done = sum(counts.values())
            accuracy = round(100 * (counts["correct"] + 0.5 * counts["partial"]) / done, 1) if done else 0
            row = {"question": q, "reference": ref, "answer": ans, "rag_used": rag_used,
                   "verdict": verdict, "score": score, "reason": reason, "latency_s": latency, "audio_url": audio_url,
                   "source_doc": item.get("source_doc"), "dimension": item.get("dimension")}
            yield "data: " + _json.dumps({"type": "result", "index": idx, "total": total,
                                          "result": row, "counts": counts, "accuracy": accuracy}) + "\n\n"
        done = sum(counts.values()) or 1
        accuracy = round(100 * (counts["correct"] + 0.5 * counts["partial"]) / done, 1)
        yield "data: " + _json.dumps({"type": "done", "accuracy": accuracy, "counts": counts,
                                      "answer_model": answer_model, "verify_model": verify_model}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/kb/eval/correct")
def kb_eval_correct(payload: dict):
    """Fix an incorrect answer by injecting a curated Q&A 'golden' fact into the
    knowledge base. This is the trusted RAG remediation pattern: the corrected
    pair becomes a high-signal chunk the retriever surfaces for similar questions."""
    from fastapi import HTTPException
    import time
    cid = (payload.get("collection_id") or "").strip()
    question = (payload.get("question") or "").strip()
    answer = (payload.get("answer") or "").strip()
    if not cid or not question or not answer:
        raise HTTPException(status_code=400, detail="collection_id, question, and answer are required.")
    text = (
        "Verified Q&A (curated correction).\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
    )
    filename = f"correction-{int(time.time())}.txt"
    try:
        doc = kb_service.ingest_file(cid, text.encode("utf-8"), filename, "text/plain")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not save correction to the knowledge base: {exc}")
    return {"ok": True, "document_id": getattr(doc, "id", None), "filename": filename}


@app.post("/kb/eval/correct-bulk")
def kb_eval_correct_bulk(payload: dict):
    """Save multiple verified evaluation corrections as one high-signal document."""
    from fastapi import HTTPException
    import time
    cid = (payload.get("collection_id") or "").strip()
    rows = payload.get("items") or []
    if not cid or not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="collection_id and non-empty items are required.")
    cleaned = []
    for item in rows[:100]:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        source = (item.get("source_doc") or "").strip()
        verdict = (item.get("verdict") or "").strip()
        if q and a:
            cleaned.append((q, a, source, verdict))
    if not cleaned:
        raise HTTPException(status_code=400, detail="No usable correction rows were provided.")
    parts = ["Verified Q&A correction pack (curated from RAG evaluation)."]
    for idx, (q, a, source, verdict) in enumerate(cleaned, start=1):
        parts.append(
            f"\nCorrection {idx}\n"
            f"Source document: {source or 'not specified'}\n"
            f"Previous verdict: {verdict or 'needs_fix'}\n"
            f"Question: {q}\n"
            f"Verified answer: {a}\n"
        )
    filename = f"correction-pack-{int(time.time())}.txt"
    try:
        doc = kb_service.ingest_file(cid, "\n".join(parts).encode("utf-8"), filename, "text/plain")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not save corrections to the knowledge base: {exc}")
    return {"ok": True, "count": len(cleaned), "document_id": getattr(doc, "id", None), "filename": filename}


@app.get("/chat/history")
def get_chat_history(limit: int = 50):
    from app.database import get_db_connection
    import json
    conn = get_db_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM chat_turns
            ORDER BY timestamp DESC
            LIMIT ?;
            """,
            (limit,)
        ).fetchall()
        
        history = []
        for r in rows:
            try:
                tts_route = json.loads(r["tts_route"]) if r["tts_route"] else []
            except Exception:
                tts_route = []
                
            try:
                timings = json.loads(r["timings"]) if r["timings"] else {}
            except Exception:
                timings = {}
                
            try:
                citations = json.loads(r["citations"]) if r["citations"] else []
            except Exception:
                citations = []
                
            ratings = {}
            if r["rating_naturalness"] is not None:
                ratings["naturalness"] = r["rating_naturalness"]
            if r["rating_voice_similarity"] is not None:
                ratings["voiceSimilarity"] = r["rating_voice_similarity"]
            if r["rating_nepali_pronunciation"] is not None:
                ratings["nepaliPronunciation"] = r["rating_nepali_pronunciation"]
            if r["rating_english_pronunciation"] is not None:
                ratings["englishPronunciation"] = r["rating_english_pronunciation"]

            history.append({
                "id": r["id"],
                "transcript": r["transcript"],
                "response": r["response"],
                "input_language": r["input_language"],
                "response_language": r["response_language"],
                "audio_url": r["audio_url"],
                "tts_route": tts_route,
                "timings": timings,
                "rag_used": bool(r["rag_used"]),
                "rag_collection_id": r["rag_collection_id"],
                "rag_fallback_used": bool(r["rag_fallback_used"]),
                "internet_used": bool(r["internet_used"]),
                "citations": citations,
                "voice_id": r["voice_id"],
                "requested_voice_id": r["requested_voice_id"],
                "requested_voice_name": r["requested_voice_name"],
                "actual_voice_id": r["actual_voice_id"],
                "actual_voice_name": r["actual_voice_name"],
                "actual_engine": r["actual_engine"],
                "actual_model_path": r["actual_model_path"],
                "fallback_used": bool(r["fallback_used"]),
                "fallback_reason": r["fallback_reason"],
                "llm_provider": r["llm_provider"],
                "rag_path": r["rag_path"],
                "ratings": ratings,
                "created_at": r["timestamp"]
            })
        return history
    finally:
        conn.close()


@app.post("/chat/turns/{turn_id}/rate")
def rate_chat_turn(turn_id: str, ratings: RatingRequest):
    from app.database import get_db_connection
    from fastapi import HTTPException
    conn = get_db_connection()
    try:
        turn = conn.execute("SELECT id FROM chat_turns WHERE id = ?;", (turn_id,)).fetchone()
        if not turn:
            raise HTTPException(status_code=404, detail="Chat turn not found")
        
        conn.execute(
            """
            UPDATE chat_turns
            SET rating_naturalness = ?,
                rating_voice_similarity = ?,
                rating_nepali_pronunciation = ?,
                rating_english_pronunciation = ?
            WHERE id = ?;
            """,
            (
                ratings.naturalness,
                ratings.voice_similarity,
                ratings.nepali_pronunciation,
                ratings.english_pronunciation,
                turn_id
            )
        )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.post("/tts/test", response_model=TtsTestResponse)
def tts_test(payload: TtsTestRequest) -> TtsTestResponse:
    language = payload.language or language_router.detect(payload.text).language
    if language in {"mixed", "unknown"}:
        parts = [TTSPart(text=chunk, language=lang) for chunk, lang in language_router.split_for_tts(payload.text)]
    else:
        parts = [TTSPart(text=payload.text, language=language)]
    result = tts_provider.synthesize(parts)
    return TtsTestResponse(language=language, audio_url=f"/audio/{result.audio_path.name}")


class TtsPreviewRequest(BaseModel):
    text: str
    voice_id: str = "openai-alloy"
    language: str = "en"


@app.post("/tts/preview")
def tts_preview(payload: TtsPreviewRequest):
    """Preview a specific voice (Piper built-in or OpenAI cloud voice)."""
    parts = [TTSPart(text=payload.text, language=payload.language)]
    try:
        result = tts_provider.synthesize(parts, voice_id=payload.voice_id, fallback_allowed=False)
        return {"ok": True, "audio_url": f"/audio/{result.audio_path.name}", "voice_id": payload.voice_id, "engine": result.actual_tts_engine}
    except Exception as exc:
        return JSONResponse(status_code=400, content={"ok": False, "detail": str(exc)})


@app.post("/stt/test")
async def stt_test(upload: UploadFile = File(...)):
    audio_path = audio_pipeline.prepare_stt_audio(await upload.read(), upload.content_type)
    try:
        result = stt_provider.transcribe_file(audio_path)
        decision = language_router.detect(result.text, result.language, result.confidence)
        return {
            "text": result.text,
            "language": decision.language,
            "whisper_language": result.language,
            "confidence": result.confidence,
            "duration_ms": result.duration_ms,
        }
    finally:
        audio_pipeline.cleanup_turn_audio(audio_path)


@app.delete("/local-data")
def delete_local_data():
    deleted: list[str] = []
    dataset_wav = settings.audio_work_dir.parent / "dataset" / "wav"
    for directory in (settings.piper_audio_cache_dir, settings.audio_work_dir, dataset_wav):
        resolved = directory.resolve()
        if ".local" not in resolved.parts:
            continue
        for path in directory.glob("*"):
            if path.is_file():
                path.unlink()
                deleted.append(str(path))
    metadata_path = settings.audio_work_dir.parent / "dataset" / "metadata.csv"
    if metadata_path.exists():
        metadata_path.unlink()
        deleted.append(str(metadata_path))
        
    # Clear SQLite database tables for chat turns & events
    from app.database import get_db_connection
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM chat_turns;")
        conn.execute("DELETE FROM voice_usage_events;")
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()
        
    return {"deleted": deleted}


@app.get("/dataset/prompts")
def get_dataset_prompts():
    return dataset_service.list_prompts()


@app.get("/dataset/recordings", response_model=list[DatasetRecordingResponse])
def get_dataset_recordings() -> list[DatasetRecordingResponse]:
    return [
        DatasetRecordingResponse(
            id=rec.id,
            text=rec.text,
            language=rec.language,
            exists=rec.exists,
            audio_url=rec.audio_url,
            quality=DatasetQualityResponse(
                score=rec.quality.score,
                verdict=rec.quality.verdict,
                duration_seconds=rec.quality.duration_seconds,
                peak=rec.quality.peak,
                rms=rec.quality.rms,
                reason=rec.quality.reason,
            ) if rec.quality else None
        )
        for rec in dataset_service.list_recordings()
    ]


@app.post("/dataset/recordings/{prompt_id}", response_model=DatasetRecordingResponse)
async def upload_dataset_recording(prompt_id: str, upload: UploadFile = File(...)) -> DatasetRecordingResponse:
    rec = dataset_service.save_recording(prompt_id, await upload.read(), upload.content_type)
    return DatasetRecordingResponse(
        id=rec.id,
        text=rec.text,
        language=rec.language,
        exists=rec.exists,
        audio_url=rec.audio_url,
        quality=DatasetQualityResponse(
            score=rec.quality.score,
            verdict=rec.quality.verdict,
            duration_seconds=rec.quality.duration_seconds,
            peak=rec.quality.peak,
            rms=rec.quality.rms,
            reason=rec.quality.reason,
        ) if rec.quality else None
    )


@app.delete("/dataset/recordings/{prompt_id}")
def delete_dataset_recording(prompt_id: str):
    dataset_service.delete_recording(prompt_id)
    return {"ok": True}


@app.get("/dataset/export")
def export_dataset():
    zip_data = dataset_service.get_zip_bytes()
    return Response(
        content=zip_data,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=voice_dataset.zip"},
    )


@app.post("/voice/turn")
async def chat_voice(
    audio: UploadFile = File(...),
    voice_id: str = Form(default=""),
    knowledge_id: str = Form(default=""),
    use_internet: bool = Form(default=False),
    llm_provider_id: str = Form(default=""),
    stt_provider_id: str = Form(default=""),
    stt_language: str = Form(default=""),
):
    """REST fallback for voice turns — accepts audio upload, returns full turn JSON."""
    import asyncio
    audio_bytes = await audio.read()
    audio_path = audio_pipeline.prepare_stt_audio(audio_bytes, audio.content_type)
    try:
        turn = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: conversation_service.handle_audio(
                audio_path,
                voice_id=voice_id or None,
                knowledge_id=knowledge_id or None,
                use_internet=use_internet,
                llm_provider_id=llm_provider_id or None,
                stt_provider_id=stt_provider_id or None,
                stt_language=stt_language or None,
            ),
        )
        return turn
    finally:
        audio_pipeline.cleanup_turn_audio(audio_path)



def _classify_transcript_language(text: str) -> str:
    """Lightweight language tag for live partials: ne | en | mixed | unknown."""
    try:
        return language_router.detect(text).language
    except Exception:
        return "unknown"


def _transcribe_blob_sync(
    audio_b64: str,
    mime_type: str | None,
    stt_provider_id: str | None,
    stt_language: str | None,
):
    """Decode a base64 audio blob to a temp wav and run the active STT router.

    Returns the STTResult. Caller handles cleanup. Used for BOTH live partials
    (transcribe-only) and the authoritative final transcription. STT provider
    selection follows the same OpenAI-first / local-fallback routing as REST.
    """
    audio_bytes = base64.b64decode(audio_b64, validate=True)
    if not audio_bytes:
        raise ValueError("audio data is empty")
    audio_path = audio_pipeline.prepare_stt_audio(audio_bytes, mime_type)
    try:
        return stt_provider.transcribe_file(
            audio_path,
            provider_name=stt_provider_id or None,
            language=stt_language or None,
        ), audio_path
    except Exception:
        audio_pipeline.cleanup_turn_audio(audio_path)
        raise


@app.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket):
    import asyncio
    await websocket.accept()
    socket_status = build_voice_socket_status(settings)
    hello_received = False
    session_voice_id = None
    session_knowledge_id = None
    session_use_internet = False
    session_llm_provider_id = None
    session_stt_provider_id = None
    session_stt_language = None
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")
            if message_type == "ping":
                await websocket.send_json({"type": "pong", "ts": message.get("ts")})
                continue
            if message_type == "hello":
                session_voice_id = message.get("selected_voice_id")
                session_knowledge_id = message.get("selected_knowledge_id")
                session_use_internet = bool(message.get("use_internet", False))
                session_llm_provider_id = message.get("selected_brain")
                session_stt_provider_id = message.get("selected_stt_provider")
                session_stt_language = message.get("selected_stt_language")
                socket_status = build_voice_socket_status(settings)
                hello_received = True
                await websocket.send_json(
                    {
                        "type": "ready",
                        "session_id": socket_status["session_id"],
                        "capabilities": socket_status["capabilities"],
                        "blocking_reasons": socket_status["blocking_reasons"],
                        "warnings": socket_status["warnings"],
                    }
                )
                continue
            if message_type == "config":
                # Real-time contract: set session params for the next turn.
                if message.get("voice_id") is not None:
                    session_voice_id = message.get("voice_id")
                if message.get("knowledge_id") is not None:
                    session_knowledge_id = message.get("knowledge_id")
                if message.get("llm_provider") is not None:
                    session_llm_provider_id = message.get("llm_provider")
                if message.get("stt_language") is not None:
                    lang = message.get("stt_language")
                    session_stt_language = None if lang in ("auto", "", None) else lang
                if message.get("use_internet") is not None:
                    session_use_internet = bool(message.get("use_internet"))
                hello_received = True
                await websocket.send_json({"type": "config_ack"})
                continue
            if message_type == "audio_partial":
                # Live partial: transcribe ONLY (no LLM) so the display stays fast.
                try:
                    stt_result, audio_path = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: _transcribe_blob_sync(
                            str(message.get("data", "")),
                            message.get("mime"),
                            session_stt_provider_id,
                            session_stt_language,
                        ),
                    )
                except (binascii.Error, ValueError) as exc:
                    await websocket.send_json({"type": "error", "detail": f"Invalid audio_partial: {exc}"})
                    continue
                except Exception as exc:
                    await websocket.send_json({"type": "error", "detail": f"Partial transcription failed: {exc}"})
                    continue
                try:
                    text = (stt_result.text or "").strip()
                    await websocket.send_json(
                        {
                            "type": "partial_transcript",
                            "text": text,
                            "language": _classify_transcript_language(text) if text else "unknown",
                        }
                    )
                finally:
                    audio_pipeline.cleanup_turn_audio(audio_path)
                continue
            if message_type == "audio_end":
                # Authoritative final transcription, then full RAG -> LLM -> TTS turn.
                if not socket_status["capabilities"].get("audio_turns"):
                    await websocket.send_json(
                        {
                            "type": "setup_required",
                            "detail": "Voice turns are blocked by runtime setup.",
                            "blocking_reasons": socket_status["blocking_reasons"],
                        }
                    )
                    continue
                await websocket.send_json({"type": "status", "status": "transcribing"})
                try:
                    audio_bytes = base64.b64decode(str(message.get("data", "")), validate=True)
                except (binascii.Error, ValueError):
                    await websocket.send_json({"type": "error", "detail": "Invalid audio_end: data must be valid base64."})
                    continue
                if not audio_bytes:
                    await websocket.send_json({"type": "error", "detail": "Invalid audio_end: data is empty."})
                    continue
                audio_path = audio_pipeline.prepare_stt_audio(audio_bytes, message.get("mime"))
                try:
                    await websocket.send_json({"type": "status", "status": "thinking"})
                    turn = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: conversation_service.handle_audio(
                            audio_path,
                            voice_id=session_voice_id,
                            knowledge_id=session_knowledge_id,
                            use_internet=session_use_internet,
                            llm_provider_id=session_llm_provider_id,
                            session_id=socket_status.get("session_id"),
                            stt_provider_id=session_stt_provider_id,
                            stt_language=session_stt_language,
                        ),
                    )
                    turn_payload = turn.model_dump()
                    await websocket.send_json(
                        {
                            "type": "final_transcript",
                            "text": turn_payload.get("transcript", ""),
                            "language": turn_payload.get("input_language", "unknown"),
                        }
                    )
                    await websocket.send_json({"type": "status", "status": "speaking"})
                    await websocket.send_json({"type": "answer", "turn": turn_payload})
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    await websocket.send_json({"type": "status", "status": "error"})
                finally:
                    audio_pipeline.cleanup_turn_audio(audio_path)
                continue
            if not hello_received:
                await websocket.send_json(
                    {
                        "type": "ready",
                        "session_id": socket_status["session_id"],
                        "capabilities": socket_status["capabilities"],
                        "blocking_reasons": socket_status["blocking_reasons"],
                        "warnings": socket_status["warnings"],
                    }
                )
                hello_received = True
            await websocket.send_json({"type": "status", "status": "thinking"})
            try:
                if message_type == "text":
                    if not socket_status["capabilities"].get("text_turns"):
                        await websocket.send_json(
                            {
                                "type": "setup_required",
                                "detail": "Text turns are blocked by runtime setup.",
                                "blocking_reasons": socket_status["blocking_reasons"],
                            }
                        )
                        continue
                    # Run blocking I/O in thread pool to avoid blocking the event loop
                    turn = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: conversation_service.handle_text(
                            str(message.get("text", "")),
                            voice_id=session_voice_id,
                            knowledge_id=session_knowledge_id,
                            use_internet=session_use_internet,
                            llm_provider_id=session_llm_provider_id,
                            session_id=socket_status.get("session_id"),
                            stt_provider_id=session_stt_provider_id,
                            synthesize=False,  # text returns instantly; audio is on-demand via /turn/speak
                        ),
                    )
                elif message_type == "audio":
                    if not socket_status["capabilities"].get("audio_turns"):
                        await websocket.send_json(
                            {
                                "type": "setup_required",
                                "detail": "Voice turns are blocked by runtime setup.",
                                "blocking_reasons": socket_status["blocking_reasons"],
                            }
                        )
                        continue
                    await websocket.send_json({"type": "status", "status": "transcribing"})
                    try:
                        audio_bytes = base64.b64decode(str(message.get("audioBase64", "")), validate=True)
                    except (binascii.Error, ValueError):
                        await websocket.send_json({"type": "error", "detail": "Invalid audio message: audioBase64 must be valid base64."})
                        continue
                    if not audio_bytes:
                        await websocket.send_json({"type": "error", "detail": "Invalid audio message: audioBase64 is empty."})
                        continue
                    audio_path = audio_pipeline.prepare_stt_audio(audio_bytes, message.get("mimeType"))
                    try:
                        # Run blocking STT+LLM+TTS in thread pool
                        turn = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: conversation_service.handle_audio(
                                audio_path,
                                voice_id=session_voice_id,
                                knowledge_id=session_knowledge_id,
                                use_internet=session_use_internet,
                                llm_provider_id=session_llm_provider_id,
                                session_id=socket_status.get("session_id"),
                                stt_provider_id=session_stt_provider_id,
                                stt_language=session_stt_language,
                            ),
                        )

                    finally:
                        audio_pipeline.cleanup_turn_audio(audio_path)

                else:
                    await websocket.send_json({"type": "error", "detail": f"Unknown message type: {message_type}"})
                    continue
                await websocket.send_json({"type": "status", "status": "speaking"})
                await websocket.send_json({"type": "turn", **turn.model_dump()})
            except Exception as exc:
                import traceback
                traceback.print_exc()
                await websocket.send_json({"type": "error", "detail": str(exc)})
                await websocket.send_json({"type": "status", "status": "error"})
    except WebSocketDisconnect:
        return


# Voice Studio endpoints
from app.schemas import VoiceCreateRequest, VoiceConsentRequest, VoiceStudioRecordingResponse

@app.get("/voices/gallery")
def get_voices_gallery():
    return voice_studio_service.get_gallery_voices()

@app.post("/voices/create")
def create_voice(payload: VoiceCreateRequest):
    return voice_studio_service.create_voice(
        name=payload.name,
        owner_name=payload.owner_name,
        owner_email=payload.owner_email,
        organization=payload.organization,
        language=payload.language,
        engine=payload.engine,
        commercial_allowed=payload.commercial_allowed
    )

@app.post("/voices/{voice_id}/consent")
async def save_voice_consent(voice_id: str, signature: str = Form(...), spoken_consent: UploadFile = File(None)):
    spoken_bytes = await spoken_consent.read() if spoken_consent else None
    mime_type = spoken_consent.content_type if spoken_consent else None
    voice_studio_service.save_consent(voice_id, signature, spoken_bytes, mime_type)
    return {"ok": True}

@app.get("/voices/{voice_id}/recordings")
def get_voice_recordings(voice_id: str):
    samples = voice_studio_service.get_voice_samples(voice_id)
    sample_map = {s["prompt_id"]: s for s in samples}
    
    records = []
    for prompt in voice_studio_service.list_prompts():
        prompt_id = prompt["id"]
        sample = sample_map.get(prompt_id)
        exists = sample is not None
        audio_url = f"/audio/voices/{voice_id}/normalized/{prompt_id}.wav" if exists else None
        
        quality = None
        if exists:
            quality = {
                "score": sample["score"],
                "verdict": sample["status"],
                "duration_seconds": 0.0,
                "peak": 0.0,
                "rms": 0.0,
                "reason": sample["reason"]
            }
        records.append({
            "id": prompt_id,
            "text": prompt["text"],
            "language": prompt["language"],
            "exists": exists,
            "audio_url": audio_url,
            "quality": quality
        })
    return records

@app.post("/voices/{voice_id}/recordings/{prompt_id}", response_model=VoiceStudioRecordingResponse)
async def upload_voice_recording(voice_id: str, prompt_id: str, upload: UploadFile = File(...)) -> VoiceStudioRecordingResponse:
    rec = voice_studio_service.save_sample(voice_id, prompt_id, await upload.read(), upload.content_type)
    
    # Verify speaker match
    voice_dir = voices_base_dir / voice_id
    wav_path = voice_dir / "normalized" / f"{prompt_id}.wav"
    
    verdict = rec.verdict
    score = rec.score
    reason = rec.reason
    
    if wav_path.exists():
        match_res = speaker_isolation_service.verify_speaker_match(voice_id, wav_path, lock_if_empty=True)
        if not match_res["matched"]:
            # Downgrade score but only hard-reject if truly a different speaker (similarity very low)
            score_penalty = 20 if match_res["score"] < 50 else 10
            score = max(0, score - score_penalty)
            reason = f"{reason}, {match_res['reason']}" if reason and reason != "clean" else match_res["reason"]
            # Hard reject only if clearly a different person
            if match_res["score"] < 40:
                verdict = "reject"
                score = min(score, 30)
            from app.database import get_db_connection
            conn = get_db_connection()
            try:
                conn.execute(
                    "UPDATE voice_samples SET status = ?, score = ?, reason = ? WHERE voice_id = ? AND prompt_id = ?;",
                    (verdict, score, reason, voice_id, prompt_id)
                )
                conn.commit()
            finally:
                conn.close()

    return VoiceStudioRecordingResponse(
        id=rec.id,
        prompt_id=rec.prompt_id,
        text=rec.text,
        language=rec.language,
        exists=rec.exists,
        audio_url=rec.audio_url,
        score=score,
        verdict=verdict,
        peak=rec.peak,
        rms=rec.rms,
        reason=reason
    )

@app.delete("/voices/{voice_id}/recordings/{prompt_id}")
def delete_voice_recording(voice_id: str, prompt_id: str):
    voice_studio_service.delete_sample(voice_id, prompt_id)
    return {"ok": True}

@app.post("/voices/{voice_id}/publish")
def publish_voice(voice_id: str):
    from fastapi import HTTPException
    try:
        return voice_studio_service.publish_voice(voice_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Publishing failed: {exc}")

@app.post("/voices/{voice_id}/clone")
def clone_voice(voice_id: str):
    """
    Triggers the voice cloning/training process. Prepares dataset, compiles/exports model artifacts,
    and updates the training job status.
    """
    from app.database import get_db_connection
    import shutil
    import uuid
    import time
    from fastapi import HTTPException
    from app.services.heavy_jobs import VOICE_BUILD

    # Reject a second build immediately (HTTP 429) if one is already running.
    VOICE_BUILD.__enter__()
    conn = get_db_connection()
    try:
        voice = conn.execute("SELECT * FROM voices WHERE id = ?;", (voice_id,)).fetchone()
        if not voice:
            raise HTTPException(status_code=404, detail="Voice not found")
        
        # Verify consent is signed
        if voice["consent_status"] != "completed":
            raise HTTPException(status_code=400, detail="Voice consent must be signed before cloning.")
            
        # Verify recordings count
        samples = conn.execute("SELECT * FROM voice_samples WHERE voice_id = ? AND status != 'reject';", (voice_id,)).fetchall()
        if len(samples) < 3:
            raise HTTPException(status_code=400, detail="At least 3 clean recordings are required to start cloning.")
            
        # Create a training job entry
        job_id = str(uuid.uuid4())
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn.execute(
            "INSERT INTO voice_training_jobs (id, voice_id, status, progress, timestamp) VALUES (?, ?, ?, ?, ?);",
            (job_id, voice_id, "running", 0.0, timestamp)
        )
        conn.commit()
        
        # Define destination paths
        voice_dir = voices_base_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        lang = voice["language"]
        
        if voice["engine"] == "elevenlabs":
            api_key = settings.elevenlabs_api_key
            if not api_key:
                raise HTTPException(status_code=400, detail="ElevenLabs API key is missing. Please configure it in settings.")
            
            files_to_upload = []
            for sample in samples:
                wav_path = Path(sample["wav_path"])
                if wav_path.exists():
                    files_to_upload.append(("files", wav_path.name, wav_path.read_bytes()))
            
            if not files_to_upload:
                raise HTTPException(status_code=400, detail="No recording files found on disk to clone.")
                
            def send_multipart_request(url: str, fields: dict[str, str], files: list[tuple[str, str, bytes]], headers: dict[str, str]) -> bytes:
                import urllib.request
                import uuid
                boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
                body = bytearray()
                for name, value in fields.items():
                    body.extend(f"--{boundary}\r\n".encode("utf-8"))
                    body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
                    body.extend(f"{value}\r\n".encode("utf-8"))
                for field_name, file_name, file_bytes in files:
                    body.extend(f"--{boundary}\r\n".encode("utf-8"))
                    body.extend(f'Content-Disposition: form-data; name="{field_name}"; filename="{file_name}"\r\n'.encode("utf-8"))
                    body.extend(b"Content-Type: audio/wav\r\n\r\n")
                    body.extend(file_bytes)
                    body.extend(b"\r\n")
                body.extend(f"--{boundary}--\r\n".encode("utf-8"))
                
                req_headers = headers.copy()
                req_headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
                req_headers["Content-Length"] = str(len(body))
                req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
                with urllib.request.urlopen(req, timeout=90) as resp:
                    return resp.read()
                    
            try:
                import json
                fields = {
                    "name": voice["name"],
                    "description": f"SwarLocal cloned voice for {voice['name']}"
                }
                headers = {"xi-api-key": api_key}
                resp_bytes = send_multipart_request("https://api.elevenlabs.io/v1/voices/add", fields, files_to_upload, headers)
                resp_json = json.loads(resp_bytes.decode("utf-8"))
                el_voice_id = resp_json.get("voice_id")
                if not el_voice_id:
                    raise ValueError("Failed to retrieve voice_id from ElevenLabs response.")
                    
                (voice_dir / "elevenlabs_id.txt").write_text(el_voice_id, encoding="utf-8")
                onnx_path = f"elevenlabs://{el_voice_id}"
                config_path = ""
            except Exception as e:
                conn.execute(
                    "UPDATE voice_training_jobs SET status = ? WHERE id = ?;",
                    ("failed", job_id)
                )
                conn.commit()
                raise HTTPException(status_code=500, detail=f"ElevenLabs cloning failed: {e}")
        elif voice["engine"] == "chatterbox":
            try:
                clone_result = voice_clone_service.create_chatterbox_reference(voice_id, samples, lang)
            except VoiceCloneError as exc:
                conn.execute(
                    "UPDATE voice_training_jobs SET status = ? WHERE id = ?;",
                    ("failed", job_id)
                )
                conn.commit()
                raise HTTPException(status_code=400, detail=str(exc))
            onnx_path = clone_result["artifact_uri"]
            config_path = clone_result["config_path"]
        elif voice["engine"] == "piper":
            try:
                prompt_lookup = {p["id"]: p["text"] for p in voice_studio_service.list_prompts()}
                dataset_result = voice_clone_service.prepare_piper_dataset(voice_id, samples, prompt_lookup)
                clone_result = voice_clone_service.run_piper_training_command(
                    settings.piper_train_command,
                    voice_id,
                    lang,
                    dataset_result["dataset_dir"],
                )
            except VoiceCloneError as exc:
                conn.execute(
                    "UPDATE voice_training_jobs SET status = ? WHERE id = ?;",
                    ("failed", job_id)
                )
                conn.commit()
                raise HTTPException(status_code=400, detail=str(exc))
            onnx_path = clone_result["artifact_uri"]
            config_path = clone_result["config_path"]
        else:
            conn.execute(
                "UPDATE voice_training_jobs SET status = ? WHERE id = ?;",
                ("failed", job_id)
            )
            conn.commit()
            raise HTTPException(
                status_code=400,
                detail=f"Voice engine '{voice['engine']}' is not implemented yet. Use Chatterbox for local zero-shot cloning or ElevenLabs with an API key."
            )
            
        # Register model artifacts in database
        conn.execute(
            "INSERT OR REPLACE INTO voice_model_artifacts (voice_id, language, onnx_path, config_path) VALUES (?, ?, ?, ?);",
            (voice_id, lang, onnx_path, config_path)
        )
        
        # Update job to completed
        conn.execute(
            "UPDATE voice_training_jobs SET status = ?, progress = ? WHERE id = ?;",
            ("completed", 100.0, job_id)
        )
        
        # Update voice status
        conn.execute(
            "UPDATE voices SET status = 'ready' WHERE id = ?;",
            (voice_id,)
        )
        conn.commit()
        
        # Log audit trail
        voice_studio_service.log_audit(None, "model_trained", f"Voice model {voice_id} cloned successfully via {voice['engine']} engine.")
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cloning failed: {exc}")
    finally:
        conn.close()
        VOICE_BUILD.__exit__()

    return {"ok": True, "job_id": job_id, "status": "completed"}

@app.post("/voices/{voice_id}/preview")
async def preview_cloned_voice(voice_id: str):
    """Synthesize a short test phrase with the cloned voice so the user can hear it."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    from app.database import get_db_connection
    conn = get_db_connection()
    try:
        voice = conn.execute("SELECT * FROM voices WHERE id = ?;", (voice_id,)).fetchone()
        if not voice:
            raise HTTPException(status_code=404, detail="Voice not found.")
        if voice["status"] not in ("ready", "published"):
            raise HTTPException(status_code=400, detail="Voice is not ready yet. Clone it first.")
    finally:
        conn.close()

    test_text = "Hello! This is how my cloned voice sounds. नमस्ते, यो मेरो क्लोन गरिएको आवाज हो।"
    parts = [TTSPart(text=chunk, language=lang) for chunk, lang in language_router.split_for_tts(test_text)]
    try:
        result = tts_provider.synthesize(parts, voice_id=voice_id, fallback_allowed=False)
    except _HeavyJobBusy:
        raise  # friendly 429 via the global handler
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preview synthesis failed: {exc}")
    return FileResponse(str(result.audio_path), media_type="audio/wav", filename="preview.wav")


@app.delete("/voices/{voice_id}")
def delete_voice(voice_id: str):
    voice_studio_service.delete_voice(voice_id)
    return {"ok": True}

@app.get("/voices/prompts")
def get_voices_prompts():
    return voice_studio_service.list_prompts()

@app.get("/audit/logs")
def get_audit_logs():
    return voice_studio_service.get_audit_logs()

# System monitoring routes
@app.get("/system/info")
def get_system_info():
    return system_monitor_service.get_static_info()

@app.get("/system/metrics")
def get_system_metrics():
    return system_monitor_service.get_realtime_metrics()

@app.websocket("/ws/system/metrics")
async def ws_system_metrics(websocket: WebSocket):
    await websocket.accept()
    try:
        import asyncio
        while True:
            metrics = system_monitor_service.get_realtime_metrics()
            await websocket.send_json(metrics)
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        return
    except Exception:
        return

# Voice cloning registry route
@app.get("/voices/cloning-engines")
def get_cloning_engines():
    return voice_cloning_registry.list_engines()

# Recording enhancements & validation routes
@app.post("/voices/{voice_id}/recordings/{prompt_id}/clean")
def clean_recording_endpoint(voice_id: str, prompt_id: str):
    """
    Cleans background noise and normalizes reference WAV files.
    """
    from fastapi import HTTPException
    voice_dir = voices_base_dir / voice_id
    raw_path = voice_dir / "raw" / f"{prompt_id}.raw"
    normalized_path = voice_dir / "normalized" / f"{prompt_id}.wav"
    
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail="Raw sample not found.")
        
    temp_wav = voice_dir / f"temp_clean_{prompt_id}.wav"
    try:
        # Convert raw to WAV using FFmpeg
        import shutil
        import subprocess
        if shutil.which("ffmpeg") is None:
            raise HTTPException(status_code=500, detail="ffmpeg is not installed on this system.")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_path), "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", str(temp_wav)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=30
        )
        
        cleaned_wav = voice_dir / f"clean_{prompt_id}.wav"
        noise_reduction_service.denoise_audio(temp_wav, cleaned_wav)
        
        final_wav = voice_dir / f"final_{prompt_id}.wav"
        audio_enhancement_service.normalize_loudness(cleaned_wav, final_wav)
        audio_enhancement_service.trim_silence(final_wav, normalized_path)
        
        for p in (temp_wav, cleaned_wav, final_wav):
            if p.exists():
                p.unlink()
    except Exception as e:
        if temp_wav.exists():
            temp_wav.unlink()
        raise HTTPException(status_code=500, detail=f"Denoise failed: {e}")

    q = voice_studio_service._evaluate_wav(normalized_path)
    
    from app.database import get_db_connection
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE voice_samples SET status = ?, score = ?, reason = ? WHERE voice_id = ? AND prompt_id = ?;",
            (q["verdict"], q["score"], q["reason"], voice_id, prompt_id)
        )
        conn.commit()
    finally:
        conn.close()
        
    return {
        "ok": True,
        "verdict": q["verdict"],
        "score": q["score"],
        "reason": q["reason"],
        "audio_url": f"/audio/voices/{voice_id}/normalized/{prompt_id}.wav"
    }

@app.post("/voices/{voice_id}/recordings/{prompt_id}/verify-speaker")
def verify_speaker_endpoint(voice_id: str, prompt_id: str):
    from fastapi import HTTPException
    voice_dir = voices_base_dir / voice_id
    wav_path = voice_dir / "normalized" / f"{prompt_id}.wav"
    
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Cleaned WAV sample not found.")
        
    res = speaker_isolation_service.verify_speaker_match(voice_id, wav_path, lock_if_empty=True)
    return res
