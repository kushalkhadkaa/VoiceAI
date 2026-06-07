from __future__ import annotations

import base64
import binascii
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Response, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import Settings
from app.providers.llm_ollama import OllamaLLMProvider
from app.providers.rag_openwebui import OpenWebUIRagProvider
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
stt_provider = FasterWhisperSTTProvider(
    settings.whisper_model_size,
    device=settings.whisper_device,
    compute_type=settings.whisper_compute_type,
)
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

conversation_service = ConversationService(
    stt_provider,
    llm_provider,
    openwebui_llm_provider,
    tts_provider,
    language_router,
    web_retrieval_provider,
)
from app.services.dataset import DatasetService

rag_provider = OpenWebUIRagProvider(settings)
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
    if {"internet_retrieval_enabled", "internet_max_sources", "internet_require_citation", "internet_fallback_allowed"} & set(changed):
        web_retrieval_provider.enabled = settings.internet_retrieval_enabled
        web_retrieval_provider.max_sources = settings.internet_max_sources
        web_retrieval_provider.require_citation = settings.internet_require_citation
        web_retrieval_provider.fallback_allowed = settings.internet_fallback_allowed
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


@app.delete("/settings/openai-key", response_model=SettingsResponse)
def delete_openai_key() -> SettingsResponse:
    settings.openai_api_key = ""
    settings._write_runtime_overrides()
    from app.services.environment import PROVIDER_TESTS
    PROVIDER_TESTS["openai"] = False
    return SettingsResponse(settings=settings.public_dict())


@app.delete("/settings/gemini-key", response_model=SettingsResponse)
def delete_gemini_key() -> SettingsResponse:
    settings.gemini_api_key = ""
    settings._write_runtime_overrides()
    from app.services.environment import PROVIDER_TESTS
    PROVIDER_TESTS["gemini"] = False
    return SettingsResponse(settings=settings.public_dict())


@app.delete("/settings/elevenlabs-key", response_model=SettingsResponse)
def delete_elevenlabs_key() -> SettingsResponse:
    settings.elevenlabs_api_key = ""
    settings._write_runtime_overrides()
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
    )


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


@app.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket):
    await websocket.accept()
    socket_status = build_voice_socket_status(settings)
    hello_received = False
    session_voice_id = None
    session_knowledge_id = None
    session_use_internet = False
    session_llm_provider_id = None
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
                    turn = conversation_service.handle_text(
                        str(message.get("text", "")),
                        voice_id=session_voice_id,
                        knowledge_id=session_knowledge_id,
                        use_internet=session_use_internet,
                        llm_provider_id=session_llm_provider_id,
                        session_id=socket_status.get("session_id"),
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
                        turn = conversation_service.handle_audio(
                            audio_path,
                            voice_id=session_voice_id,
                            knowledge_id=session_knowledge_id,
                            use_internet=session_use_internet,
                            llm_provider_id=session_llm_provider_id,
                            session_id=socket_status.get("session_id"),
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
            verdict = "reject"
            score = min(score, 40)
            reason = f"{reason}, {match_res['reason']}" if reason and reason != "clean" else match_res["reason"]
            
            # Update SQLite database with rejection status
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
        
    return {"ok": True, "job_id": job_id, "status": "completed"}

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
            ["ffmpeg", "-y", "-i", str(raw_path), "-ac", "1", "-ar", "22050", "-sample_fmt", "s16", str(temp_wav)],
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
