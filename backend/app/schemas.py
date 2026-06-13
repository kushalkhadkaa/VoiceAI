from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LanguageCode = Literal["ne", "en", "mixed", "unknown"]


class HealthResponse(BaseModel):
    ok: bool
    app: str
    version: str


class SettingsResponse(BaseModel):
    settings: dict


class SettingsUpdateRequest(BaseModel):
    ollama_base_url: str | None = None
    ollama_model: str | None = None
    ollama_temperature: float | None = None
    ollama_num_predict: int | None = None
    ollama_keep_alive: str | None = None
    system_prompt: str | None = None
    bank_instruction: str | None = None
    whisper_model_size: str | None = None
    piper_nepali_voice: str | None = None
    piper_english_voice: str | None = None
    piper_train_command: str | None = None
    max_recording_seconds: float | None = None
    low_latency_mode: bool | None = None
    quality_mode: bool | None = None
    open_webui_base_url: str | None = None
    open_webui_api_key: str | None = None
    rag_enabled: bool | None = None
    rag_default_collection: str | None = None
    rag_fallback_to_ollama: bool | None = None
    rag_mode: str | None = None
    kb_chromadb_path: str | None = None
    kb_embedding_provider: str | None = None
    kb_embedding_model: str | None = None
    kb_embedding_model_st: str | None = None
    kb_chunk_size: int | None = None
    kb_chunk_overlap: int | None = None
    kb_max_results: int | None = None
    kb_similarity_threshold: float | None = None
    kb_search_mode: str | None = None
    kb_chunk_strategy: str | None = None
    kb_reranking_enabled: bool | None = None
    kb_reranking_model: str | None = None
    kb_query_analytics: bool | None = None
    internet_retrieval_enabled: bool | None = None
    internet_max_sources: int | None = None
    internet_require_citation: bool | None = None
    internet_fallback_allowed: bool | None = None
    llm_provider: str | None = None
    stt_provider: str | None = None
    tts_provider: str | None = None
    local_model: str | None = None
    local_fallback_model: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    elevenlabs_api_key: str | None = None
    cloud_fallback_to_local: bool | None = None
    cloud_timeout_seconds: float | None = None
    cloud_temperature: float | None = None
    cloud_max_tokens: int | None = None
    force_selected_voice: bool | None = None
    fallback_allowed: bool | None = None
    single_tts_voice_model: bool | None = None
    openai_tts_voice: str | None = None


class ProviderStatus(BaseModel):
    name: str
    ok: bool
    detail: str
    critical: bool = True
    fix: str | None = None


class ModelStatusResponse(BaseModel):
    ready: bool
    providers: list[ProviderStatus]


class ChatTestRequest(BaseModel):
    text: str = Field(min_length=1)
    voice_id: str | None = None
    knowledge_id: str | None = None
    use_internet: bool = False
    llm_provider_id: str | None = None
    stt_provider_id: str | None = None
    document_id: str | None = None
    temperature: float | None = None



class TtsTestRequest(BaseModel):
    text: str = Field(min_length=1)
    language: LanguageCode | None = None


class TimingMetrics(BaseModel):
    audio_received_to_transcript_ms: float | None = None
    llm_first_token_ms: float | None = None
    llm_total_ms: float | None = None
    llm_load_ms: float | None = None
    prompt_eval_ms: float | None = None
    generation_ms: float | None = None
    tts_generation_ms: float | None = None
    total_turn_ms: float


class ConversationResponse(BaseModel):
    id: str
    transcript: str
    response: str
    transcript_translation: str | None = None
    response_translation: str | None = None
    input_language: LanguageCode
    response_language: LanguageCode

    audio_url: str | None = None
    tts_route: list[dict] = []
    timings: TimingMetrics
    rag_used: bool | None = None
    rag_collection_id: str | None = None
    rag_fallback_used: bool | None = None
    internet_used: bool | None = None
    citations: list[dict] | None = None
    voice_id: str | None = None
    requested_voice_id: str | None = None
    requested_voice_name: str | None = None
    actual_voice_id: str | None = None
    actual_voice_name: str | None = None
    actual_engine: str | None = None
    actual_model_path: str | None = None
    fallback_used: bool | None = None
    fallback_reason: str | None = None
    audio_sidecar: dict | None = None
    llm_provider: str | None = None
    rag_path: str | None = None
    ratings: dict | None = None


class RatingRequest(BaseModel):
    naturalness: int | None = None
    voice_similarity: int | None = None
    nepali_pronunciation: int | None = None
    english_pronunciation: int | None = None



class TtsTestResponse(BaseModel):
    language: LanguageCode
    audio_url: str


class VoicesResponse(BaseModel):
    voices: list[dict]
    selected: dict


class DoctorResponse(BaseModel):
    ready: bool
    checks: list[dict]


class VoiceSocketStatusResponse(BaseModel):
    ok: bool
    session_id: str
    capabilities: dict
    blocking_reasons: list[str]
    warnings: list[str]
    checks: list[dict]


class RagTestRequest(BaseModel):
    query: str = Field(min_length=1)
    collection_id: str | None = None


class DatasetQualityResponse(BaseModel):
    score: int
    verdict: str
    duration_seconds: float
    peak: float
    rms: float
    reason: str


class DatasetRecordingResponse(BaseModel):
    id: str
    text: str
    language: str
    exists: bool
    audio_url: str | None = None
    quality: DatasetQualityResponse | None = None


class VoiceCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    owner_name: str = Field(min_length=1)
    owner_email: str | None = None
    organization: str | None = None
    language: str = Field(min_length=2)
    engine: str = "piper"
    commercial_allowed: bool = False


class VoiceConsentRequest(BaseModel):
    signature: str = Field(min_length=1)


class VoiceStudioRecordingResponse(BaseModel):
    id: str
    prompt_id: str
    text: str
    language: str
    exists: bool
    audio_url: str | None = None
    score: int
    verdict: str
    peak: float
    rms: float
    reason: str
