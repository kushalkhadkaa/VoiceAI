export type LanguageCode = "ne" | "en" | "mixed" | "unknown";
export type AssistantStatus = "idle" | "listening" | "transcribing" | "thinking" | "speaking" | "error";
export type ViewId = "conversation" | "voice_studio" | "knowledge" | "setup" | "evaluation" | "admin" | "logs" | "settings";

export interface TimingMetrics {
  audio_received_to_transcript_ms?: number | null;
  llm_first_token_ms?: number | null;
  llm_total_ms?: number | null;
  llm_load_ms?: number | null;
  prompt_eval_ms?: number | null;
  generation_ms?: number | null;
  tts_generation_ms?: number | null;
  total_turn_ms: number;
}

export interface ConversationTurn {
  transcript: string;
  response: string;
  input_language: LanguageCode;
  response_language: LanguageCode;
  audio_url?: string | null;
  user_audio_url?: string | null;
  tts_route?: Array<{ text: string; language: LanguageCode }>;
  timings: TimingMetrics;
  created_at?: string;
  
  // Custom upgrade fields
  rag_used?: boolean | null;
  rag_collection_id?: string | null;
  rag_fallback_used?: boolean | null;
  internet_used?: boolean | null;
  citations?: Array<{ title: string; url: string; snippet: string }> | null;
  voice_id?: string | null;
  requested_voice_id?: string | null;
  requested_voice_name?: string | null;
  actual_voice_id?: string | null;
  actual_voice_name?: string | null;
  actual_engine?: string | null;
  actual_model_path?: string | null;
  fallback_used?: boolean | null;
  fallback_reason?: string | null;
  llm_provider?: string | null;
  rag_path?: string | null;
}

export interface SystemMetrics {
  os?: string;
  cpu_percent?: number;
  ram_used_gb?: number;
  ram_available_gb?: number;
  disk_free_gb?: number;
  gpu_available?: boolean;
  gpu_name?: string;
  ollama_running?: boolean;
  open_webui_running?: boolean;
  recommendations?: string[];
  recommendation?: string;
  [key: string]: unknown;
}

export interface SystemInfo {
  os?: string;
  os_version?: string;
  architecture?: string;
  cpu_model?: string;
  cpu_cores?: number;
  ram_total_gb?: number;
  disk_total_gb?: number;
  disk_free_gb?: number;
  gpu_mps_available?: boolean;
  cuda_available?: boolean;
  python_version?: string;
  node_version?: string;
  ffmpeg_version?: string;
  ollama_version?: string;
  piper_version?: string;
  [key: string]: unknown;
}

export interface RagStatus {
  ok: boolean;
  base_url: string;
  version?: string | null;
  api_key_configured: boolean;
  enabled: boolean;
  default_collection?: string | null;
  fallback_to_ollama: boolean;
}

export interface ProviderStatus {
  name: string;
  ok: boolean;
  detail: string;
  critical?: boolean;
  fix?: string | null;
}

export type VoiceSocketConnectionState = "untested" | "connecting" | "ready" | "setup_required" | "error" | "closed";

export interface VoiceSocketStatus {
  ok: boolean;
  session_id: string;
  capabilities: Record<string, boolean>;
  blocking_reasons: string[];
  warnings: string[];
  checks: ProviderStatus[];
}

export interface BackendSettings {
  ollama_base_url: string;
  ollama_model: string;
  ollama_temperature: number;
  ollama_num_predict: number;
  ollama_keep_alive: string;
  system_prompt: string;
  whisper_model_size: string;
  piper_nepali_voice: string;
  piper_english_voice: string;
  piper_train_command: string;
  max_recording_seconds: number;
  low_latency_mode: boolean;
  quality_mode: boolean;
  open_webui_base_url: string;
  open_webui_api_key: string;
  rag_enabled: boolean;
  rag_default_collection: string;
  rag_fallback_to_ollama: boolean;
  internet_retrieval_enabled: boolean;
  internet_max_sources: number;
  internet_require_citation: boolean;
  internet_fallback_allowed: boolean;
  llm_provider: string;
  local_model: string;
  local_fallback_model: string;
  openai_api_key: string;
  openai_model: string;
  gemini_api_key: string;
  gemini_model: string;
  elevenlabs_api_key: string;
  cloud_fallback_to_local: boolean;
  cloud_timeout_seconds: number;
  cloud_temperature: number;
  cloud_max_tokens: number;
  force_selected_voice: boolean;
  fallback_allowed: boolean;
  single_tts_voice_model: boolean;
}

export interface VoiceInfo {
  id: string;
  model_path: string;
  config_path: string;
  model_exists: boolean;
  config_exists: boolean;
  language: string;
  engine?: string;
  status: string;
  missing_files: string[];
  license_path?: string | null;
  model_card_path?: string | null;
  name?: string;
  owner_name?: string;
  quality_score?: number;
  publish_status?: string;
  consent_status?: string;
  commercial_allowed?: boolean;
  disabled_reason?: string | null;
}

export interface VoicesResponse {
  voices: VoiceInfo[];
  selected: {
    nepali: VoiceInfo;
    english: VoiceInfo;
  };
}

export interface QualityScore {
  score: number;
  verdict: "good" | "review" | "reject";
  durationSeconds: number;
  peak: number;
  rms: number;
  reason: string;
}

export interface DatasetRecording {
  id: string;
  text: string;
  language: "ne" | "en";
  url: string;
  wavBlob?: Blob;
  quality: QualityScore;
}
