import {
  Activity,
  CheckCircle2,
  CircleAlert,
  Database,
  Download,
  Mic,
  Play,
  Radio,
  MicOff,
  Save,
  Send,
  Settings,
  SlidersHorizontal,
  Square,
  Trash2,
  Volume2,
  Plus,
  User,
  FileText,
  Check,
  ShieldCheck,
  Search,
  Globe,
  AlertTriangle,
  BookOpen,
  Cpu,
  Gauge,
  HardDrive,
  KeyRound,
  Sparkles,
  Wand2,
  Wrench,
  FolderOpen,
  Upload,
  RefreshCw,
  Pencil,
  Lightbulb,
  Info,
  Languages,
  Headphones,
  ListChecks,
  ChevronRight,
  Building2,
  Copy,
  LifeBuoy
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  API_HTTP,
  API_WS,
  absoluteAudioUrl,
  deleteLocalData,
  getHealth,
  getModelStatus,
  getSettings,
  getVoiceSocketStatus,
  getVoices,
  sendTextTurn,
  testTts,
  updateSettings,
  getVoicesGallery,
  createVoice,
  saveVoiceConsent,
  getVoiceRecordings,
  uploadVoiceRecording,
  deleteVoiceRecording,
  publishVoice,
  deleteVoice,
  cleanRecording,
  cloneVoice,
  getVoicesPrompts,
  getCloningEngines,
  getAuditLogs,
  getRagStatus,
  getSystemInfo,
  getSystemMetrics,
  downloadDefaultVoices,
  getRagCollections,
  getKBStatus,
  getKBCollections,
  createKBCollection,
  deleteKBCollection,
  getKBDocuments,
  ingestKBFile,
  ingestKBUrl,
  deleteKBDocument,
  renameKBDocument,
  getKBDocumentChunks,
  queryKB,
  getDatasetRecordings,
  uploadDatasetRecording,
  deleteDatasetRecording,
  deriveDatasetExportUrl,
  testProvider,
  testRagQuestion,
  deleteOpenAIKey,
  deleteGeminiKey,
  deleteElevenLabsKey,
  getChatHistory,
  rateChatTurn,
  previewTts,
  getKBEmbeddingStatus,
  queryKBAdvanced,
  getKBAnalytics,
  clearKBAnalytics,
  getKBCollectionStats,
  exportKBCollection,
  kbEvalGenerate,
  kbEvalRun,
  chatWithDocument,
  voiceTurnRest,
  setActiveAIProvider,
  getOpenAIModels,
  getAiProviders,
  getAiProvidersStatus,
  crawlKBSite,
  adminLogin,
  adminLogout,
  adminVerify,
  kycQuery,
  kycStatus,
  type KycQueryResult,
  type CrawlEvent,
  type CrawlSiteOptions
} from "./api";
import type { AdvancedQueryOptions, EvalQA, EvalRow } from "./api";
import { blobToBase64, downloadBlob, scoreRecording } from "./audio";
import type {
  AssistantStatus,
  BackendSettings,
  ConversationTurn,
  DatasetRecording,
  KBCollection,
  KBDocument,
  KBSearchResult,
  KBStatus,
  LanguageCode,
  ProviderStatus,
  RagStatus,
  SystemInfo,
  SystemMetrics,
  VoiceSocketConnectionState,
  VoiceSocketStatus,
  VoicesResponse,
  ViewId
} from "./types";

const views: Array<{ id: ViewId; label: string; icon: any }> = [
  { id: "conversation", label: "Voice", icon: Radio },
  { id: "voice_studio", label: "Voice Studio", icon: Database },
  { id: "knowledge", label: "RAG", icon: Database },
  { id: "setup", label: "Setup", icon: CheckCircle2 },
  { id: "evaluation", label: "Eval", icon: Activity },
  { id: "admin", label: "Admin", icon: Wrench },
  { id: "logs", label: "Logs", icon: FileText },
  { id: "system_map", label: "System Map", icon: Globe },
  { id: "settings", label: "Settings", icon: Settings }
];

const prompts = [
  { id: "000001", language: "ne" as const, text: "नमस्ते, आज म मेरो आवाजको नमूना रेकर्ड गर्दैछु।" },
  { id: "000002", language: "ne" as const, text: "कृपया मलाई छोटो र स्पष्ट जवाफ दिनुहोस्।" },
  { id: "000003", language: "en" as const, text: "Hello, this is a clean sample of my speaking voice." },
  { id: "000004", language: "en" as const, text: "Please answer naturally and keep the response concise." },
  { id: "000005", language: "ne" as const, text: "मेरो आवाज शान्त कोठामा रेकर्ड गरिएको हो।" },
  { id: "000006", language: "en" as const, text: "The quick local assistant should work without cloud services." }
];

const languageLabels: Record<LanguageCode, string> = {
  ne: "Nepali",
  en: "English",
  mixed: "Mixed",
  unknown: "Unknown"
};

const statusLabels: Record<AssistantStatus, string> = {
  idle: "Idle",
  listening: "Listening",
  transcribing: "Transcribing",
  thinking: "Thinking",
  speaking: "Speaking",
  error: "Needs attention"
};

const defaultSettings: BackendSettings = {
  ollama_base_url: "http://localhost:11434",
  ollama_model: "qwen3:1.7b",
  ollama_temperature: 0.35,
  ollama_num_predict: 180,
  ollama_keep_alive: "10m",
  system_prompt:
    "You are Nabil Voice AI, a helpful Nepali-English banking assistant for Nabil Bank. Reply naturally in English, Nepali, or mixed Nepali-English only. Use provided knowledge base context before answering. If the knowledge base does not contain enough information, say so clearly.",
  bank_instruction: "",
  whisper_model_size: "small",
  piper_nepali_voice: "./models/piper/ne_NP-chitwan-medium.onnx",
  piper_english_voice: "./models/piper/en_US-lessac-medium.onnx",
  piper_train_command: "",
  max_recording_seconds: 30,
  low_latency_mode: true,
  quality_mode: false,
  open_webui_base_url: "http://127.0.0.1:8080",
  open_webui_api_key: "",
  rag_enabled: false,
  rag_default_collection: "",
  rag_fallback_to_ollama: true,
  internet_retrieval_enabled: false,
  internet_max_sources: 5,
  internet_require_citation: true,
  internet_fallback_allowed: false,
  llm_provider: "openai",
  stt_provider: "openai",
  tts_provider: "openai",
  local_model: "qwen2.5:7b",
  local_fallback_model: "gemma3:4b",
  openai_api_key: "",
  openai_model: "gpt-4o-mini",
  gemini_api_key: "",
  gemini_model: "gemini-1.5-flash",
  elevenlabs_api_key: "",
  cloud_fallback_to_local: false,
  cloud_timeout_seconds: 30.0,
  cloud_temperature: 0.35,
  cloud_max_tokens: 180,
  force_selected_voice: false,
  fallback_allowed: true,
  single_tts_voice_model: true,
  openai_tts_voice: "alloy",
  rag_mode: "local",
  kb_embedding_provider: "ollama",
  kb_embedding_model: "nomic-embed-text",
  kb_embedding_model_st: "all-MiniLM-L6-v2",
  kb_chunk_size: 512,
  kb_chunk_overlap: 50,
  kb_max_results: 5,
  kb_similarity_threshold: 0.3,
  chatterbox_exaggeration: 0.5,
  chatterbox_cfg_weight: 0.5,
  chatterbox_temperature: 0.8,
  chatterbox_repetition_penalty: 1.2,
};

interface TypingTextProps {
  text: string;
  speedMs?: number;
}

const TypingText: React.FC<TypingTextProps> = ({ text, speedMs = 12 }) => {
  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    setDisplayedText("");
    setIsTyping(true);
    if (!text) {
      setIsTyping(false);
      return;
    }

    let index = 0;
    const interval = setInterval(() => {
      setDisplayedText((prev) => {
        const next = prev + text.charAt(index);
        setTimeout(() => {
          const container = document.querySelector(".chat-messages");
          if (container) {
            container.scrollTop = container.scrollHeight;
          }
        }, 0);
        return next;
      });
      index++;
      if (index >= text.length) {
        clearInterval(interval);
        setIsTyping(false);
      }
    }, speedMs);

    return () => {
      clearInterval(interval);
      setIsTyping(false);
    };
  }, [text, speedMs]);

  return (
    <>
      {displayedText}
      {isTyping && (
        <span 
          style={{
            display: "inline-block",
            width: "8px",
            height: "15px",
            background: "var(--teal)",
            marginLeft: "4px",
            verticalAlign: "middle",
            animation: "prog-pulse 0.8s ease-in-out infinite",
            borderRadius: "1px"
          }}
        />
      )}
    </>
  );
};

function App() {
  const [activeView, setActiveView] = useState<ViewId>("conversation");
  const [status, setStatus] = useState<AssistantStatus>("idle");
  const [autoVad, setAutoVad] = useLocalStorage("swarlocal.autoVad", true);
  const [history, setHistory] = useState<ConversationTurn[]>([]);
  const [settings, setSettings] = useState<BackendSettings>(defaultSettings);
  const [providerStatus, setProviderStatus] = useState<ProviderStatus[]>([]);
  const [voiceSocketStatus, setVoiceSocketStatus] = useState<VoiceSocketStatus | null>(null);
  const [voiceSocketState, setVoiceSocketState] = useState<VoiceSocketConnectionState>("untested");
  const [voices, setVoices] = useState<VoicesResponse | null>(null);
  const [manualText, setManualText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [ttsTestingId, setTtsTestingId] = useState<string | null>(null);
  const [lastAudioUrl, setLastAudioUrl] = useState<string | null>(null);
  const [datasetRecordings, setDatasetRecordings] = useState<DatasetRecording[]>([]);
  const [datasetActivePrompt, setDatasetActivePrompt] = useState<string | null>(null);

  const [selectedVoiceId, setSelectedVoiceId] = useLocalStorage("swarlocal.selectedVoiceId", "openai-alloy");
  const [selectedKnowledgeId, setSelectedKnowledgeId] = useLocalStorage("swarlocal.selectedKnowledgeId", "none");
  const [useInternet, setUseInternet] = useLocalStorage("swarlocal.useInternet", false);
  const [selectedBrain, setSelectedBrain] = useLocalStorage("swarlocal.selectedBrain", "openai");
  const [selectedSttProvider, setSelectedSttProvider] = useLocalStorage("swarlocal.selectedSttProvider", "openai");
  // STT language for the next turn — flows to backend via WS config + REST/text calls. Default "auto".
  const [sttLanguage, setSttLanguage] = useLocalStorage<"auto" | "ne" | "en">("swarlocal.sttLanguage", "auto");

  // Live voice state — single source of truth is the backend transcription
  // delivered over the WebSocket (partial_transcript / final_transcript / partial_translation).
  const [liveTranscript, setLiveTranscript] = useState("");
  const [liveTranslation, setLiveTranslation] = useState("");
  const [liveLanguage, setLiveLanguage] = useState<LanguageCode>("unknown");
  const [transcriptFinal, setTranscriptFinal] = useState(false);
  // Real mic level (0..1) sampled from the SAME MediaStream used for recording.
  const [micLevel, setMicLevel] = useState(0);

  useEffect(() => {
    const marker = "swarlocal.openaiDefaultsApplied.v1";
    if (window.localStorage.getItem(marker)) return;
    setSelectedBrain("openai");
    setSelectedSttProvider("openai");
    setSelectedVoiceId("openai-alloy");
    window.localStorage.setItem(marker, "1");
  }, [setSelectedBrain, setSelectedSttProvider, setSelectedVoiceId]);

  const [ragCollections, setRagCollections] = useState<any[]>([]);
  const [ragStatus, setRagStatus] = useState<RagStatus | null>(null);
  const [galleryVoices, setGalleryVoices] = useState<any[]>([]);
  const [auditLogs, setAuditLogs] = useState<any[]>([]);
  const [appLogs, setAppLogs] = useLocalStorage<any[]>("swarlocal.logs", []);
  const [showSettings, setShowSettings] = useLocalStorage("swarlocal.showSettings", false);
  const [playingAudioUrl, setPlayingAudioUrl] = useState<string | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const audioTurnRef = useRef<HTMLAudioElement | null>(null);
  const lastUserAudioUrlRef = useRef<string | null>(null);

  const handlePlayTurnAudio = useCallback((rawUrl: string | null | undefined) => {
    if (!rawUrl) return;
    const url = rawUrl.startsWith("blob:") ? rawUrl : absoluteAudioUrl(rawUrl || undefined);
    if (!url) return;
    
    if (playingAudioUrl === url && audioTurnRef.current) {
      audioTurnRef.current.pause();
      setPlayingAudioUrl(null);
      return;
    }

    if (audioTurnRef.current) {
      audioTurnRef.current.pause();
    }
    if (lastAudioElementRef.current) {
      lastAudioElementRef.current.pause();
      setStatus("idle");
    }

    const audio = new Audio(url);
    audioTurnRef.current = audio;
    setPlayingAudioUrl(url);
    
    audio.onended = () => {
      setPlayingAudioUrl(null);
    };
    audio.onerror = () => {
      setPlayingAudioUrl(null);
    };

    audio.play().catch(() => {
      setPlayingAudioUrl(null);
    });
  }, [playingAudioUrl]);

  const socketRef = useRef<WebSocket | null>(null);
  const heartbeatRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);
  const lastAudioElementRef = useRef<HTMLAudioElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const vadIntervalRef = useRef<number | null>(null);
  const vadAudioContextRef = useRef<AudioContext | null>(null);
  // Real-time partials: timer that periodically ships the full accumulated blob,
  // and a monotonically increasing sequence number.
  const partialTimerRef = useRef<number | null>(null);
  const partialSeqRef = useRef(0);
  const partialInFlightRef = useRef(false);
  const recordMimeRef = useRef<string>("audio/webm");
  // Mic-level analyser bound to the recording stream (drives the orb --voice-level).
  const levelAudioContextRef = useRef<AudioContext | null>(null);
  const levelRafRef = useRef<number | null>(null);
  const datasetRecorderRef = useRef<MediaRecorder | null>(null);
  const datasetChunksRef = useRef<Blob[]>([]);

  const latestTurn = history[0];

  const logEvent = useCallback((level: "info" | "success" | "warning" | "error", event: string, detail: string, meta?: Record<string, unknown>) => {
    setAppLogs((items) => [
      {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        timestamp: new Date().toISOString(),
        level,
        event,
        detail,
        meta: meta ?? null,
      },
      ...items,
    ].slice(0, 250));
  }, [setAppLogs]);

  const handleRatingChange = useCallback(async (newRatings: Record<string, number>) => {
    const latestTurn = history[0];
    if (latestTurn && latestTurn.id) {
      try {
        await rateChatTurn(latestTurn.id, newRatings);
        setHistory(prev => prev.map((turn, i) => i === 0 ? { ...turn, ratings: newRatings } : turn));
      } catch (err: any) {
        logEvent("error", "save_rating_failed", `Failed to save rating to SQLite: ${err.message}`);
      }
    }
  }, [history, logEvent]);

  useEffect(() => {
    if (error) {
      logEvent("error", "ui_error", error);
    }
  }, [error, logEvent]);

  const refreshStatus = useCallback(async () => {
    setError(null);
    try {
      const [
        healthOk,
        providers,
        socketPayload,
        backendSettings,
        voicePayload,
        datasetPayload,
        galleryPayload,
        ragPayload,
        auditPayload,
        systemPayload,
        systemInfoPayload,
        ragStatusPayload,
        historyPayload
      ] = await Promise.all([
        getHealth().catch(() => false),
        getModelStatus().catch(() => []),
        getVoiceSocketStatus().catch(() => null),
        getSettings().catch(() => defaultSettings),
        getVoices().catch(() => null),
        getDatasetRecordings().catch(() => []),
        getVoicesGallery().catch(() => []),
        getRagCollections().catch(() => []),
        getAuditLogs().catch(() => []),
        getSystemMetrics().catch(() => null),
        getSystemInfo().catch(() => null),
        getRagStatus().catch(() => null),
        getChatHistory().catch(() => [])
      ]);
      const allProviders = [
        { name: "browser_backend_link", ok: healthOk, detail: healthOk ? "Backend reachable" : "Backend unavailable" },
        ...providers
      ];
      setProviderStatus(allProviders.sort((a, b) => Number(a.ok) - Number(b.ok)));
      setVoiceSocketStatus(socketPayload);
      if (socketPayload?.blocking_reasons.length) {
        setVoiceSocketState("setup_required");
      }
      setSettings({ ...defaultSettings, ...backendSettings });
      setVoices(voicePayload);
      setDatasetRecordings(datasetPayload);
      setGalleryVoices(galleryPayload);
      setRagCollections(ragPayload);
      setAuditLogs(auditPayload);
      setSystemMetrics(systemPayload);
      setSystemInfo(systemInfoPayload);
      setRagStatus(ragStatusPayload);
      setHistory(historyPayload);
      logEvent("success", "status_refreshed", "Runtime status, settings, voices, collections, and logs refreshed.");
    } catch (refreshError) {
      setError(refreshError instanceof Error ? refreshError.message : "Unable to refresh status.");
    }
  }, [logEvent]);

  useEffect(() => {
    refreshStatus();
    return () => {
      socketRef.current?.close();
      stopHeartbeat();
      stopReconnect();
      stopTracks();
      stopVadWatch();
      stopPartials();
      stopLevelMeter();
    };
  }, [refreshStatus]);

  const handleTurn = useCallback(
    async (turn: ConversationTurn) => {
      const stamped = { 
        ...turn, 
        user_audio_url: lastUserAudioUrlRef.current,
        created_at: new Date().toISOString() 
      };
      lastUserAudioUrlRef.current = null;
      // The turn is now in history — clear the transient live transcript buffers.
      setLiveTranscript("");
      setLiveTranslation("");
      setTranscriptFinal(false);
      setHistory((items) => {
        const filtered = items.filter(item => !item.is_pending);
        return [stamped, ...filtered].slice(0, 80);
      });

      logEvent("success", "conversation_turn", "Conversation turn completed.", {
        provider: turn.llm_provider,
        voice: turn.actual_voice_id,
        audio_url: turn.audio_url,
        fallback: Boolean(turn.fallback_used),
      });
      const audioUrl = absoluteAudioUrl(turn.audio_url);
      setLastAudioUrl(audioUrl);
      if (audioUrl) {
        setStatus("speaking");
        const audio = new Audio(audioUrl);
        lastAudioElementRef.current = audio;
        audio.onended = () => setStatus("idle");
        try {
          await audio.play();
        } catch {
          setStatus("idle");
        }
      } else {
        setStatus("idle");
      }
    },
    [setHistory, logEvent]
  );

  async function diagnoseSocketFailure(): Promise<string> {
    try {
      const healthOk = await getHealth().catch(() => false);
      if (!healthOk) {
        return "backend unavailable: Nabil Voice AI FastAPI server is not running or unreachable.";
      }
      const socketStatus = await getVoiceSocketStatus().catch(() => null);
      if (!socketStatus) {
        return "websocket route unavailable: FastAPI is running but the WebSocket server path is unreachable.";
      }
      if (socketStatus.blocking_reasons && socketStatus.blocking_reasons.length) {
        return `setup required: The following blockers must be fixed: ${socketStatus.blocking_reasons.join(", ")}`;
      }
      return "websocket connection blocked: Verify your browser settings, local origin CORS, or firewall rules.";
    } catch (e) {
      return "backend unavailable: Cannot connect to Nabil Voice AI local server.";
    }
  }

  const ensureSocket = useCallback((allowReconnect = true) => {
    const existing = socketRef.current;
    if (existing && existing.readyState === WebSocket.OPEN) {
      return Promise.resolve(existing);
    }
    return new Promise<WebSocket>((resolve, reject) => {
      setVoiceSocketState("connecting");
      const socket = new WebSocket(API_WS);
      socketRef.current = socket;
      let ready = false;
      socket.onopen = () => {
        socket.send(
          JSON.stringify({
            type: "hello",
            client_version: "web",
            selected_voice_id: selectedVoiceId,
            selected_knowledge_id: selectedKnowledgeId,
            use_internet: useInternet,
            selected_brain: selectedBrain,
            selected_stt_provider: selectedSttProvider,
          })
        );
        startHeartbeat(socket);
      };
      socket.onerror = async () => {
        setVoiceSocketState("error");
        const diagnosis = await diagnoseSocketFailure();
        reject(new Error(diagnosis));
      };
      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "ready") {
          ready = true;
          const nextStatus = {
            ok: true,
            session_id: payload.session_id,
            capabilities: payload.capabilities ?? {},
            blocking_reasons: payload.blocking_reasons ?? [],
            warnings: payload.warnings ?? [],
            checks: []
          };
          setVoiceSocketStatus(nextStatus);
          setVoiceSocketState(nextStatus.blocking_reasons.length ? "setup_required" : "ready");
          reconnectAttemptRef.current = 0;
          resolve(socket);
        }
        if (payload.type === "pong") {
          return;
        }
        if (payload.type === "status") {
          setStatus(payload.status);
        }
        if (payload.type === "setup_required") {
          setStatus("error");
          setVoiceSocketState("setup_required");
          setError([payload.detail, ...(payload.blocking_reasons ?? [])].filter(Boolean).join(" "));
        }
        // ── Real-time voice contract ─────────────────────────────────────────
        // Live, may update repeatedly. Single source of truth for the live display.
        if (payload.type === "partial_transcript") {
          setTranscriptFinal(false);
          setLiveTranscript(typeof payload.text === "string" ? payload.text : "");
          if (payload.language) setLiveLanguage(payload.language as LanguageCode);
        }
        // Authoritative transcript after audio_end.
        if (payload.type === "final_transcript") {
          setTranscriptFinal(true);
          setLiveTranscript(typeof payload.text === "string" ? payload.text : "");
          if (payload.language) setLiveLanguage(payload.language as LanguageCode);
        }
        // Optional live translation of the transcript (best-effort).
        if (payload.type === "partial_translation") {
          setLiveTranslation(typeof payload.text === "string" ? payload.text : "");
        }
        // Final answer turn — identical shape to REST /voice/turn. Render text + play TTS.
        if (payload.type === "answer") {
          const turn = (payload.turn ?? payload) as ConversationTurn;
          setLiveTranslation("");
          handleTurn(turn);
        }
        // Backward-compat: existing blocking audio/text turn message.
        if (payload.type === "turn") {
          handleTurn(payload as ConversationTurn);
        }
        if (payload.type === "error") {
          setStatus("error");
          setError(payload.detail);
        }
      };
      socket.onclose = () => {
        stopHeartbeat();
        if (socketRef.current === socket) {
          socketRef.current = null;
        }
        setVoiceSocketState((value) => (value === "setup_required" ? value : "closed"));
        if (allowReconnect && ready) {
          scheduleReconnect();
        }
      };
    });
  }, [handleTurn, selectedVoiceId, selectedKnowledgeId, useInternet, selectedBrain, selectedSttProvider]);

  useEffect(() => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.close();
    }
  }, [selectedVoiceId, selectedKnowledgeId, useInternet, selectedBrain, selectedSttProvider]);

  // Reset a stale knowledge selection (e.g. a deleted collection id left in
  // localStorage) so we never send a phantom knowledge_id that silently kills RAG.
  useEffect(() => {
    if (selectedKnowledgeId !== "none" && ragCollections.length > 0
        && !ragCollections.find((c: any) => c.id === selectedKnowledgeId)) {
      setSelectedKnowledgeId("none");
    }
  }, [ragCollections, selectedKnowledgeId, setSelectedKnowledgeId]);


  useEffect(() => {
    if (!voices || selectedVoiceId === "auto" || selectedVoiceId.startsWith("openai-")) {
      return;
    }
    const selected = voices.voices.find((voice) => voice.id === selectedVoiceId);
    if (!selected || selected.model_exists === false || selected.config_exists === false) {
      setSelectedVoiceId("auto");
    }
  }, [voices, selectedVoiceId, setSelectedVoiceId]);

  function startHeartbeat(socket: WebSocket) {
    stopHeartbeat();
    heartbeatRef.current = window.setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "ping", ts: Date.now() }));
      }
    }, 15000);
  }

  function stopHeartbeat() {
    if (heartbeatRef.current) {
      window.clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    }
  }

  function stopReconnect() {
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }

  function scheduleReconnect() {
    stopReconnect();
    const attempt = Math.min(reconnectAttemptRef.current + 1, 6);
    reconnectAttemptRef.current = attempt;
    const delayMs = Math.min(30000, 1000 * 2 ** (attempt - 1));
    reconnectTimerRef.current = window.setTimeout(() => {
      ensureSocket(true).catch(() => scheduleReconnect());
    }, delayMs);
  }

  async function testVoiceSocket() {
    setError(null);
    try {
      socketRef.current?.close();
      socketRef.current = null;
      setVoiceSocketStatus(await getVoiceSocketStatus());
      await ensureSocket(false);
      logEvent("success", "voice_socket_test", "Voice socket connected successfully.");
    } catch (socketError) {
      setVoiceSocketState("error");
      const diagnosis = await diagnoseSocketFailure();
      setError(diagnosis);
      logEvent("error", "voice_socket_test_failed", diagnosis);
    }
  }

  async function sendText() {
    const text = manualText.trim();
    if (!text) {
      return;
    }
    setStatus("thinking");
    setError(null);
    setManualText("");

    const pendingTurn: ConversationTurn = {
      transcript: text,
      response: "",
      input_language: "en",
      response_language: "en",
      timings: { total_turn_ms: 0 },
      created_at: new Date().toISOString(),
      is_pending: true,
    };
    setHistory((items) => [pendingTurn, ...items]);

    try {
      const socket = await ensureSocket();
      socket.send(JSON.stringify({ type: "text", text, stt_language: sttLanguage }));
    } catch {
      try {
        await handleTurn(await sendTextTurn(text, {
          voice_id: selectedVoiceId === "auto" ? undefined : selectedVoiceId,
          knowledge_id: selectedKnowledgeId === "none" ? undefined : selectedKnowledgeId,
          use_internet: useInternet,
          llm_provider_id: selectedBrain,
          stt_language: sttLanguage,
        }));
      } catch (chatError) {
        setHistory((items) => items.filter((item) => !item.is_pending));
        setStatus("error");
        setError(chatError instanceof Error ? chatError.message : "Text turn failed.");
      }
    }
  }


  async function startRecording() {
    if (recorderRef.current?.state === "recording") {
      stopRecording();
      return;
    }
    setError(null);
    try {
      // Always fetch fresh status — never use stale cache that may have old blocking reasons
      const socketStatus = await getVoiceSocketStatus();
      setVoiceSocketStatus(socketStatus);
      if (!socketStatus.capabilities.audio_turns) {
        const reasons = socketStatus.blocking_reasons.filter((r: string) => r && r.trim());
        if (reasons.length > 0) {
          setVoiceSocketState("setup_required");
          setStatus("error");
          setError(`Setup required before recording: ${reasons.join(", ")}`);
          logEvent("warning", "recording_blocked", `Setup required before recording: ${reasons.join(", ")}`);
          return;
        }
        // audio_turns false but no blocking reasons — proceed anyway
      }
      const socket = await ensureSocket();
      // Reset live transcript state — the backend is the single source of truth.
      setLiveTranscript("");
      setLiveTranslation("");
      setLiveLanguage("unknown");
      setTranscriptFinal(false);
      partialSeqRef.current = 0;
      partialInFlightRef.current = false;
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
      recordMimeRef.current = mimeType;
      const recorder = new MediaRecorder(stream, { mimeType });
      streamRef.current = stream;
      recorderRef.current = recorder;
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onstop = async () => {
        stopVadWatch();
        stopPartials();
        stopLevelMeter();
        stopTracks();
        const blob = new Blob(chunksRef.current, { type: mimeType });
        chunksRef.current = [];
        recorderRef.current = null;
        if (blob.size < 1024) {
          setStatus("idle");
          return;
        }
        try {
          lastUserAudioUrlRef.current = URL.createObjectURL(blob);
        } catch (e) {
          console.error(e);
        }
        await sendAudio(blob, mimeType);
      };
      // Send session params for this turn (STT language, voice, knowledge).
      try {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({
            type: "config",
            stt_language: sttLanguage,
            llm_provider: selectedBrain,
            voice_id: selectedVoiceId === "auto" ? undefined : selectedVoiceId,
            knowledge_id: selectedKnowledgeId === "none" ? undefined : selectedKnowledgeId,
          }));
        }
      } catch { /* config best-effort */ }
      // Emit chunks every 1s so the accumulated buffer always includes the header chunk.
      recorder.start(1000);
      setStatus("listening");
      logEvent("info", "recording_started", "Microphone recording started.");
      // Drive the orb from the SAME recording stream (no second getUserMedia).
      startLevelMeter(stream);
      // Stream partial transcriptions every ~2.5s while the user speaks.
      startPartials();
      if (autoVad) {
        startVadWatch(stream, recorder);
      }
    } catch (recordError) {
      setStatus("error");
      setError(recordError instanceof Error ? recordError.message : "Microphone is unavailable.");
      logEvent("error", "recording_failed", recordError instanceof Error ? recordError.message : "Microphone is unavailable.");
    }
  }

  function stopRecording() {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      logEvent("info", "recording_stopped", "Microphone recording stopped.");
    }
  }

  // Cancel = discard the in-progress recording without transcribing/answering.
  function cancelRecording() {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      // Empty the buffer first so onstop sees < 1KB and bails out without sending.
      chunksRef.current = [];
      recorder.stop();
      logEvent("info", "recording_cancelled", "Microphone recording cancelled (discarded).");
    }
    stopPartials();
    stopLevelMeter();
    setLiveTranscript("");
    setLiveTranslation("");
    setTranscriptFinal(false);
    setStatus("idle");
  }

  function interruptPlayback() {
    lastAudioElementRef.current?.pause();
    lastAudioElementRef.current = null;
    setStatus("idle");
  }

  async function replayLastAnswer() {
    if (!lastAudioUrl) {
      return;
    }
    const audio = new Audio(lastAudioUrl);
    lastAudioElementRef.current = audio;
    setStatus("speaking");
    audio.onended = () => setStatus("idle");
    try {
      await audio.play();
    } catch {
      setStatus("idle");
    }
  }

  async function sendAudio(blob: Blob, mimeType: string) {
    setStatus("transcribing");
    try {
      // Try WebSocket first — final, authoritative transcription per the real-time contract.
      const socket = await ensureSocket();
      logEvent("info", "audio_sent", "Audio sent to the voice pipeline via WebSocket.", { bytes: blob.size, mimeType });
      setStatus("thinking");
      const data = await blobToBase64(blob);
      socket.send(
        JSON.stringify({
          type: "audio_end",
          mime: mimeType,
          data,
          // Back-compat aliases for handlers that still read the old field names.
          mimeType,
          audioBase64: data,
        })
      );
    } catch {
      // REST fallback — works even if WebSocket is unavailable
      logEvent("info", "audio_rest_fallback", "WebSocket unavailable, using REST audio endpoint.");
      try {
        const turn = await voiceTurnRest(blob, {
          voice_id: selectedVoiceId === "auto" ? undefined : selectedVoiceId,
          knowledge_id: selectedKnowledgeId === "none" ? undefined : selectedKnowledgeId,
          use_internet: useInternet,
          llm_provider_id: selectedBrain,
          stt_provider_id: selectedSttProvider,
          stt_language: sttLanguage,
        });

        await handleTurn(turn);
      } catch (restError) {
        setStatus("error");
        setError(restError instanceof Error ? restError.message : "Voice turn failed (REST fallback).");
        logEvent("error", "audio_rest_failed", restError instanceof Error ? restError.message : "Voice turn REST failed.");
      }
    }
  }

  // ── Real-time partial transcription ─────────────────────────────────────────
  // Every ~2.5s ship the FULL accumulated recording-so-far (chunks[0..n] joined,
  // including the header chunk so the blob is decodable). The backend transcribes
  // the whole blob and emits a partial_transcript. No LLM is invoked for partials.
  function startPartials() {
    stopPartials();
    partialTimerRef.current = window.setInterval(() => {
      void sendPartial();
    }, 2500);
  }

  function stopPartials() {
    if (partialTimerRef.current) {
      window.clearInterval(partialTimerRef.current);
      partialTimerRef.current = null;
    }
  }

  async function sendPartial() {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    if (partialInFlightRef.current) return; // avoid piling up if STT is slow
    const chunks = chunksRef.current;
    if (!chunks.length) return;
    const mimeType = recordMimeRef.current;
    const blob = new Blob(chunks, { type: mimeType });
    if (blob.size < 2048) return; // too small to decode
    partialInFlightRef.current = true;
    try {
      const data = await blobToBase64(blob);
      socket.send(JSON.stringify({
        type: "audio_partial",
        seq: partialSeqRef.current++,
        mime: mimeType,
        data,
      }));
    } catch {
      /* best-effort — final transcription is authoritative */
    } finally {
      partialInFlightRef.current = false;
    }
  }

  // ── Live mic-level meter ────────────────────────────────────────────────────
  // Bound to the SAME MediaStream used for recording so the orb pulses with the
  // actual captured audio (not a separate getUserMedia tap).
  function startLevelMeter(stream: MediaStream) {
    stopLevelMeter();
    const AudioContextClass = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextClass) return;
    const ctx = new AudioContextClass();
    levelAudioContextRef.current = ctx;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const data = new Float32Array(analyser.fftSize);
    const tick = () => {
      analyser.getFloatTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
      const level = Math.min(1, Math.sqrt(sum / data.length) * 8);
      setMicLevel(level);
      levelRafRef.current = requestAnimationFrame(tick);
    };
    tick();
  }

  function stopLevelMeter() {
    if (levelRafRef.current) {
      cancelAnimationFrame(levelRafRef.current);
      levelRafRef.current = null;
    }
    levelAudioContextRef.current?.close().catch(() => {});
    levelAudioContextRef.current = null;
    setMicLevel(0);
  }

  function startVadWatch(stream: MediaStream, recorder: MediaRecorder) {
    const AudioContextClass = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextClass) {
      return;
    }
    const context = new AudioContextClass();
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    const data = new Float32Array(analyser.fftSize);
    const started = performance.now();
    let heardSpeech = false;
    let silenceSince: number | null = null;
    vadAudioContextRef.current = context;
    vadIntervalRef.current = window.setInterval(() => {
      analyser.getFloatTimeDomainData(data);
      const rms = Math.sqrt(data.reduce((sum, value) => sum + value * value, 0) / data.length);
      const now = performance.now();
      if (rms > 0.035) {
        heardSpeech = true;
        silenceSince = null;
      } else if (heardSpeech && now - started > 1000) {
        silenceSince ??= now;
        if (now - silenceSince > 1200 && recorder.state === "recording") {
          recorder.stop();
        }
      }
      if (now - started > 14000 && recorder.state === "recording") {
        recorder.stop();
      }
    }, 180);
  }

  function stopVadWatch() {
    if (vadIntervalRef.current) {
      window.clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = null;
    }
    vadAudioContextRef.current?.close();
    vadAudioContextRef.current = null;
  }

  function stopTracks() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  async function runTtsTest(language: "ne" | "en", voiceId?: string) {
    setError(null);
    setTtsTestingId(voiceId ?? `__${language}`);
    try {
      if (voiceId) {
        const text = language === "ne"
          ? "नमस्ते, यो नयाँ आवाजको पूर्वअवलोकन हो।"
          : "Hello, this is a preview of the selected voice.";
        const res = await previewTts(text, voiceId, language);
        if (res.ok && res.audio_url) {
          const audioUrl = absoluteAudioUrl(res.audio_url);
          setLastAudioUrl(audioUrl);
          if (audioUrl) {
            await new Audio(audioUrl).play();
          }
        } else {
          setError(res.detail ?? "Voice preview failed.");
          throw new Error(res.detail ?? "Voice preview failed.");
        }
      } else {
        const text = language === "ne" ? "नमस्ते, Piper आवाज परीक्षण सफल भयो।" : "Hello, Piper voice test was successful.";
        const result = await testTts(text, language);
        const audioUrl = absoluteAudioUrl(result.audio_url);
        setLastAudioUrl(audioUrl);
        if (audioUrl) {
          await new Audio(audioUrl).play();
        }
      }
    } catch (ttsError) {
      setError(ttsError instanceof Error ? ttsError.message : "TTS test failed.");
      throw ttsError; // let callers (e.g. Voice Studio) show their own busy/error UI
    } finally {
      setTtsTestingId(null);
    }
  }

  async function startDatasetRecording(promptId: string) {
    if (datasetRecorderRef.current?.state === "recording") {
      stopDatasetRecording();
      return;
    }
    const prompt = prompts.find((item) => item.id === promptId);
    if (!prompt) {
      return;
    }
    setDatasetActivePrompt(promptId);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      datasetRecorderRef.current = recorder;
      datasetChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          datasetChunksRef.current.push(event.data);
        }
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(datasetChunksRef.current, { type: mimeType });
        datasetChunksRef.current = [];
        datasetRecorderRef.current = null;
        setDatasetActivePrompt(null);
        if (blob.size < 1024) {
          return;
        }
        const { wavBlob } = await scoreRecording(blob);
        try {
          const rec = await uploadDatasetRecording(prompt.id, wavBlob);
          setDatasetRecordings((items) => [
            ...items.filter((item) => item.id !== prompt.id),
            rec
          ]);
          setError(null);
          logEvent("success", "dataset_recording_saved", `Dataset recording saved for ${prompt.id}.`);
        } catch (uploadError) {
          setError(uploadError instanceof Error ? uploadError.message : "Failed to save recording to backend.");
          logEvent("error", "dataset_recording_failed", uploadError instanceof Error ? uploadError.message : "Failed to save recording to backend.");
        }
      };
      recorder.start();
    } catch (datasetError) {
      setDatasetActivePrompt(null);
      setError(datasetError instanceof Error ? datasetError.message : "Dataset recording failed.");
      logEvent("error", "dataset_recording_failed", datasetError instanceof Error ? datasetError.message : "Dataset recording failed.");
    }
  }

  function stopDatasetRecording() {
    const recorder = datasetRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
    }
  }

  async function deletePrompt(promptId: string) {
    try {
      await deleteDatasetRecording(promptId);
      setDatasetRecordings((items) => items.filter((item) => item.id !== promptId));
      setError(null);
      logEvent("warning", "dataset_recording_deleted", `Dataset recording deleted for ${promptId}.`);
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Failed to delete recording.");
      logEvent("error", "dataset_recording_delete_failed", deleteError instanceof Error ? deleteError.message : "Failed to delete recording.");
    }
  }

  function exportDatasetZip() {
    window.open(deriveDatasetExportUrl(), "_blank");
  }

  async function saveSettings(nextSettings: BackendSettings) {
    try {
      setSettings(await updateSettings(nextSettings));
      setError(null);
      logEvent("success", "settings_saved", "Settings saved successfully.");
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Settings save failed.");
      logEvent("error", "settings_save_failed", saveError instanceof Error ? saveError.message : "Settings save failed.");
    }
  }

  async function clearLocalData() {
    try {
      await deleteLocalData();
      setHistory([]);
      localStorage.removeItem("swarlocal.history");
      setLastAudioUrl(null);
      setError(null);
      logEvent("warning", "local_data_deleted", "Local turn data and browser chat history were cleared.");
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Local data delete failed.");
      logEvent("error", "local_data_delete_failed", deleteError instanceof Error ? deleteError.message : "Local data delete failed.");
    }
  }

  function applyPreset(preset: "fast" | "quality" | "noisy" | "cloned" | "local") {
    if (preset === "fast") {
      setUseInternet(false);
      setSelectedBrain("openai");
      setSelectedSttProvider("openai");
      setSelectedVoiceId("openai-alloy");
      setAutoVad(false);
      void saveSettings({ ...settings, llm_provider: "openai", stt_provider: "openai", tts_provider: "openai", low_latency_mode: true, quality_mode: false });
    }
    if (preset === "quality") {
      setAutoVad(true);
      void saveSettings({ ...settings, low_latency_mode: false, quality_mode: true });
    }
    if (preset === "noisy") {
      setAutoVad(false);
      void saveSettings({ ...settings, max_recording_seconds: 20, low_latency_mode: true });
    }
    if (preset === "cloned") {
      void saveSettings({ ...settings, force_selected_voice: true, fallback_allowed: false });
    }
    if (preset === "local") {
      setSelectedBrain("local");
      setUseInternet(false);
      void saveSettings({ ...settings, llm_provider: "local", cloud_fallback_to_local: true });
    }
  }

  const renderedView = useMemo(() => {
    switch (activeView) {
      case "setup":
        return (
          <SetupView
            providerStatus={providerStatus}
            voiceSocketState={voiceSocketState}
            voiceSocketStatus={voiceSocketStatus}
            onRefresh={refreshStatus}
            onTestVoiceSocket={testVoiceSocket}
            onTtsTest={runTtsTest}
          />
        );
      case "voice_studio":
        return (
          <VoiceStudioView
            voices={voices}
            galleryVoices={galleryVoices}
            auditLogs={auditLogs}
            onRefresh={refreshStatus}
            onTtsTest={runTtsTest}
            ttsTestingId={ttsTestingId}
          />
        );
      case "knowledge":
        return (
          <KnowledgeView
            settings={settings}
            ragStatus={ragStatus}
            ragCollections={ragCollections}
            selectedKnowledgeId={selectedKnowledgeId}
            onSelectKnowledge={setSelectedKnowledgeId}
            onSave={saveSettings}
          />
        );
      case "evaluation":
        const currentRatings = history[0]?.ratings ?? {
          naturalness: 3,
          voiceSimilarity: 3,
          nepaliPronunciation: 3,
          englishPronunciation: 3
        };
        return <EvaluationView ratings={currentRatings} onChange={handleRatingChange} history={history} />;
      case "admin":
        return (
          <AdminLoginGate>
          <AdminView
            providerStatus={providerStatus}
            voiceSocketStatus={voiceSocketStatus}
            systemMetrics={systemMetrics}
            systemInfo={systemInfo}
            voices={voices}
            galleryVoices={galleryVoices}
            auditLogs={auditLogs}
            settings={settings}
          />
          </AdminLoginGate>
        );
      case "logs":
        return (
          <LogsView
            appLogs={appLogs}
            auditLogs={auditLogs}
            providerStatus={providerStatus}
            voiceSocketStatus={voiceSocketStatus}
            onClear={() => setAppLogs([])}
          />
        );
      case "system_map":
        return <SystemExplorerView />;
      case "settings":
        return <SettingsView settings={settings} voices={voices} onSave={saveSettings} onDelete={clearLocalData} onTtsTest={runTtsTest} />;
      default:
        return (
          <ConversationView
            status={status}
            latestTurn={latestTurn}
            history={history}
            manualText={manualText}
            autoVad={autoVad}
            settings={settings}
            voiceSocketState={voiceSocketState}
            voiceSocketStatus={voiceSocketStatus}
            lastAudioUrl={lastAudioUrl}
            onManualText={setManualText}
            onSendText={sendText}
            onToggleVad={() => setAutoVad((value) => !value)}
            onRecord={startRecording}
            onStop={stopRecording}
            onInterrupt={interruptPlayback}
            onReplay={replayLastAnswer}
            voices={voices}
            selectedVoiceId={selectedVoiceId}
            onSelectVoice={setSelectedVoiceId}
            selectedKnowledgeId={selectedKnowledgeId}
            onSelectKnowledge={setSelectedKnowledgeId}
            useInternet={useInternet}
            onToggleInternet={() => setUseInternet((value) => !value)}
            ragCollections={ragCollections}
            selectedBrain={selectedBrain}
            onSelectBrain={setSelectedBrain}
            selectedSttProvider={selectedSttProvider}
            onSelectSttProvider={setSelectedSttProvider}
            systemMetrics={systemMetrics}
            onApplyPreset={applyPreset}
            showSettings={showSettings}
            onToggleSettings={() => setShowSettings((v) => !v)}
            playingAudioUrl={playingAudioUrl}
            onPlayAudio={handlePlayTurnAudio}
            onSaveSettings={saveSettings}
            sttLanguage={sttLanguage}
            onSelectSttLanguage={setSttLanguage}
            liveTranscript={liveTranscript}
            liveTranslation={liveTranslation}
            liveLanguage={liveLanguage}
            transcriptFinal={transcriptFinal}
            micLevel={micLevel}
            onCancelRecording={cancelRecording}
          />
        );
    }
  }, [
    activeView,
    autoVad,
    history,
    lastAudioUrl,
    latestTurn,
    manualText,
    providerStatus,
    refreshStatus,
    settings,
    setAutoVad,
    setHistory,
    handleRatingChange,
    status,
    voiceSocketState,
    voiceSocketStatus,
    voices,
    selectedVoiceId,
    setSelectedVoiceId,
    selectedKnowledgeId,
    setSelectedKnowledgeId,
    useInternet,
    setUseInternet,
    selectedBrain,
    systemMetrics,
    setSelectedBrain,
    selectedSttProvider,
    setSelectedSttProvider,
    ragCollections,
    ragStatus,
    galleryVoices,
    auditLogs,
    systemInfo,
    appLogs,
    setAppLogs,
    sttLanguage,
    setSttLanguage,
    liveTranscript,
    liveTranslation,
    liveLanguage,
    transcriptFinal,
    micLevel
  ]);


  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand nabil-brand">
          <div className="brand-mark">N</div>
          <div className="nabil-brand-text">
            <h1 className="nabil-wordmark">NABIL</h1>
            <span>Voice AI</span>
            <div className="nabil-gold-rule" />
          </div>
        </div>
        <nav className="nav-tabs" aria-label="Primary">
          {views.map((view) => {
            const Icon = view.icon;
            return (
              <button
                className={activeView === view.id ? "nav-item active" : "nav-item"}
                key={view.id}
                onClick={() => setActiveView(view.id)}
                type="button"
                title={view.label}
              >
                <Icon size={18} />
                <span>{view.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-status">
          <StatusDot status={status} />
          <span>{statusLabels[status]}</span>
        </div>
      </aside>
      <main className="workspace">
        {error ? (
          <div className="notice error">
            <CircleAlert size={18} />
            <span>{error}</span>
          </div>
        ) : null}
        {renderedView}
      </main>
    </div>
  );
}

function VoiceOverlay({
  status,
  latestTurn,
  liveTranscript,
  liveTranslation,
  liveLanguage,
  transcriptFinal,
  micLevel,
  sttLanguage,
  onSelectSttLanguage,
  onStart,
  onStop,
  onCancel,
  onClose
}: {
  status: AssistantStatus;
  latestTurn?: ConversationTurn;
  liveTranscript: string;
  liveTranslation: string;
  liveLanguage: LanguageCode;
  transcriptFinal: boolean;
  micLevel: number;
  sttLanguage: "auto" | "ne" | "en";
  onSelectSttLanguage: (value: "auto" | "ne" | "en") => void;
  onStart: () => void;
  onStop: () => void;
  onCancel: () => void;
  onClose: () => void;
}) {
  const orbWrapRef = useRef<HTMLDivElement>(null);
  const listening = status === "listening";
  const overlayState: "idle" | "listening" | "thinking" | "speaking" | "error" =
    listening ? "listening"
    : status === "thinking" || status === "transcribing" ? "thinking"
    : status === "speaking" ? "speaking"
    : status === "error" ? "error"
    : "idle";

  // ESC closes the overlay
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Orb mic level is driven by the SAME MediaStream used for recording — App
  // samples it and passes micLevel (0..1) down. We just write it to a CSS var.
  useEffect(() => {
    const level = listening ? Math.max(0, Math.min(1, micLevel)) : 0;
    orbWrapRef.current?.style.setProperty("--voice-level", level.toFixed(3));
  }, [micLevel, listening]);

  const statusText: Record<typeof overlayState, string> = {
    idle: "Ready",
    listening: "Listening",
    thinking: "Thinking",
    speaking: "Speaking",
    error: "Error"
  };

  // Single source of truth for the live display = backend transcript.
  const showLive = liveTranscript.trim().length > 0;
  const langChips: { id: "en" | "ne" | "auto"; label: string }[] = [
    { id: "en", label: "English" },
    { id: "ne", label: "नेपाली" },
    { id: "auto", label: "Mixed" },
  ];

  return (
    <div className={`voice-overlay ${overlayState}`} role="dialog" aria-label="Voice mode">
      <button type="button" className="voice-overlay-close" onClick={onClose} title="Close (Esc)">✕</button>

      {/* Language chip group — sets the WS config stt_language for the next turn */}
      <div className="voice-lang-group" role="group" aria-label="Transcription language">
        {langChips.map((chip) => (
          <button
            key={chip.id}
            type="button"
            className={`voice-lang-chip ${sttLanguage === chip.id ? "active" : ""}`}
            onClick={() => onSelectSttLanguage(chip.id)}
            title="Set transcription language for the next turn"
          >
            {chip.label}
          </button>
        ))}
      </div>

      <div className="voice-orb-wrap" ref={orbWrapRef}>
        <div className="voice-orb-halo" />
        <div className="voice-orb-ripple" />
        <div className="voice-orb-ripple" />
        <div className="voice-orb-ripple" />
        <div className="voice-orb-core">
          {overlayState === "speaking" ? (
            <div className="voice-orb-bars">{[1, 2, 3, 4, 5].map((i) => <i key={i} />)}</div>
          ) : (
            <Mic size={34} color="rgba(255,255,255,0.92)" />
          )}
        </div>
      </div>

      <div className="voice-status-label">
        {statusText[overlayState]}
        {liveLanguage && liveLanguage !== "unknown" && (showLive || overlayState === "thinking") && (
          <span className="voice-lang-tag"> · {liveLanguage}</span>
        )}
      </div>

      {/* Live transcript panel — backend partial/final transcript */}
      <div className="voice-transcript">
        {showLive ? (
          <>
            {liveTranscript}
            {!transcriptFinal && listening && <span className="interim typing-dots"><i /><i /><i /></span>}
          </>
        ) : listening ? (
          <span className="interim">Speak now…</span>
        ) : overlayState === "thinking" ? (
          <span className="interim">Working on it…</span>
        ) : null}
      </div>

      {/* Live translation panel (best-effort, optional) */}
      {liveTranslation && (overlayState === "listening" || overlayState === "thinking") && (
        <div className="voice-translation">{liveTranslation}</div>
      )}
      {!liveTranslation && latestTurn?.transcript_translation && overlayState !== "listening" && (
        <div className="voice-translation">{latestTurn.transcript_translation}</div>
      )}

      {(overlayState === "speaking" || overlayState === "idle") && latestTurn?.response && (
        <div className="voice-reply">
          {latestTurn.response}
          {latestTurn.response_translation && (
            <div className="voice-reply-translation">{latestTurn.response_translation}</div>
          )}
        </div>
      )}

      <div className="voice-overlay-controls">
        {listening ? (
          <>
            <button type="button" className="voice-mic-btn recording" onClick={onStop} title="Stop and send">
              <Square size={26} />
            </button>
            <button type="button" className="voice-cancel-btn" onClick={onCancel} title="Cancel (discard)">
              Cancel
            </button>
          </>
        ) : overlayState === "thinking" ? (
          <button type="button" className="voice-cancel-btn" onClick={onClose}>
            Cancel
          </button>
        ) : (
          <>
            <button type="button" className="voice-mic-btn" onClick={onStart} title="Start speaking">
              <Mic size={26} />
            </button>
            <button type="button" className="voice-cancel-btn" onClick={onClose} title="Close">
              Close
            </button>
          </>
        )}
      </div>

      <div className="voice-overlay-hint">
        {listening ? "Tap the square to stop & send" : overlayState === "thinking" ? "Transcribing & generating…" : overlayState === "speaking" ? "Playing response…" : "Tap the mic to speak"}
      </div>
    </div>
  );
}

function ConversationView({
  status,
  latestTurn,
  history,
  manualText,
  autoVad,
  settings,
  voiceSocketState,
  voiceSocketStatus,
  lastAudioUrl,
  onManualText,
  onSendText,
  onToggleVad,
  onRecord,
  onStop,
  onInterrupt,
  onReplay,
  voices,
  selectedVoiceId,
  onSelectVoice,
  selectedKnowledgeId,
  onSelectKnowledge,
  useInternet,
  onToggleInternet,
  ragCollections,
  selectedBrain,
  onSelectBrain,
  selectedSttProvider,
  onSelectSttProvider,
  systemMetrics,
  onApplyPreset,
  showSettings,
  onToggleSettings,
  playingAudioUrl,
  onPlayAudio,
  onSaveSettings,
  sttLanguage,
  onSelectSttLanguage,
  liveTranscript,
  liveTranslation,
  liveLanguage,
  transcriptFinal,
  micLevel,
  onCancelRecording
}: {
  status: AssistantStatus;
  latestTurn?: ConversationTurn;
  history: ConversationTurn[];
  manualText: string;
  autoVad: boolean;
  settings: BackendSettings;
  voiceSocketState: VoiceSocketConnectionState;
  voiceSocketStatus: VoiceSocketStatus | null;
  lastAudioUrl: string | null;
  onManualText: (value: string) => void;
  onSendText: () => void;
  onToggleVad: () => void;
  onRecord: () => void;
  onStop: () => void;
  onInterrupt: () => void;
  onReplay: () => void;
  voices: VoicesResponse | null;
  selectedVoiceId: string;
  onSelectVoice: (value: string) => void;
  selectedKnowledgeId: string;
  onSelectKnowledge: (value: string) => void;
  useInternet: boolean;
  onToggleInternet: () => void;
  ragCollections: any[];
  selectedBrain: string;
  onSelectBrain: (value: string) => void;
  selectedSttProvider: string;
  onSelectSttProvider: (value: string) => void;
  systemMetrics: SystemMetrics | null;
  onApplyPreset: (preset: "fast" | "quality" | "noisy" | "cloned" | "local") => void;
  showSettings: boolean;
  onToggleSettings: () => void;
  playingAudioUrl: string | null;
  onPlayAudio: (url: string | null | undefined) => void;
  onSaveSettings: (next: BackendSettings) => void;
  sttLanguage: "auto" | "ne" | "en";
  onSelectSttLanguage: (value: "auto" | "ne" | "en") => void;
  liveTranscript: string;
  liveTranslation: string;
  liveLanguage: LanguageCode;
  transcriptFinal: boolean;
  micLevel: number;
  onCancelRecording: () => void;
}) {

  // -------- local KB collections (from KnowledgeView's refresh) --------
  const [localKbCollections, setLocalKbCollections] = useState<KBCollection[]>([]);
  useEffect(() => {
    getKBCollections().then(setLocalKbCollections).catch(() => {});
  }, []);

  const recording = status === "listening";
  const socketBlocked = voiceSocketStatus ? !voiceSocketStatus.capabilities.audio_turns : false;
  const builtInVoiceIds = ["ne_NP-chitwan-medium", "ne_NP-google-medium", "en_US-lessac-medium", "en_US-ryan-medium"];
  const customVoices = voices?.voices?.filter((voice) => !builtInVoiceIds.includes(voice.id) && !voice.id.startsWith("openai-")) ?? [];
  const openaiVoices = voices?.voices?.filter((voice) => voice.id.startsWith("openai-")) ?? [];
  const selectedVoice = voices?.voices?.find((voice) => voice.id === selectedVoiceId);
  const allKbCollections = [...ragCollections, ...localKbCollections.filter(lk => !ragCollections.find((r: any) => r.id === lk.id))];
  const selectedKnowledge = allKbCollections.find((c: any) => c.id === selectedKnowledgeId);
  const activeVoiceName = latestTurn?.actual_voice_name ||
    (selectedVoiceId.startsWith("openai-") ? `OpenAI ${selectedVoiceId.split("-")[1]?.charAt(0).toUpperCase() + selectedVoiceId.split("-")[1]?.slice(1)}` : "") ||
    selectedVoice?.name || selectedVoice?.id || "Auto bilingual";
  const orbHint: Record<AssistantStatus, string> = {
    idle: socketBlocked ? "Setup required" : "Click orb or type below",
    listening: autoVad ? "Speak now — auto stop on silence" : "Recording… click to stop",
    transcribing: "Converting speech to text…",
    thinking: selectedKnowledgeId !== "none" ? "Searching knowledge base…" : "Generating response…",
    speaking: "Speaking — click to interrupt",
    error: "Something went wrong",
  };
  // ---- voice mode overlay ----
  const [voiceOverlayOpen, setVoiceOverlayOpen] = useState(false);
  // ---- chat textarea auto-expand ----
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history.length]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSendText();
    }
  };

  // Chronological order for display (oldest → newest)
  const chronoHistory = [...history].reverse();

  const orbColorMap: Record<AssistantStatus, string> = {
    idle: "var(--teal)", listening: "#4ade80", transcribing: "#818cf8",
    thinking: "#c084fc", speaking: "#fb923c", error: "#f87171"
  };
  const orbColor = orbColorMap[status];

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", background: "var(--bg)" }}>

      {/* ── LEFT SIDEBAR: Config ── */}
      {showSettings && (
        <div className="conv-config" style={{ width: 240, flexShrink: 0 }}>
          {/* Status pill */}
          <div style={{
            padding: "7px 10px", borderRadius: 8, marginBottom: 10,
            background: socketBlocked ? "rgba(248,113,113,0.08)" : "rgba(74,222,128,0.07)",
            border: `1px solid ${socketBlocked ? "rgba(248,113,113,0.2)" : "rgba(74,222,128,0.15)"}`,
            display: "flex", alignItems: "center", gap: 6, fontSize: 12
          }}>
            <StatusDot status={socketBlocked ? "error" : status} />
            <span style={{ fontWeight: 600, color: socketBlocked ? "var(--rose)" : "var(--green)" }}>
              {socketBlocked ? "Setup needed" : voiceSocketState === "ready" ? "Voice ready" : voiceSocketState.replace(/_/g, " ")}
            </span>
          </div>

          {/* Quick presets */}
          <p style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", margin: "0 0 6px" }}>Presets</p>
          <div className="preset-grid" style={{ marginBottom: 14 }}>
            <button className="preset-btn" type="button" onClick={() => onApplyPreset("fast")}><Sparkles size={12} /><span>Fast</span></button>
            <button className="preset-btn" type="button" onClick={() => onApplyPreset("quality")}><Gauge size={12} /><span>Quality</span></button>
            <button className="preset-btn" type="button" onClick={() => onApplyPreset("noisy")}><Activity size={12} /><span>Noisy</span></button>
            <button className="preset-btn" type="button" onClick={() => onApplyPreset("local")}><Cpu size={12} /><span>Local</span></button>
          </div>

          <div className="ctrl-group">
            <label style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", display: "block", marginBottom: 5 }}>🧠 Brain / LLM</label>
            <select value={selectedBrain} onChange={(e) => onSelectBrain(e.target.value)} style={{ fontSize: 12 }}>
              <option value="openai">OpenAI Cloud</option>
              <option value="local">Local AI (Ollama)</option>
              <option value="gemini">Google Gemini</option>
              <option value="auto">Auto - best available</option>
            </select>
          </div>

          <div className="ctrl-group">
            <label style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", display: "flex", alignItems: "center", gap: 5, marginBottom: 5 }}>
              <LifeBuoy size={12} /> Fallback brain
            </label>
            <select
              value={settings.cloud_fallback_to_local ? "local" : "none"}
              onChange={(e) => onSaveSettings({ ...settings, cloud_fallback_to_local: e.target.value === "local" })}
              style={{ fontSize: 12 }}
            >
              <option value="local">Local AI (Ollama) — recommended</option>
              <option value="none">None — show an error</option>
            </select>
            <p style={{ fontSize: 10, color: "var(--muted)", margin: "4px 0 0", lineHeight: 1.5 }}>
              Used only if the main brain fails or times out, so chat keeps working.
            </p>
          </div>

          <div className="ctrl-group">
            <label style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", display: "block", marginBottom: 5 }}>🎙️ Voice</label>
            <select value={selectedVoiceId} onChange={(e) => onSelectVoice(e.target.value)} style={{ fontSize: 12 }}>
              <option value="openai-alloy">OpenAI Alloy</option>
              <option value="auto">Auto bilingual</option>
              {openaiVoices.length > 0 && <optgroup label="OpenAI Cloud">
                {openaiVoices.filter(v => v.id !== "openai-alloy").map(v => <option key={v.id} value={v.id} disabled={!!v.disabled_reason}>{v.name || v.id}{v.disabled_reason ? " - key missing" : ""}</option>)}
              </optgroup>}
              {customVoices.length > 0 && <optgroup label="My cloned voices">
                {customVoices.map(v => <option key={v.id} value={v.id} disabled={!v.model_exists}>{v.name || v.id}{v.model_exists ? "" : " - untrained"}</option>)}
              </optgroup>}
              <optgroup label="Built-in (Piper)">
                <option value="ne_NP-chitwan-medium">Chitwan Nepali</option>
                <option value="en_US-lessac-medium">Lessac English</option>
                <option value="ne_NP-google-medium">Google Nepali</option>
                <option value="en_US-ryan-medium">Ryan English</option>
              </optgroup>
            </select>
          </div>

          <div className="ctrl-group">
            <label style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", display: "block", marginBottom: 5 }}>🎙️ Transcribing / STT</label>
            <select value={selectedSttProvider} onChange={(e) => onSelectSttProvider(e.target.value)} style={{ fontSize: 12 }}>
              <option value="openai">OpenAI Cloud</option>
              <option value="local">Local Whisper</option>
            </select>
          </div>

          <div className="ctrl-group">
            <label style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", display: "block", marginBottom: 5 }}>🗣️ STT Language Mode</label>
            <select value={sttLanguage} onChange={(e) => onSelectSttLanguage(e.target.value as "auto" | "ne" | "en")} style={{ fontSize: 12 }}>
              <option value="auto">Auto Detect</option>
              <option value="ne">Strict Nepali</option>
              <option value="en">Strict English</option>
            </select>
          </div>



          <div className="ctrl-group">
            <label style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", display: "block", marginBottom: 5 }}>📚 Knowledge</label>
            <select value={selectedKnowledgeId} onChange={(e) => onSelectKnowledge(e.target.value)} style={{ fontSize: 12 }}>
              <option value="none">Auto-check Nabil KB</option>
              {allKbCollections.map((c: any) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
          </div>

          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, cursor: "pointer", padding: "7px 10px", border: "1px solid var(--line)", borderRadius: 6, background: useInternet ? "var(--teal-soft)" : "var(--panel)", marginTop: 4 }}>
            <input type="checkbox" checked={useInternet} onChange={onToggleInternet} style={{ width: "auto" }} />
            <Globe size={13} />
            <span>{useInternet ? "Internet ON" : "Internet OFF"}</span>
          </label>

          {systemMetrics && (
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
              {typeof systemMetrics.cpu_percent === "number" && <div className="metric-item"><span className="val">{systemMetrics.cpu_percent.toFixed(0)}%</span><span className="lbl">CPU</span></div>}
              {typeof systemMetrics.ram_used_gb === "number" && <div className="metric-item"><span className="val">{systemMetrics.ram_used_gb.toFixed(1)}G</span><span className="lbl">RAM</span></div>}
            </div>
          )}
        </div>
      )}

      {/* ── MAIN CHAT AREA ── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>

        {/* Chat top bar */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8, padding: "10px 16px",
          borderBottom: "1px solid var(--line)", background: "var(--surface-2)", flexShrink: 0, flexWrap: "wrap"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            {/* Animated status dot in top bar */}
            <div style={{
              width: 10, height: 10, borderRadius: "50%", background: orbColor, flexShrink: 0,
              boxShadow: status !== "idle" ? `0 0 6px ${orbColor}` : "none",
              animation: status === "listening" ? "pulse-red 1s infinite" : status === "thinking" || status === "transcribing" ? "orb-breathe 1.2s infinite" : "none"
            }} />
            <span style={{ fontWeight: 600, fontSize: 13, color: "var(--ink)" }}>Nabil Voice AI</span>
            <span style={{ fontSize: 11, color: orbColor, fontWeight: 600 }}>{statusLabels[status]}</span>
          </div>
          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
            <span className="pill" style={{ fontSize: 10 }}>🧠 {selectedBrain === "local" ? "Local" : selectedBrain === "openai" ? "OpenAI" : selectedBrain === "gemini" ? "Gemini" : "Auto"}</span>
            <span className="pill" style={{ fontSize: 10 }}>🎙 {activeVoiceName}</span>
            {selectedKnowledgeId !== "none" && <span className="pill" style={{ fontSize: 10 }}>📚 {selectedKnowledge?.name || "KB"}</span>}
            {latestTurn?.rag_used && <span className="pill good" style={{ fontSize: 10 }}>RAG</span>}
            {latestTurn?.internet_used && <span className="pill good" style={{ fontSize: 10 }}>🌐</span>}
          </div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
            {history.length > 0 && <span style={{ fontSize: 11, color: "var(--muted)" }}>{history.length} messages</span>}
            <button type="button" className="btn sm" onClick={onToggleSettings}
              style={{ background: showSettings ? "var(--teal-soft)" : undefined, color: showSettings ? "var(--teal)" : undefined, borderColor: showSettings ? "rgba(0,166,81,0.3)" : undefined }}>
              <SlidersHorizontal size={13} />
              {showSettings ? "Hide panel" : "Configure"}
            </button>
          </div>
        </div>

        {/* Active status banner — clean, contained indicator while not idle */}
        {status !== "idle" && (
          <div className={`status-banner ${status}`}>
            <div
              className="status-orb"
              onClick={recording ? onStop : status === "speaking" ? onInterrupt : undefined}
              style={{ cursor: recording || status === "speaking" ? "pointer" : "default" }}
              title={recording ? "Click to stop" : status === "speaking" ? "Click to interrupt" : undefined}
            >
              {status === "listening" || status === "speaking" ? (
                <div className="status-wave">{[0, 1, 2, 3, 4].map((i) => <i key={i} />)}</div>
              ) : status === "thinking" || status === "transcribing" ? (
                <RefreshCw size={20} className="status-spin" />
              ) : status === "error" ? (
                <CircleAlert size={20} />
              ) : (
                <Mic size={20} />
              )}
            </div>
            <div className="status-meta">
              <div className="status-title">{statusLabels[status]}</div>
              <div className="status-sub">{orbHint[status]}</div>
              {status === "transcribing" && latestTurn?.transcript && (
                <div className="status-transcript">“{latestTurn.transcript}”</div>
              )}
            </div>
            {status === "speaking" && (
              <button type="button" className="btn sm" onClick={onInterrupt}
                style={{ marginLeft: "auto", color: "var(--rose)", borderColor: "var(--rose)" }}>
                <Square size={12} /> Interrupt
              </button>
            )}
          </div>
        )}

        {/* Messages area */}
        <div className="chat-messages" style={{ flex: 1, overflowY: "auto", padding: "20px 16px", display: "flex", flexDirection: "column", gap: 20 }}>
          {!chronoHistory.length ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", flex: 1, gap: 12, color: "var(--muted)" }}>
              <div style={{
                width: 72, height: 72, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center",
                background: "var(--panel)", border: "2px solid var(--teal-soft)", color: "var(--teal)"
              }}>
                <Radio size={28} />
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontWeight: 600, fontSize: 15, color: "var(--ink)", marginBottom: 4 }}>Start a conversation</div>
                <div style={{ fontSize: 13 }}>Type a message below, press the mic to speak, or try one of these:</div>
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center", maxWidth: 480, marginTop: 4 }}>
                {[
                  "What savings accounts do you offer?",
                  "How do I open an account?",
                  "Tell me about your loan types",
                  "नबिल बैंकको सेवाहरू के के हुन्?",
                ].map((q) => (
                  <button key={q} type="button" className="example-chip"
                    onClick={() => { onManualText(q); textareaRef.current?.focus(); }}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            chronoHistory.map((turn, index) => (
              <div key={`${turn.created_at ?? index}-${index}`} className="chat-turn" style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {/* User bubble */}
                <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "flex-end", gap: 8 }}>
                  {turn.user_audio_url && (
                    <button className={`play-btn ${playingAudioUrl === turn.user_audio_url ? "playing" : ""}`} type="button"
                      onClick={() => onPlayAudio(turn.user_audio_url)} title="Play your recording" style={{ flexShrink: 0 }}>
                      {playingAudioUrl === turn.user_audio_url ? <Square size={11} /> : <Play size={11} />}
                    </button>
                  )}
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4, maxWidth: "75%" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                      {turn.input_language && <span className="pill" style={{ fontSize: 10 }}>{turn.input_language}</span>}
                      <span style={{ fontSize: 11, fontWeight: 600, color: "var(--teal)" }}>You</span>
                    </div>
                    <div style={{
                      background: "linear-gradient(135deg, rgba(0,166,81,0.18) 0%, rgba(0,166,81,0.10) 100%)",
                      border: "1px solid rgba(0,166,81,0.25)", borderRadius: "16px 16px 4px 16px",
                      padding: "10px 14px", fontSize: 14, lineHeight: 1.55, color: "var(--ink)",
                      wordBreak: "break-word"
                    }}>
                      {turn.transcript}
                    </div>
                    {turn.transcript_translation && (
                      <div className="translation-note user-translation">
                        {turn.transcript_translation}
                      </div>
                    )}
                    {turn.created_at && <span className="msg-time">{new Date(turn.created_at).toLocaleTimeString()}</span>}
                  </div>
                  <div className="chat-avatar user">👤</div>
                </div>

                {/* Assistant bubble */}
                {!turn.is_pending && (
                  <div style={{ display: "flex", alignItems: "flex-end", gap: 8 }}>
                    <div className="chat-avatar assistant">🏦</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4, maxWidth: "80%", flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap" }}>
                        <span style={{ fontSize: 11, fontWeight: 600, color: "var(--amber)" }}>Nabil Voice AI</span>
                        {turn.response_language && <span className="pill" style={{ fontSize: 10 }}>{turn.response_language}</span>}
                        {turn.llm_provider && <span className="pill" style={{ fontSize: 10 }}>🧠 {turn.llm_provider}</span>}
                        {turn.rag_used && <span className="pill good" style={{ fontSize: 10 }}>RAG</span>}
                        {turn.internet_used && <span className="pill good" style={{ fontSize: 10 }}>🌐</span>}
                        {turn.created_at && <span className="msg-time">{new Date(turn.created_at).toLocaleTimeString()}</span>}
                      </div>
                      <div style={{
                        background: "var(--panel)", border: "1px solid rgba(0,166,81,0.18)",
                        borderRadius: "4px 16px 16px 16px", padding: "10px 14px",
                        fontSize: 14, lineHeight: 1.6, color: "var(--ink)", wordBreak: "break-word"
                      }}>

                        {index === chronoHistory.length - 1 ? (
                          <TypingText text={turn.response} />
                        ) : (
                          turn.response
                        )}
                        {turn.response_translation && (
                          <div className="translation-note assistant-translation">
                            {turn.response_translation}
                          </div>
                        )}
                        {/* Footer */}
                        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", marginTop: 8, paddingTop: 8, borderTop: "1px solid var(--line-2)", fontSize: 11, color: "var(--muted)" }}>
                          {turn.actual_voice_name && <span title={turn.actual_voice_id ?? undefined}>🗣 {turn.actual_voice_name}</span>}
                          {turn.timings && <span style={{ color: latencyColor(turn.timings.total_turn_ms), fontWeight: 600 }}>⏱ {formatMs(turn.timings.total_turn_ms)}</span>}
                          {turn.fallback_used && <span className="pill warn" style={{ fontSize: 10 }}>Fallback</span>}
                          {turn.citations?.length ? <span>📎 {turn.citations.length}</span> : null}
                          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 6 }}>
                            <button className="msg-action" type="button" title="Copy response"
                              onClick={() => { try { navigator.clipboard?.writeText(turn.response); } catch { /* ignore */ } }}>
                              <Copy size={12} />
                            </button>
                            {turn.audio_url && (
                              <button className={`play-btn ${playingAudioUrl === absoluteAudioUrl(turn.audio_url) ? "playing" : ""}`} type="button"
                                onClick={() => onPlayAudio(turn.audio_url)} title="Play audio response">
                                {playingAudioUrl === absoluteAudioUrl(turn.audio_url) ? <Square size={12} /> : <Play size={12} />}
                                <span style={{ fontSize: 11, marginLeft: 3 }}>{playingAudioUrl === absoluteAudioUrl(turn.audio_url) ? "Stop" : "Play"}</span>
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

              </div>
            ))
          )}
          {/* Thinking indicator */}
          {(status === "thinking" || status === "transcribing") && (
            <div style={{ display: "flex", alignItems: "flex-end", gap: 8 }}>
              <div className="chat-avatar assistant">🏦</div>
              <div style={{ background: "var(--panel)", border: "1px solid rgba(0,166,81,0.18)", borderRadius: "4px 16px 16px 16px", padding: "12px 16px" }}>
                <div className="typing-dots"><i /><i /><i /></div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input bar */}
        <div style={{
          padding: "12px 16px", borderTop: "1px solid var(--line)", background: "var(--surface-2)",
          flexShrink: 0, display: "flex", flexDirection: "column", gap: 8
        }}>
          {/* Voice controls row */}
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button type="button"
              className={`btn ${autoVad ? "primary" : ""}`}
              onClick={onToggleVad}
              style={{ fontSize: 11, padding: "4px 10px", height: 30, borderRadius: 6 }}>
              <Activity size={12} />
              {autoVad ? "Auto-Detect" : "Push-to-Talk"}
            </button>
            {status === "speaking" && (
              <button type="button" className="btn" onClick={onInterrupt} style={{ fontSize: 11, padding: "4px 10px", height: 30, borderRadius: 6, color: "var(--rose)", borderColor: "var(--rose)" }}>
                <Square size={12} /> Interrupt
              </button>
            )}
            {lastAudioUrl && (
              <button type="button" className="btn" onClick={onReplay} style={{ fontSize: 11, padding: "4px 10px", height: 30, borderRadius: 6 }}>
                <Play size={12} /> Replay
              </button>
            )}
            {socketBlocked && <span style={{ fontSize: 11, color: "var(--rose)" }}>⚠ Voice setup needed</span>}
          </div>
          {/* Text input row */}
          <div className="chat-input-bar" style={{ padding: 0, border: "none", background: "none", gap: 8 }}>
            <textarea
              ref={textareaRef}
              value={manualText}
              onChange={e => {
                onManualText(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
              }}
              onKeyDown={handleKeyDown}
              placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
              rows={1}
              style={{ resize: "none", flex: 1, borderRadius: 10, fontSize: 14, lineHeight: 1.5 }}
            />
            <button
              type="button"
              className={`voice-btn btn${recording ? " recording" : ""}`}
              onClick={() => {
                setVoiceOverlayOpen(true);
                if (!recording) onRecord();
              }}
              disabled={!recording && socketBlocked}
              title={recording ? "Open voice mode" : "Voice input"}
            >
              {recording ? <Square size={16} /> : <Mic size={16} />}
            </button>
            <button
              type="button" className="send-btn btn"
              onClick={onSendText}
              disabled={!manualText.trim() || status === "thinking"}
              title="Send message"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Voice mode overlay */}
      {voiceOverlayOpen && (
        <VoiceOverlay
          status={status}
          latestTurn={latestTurn}
          liveTranscript={liveTranscript}
          liveTranslation={liveTranslation}
          liveLanguage={liveLanguage}
          transcriptFinal={transcriptFinal}
          micLevel={micLevel}
          sttLanguage={sttLanguage}
          onSelectSttLanguage={onSelectSttLanguage}
          onStart={onRecord}
          onStop={onStop}
          onCancel={() => {
            if (recording) onCancelRecording();
            setVoiceOverlayOpen(false);
          }}
          onClose={() => {
            if (recording) onCancelRecording();
            setVoiceOverlayOpen(false);
          }}
        />
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: extension → { emoji, className, chip }
// ─────────────────────────────────────────────────────────────────────────────
function extInfo(filename: string, sourceType?: string) {
  if (sourceType === "url") return { emoji: "🌐", cls: "ext-url", chip: "url", label: "URL" };
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  const map: Record<string, { emoji: string; cls: string; chip: string; label: string }> = {
    pdf:  { emoji: "📄", cls: "ext-pdf",  chip: "pdf",  label: "PDF" },
    docx: { emoji: "📝", cls: "ext-docx", chip: "docx", label: "DOCX" },
    doc:  { emoji: "📝", cls: "ext-doc",  chip: "doc",  label: "DOC" },
    txt:  { emoji: "📃", cls: "ext-txt",  chip: "txt",  label: "TXT" },
    md:   { emoji: "📖", cls: "ext-md",   chip: "md",   label: "MD" },
    html: { emoji: "🌍", cls: "ext-html", chip: "html", label: "HTML" },
    htm:  { emoji: "🌍", cls: "ext-htm",  chip: "htm",  label: "HTM" },
    csv:  { emoji: "📊", cls: "ext-csv",  chip: "csv",  label: "CSV" },
    json: { emoji: "🔧", cls: "ext-json", chip: "json", label: "JSON" },
  };
  return map[ext] ?? { emoji: "📎", cls: "ext-other", chip: "other", label: ext.toUpperCase() || "FILE" };
}

function fmtBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main RAG / KnowledgeView component
// ─────────────────────────────────────────────────────────────────────────────
function KnowledgeView({
  settings,
  selectedKnowledgeId,
  onSelectKnowledge,
  onSave
}: {
  settings: BackendSettings;
  ragStatus: RagStatus | null;
  ragCollections: any[];
  selectedKnowledgeId: string;
  onSelectKnowledge: (value: string) => void;
  onSave: (settings: BackendSettings) => void;
}) {
  // ── Core ──────────────────────────────────────────────────────────────────
  const [kbStatus, setKbStatus] = useState<KBStatus | null>(null);
  const [collections, setCollections] = useState<KBCollection[]>([]);
  const [selectedCol, setSelectedCol] = useState<KBCollection | null>(null);
  const [documents, setDocuments] = useState<KBDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // True when getKBStatus() returned null — the KB service is unreachable.
  const [serviceDown, setServiceDown] = useState(false);

  // ── Panel mode: null = docs grid ─────────────────────────────────────────
  const [panel, setPanel] = useState<null | "search" | "analytics" | "config" | "newcol" | "eval">(null);

  // ── Evaluation & document chat ──────────────────────────────────────────
  const [evalDocId, setEvalDocId] = useState<string>("");          // "" = whole collection
  const [evalAnswerModel, setEvalAnswerModel] = useState("local");
  const [evalVerifyModel, setEvalVerifyModel] = useState("openai");
  const [evalTemp, setEvalTemp] = useState(0.2);
  const [evalN, setEvalN] = useState(5);
  const [evalSpeak, setEvalSpeak] = useState(false);
  const [evalQuestions, setEvalQuestions] = useState<EvalQA[]>([]);
  const [evalResults, setEvalResults] = useState<EvalRow[] | null>(null);
  const [evalSummary, setEvalSummary] = useState<{ accuracy: number; counts: Record<string, number>; answer_model: string; verify_model: string } | null>(null);
  const [evalBusy, setEvalBusy] = useState<null | "gen" | "run">(null);
  const [evalErr, setEvalErr] = useState<string | null>(null);
  // Document chat
  const [docChatInput, setDocChatInput] = useState("");
  const [docChatLog, setDocChatLog] = useState<{ role: "user" | "ai"; text: string; rag?: boolean; audio?: string | null }[]>([]);
  const [docChatBusy, setDocChatBusy] = useState(false);

  // ── New collection form ───────────────────────────────────────────────────
  const [newColName, setNewColName] = useState("");
  const [newColDesc, setNewColDesc] = useState("");
  const [creating, setCreating] = useState(false);

  // ── Ingest ────────────────────────────────────────────────────────────────
  const [ingestUrl, setIngestUrl] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ingestMsg, setIngestMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [uploadProgress, setUploadProgress] = useState<{ name: string; status: "pending" | "ok" | "fail" }[]>([]);
  const [dragOver, setDragOver] = useState(false);

  // ── Site crawl ────────────────────────────────────────────────────────────
  const [crawlMode, setCrawlMode] = useState(false);
  const [crawlMaxPages, setCrawlMaxPages] = useState(200);
  const [crawlDelay, setCrawlDelay] = useState(150);
  const [crawling, setCrawling] = useState(false);
  const [crawlEvents, setCrawlEvents] = useState<CrawlEvent[]>([]);
  const crawlAbortRef = useRef<AbortController | null>(null);

  // ── Doc grid ──────────────────────────────────────────────────────────────
  const [docSearch, setDocSearch] = useState("");
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  // ── Detail drawer ─────────────────────────────────────────────────────────
  const [docChunks, setDocChunks] = useState<{ text: string; chunk_index: number }[]>([]);
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [renamingDocId, setRenamingDocId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  // ── Search panel ─────────────────────────────────────────────────────────
  const [searchQuery, setSearchQuery] = useState("");
  const [searchMode, setSearchMode] = useState<"semantic" | "keyword" | "hybrid">("hybrid");
  const [searchNResults, setSearchNResults] = useState(8);
  const [searchMinScore, setSearchMinScore] = useState(0.2);
  const [searchSourceFilter, setSearchSourceFilter] = useState<"" | "file" | "url">("");
  const [searchRerank, setSearchRerank] = useState(false);
  const [searchResults, setSearchResults] = useState<KBSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchMeta, setSearchMeta] = useState<{ mode: string; elapsed_ms: number; reranked: boolean } | null>(null);
  const [queryHistory, setQueryHistory] = useState<string[]>([]);

  // ── Analytics panel ───────────────────────────────────────────────────────
  const [analytics, setAnalytics] = useState<any>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);

  // ── Config panel ──────────────────────────────────────────────────────────
  const [configDraft, setConfigDraft] = useState({
    kb_embedding_provider: settings.kb_embedding_provider || "ollama",
    kb_embedding_model: settings.kb_embedding_model || "nomic-embed-text",
    kb_embedding_model_st: settings.kb_embedding_model_st || "all-MiniLM-L6-v2",
    kb_chunk_size: settings.kb_chunk_size || 512,
    kb_chunk_overlap: settings.kb_chunk_overlap || 50,
    kb_max_results: settings.kb_max_results || 5,
    kb_similarity_threshold: settings.kb_similarity_threshold || 0.3,
    kb_search_mode: settings.kb_search_mode || "hybrid",
    kb_chunk_strategy: settings.kb_chunk_strategy || "sentence",
    kb_reranking_enabled: settings.kb_reranking_enabled ?? false,
    kb_reranking_model: settings.kb_reranking_model || "cross-encoder/ms-marco-MiniLM-L-6-v2",
    kb_query_analytics: settings.kb_query_analytics ?? true,
  });

  // ── Data loading ──────────────────────────────────────────────────────────
  const refresh = async () => {
    setLoading(true);
    try {
      const [status, cols] = await Promise.all([getKBStatus(), getKBCollections()]);
      setKbStatus(status);                       // may be null → "service unavailable" banner
      setServiceDown(status === null);
      setCollections(Array.isArray(cols) ? cols : []);
      if (selectedCol) {
        const docs = await getKBDocuments(selectedCol.id);
        setDocuments(Array.isArray(docs) ? docs : []);
      }
      setError(null);
    } catch (e: any) {
      setServiceDown(true);
      setError(e?.message ?? "Knowledge service unavailable.");
    }
    finally { setLoading(false); }
  };

  useEffect(() => { void refresh(); }, []);

  const openCollection = async (col: KBCollection) => {
    setSelectedCol(col);
    setSelectedDocId(null);
    setPanel(null);
    onSelectKnowledge(col.id);
    try {
      const docs = await getKBDocuments(col.id);
      setDocuments(Array.isArray(docs) ? docs : []);
    }
    catch (e: any) { setError(e?.message ?? "Failed to load documents."); }
  };

  // ── Collection create ─────────────────────────────────────────────────────
  const handleCreateCollection = async () => {
    if (!newColName.trim()) return;
    setCreating(true);
    try {
      const col = await createKBCollection(newColName.trim(), newColDesc.trim());
      setNewColName(""); setNewColDesc("");
      setCollections(prev => [...(prev ?? []), col]);
      setPanel(null);
      setError(null);              // clear any stale error on success
      await openCollection(col);   // await so docs load before we settle
    } catch (e: any) { setError(e?.message ?? "Failed to create collection."); }
    finally { setCreating(false); }
  };

  const handleDeleteCollection = async (colId: string) => {
    if (!confirm("Delete this knowledge base and ALL its documents?")) return;
    try {
      await deleteKBCollection(colId);
      setCollections(prev => prev.filter(c => c.id !== colId));
      if (selectedCol?.id === colId) { setSelectedCol(null); setDocuments([]); }
    } catch (e: any) { setError(e.message); }
  };

  // ── File upload ───────────────────────────────────────────────────────────
  const doUpload = async (files: File[]) => {
    if (!selectedCol || !files.length) return;
    setIngesting(true); setIngestMsg(null);
    setUploadProgress(files.map(f => ({ name: f.name, status: "pending" as const })));
    let ok = 0; let fail = 0;
    for (let i = 0; i < files.length; i++) {
      try {
        await ingestKBFile(selectedCol.id, files[i]);
        ok++;
        setUploadProgress(prev => prev.map((p, idx) => idx === i ? { ...p, status: "ok" } : p));
      } catch {
        fail++;
        setUploadProgress(prev => prev.map((p, idx) => idx === i ? { ...p, status: "fail" } : p));
      }
    }
    setIngestMsg({ ok: fail === 0, text: `${ok} file(s) added${fail ? `, ${fail} failed` : ""}.` });
    try { setDocuments(await getKBDocuments(selectedCol.id)); } catch {}
    setIngesting(false);
    void refresh();
  };

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) await doUpload(Array.from(e.target.files));
    e.target.value = "";
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault(); setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length) await doUpload(files);
  };

  // ── URL ingest ────────────────────────────────────────────────────────────
  const handleIngestUrl = async () => {
    if (!selectedCol || !ingestUrl.trim()) return;
    setIngesting(true); setIngestMsg(null);
    try {
      await ingestKBUrl(selectedCol.id, ingestUrl.trim());
      setIngestUrl("");
      setIngestMsg({ ok: true, text: "URL ingested successfully." });
      try { setDocuments(await getKBDocuments(selectedCol.id)); } catch {}
      void refresh();
    } catch (e: any) { setIngestMsg({ ok: false, text: e.message }); }
    finally { setIngesting(false); }
  };

  // ── Site crawl ────────────────────────────────────────────────────────────
  const handleStartCrawl = async () => {
    if (!selectedCol || !ingestUrl.trim()) return;
    setCrawling(true); setCrawlEvents([]);
    const abort = new AbortController();
    crawlAbortRef.current = abort;
    try {
      await crawlKBSite(
        selectedCol.id,
        { url: ingestUrl.trim(), max_pages: crawlMaxPages, same_domain_only: true, delay_ms: crawlDelay },
        (evt) => {
          setCrawlEvents(prev => [...prev.slice(-199), evt]);
          if (evt.status === "done") {
            setIngestMsg({ ok: true, text: `Crawl done — ${(evt as any).ingested} pages ingested, ${(evt as any).failed} failed.` });
            try { getKBDocuments(selectedCol.id).then(setDocuments); } catch {}
            void refresh();
          }
        },
        abort.signal,
      );
    } catch (e: any) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      setCrawling(false); crawlAbortRef.current = null;
    }
  };

  const handleStopCrawl = () => { crawlAbortRef.current?.abort(); };

  // ── Doc actions ───────────────────────────────────────────────────────────
  const openDocDetail = async (docId: string) => {
    setSelectedDocId(docId);
    setDocChunks([]); setLoadingChunks(true);
    if (!selectedCol) return;
    try { setDocChunks((await getKBDocumentChunks(selectedCol.id, docId, 12)).chunks ?? []); }
    catch (e: any) { setError(e.message); }
    finally { setLoadingChunks(false); }
  };

  const handleDeleteDoc = async (docId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!selectedCol || !confirm("Delete this document and all its chunks?")) return;
    try {
      await deleteKBDocument(selectedCol.id, docId);
      setDocuments(prev => prev.filter(d => d.id !== docId));
      if (selectedDocId === docId) setSelectedDocId(null);
      void refresh();
    } catch (e2: any) { setError((e2 as any).message); }
  };

  const commitRename = async (docId: string) => {
    if (!selectedCol || !renameValue.trim()) { setRenamingDocId(null); return; }
    try {
      await renameKBDocument(selectedCol.id, docId, renameValue.trim());
      setDocuments(prev => prev.map(d => d.id === docId ? { ...d, filename: renameValue.trim() } : d));
    } catch (e: any) { setError(e.message); }
    finally { setRenamingDocId(null); }
  };

  // ── Advanced search ───────────────────────────────────────────────────────
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const res = await queryKBAdvanced({
        query: searchQuery,
        collection_ids: selectedCol ? [selectedCol.id] : undefined,
        n_results: searchNResults,
        mode: searchMode,
        source_type_filter: searchSourceFilter || undefined,
        rerank: searchRerank,
        min_score: searchMinScore,
      } as AdvancedQueryOptions);
      setSearchResults(res.results ?? []);
      setSearchMeta({ mode: res.mode, elapsed_ms: res.elapsed_ms, reranked: res.reranked });
      setQueryHistory(prev => [searchQuery, ...prev.filter(q => q !== searchQuery)].slice(0, 15));
    } catch (e: any) { setError(e.message); }
    finally { setSearching(false); }
  };

  // ── Evaluation & document chat ──────────────────────────────────────────
  const evalScopeLabel = evalDocId
    ? (documents.find(d => d.id === evalDocId)?.filename ?? "selected document")
    : `whole collection (${documents.length} docs)`;

  const handleGenerateQuestions = async () => {
    if (!selectedCol) return;
    setEvalBusy("gen"); setEvalErr(null); setEvalResults(null); setEvalSummary(null);
    try {
      const res = await kbEvalGenerate({
        collection_id: selectedCol.id, document_id: evalDocId || null, n: evalN, gen_model: "openai",
      });
      setEvalQuestions(res.questions);
      if (!res.questions.length) setEvalErr("No questions could be generated from this scope.");
    } catch (e: any) { setEvalErr(e.message); }
    finally { setEvalBusy(null); }
  };

  const handleRunEval = async () => {
    if (!selectedCol || !evalQuestions.length) return;
    setEvalBusy("run"); setEvalErr(null);
    try {
      const res = await kbEvalRun({
        collection_id: selectedCol.id, document_id: evalDocId || null, questions: evalQuestions,
        answer_model: evalAnswerModel, verify_model: evalVerifyModel, temperature: evalTemp,
        voice_id: evalSpeak ? "openai-alloy" : null,
      });
      setEvalResults(res.results);
      setEvalSummary({ accuracy: res.accuracy, counts: res.counts, answer_model: res.answer_model, verify_model: res.verify_model });
    } catch (e: any) { setEvalErr(e.message); }
    finally { setEvalBusy(null); }
  };

  const handleDocChatSend = async () => {
    const text = docChatInput.trim();
    if (!text || !selectedCol) return;
    setDocChatInput("");
    setDocChatLog(l => [...l, { role: "user", text }]);
    setDocChatBusy(true);
    try {
      const res = await chatWithDocument({
        text, collection_id: selectedCol.id, document_id: evalDocId || null,
        llm_provider_id: evalAnswerModel, temperature: evalTemp,
        voice_id: evalSpeak ? "openai-alloy" : null,
      });
      setDocChatLog(l => [...l, { role: "ai", text: res.response || "(no answer)", rag: !!res.rag_used, audio: res.audio_url }]);
      if (evalSpeak && res.audio_url) { try { new Audio(absoluteAudioUrl(res.audio_url) || "").play(); } catch {} }
    } catch (e: any) {
      setDocChatLog(l => [...l, { role: "ai", text: "Error: " + e.message }]);
    } finally { setDocChatBusy(false); }
  };

  // ── Analytics ─────────────────────────────────────────────────────────────
  const loadAnalytics = async () => {
    setAnalyticsLoading(true);
    try { setAnalytics(await getKBAnalytics()); }
    catch (e: any) { setError(e.message); }
    finally { setAnalyticsLoading(false); }
  };

  useEffect(() => { if (panel === "analytics") void loadAnalytics(); }, [panel]);

  // ── Config save ───────────────────────────────────────────────────────────
  const handleSaveConfig = async () => {
    try { await onSave({ ...settings, ...configDraft }); }
    catch (e: any) { setError(e.message); }
  };

  // ── Export ────────────────────────────────────────────────────────────────
  const handleExport = async (colId: string) => {
    try {
      const data = await exportKBCollection(colId);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = `kb_${colId}.json`; a.click();
      URL.revokeObjectURL(url);
    } catch (e: any) { setError(e.message); }
  };

  // ── Derived ────────────────────────────────────────────────────────────────
  const embOk = kbStatus?.embedding?.ok;
  const filteredDocs = documents.filter(d =>
    d.filename.toLowerCase().includes(docSearch.toLowerCase()) ||
    (d.source_url ?? "").toLowerCase().includes(docSearch.toLowerCase())
  );
  const selectedDoc = documents.find(d => d.id === selectedDocId) ?? null;

  // ═══════════════════════════════════════════════════════════════════════════
  // Render
  // ═══════════════════════════════════════════════════════════════════════════
  return (
    <div className="rag-shell">
      {/* ════════════════════════════════════════════
          LEFT — Collections sidebar
         ════════════════════════════════════════════ */}
      <div className="rag-sidebar">
        <div className="rag-sidebar-header">
          <h2>
            <Database size={17} />
            RAG Knowledge
            {embOk
              ? <span className="rag-status-badge ok" style={{ marginLeft: "auto", fontSize: 10 }}><CheckCircle2 size={10} />Ready</span>
              : <span className="rag-status-badge err" style={{ marginLeft: "auto", fontSize: 10 }}><CircleAlert size={10} />Offline</span>}
          </h2>
          <p>{collections.length} knowledge base{collections.length !== 1 ? "s" : ""}</p>
        </div>

        <div className="rag-col-list">
          {collections.length === 0 && (
            <div style={{ padding: "20px 8px", textAlign: "center", color: "var(--muted)", fontSize: 12 }}>
              <Database size={28} style={{ opacity: 0.3, marginBottom: 8 }} />
              <div>No knowledge bases yet.</div>
              <div>Create one below.</div>
            </div>
          )}
          {collections.map((col, i) => (
            <div key={col.id}
              className={`rag-col-item ${selectedCol?.id === col.id ? "active" : ""}`}
              onClick={() => void openCollection(col)}>
              <div className={`rag-col-icon idx-${i % 6}`}>
                <Database size={16} />
              </div>
              <div className="rag-col-meta">
                <strong>{col.name}</strong>
                <span>{col.document_count} doc{col.document_count !== 1 ? "s" : ""} · {col.chunk_count} chunks</span>
              </div>
              {selectedKnowledgeId === col.id && <div className="rag-col-active-dot" title="Active for voice" />}
            </div>
          ))}
        </div>

        {/* New collection form / button */}
        {panel === "newcol" ? (
          <div className="rag-new-col-form">
            <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8, color: "var(--teal)" }}>New Knowledge Base</div>
            <input value={newColName} onChange={e => setNewColName(e.target.value)}
              placeholder="Name (e.g. Company Docs)" style={{ marginBottom: 6 }}
              onKeyDown={e => e.key === "Enter" && void handleCreateCollection()} autoFocus />
            <input value={newColDesc} onChange={e => setNewColDesc(e.target.value)}
              placeholder="Description (optional)" style={{ marginBottom: 8 }} />
            <div className="form-row">
              <button className="rag-toolbar-btn primary" style={{ flex: 1 }} onClick={handleCreateCollection} disabled={creating || !newColName.trim()}>
                {creating ? "Creating…" : "Create"}
              </button>
              <button className="rag-toolbar-btn" onClick={() => setPanel(null)}>Cancel</button>
            </div>
          </div>
        ) : (
          <button className="rag-new-col-btn" onClick={() => setPanel("newcol")}>
            <Plus size={14} /> New Knowledge Base
          </button>
        )}
      </div>

      {/* ════════════════════════════════════════════
          RIGHT — Main content
         ════════════════════════════════════════════ */}
      <div className="rag-main" style={{ position: "relative" }}>

        {/* ── Toolbar ── */}
        <div className="rag-toolbar">
          {selectedCol ? (
            <div className="rag-toolbar-title">
              <h3>{selectedCol.name}</h3>
              <p>{selectedCol.document_count} document{selectedCol.document_count !== 1 ? "s" : ""} · {selectedCol.chunk_count} chunks</p>
            </div>
          ) : (
            <div className="rag-toolbar-title">
              <h3>RAG — Retrieval Augmented Generation</h3>
              <p>Select a knowledge base to browse documents</p>
            </div>
          )}

          <div className="rag-toolbar-actions">
            {/* Search box */}
            {selectedCol && (
              <div className="rag-search-bar">
                <Search size={13} style={{ color: "var(--muted)", flexShrink: 0 }} />
                <input value={docSearch} onChange={e => setDocSearch(e.target.value)} placeholder="Filter documents…" />
                {docSearch && (
                  <button onClick={() => setDocSearch("")} style={{ background: "none", border: "none", padding: 0, color: "var(--muted)", cursor: "pointer", lineHeight: 1 }}>✕</button>
                )}
              </div>
            )}
            {/* Upload */}
            {selectedCol && (
              <label className="rag-toolbar-btn primary" style={{ cursor: "pointer" }}>
                <Upload size={14} /><span>{ingesting ? "Uploading…" : "Add Files"}</span>
                <input type="file" multiple accept=".pdf,.docx,.txt,.md,.html,.htm,.csv,.json"
                  style={{ display: "none" }} onChange={handleFileInput} disabled={ingesting} />
              </label>
            )}
            {/* Semantic search */}
            <button className={`rag-toolbar-btn ${panel === "search" ? "active" : ""}`} onClick={() => setPanel(p => p === "search" ? null : "search")}>
              <Search size={14} /><span>Search</span>
            </button>
            {/* Evaluate */}
            {selectedCol && (
              <button className={`rag-toolbar-btn ${panel === "eval" ? "active" : ""}`} onClick={() => setPanel(p => p === "eval" ? null : "eval")}>
                <CheckCircle2 size={14} /><span>Evaluate</span>
              </button>
            )}
            {/* Analytics */}
            <button className={`rag-toolbar-btn ${panel === "analytics" ? "active" : ""}`} onClick={() => setPanel(p => p === "analytics" ? null : "analytics")}>
              <Activity size={14} /><span>Analytics</span>
            </button>
            {/* Config */}
            <button className={`rag-toolbar-btn ${panel === "config" ? "active" : ""}`} onClick={() => setPanel(p => p === "config" ? null : "config")}>
              <Settings size={14} /><span>Config</span>
            </button>
            {selectedCol && (
              <button className="rag-toolbar-btn" onClick={() => void handleExport(selectedCol.id)} title="Export collection">
                <Download size={14} />
              </button>
            )}
            {selectedCol && (
              <button className="rag-toolbar-btn" style={{ color: "var(--rose)" }}
                onClick={() => void handleDeleteCollection(selectedCol.id)} title="Delete knowledge base">
                <Trash2 size={14} />
              </button>
            )}
            <button className="rag-toolbar-btn" onClick={refresh} disabled={loading}>
              <RefreshCw size={14} className={loading ? "spin" : ""} />
            </button>
          </div>
        </div>

        {/* ── Error bar ── */}
        {error && (
          <div className="notice error" style={{ margin: "12px 20px 0", borderRadius: 8 }}>
            <CircleAlert size={14} /><span>{error}</span>
            <button onClick={() => setError(null)} style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer", color: "inherit" }}>✕</button>
          </div>
        )}

        {/* ── Knowledge service unavailable banner ── */}
        {serviceDown && (
          <div className="notice warn" style={{ margin: "12px 20px 0", borderRadius: 8 }}>
            <AlertTriangle size={14} />
            <span>Knowledge service unavailable — the backend RAG API did not respond. Collections and search are read-only until it recovers.</span>
            <button onClick={() => void refresh()} disabled={loading} style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 4, background: "none", border: "1px solid currentColor", borderRadius: 6, padding: "2px 8px", cursor: "pointer", color: "inherit" }}>
              <RefreshCw size={12} className={loading ? "spin" : ""} /> Retry
            </button>
          </div>
        )}

        {/* ── Ingest message ── */}
        {ingestMsg && (
          <div className={`notice ${ingestMsg.ok ? "success" : "error"}`} style={{ margin: "10px 20px 0", borderRadius: 8 }}>
            {ingestMsg.ok ? <CheckCircle2 size={14} /> : <CircleAlert size={14} />}
            <span>{ingestMsg.text}</span>
            <button onClick={() => setIngestMsg(null)} style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer", color: "inherit" }}>✕</button>
          </div>
        )}

        {/* ── Upload progress ── */}
        {uploadProgress.length > 0 && (
          <div className="upload-progress-list" style={{ margin: "10px 20px 0" }}>
            {uploadProgress.map((f, i) => (
              <div key={i} className={`upload-prog-item ${f.status}`}>
                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: 12 }}>{f.name}</span>
                <div className="prog-bar-track" style={{ width: 80 }}>
                  <div className={`prog-bar-fill ${f.status}`} />
                </div>
                <span style={{ fontSize: 11, width: 36, textAlign: "right", fontWeight: 700, textTransform: "uppercase",
                  color: f.status === "ok" ? "var(--green)" : f.status === "fail" ? "var(--rose)" : "var(--amber)" }}>
                  {f.status}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* ── Document content ── */}
        <div className="rag-content">
          {!selectedCol ? (
            /* ── No collection selected ── */
            <div className="rag-empty" style={{ paddingTop: 80 }}>
              <div className="rag-empty-icon"><Database size={28} /></div>
              <h3>Select a Knowledge Base</h3>
              <p>Choose a knowledge base from the sidebar, or create one to start adding documents.</p>
              <button className="rag-toolbar-btn primary" style={{ marginTop: 4 }} onClick={() => setPanel("newcol")}>
                <Plus size={14} /><span>Create Knowledge Base</span>
              </button>
            </div>
          ) : documents.length === 0 && !ingesting ? (
            /* ── Empty collection ── */
            <>
              {/* URL ingest row */}
              <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
                <input value={ingestUrl} onChange={e => setIngestUrl(e.target.value)}
                  placeholder="Paste a URL to crawl and add (e.g. https://docs.example.com)"
                  onKeyDown={e => e.key === "Enter" && void handleIngestUrl()} style={{ flex: 1 }} />
                <button className="rag-toolbar-btn" onClick={handleIngestUrl} disabled={ingesting || !ingestUrl.trim()}>
                  <Globe size={14} /><span>Add URL</span>
                </button>
              </div>
              {/* Drop zone */}
              <label
                className={`rag-dropzone ${dragOver ? "drag-over" : ""}`}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                style={{ cursor: "pointer" }}>
                <Upload size={32} style={{ opacity: 0.4, marginBottom: 10 }} />
                <div style={{ fontWeight: 600, marginBottom: 4 }}>Drop files here or click to upload</div>
                <div style={{ fontSize: 12 }}>PDF, DOCX, TXT, MD, HTML, CSV, JSON</div>
                <input type="file" multiple accept=".pdf,.docx,.txt,.md,.html,.htm,.csv,.json"
                  style={{ display: "none" }} onChange={handleFileInput} />
              </label>
              <div className="rag-empty">
                <div className="rag-empty-icon"><FileText size={28} /></div>
                <h3>No documents yet</h3>
                <p>Upload files above or paste a URL to start building this knowledge base.</p>
              </div>
            </>
          ) : (
            /* ── Document grid ── */
            <>
              {/* URL ingest / site crawl row */}
              <div style={{ marginBottom: 16, background: "var(--panel)", borderRadius: 10, border: "1px solid var(--line)", overflow: "hidden" }}>
                {/* URL input row */}
                <div style={{ display: "flex", gap: 0 }}>
                  <input value={ingestUrl} onChange={e => setIngestUrl(e.target.value)}
                    placeholder="Paste URL — single page or entire website (https://example.com)"
                    onKeyDown={e => e.key === "Enter" && !crawlMode && void handleIngestUrl()}
                    style={{ flex: 1, fontSize: 12, border: "none", borderRadius: 0, background: "transparent", padding: "10px 14px" }} />
                  {/* Single page */}
                  <button className="rag-toolbar-btn" onClick={handleIngestUrl}
                    disabled={ingesting || crawling || !ingestUrl.trim() || crawlMode}
                    style={{ borderRadius: 0, borderLeft: "1px solid var(--line)", flexShrink: 0, padding: "8px 14px" }}>
                    <Globe size={13} /><span>Add Page</span>
                  </button>
                  {/* Toggle crawl mode */}
                  <button className={`rag-toolbar-btn ${crawlMode ? "active" : ""}`}
                    onClick={() => setCrawlMode(m => !m)}
                    style={{ borderRadius: 0, borderLeft: "1px solid var(--line)", flexShrink: 0, padding: "8px 14px" }}
                    title="Crawl entire website — follows all links">
                    <Sparkles size={13} /><span>Crawl Site</span>
                  </button>
                </div>

                {/* Crawl options panel */}
                {crawlMode && (
                  <div style={{ borderTop: "1px solid var(--line)", padding: "12px 14px", background: "var(--surface-2)" }}>
                    <div style={{ fontSize: 11, color: "var(--teal)", fontWeight: 700, marginBottom: 8 }}>
                      🕷 Site Crawler — follows all same-domain links automatically
                    </div>
                    <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: crawling ? 10 : 0 }}>
                      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                        <span style={{ color: "var(--muted)" }}>Max pages</span>
                        <input type="number" value={crawlMaxPages} min={1} max={1000}
                          onChange={e => setCrawlMaxPages(Number(e.target.value))}
                          style={{ width: 70, fontSize: 12 }} disabled={crawling} />
                      </label>
                      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                        <span style={{ color: "var(--muted)" }}>Delay (ms)</span>
                        <input type="number" value={crawlDelay} min={50} max={2000} step={50}
                          onChange={e => setCrawlDelay(Number(e.target.value))}
                          style={{ width: 70, fontSize: 12 }} disabled={crawling} />
                      </label>
                      {!crawling ? (
                        <button className="rag-toolbar-btn primary" onClick={handleStartCrawl}
                          disabled={!ingestUrl.trim()} style={{ marginLeft: "auto" }}>
                          <Sparkles size={13} /><span>Start Crawl</span>
                        </button>
                      ) : (
                        <button className="rag-toolbar-btn" onClick={handleStopCrawl}
                          style={{ marginLeft: "auto", color: "var(--rose)", borderColor: "var(--rose)" }}>
                          <Square size={13} /><span>Stop</span>
                        </button>
                      )}
                    </div>

                    {/* Live crawl log */}
                    {crawlEvents.length > 0 && (() => {
                      const last = crawlEvents[crawlEvents.length - 1] as any;
                      const done = last?.status === "done";
                      const ingested = last?.ingested ?? 0;
                      const failed = last?.failed ?? 0;
                      const total = last?.total_queued ?? crawlMaxPages;
                      const pct = Math.min(100, Math.round((ingested / Math.max(crawlMaxPages, 1)) * 100));
                      return (
                        <div>
                          {/* Progress bar */}
                          <div style={{ height: 4, background: "var(--line)", borderRadius: 2, marginBottom: 8, overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${pct}%`, background: done ? "var(--green)" : "var(--teal)", borderRadius: 2, transition: "width 0.3s" }} />
                          </div>
                          {/* Summary */}
                          <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 8, display: "flex", gap: 12 }}>
                            <span>Ingested: <strong style={{ color: "var(--green)" }}>{ingested}</strong></span>
                            {failed > 0 && <span>Failed: <strong style={{ color: "var(--rose)" }}>{failed}</strong></span>}
                            <span>Queue: <strong>{total - ingested - failed}</strong></span>
                            {crawling && <span style={{ color: "var(--teal)" }}><RefreshCw size={10} className="spin" style={{ marginRight: 4 }} />Crawling…</span>}
                            {done && <span className="rag-status-badge ok" style={{ fontSize: 10 }}>Done ✓</span>}
                          </div>
                          {/* Last 5 events */}
                          <div style={{ maxHeight: 120, overflowY: "auto", display: "flex", flexDirection: "column", gap: 3 }}>
                            {[...crawlEvents].reverse().slice(0, 8).map((evt, i) => {
                              const e = evt as any;
                              const color = e.status === "saved" ? "var(--green)" : e.status === "error" ? "var(--rose)" : e.status === "done" ? "var(--teal)" : "var(--muted)";
                              return (
                                <div key={i} style={{ fontSize: 10, display: "flex", alignItems: "center", gap: 6, padding: "2px 0" }}>
                                  <span style={{ color, fontWeight: 700, width: 50, flexShrink: 0, textTransform: "uppercase" }}>{e.status}</span>
                                  <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--muted)" }}>
                                    {e.url ?? e.start_url ?? ""}
                                  </span>
                                  {e.chunks != null && <span style={{ color: "var(--teal)", fontSize: 9 }}>{e.chunks} chunks</span>}
                                  {e.error && <span style={{ color: "var(--rose)", fontSize: 9, maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis" }}>{e.error}</span>}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>

              {/* Drop zone (compact) */}
              <label
                className={`rag-dropzone ${dragOver ? "drag-over" : ""}`}
                style={{ padding: "14px 20px", marginBottom: 16, fontSize: 12, cursor: "pointer" }}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, justifyContent: "center" }}>
                  <Upload size={16} style={{ opacity: 0.6 }} />
                  <span>Drop files here or <strong>click to upload more</strong></span>
                </div>
                <input type="file" multiple accept=".pdf,.docx,.txt,.md,.html,.htm,.csv,.json"
                  style={{ display: "none" }} onChange={handleFileInput} />
              </label>

              {/* Filter indicator */}
              {docSearch && (
                <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 10 }}>
                  Showing {filteredDocs.length} of {documents.length} documents matching "{docSearch}"
                </div>
              )}

              {filteredDocs.length === 0 ? (
                <div className="rag-empty" style={{ paddingTop: 30 }}>
                  <div className="rag-empty-icon"><Search size={22} /></div>
                  <h3>No matches</h3>
                  <p>Try a different search term.</p>
                </div>
              ) : (
                <div className="doc-grid">
                  {filteredDocs.map(doc => {
                    const info = extInfo(doc.filename, doc.source_type);
                    return (
                      <div key={doc.id}
                        className={`doc-card ${selectedDocId === doc.id ? "selected" : ""}`}
                        onClick={() => void openDocDetail(doc.id)}>
                        {/* Thumbnail */}
                        <div className={`doc-thumb ${info.cls}`}>
                          <span className="doc-thumb-icon">{info.emoji}</span>
                          <span className="doc-thumb-ext">{info.label}</span>
                        </div>
                        {/* Body */}
                        <div className="doc-card-body">
                          {renamingDocId === doc.id ? (
                            <input value={renameValue}
                              onChange={e => setRenameValue(e.target.value)}
                              onBlur={() => void commitRename(doc.id)}
                              onKeyDown={e => {
                                e.stopPropagation();
                                if (e.key === "Enter") void commitRename(doc.id);
                                if (e.key === "Escape") setRenamingDocId(null);
                              }}
                              onClick={e => e.stopPropagation()}
                              autoFocus style={{ fontSize: 12, marginBottom: 4 }} />
                          ) : (
                            <div className="doc-card-name">{doc.filename}</div>
                          )}
                          <div className="doc-card-meta">
                            <span className={`doc-chip ${info.chip}`}>{info.label}</span>
                            <span>{doc.chunk_count} chunks</span>
                            {doc.size_bytes > 0 && <span>{fmtBytes(doc.size_bytes)}</span>}
                          </div>
                          <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>
                            {new Date(doc.created_at).toLocaleDateString()}
                          </div>
                        </div>
                        {/* Hover actions */}
                        <div className="doc-card-actions">
                          <button className="doc-action-btn" title="Rename"
                            onClick={e => { e.stopPropagation(); setRenamingDocId(doc.id); setRenameValue(doc.filename); }}>
                            <Pencil size={11} />
                          </button>
                          {doc.source_url && (
                            <a href={doc.source_url} target="_blank" rel="noreferrer"
                              className="doc-action-btn" title="Open URL" onClick={e => e.stopPropagation()} style={{ textDecoration: "none" }}>
                              <Globe size={11} />
                            </a>
                          )}
                          <button className="doc-action-btn danger" title="Delete"
                            onClick={e => void handleDeleteDoc(doc.id, e)}>
                            <Trash2 size={11} />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}
        </div>

        {/* ════════════════════════════════════════════
            DOCUMENT DETAIL DRAWER
           ════════════════════════════════════════════ */}
        {selectedDoc && (
          <div className="doc-detail-panel">
            <div className="doc-detail-header">
              {(() => {
                const info = extInfo(selectedDoc.filename, selectedDoc.source_type);
                return (
                  <>
                    <div className={`detail-thumb doc-thumb ${info.cls}`} style={{ width: 44, height: 44 }}>
                      <span style={{ fontSize: 22 }}>{info.emoji}</span>
                    </div>
                    <div className="detail-info">
                      <h3>{selectedDoc.filename}</h3>
                      <p>
                        <span className={`doc-chip ${info.chip}`}>{info.label}</span>
                        {" · "}{selectedDoc.chunk_count} chunks
                        {selectedDoc.size_bytes > 0 && ` · ${fmtBytes(selectedDoc.size_bytes)}`}
                        {" · "}{new Date(selectedDoc.created_at).toLocaleDateString()}
                      </p>
                      {selectedDoc.source_url && (
                        <a href={selectedDoc.source_url} target="_blank" rel="noreferrer"
                          style={{ fontSize: 11, color: "var(--teal)", display: "block", marginTop: 3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {selectedDoc.source_url}
                        </a>
                      )}
                    </div>
                    <button onClick={() => setSelectedDocId(null)}
                      style={{ background: "none", border: "none", color: "var(--muted)", cursor: "pointer", padding: 4, marginLeft: 4, flexShrink: 0 }}>✕</button>
                  </>
                );
              })()}
            </div>

            {/* Actions */}
            <div style={{ display: "flex", gap: 6, padding: "10px 18px", borderBottom: "1px solid var(--line)", flexShrink: 0 }}>
              <button className="rag-toolbar-btn" style={{ flex: 1 }}
                onClick={() => { setRenamingDocId(selectedDoc.id); setRenameValue(selectedDoc.filename); }}>
                <Pencil size={13} /><span>Rename</span>
              </button>
              <button className="rag-toolbar-btn" style={{ flex: 1, color: "var(--rose)" }}
                onClick={async e => {
                  if (!confirm("Delete this document?")) return;
                  await handleDeleteDoc(selectedDoc.id, e);
                  setSelectedDocId(null);
                }}>
                <Trash2 size={13} /><span>Delete</span>
              </button>
            </div>

            {/* Chunks */}
            <div className="doc-detail-body">
              <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 12, fontWeight: 700, textTransform: "uppercase", letterSpacing: 0.5 }}>
                Content Chunks ({loadingChunks ? "loading…" : `${docChunks.length} of ${selectedDoc.chunk_count}`})
              </div>
              {loadingChunks ? (
                <div style={{ textAlign: "center", padding: 24, color: "var(--muted)" }}>
                  <RefreshCw size={18} className="spin" />
                </div>
              ) : docChunks.length === 0 ? (
                <div style={{ color: "var(--muted)", fontSize: 13, textAlign: "center", padding: 24 }}>No chunks loaded.</div>
              ) : (
                docChunks.map(c => (
                  <div key={c.chunk_index} className="chunk-card">
                    <div className="chunk-card-label">CHUNK #{c.chunk_index}</div>
                    <p>{c.text}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* ════════════════════════════════════════════
            PANEL OVERLAYS (Search / Analytics / Config)
           ════════════════════════════════════════════ */}
        {panel === "search" && (
          <div className="rag-panel-overlay">
            <div className="rag-panel-header">
              <Search size={18} style={{ color: "var(--teal)" }} />
              <h2>Semantic Search</h2>
              <span style={{ fontSize: 12, color: "var(--muted)" }}>
                {selectedCol ? `Searching in: ${selectedCol.name}` : "All knowledge bases"}
              </span>
              <button onClick={() => setPanel(null)} style={{ marginLeft: "auto", background: "none", border: "none", color: "var(--muted)", cursor: "pointer" }}>✕</button>
            </div>
            <div className="rag-panel-body">
              {/* Mode bar */}
              <div style={{ display: "flex", gap: 4, marginBottom: 14, background: "var(--panel)", borderRadius: 8, padding: 4 }}>
                {(["hybrid", "semantic", "keyword"] as const).map(m => (
                  <button key={m} onClick={() => setSearchMode(m)} style={{
                    flex: 1, padding: "7px 10px", borderRadius: 6, border: "none", cursor: "pointer", fontSize: 12, fontWeight: 600,
                    background: searchMode === m ? "var(--teal)" : "transparent",
                    color: searchMode === m ? "#0a1412" : "var(--muted)"
                  }}>{m === "hybrid" ? "Hybrid (RRF)" : m === "semantic" ? "Semantic" : "Keyword (BM25)"}</button>
                ))}
              </div>
              {/* Controls */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 80px 80px 80px auto", gap: 8, alignItems: "end", marginBottom: 14 }}>
                <div>
                  <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 3 }}>Scope</div>
                  <select value={selectedCol?.id ?? ""} onChange={e => setSelectedCol(collections.find(c => c.id === e.target.value) ?? null)}>
                    <option value="">All collections</option>
                    {collections.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                  </select>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 3 }}>Results</div>
                  <input type="number" min={1} max={20} value={searchNResults} onChange={e => setSearchNResults(Number(e.target.value))} />
                </div>
                <div>
                  <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 3 }}>Min score</div>
                  <input type="number" min={0} max={1} step={0.05} value={searchMinScore} onChange={e => setSearchMinScore(Number(e.target.value))} />
                </div>
                <div>
                  <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 3 }}>Source</div>
                  <select value={searchSourceFilter} onChange={e => setSearchSourceFilter(e.target.value as any)}>
                    <option value="">All</option>
                    <option value="file">Files</option>
                    <option value="url">URLs</option>
                  </select>
                </div>
                <div style={{ paddingTop: 16 }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 5, cursor: "pointer", fontSize: 12, whiteSpace: "nowrap" }}>
                    <input type="checkbox" checked={searchRerank} onChange={e => setSearchRerank(e.target.checked)} />
                    Rerank
                  </label>
                </div>
              </div>
              {/* Query input */}
              <form style={{ display: "flex", gap: 8, marginBottom: 14 }} onSubmit={e => { e.preventDefault(); void handleSearch(); }}>
                <input value={searchQuery} onChange={e => setSearchQuery(e.target.value)} placeholder="Ask something your documents should know…" style={{ flex: 1 }} />
                <button className="rag-toolbar-btn primary" type="submit" disabled={searching || !searchQuery.trim()}>
                  <Search size={14} /><span>{searching ? "Searching…" : "Search"}</span>
                </button>
              </form>
              {/* History chips */}
              {queryHistory.length > 0 && searchResults.length === 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 14 }}>
                  {queryHistory.map(q => (
                    <button key={q} onClick={() => { setSearchQuery(q); void handleSearch(); }}
                      style={{ fontSize: 11, padding: "4px 10px", borderRadius: 20, background: "var(--panel)", border: "1px solid var(--line)", cursor: "pointer", color: "var(--muted)" }}>
                      {q.length > 50 ? q.slice(0, 50) + "…" : q}
                    </button>
                  ))}
                </div>
              )}
              {/* Metadata */}
              {searchMeta && (
                <div style={{ display: "flex", gap: 12, fontSize: 11, color: "var(--muted)", marginBottom: 12, alignItems: "center" }}>
                  <span>Mode: <strong style={{ color: "var(--teal)" }}>{searchMeta.mode}</strong></span>
                  <span>{searchMeta.elapsed_ms}ms</span>
                  {searchMeta.reranked && <span className="rag-status-badge ok" style={{ fontSize: 10 }}>Reranked</span>}
                  <span>{searchResults.length} results</span>
                </div>
              )}
              {/* Results */}
              {searching && <div style={{ textAlign: "center", padding: 40, color: "var(--muted)" }}><RefreshCw size={20} className="spin" /></div>}
              {searchResults.map((r, i) => {
                const pct = Math.round(r.score * 100);
                const col = r.score > 0.7 ? "var(--green)" : r.score > 0.5 ? "var(--amber)" : "var(--rose)";
                return (
                  <div key={r.chunk_id} style={{
                    padding: "13px 16px", background: "var(--panel)", borderRadius: 8,
                    border: "1px solid var(--line)", marginBottom: 10, position: "relative", overflow: "hidden"
                  }}>
                    <div style={{ position: "absolute", inset: 0, background: col, opacity: 0.04, pointerEvents: "none" }} />
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ background: col, color: "#0a1412", borderRadius: 4, padding: "2px 7px", fontSize: 11, fontWeight: 800 }}>{pct}%</span>
                        <span style={{ fontWeight: 600, fontSize: 13 }}>{r.doc_name}</span>
                        <span style={{ fontSize: 11, color: "var(--muted)" }}>#{r.chunk_index}</span>
                        {r.source_type === "url" && <span className="doc-chip url">URL</span>}
                      </div>
                      <div style={{ display: "flex", gap: 8, fontSize: 11, color: "var(--muted)" }}>
                        {r.semantic_score != null && <span>sem:{(r.semantic_score * 100).toFixed(0)}%</span>}
                        {r.bm25_score != null && <span>bm25:{(r.bm25_score * 100).toFixed(0)}%</span>}
                        {r.rerank_score != null && <span className="rag-status-badge ok" style={{ fontSize: 10 }}>↑{r.rerank_score.toFixed(2)}</span>}
                        <span style={{ color: "var(--muted)" }}>#{i + 1}</span>
                      </div>
                    </div>
                    <div className="score-bar" style={{ marginBottom: 8 }}>
                      <div className="score-bar-track">
                        <div className="score-bar-fill" style={{ width: `${pct}%`, background: col }} />
                      </div>
                    </div>
                    <p style={{ fontSize: 13, lineHeight: 1.55, margin: 0 }}>{r.text}</p>
                    {r.source_url && <a href={r.source_url} target="_blank" rel="noreferrer" style={{ fontSize: 11, color: "var(--teal)", display: "block", marginTop: 6 }}>{r.source_url}</a>}
                  </div>
                );
              })}
              {searchResults.length === 0 && searchMeta && !searching && (
                <div className="rag-empty"><div className="rag-empty-icon"><Search size={22} /></div><h3>No results</h3><p>Lower min score or try different terms.</p></div>
              )}
            </div>
          </div>
        )}

        {panel === "eval" && selectedCol && (
          <div className="rag-panel-overlay">
            <div className="rag-panel-header">
              <CheckCircle2 size={18} style={{ color: "var(--teal)" }} />
              <h2>Evaluate &amp; Chat — {selectedCol.name}</h2>
              <button onClick={() => setPanel(null)} style={{ marginLeft: "auto", background: "none", border: "none", color: "var(--muted)", cursor: "pointer", padding: "0 4px" }}>✕</button>
            </div>
            <div className="rag-panel-body">
              {/* Controls */}
              <div className="eval-controls">
                <div className="eval-field">
                  <label>Scope</label>
                  <select value={evalDocId} onChange={e => { setEvalDocId(e.target.value); setEvalQuestions([]); setEvalResults(null); setEvalSummary(null); }}>
                    <option value="">Whole collection ({documents.length} docs)</option>
                    {documents.map(d => <option key={d.id} value={d.id}>{d.filename} ({d.chunk_count} chunks)</option>)}
                  </select>
                </div>
                <div className="eval-field">
                  <label>Answer model</label>
                  <select value={evalAnswerModel} onChange={e => setEvalAnswerModel(e.target.value)}>
                    <option value="local">Local (Ollama)</option>
                    <option value="openai">OpenAI</option>
                    <option value="gemini">Gemini</option>
                  </select>
                </div>
                <div className="eval-field">
                  <label>Verify model</label>
                  <select value={evalVerifyModel} onChange={e => setEvalVerifyModel(e.target.value)}>
                    <option value="openai">OpenAI</option>
                    <option value="gemini">Gemini</option>
                    <option value="local">Local (Ollama)</option>
                  </select>
                </div>
                <div className="eval-field">
                  <label># Questions: <b>{evalN}</b></label>
                  <input type="range" min={1} max={20} value={evalN} onChange={e => setEvalN(Number(e.target.value))} />
                </div>
                <div className="eval-field">
                  <label>Temperature: <b>{evalTemp.toFixed(2)}</b></label>
                  <input type="range" min={0} max={1} step={0.05} value={evalTemp} onChange={e => setEvalTemp(Number(e.target.value))} />
                </div>
                <label className="eval-speak">
                  <input type="checkbox" checked={evalSpeak} onChange={e => setEvalSpeak(e.target.checked)} />
                  <span>🔊 Speak answers (voice)</span>
                </label>
              </div>
              <div className="eval-actions">
                <button className="rag-toolbar-btn primary" onClick={handleGenerateQuestions} disabled={evalBusy !== null}>
                  {evalBusy === "gen" ? <RefreshCw size={13} className="spin" /> : <Sparkles size={13} />}
                  <span>{evalBusy === "gen" ? "Generating…" : "Generate questions"}</span>
                </button>
                <button className="rag-toolbar-btn" onClick={handleRunEval} disabled={evalBusy !== null || !evalQuestions.length}>
                  {evalBusy === "run" ? <RefreshCw size={13} className="spin" /> : <Activity size={13} />}
                  <span>{evalBusy === "run" ? "Evaluating…" : `Run evaluation (${evalQuestions.length})`}</span>
                </button>
                <span className="eval-scope-note">Scope: {evalScopeLabel}</span>
              </div>
              {evalErr && <div className="eval-err"><CircleAlert size={14} /> {evalErr}</div>}

              {/* Generated questions preview */}
              {evalQuestions.length > 0 && !evalResults && (
                <div className="eval-section">
                  <div className="eval-section-title">{evalQuestions.length} questions delivered from {evalScopeLabel}</div>
                  <ol className="eval-qlist">
                    {evalQuestions.map((q, i) => <li key={i}><b>{q.q}</b><span>{q.a}</span></li>)}
                  </ol>
                </div>
              )}

              {/* Results scoreboard */}
              {evalSummary && (
                <div className="eval-scoreboard">
                  <div className="eval-score-main">
                    <div className="eval-accuracy">{evalSummary.accuracy}%</div>
                    <div className="eval-score-sub">accuracy<br /><span>{evalSummary.answer_model} answered · {evalSummary.verify_model} verified</span></div>
                  </div>
                  <div className="eval-score-pills">
                    <span className="pill good">✓ {evalSummary.counts.correct} correct</span>
                    <span className="pill warn">~ {evalSummary.counts.partial} partial</span>
                    <span className="pill" style={{ color: "var(--rose)" }}>✗ {evalSummary.counts.incorrect} incorrect</span>
                    {evalSummary.counts.error > 0 && <span className="pill">⚠ {evalSummary.counts.error} error</span>}
                  </div>
                </div>
              )}

              {/* Results table */}
              {evalResults && (
                <div className="eval-section">
                  <div className="eval-section-title">Detailed results</div>
                  {evalResults.map((r, i) => (
                    <div key={i} className={`eval-row v-${r.verdict}`}>
                      <div className="eval-row-head">
                        <span className={`eval-verdict v-${r.verdict}`}>{r.verdict}</span>
                        <span className="eval-q">{r.question}</span>
                        {r.rag_used && <span className="pill good">RAG</span>}
                        <span className="eval-lat">{r.latency_s}s</span>
                      </div>
                      <div className="eval-row-body">
                        <div><b>Expected:</b> {r.reference}</div>
                        <div><b>Answer:</b> {r.answer}</div>
                        {r.reason && <div className="eval-reason"><b>Judge:</b> {r.reason}</div>}
                        {r.audio_url && <button className="rag-toolbar-btn" onClick={() => { try { new Audio(absoluteAudioUrl(r.audio_url!) || "").play(); } catch {} }}><Volume2 size={12} /> Play</button>}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Chat with this document */}
              <div className="eval-section">
                <div className="eval-section-title"><Radio size={13} /> Chat with {evalDocId ? "this document" : "this collection"} ({evalAnswerModel}, temp {evalTemp.toFixed(2)})</div>
                <div className="doc-chat-log">
                  {docChatLog.length === 0 && <div className="doc-chat-empty">Ask a question grounded in {evalScopeLabel}.</div>}
                  {docChatLog.map((m, i) => (
                    <div key={i} className={`doc-chat-msg ${m.role}`}>
                      <span className="doc-chat-role">{m.role === "user" ? "You" : "AI"}{m.rag ? " · RAG" : ""}</span>
                      <div>{m.text}</div>
                      {m.audio && <button className="rag-toolbar-btn" onClick={() => { try { new Audio(absoluteAudioUrl(m.audio!) || "").play(); } catch {} }}><Volume2 size={12} /></button>}
                    </div>
                  ))}
                  {docChatBusy && <div className="doc-chat-msg ai"><RefreshCw size={13} className="spin" /> thinking…</div>}
                </div>
                <form className="doc-chat-input" onSubmit={e => { e.preventDefault(); void handleDocChatSend(); }}>
                  <input value={docChatInput} onChange={e => setDocChatInput(e.target.value)} placeholder="Ask about this document…" disabled={docChatBusy} />
                  <button type="submit" className="rag-toolbar-btn primary" disabled={docChatBusy || !docChatInput.trim()}><Send size={13} /></button>
                </form>
              </div>
            </div>
          </div>
        )}

        {panel === "analytics" && (
          <div className="rag-panel-overlay">
            <div className="rag-panel-header">
              <Activity size={18} style={{ color: "var(--teal)" }} />
              <h2>Query Analytics</h2>
              <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                <button className="rag-toolbar-btn" onClick={loadAnalytics} disabled={analyticsLoading}><RefreshCw size={13} className={analyticsLoading ? "spin" : ""} /></button>
                <button className="rag-toolbar-btn" style={{ color: "var(--rose)" }}
                  onClick={async () => { if (confirm("Clear all analytics?")) { await clearKBAnalytics(); void loadAnalytics(); } }}>
                  <Trash2 size={13} />
                </button>
                <button onClick={() => setPanel(null)} style={{ background: "none", border: "none", color: "var(--muted)", cursor: "pointer", padding: "0 4px" }}>✕</button>
              </div>
            </div>
            <div className="rag-panel-body">
              {analyticsLoading ? <div style={{ textAlign: "center", padding: 40 }}><RefreshCw size={20} className="spin" /></div>
                : analytics ? (
                  <>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 24 }}>
                      {[
                        { label: "Total Queries", value: analytics.total_queries, color: "var(--teal)" },
                        { label: "Avg Results", value: analytics.avg_results },
                        { label: "Avg Latency", value: `${analytics.avg_latency_ms}ms` },
                        { label: "Zero Results", value: analytics.zero_result_queries, color: analytics.zero_result_queries > 0 ? "var(--rose)" : undefined },
                      ].map(({ label, value, color }) => (
                        <div key={label} className="analytics-stat">
                          <div className="stat-val" style={color ? { color } : {}}>{String(value)}</div>
                          <div className="stat-lbl">{label}</div>
                        </div>
                      ))}
                    </div>
                    {analytics.top_queries.length > 0 && (
                      <div style={{ marginBottom: 24 }}>
                        <h3 style={{ fontSize: 13, marginBottom: 10, fontWeight: 700 }}>Top Queries</h3>
                        {analytics.top_queries.map((q: { query: string; count: number }, i: number) => (
                          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 12px",
                            background: "var(--panel)", borderRadius: 6, border: "1px solid var(--line)", marginBottom: 5 }}>
                            <span style={{ fontWeight: 800, color: "var(--muted)", fontSize: 11, width: 18 }}>#{i + 1}</span>
                            <span style={{ flex: 1, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{q.query}</span>
                            <span className="rag-status-badge ok" style={{ fontSize: 10 }}>{q.count}×</span>
                            <button className="rag-toolbar-btn" style={{ fontSize: 11, padding: "3px 8px" }}
                              onClick={() => { setSearchQuery(q.query); setPanel("search"); }}>
                              <Search size={11} />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                    {analytics.recent.length > 0 && (
                      <div>
                        <h3 style={{ fontSize: 13, marginBottom: 10, fontWeight: 700 }}>Recent</h3>
                        {analytics.recent.map((q: any, i: number) => (
                          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 12px",
                            background: "var(--panel)", borderRadius: 6, border: "1px solid var(--line)", marginBottom: 4, fontSize: 12 }}>
                            <span style={{ color: "var(--muted)", fontSize: 11, width: 60, flexShrink: 0 }}>{new Date(q.ts).toLocaleTimeString()}</span>
                            <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{q.query}</span>
                            <span className="doc-chip other">{q.mode}</span>
                            <span style={{ fontSize: 11, color: q.result_count === 0 ? "var(--rose)" : "var(--green)", fontWeight: 700 }}>{q.result_count}</span>
                            <span style={{ fontSize: 11, color: "var(--muted)" }}>{q.elapsed_ms}ms</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {analytics.total_queries === 0 && <div className="rag-empty"><div className="rag-empty-icon"><Activity size={22} /></div><h3>No queries yet</h3><p>Use Search to generate analytics.</p></div>}
                  </>
                ) : <div className="rag-empty"><div className="rag-empty-icon"><Activity size={22} /></div><h3>Analytics unavailable</h3><p>Enable query_analytics in Config.</p></div>}
            </div>
          </div>
        )}

        {panel === "config" && (
          <div className="rag-panel-overlay">
            <div className="rag-panel-header">
              <Settings size={18} style={{ color: "var(--teal)" }} />
              <h2>RAG Configuration</h2>
              <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                <button className="rag-toolbar-btn primary" onClick={handleSaveConfig}><Save size={13} /><span>Save</span></button>
                <button onClick={() => setPanel(null)} style={{ background: "none", border: "none", color: "var(--muted)", cursor: "pointer", padding: "0 4px" }}>✕</button>
              </div>
            </div>
            <div className="rag-panel-body">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28 }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <h4 style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 0.6, fontWeight: 700 }}>Embedding</h4>
                  <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 13 }}>
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>Provider</span>
                    <select value={configDraft.kb_embedding_provider} onChange={e => setConfigDraft(d => ({ ...d, kb_embedding_provider: e.target.value }))}>
                      <option value="ollama">Ollama</option>
                      <option value="sentence-transformers">Sentence Transformers</option>
                      <option value="openai">OpenAI</option>
                    </select>
                  </label>
                  <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 13 }}>
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>
                      {configDraft.kb_embedding_provider === "ollama" ? "Ollama model" : configDraft.kb_embedding_provider === "openai" ? "OpenAI embedding model" : "ST model"}
                    </span>
                    <input value={configDraft.kb_embedding_provider === "ollama" || configDraft.kb_embedding_provider === "openai" ? configDraft.kb_embedding_model : configDraft.kb_embedding_model_st}
                      placeholder={configDraft.kb_embedding_provider === "openai" ? "text-embedding-3-small" : undefined}
                      onChange={e => setConfigDraft(d => configDraft.kb_embedding_provider === "ollama" || configDraft.kb_embedding_provider === "openai"
                        ? { ...d, kb_embedding_model: e.target.value }
                        : { ...d, kb_embedding_model_st: e.target.value })} />
                  </label>

                  <h4 style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 0.6, fontWeight: 700, marginTop: 8 }}>Chunking</h4>
                  <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 13 }}>
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>Strategy</span>
                    <select value={configDraft.kb_chunk_strategy} onChange={e => setConfigDraft(d => ({ ...d, kb_chunk_strategy: e.target.value }))}>
                      <option value="sentence">Sentence-aware ★</option>
                      <option value="word">Word-based</option>
                      <option value="paragraph">Paragraph-based</option>
                    </select>
                  </label>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 13 }}>
                      <span style={{ fontSize: 11, color: "var(--muted)" }}>Chunk size (words)</span>
                      <input type="number" value={configDraft.kb_chunk_size} min={50} max={2000}
                        onChange={e => setConfigDraft(d => ({ ...d, kb_chunk_size: Number(e.target.value) }))} />
                    </label>
                    <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 13 }}>
                      <span style={{ fontSize: 11, color: "var(--muted)" }}>Overlap (words)</span>
                      <input type="number" value={configDraft.kb_chunk_overlap} min={0} max={500}
                        onChange={e => setConfigDraft(d => ({ ...d, kb_chunk_overlap: Number(e.target.value) }))} />
                    </label>
                  </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <h4 style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 0.6, fontWeight: 700 }}>Search</h4>
                  <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 13 }}>
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>Default mode</span>
                    <select value={configDraft.kb_search_mode} onChange={e => setConfigDraft(d => ({ ...d, kb_search_mode: e.target.value }))}>
                      <option value="hybrid">Hybrid — semantic + BM25 (RRF) ★</option>
                      <option value="semantic">Semantic only</option>
                      <option value="keyword">Keyword (BM25) only</option>
                    </select>
                    <span style={{ fontSize: 10, color: "var(--muted)" }}>Hybrid needs: pip install rank-bm25</span>
                  </label>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 13 }}>
                      <span style={{ fontSize: 11, color: "var(--muted)" }}>Max results</span>
                      <input type="number" value={configDraft.kb_max_results} min={1} max={20}
                        onChange={e => setConfigDraft(d => ({ ...d, kb_max_results: Number(e.target.value) }))} />
                    </label>
                    <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 13 }}>
                      <span style={{ fontSize: 11, color: "var(--muted)" }}>Min score threshold</span>
                      <input type="number" value={configDraft.kb_similarity_threshold} min={0} max={1} step={0.05}
                        onChange={e => setConfigDraft(d => ({ ...d, kb_similarity_threshold: Number(e.target.value) }))} />
                    </label>
                  </div>

                  <h4 style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 0.6, fontWeight: 700, marginTop: 8 }}>Reranking</h4>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 13 }}>
                    <input type="checkbox" checked={configDraft.kb_reranking_enabled}
                      onChange={e => setConfigDraft(d => ({ ...d, kb_reranking_enabled: e.target.checked }))} />
                    <span>Enable cross-encoder reranking</span>
                  </label>
                  <label style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 13 }}>
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>Reranker model</span>
                    <input value={configDraft.kb_reranking_model}
                      onChange={e => setConfigDraft(d => ({ ...d, kb_reranking_model: e.target.value }))} />
                    <span style={{ fontSize: 10, color: "var(--muted)" }}>pip install sentence-transformers</span>
                  </label>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 13, marginTop: 4 }}>
                    <input type="checkbox" checked={configDraft.kb_query_analytics}
                      onChange={e => setConfigDraft(d => ({ ...d, kb_query_analytics: e.target.checked }))} />
                    <span>Log query analytics</span>
                  </label>

                  {/* Live status */}
                  <div style={{ marginTop: 12, padding: "12px 14px", background: "var(--panel)", borderRadius: 8, border: "1px solid var(--line)" }}>
                    <div style={{ fontSize: 11, color: "var(--muted)", fontWeight: 700, marginBottom: 8 }}>LIVE STATUS</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                      {[
                        ["Embedding", embOk ? "Ready" : "Not ready", embOk ? "ok" : "err"],
                        ["BM25", kbStatus?.capabilities?.bm25_available ? "Installed" : "Missing", kbStatus?.capabilities?.bm25_available ? "ok" : "warn"],
                        ["Reranker", kbStatus?.capabilities?.reranker_available ? "Installed" : "Missing", kbStatus?.capabilities?.reranker_available ? "ok" : "warn"],
                        ["Search mode", kbStatus?.search_mode ?? "—", ""],
                        ["Chunk strategy", kbStatus?.chunk_strategy ?? "—", ""],
                      ].map(([k, v, badge]) => (
                        <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, alignItems: "center" }}>
                          <span style={{ color: "var(--muted)" }}>{k}</span>
                          {badge ? <span className={`rag-status-badge ${badge}`} style={{ fontSize: 10 }}>{v}</span> : <span style={{ fontWeight: 600 }}>{v}</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
const ADMIN_TOKEN_KEY = "nabil_admin_token";

function AdminLoginGate({ children }: { children: React.ReactNode }) {
  const [phase, setPhase] = useState<"verifying" | "login" | "authed">("verifying");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [adminUser, setAdminUser] = useState<string | null>(null);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem(ADMIN_TOKEN_KEY);
    if (!token) {
      setPhase("login");
      return;
    }
    let cancelled = false;
    adminVerify(token)
      .then((ok) => {
        if (cancelled) return;
        if (ok) {
          setPhase("authed");
        } else {
          localStorage.removeItem(ADMIN_TOKEN_KEY);
          setPhase("login");
        }
      })
      .catch(() => {
        if (!cancelled) setPhase("login");
      });
    return () => { cancelled = true; };
  }, []);

  const handleLogin = async () => {
    if (!username.trim() || !password || busy) return;
    setBusy(true);
    setLoginError(null);
    try {
      const result = await adminLogin(username.trim(), password);
      if (result.ok && result.token) {
        localStorage.setItem(ADMIN_TOKEN_KEY, result.token);
        setAdminUser(result.username ?? username.trim());
        setPassword("");
        setPhase("authed");
      } else {
        setLoginError(result.detail ?? "Invalid username or password.");
      }
    } catch (err) {
      setLoginError(err instanceof Error ? err.message : "Login failed. Is the backend running?");
    } finally {
      setBusy(false);
    }
  };

  const handleSignOut = async () => {
    const token = localStorage.getItem(ADMIN_TOKEN_KEY);
    localStorage.removeItem(ADMIN_TOKEN_KEY);
    setPhase("login");
    setAdminUser(null);
    if (token) {
      adminLogout(token).catch(() => {});
    }
  };

  if (phase === "verifying") {
    return (
      <div className="admin-login-wrap">
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, color: "var(--muted)" }}>
          <RefreshCw size={22} style={{ animation: "spin 1.2s linear infinite", color: "var(--teal)" }} />
          <span style={{ fontSize: 13 }}>Verifying admin session…</span>
        </div>
      </div>
    );
  }

  if (phase === "login") {
    return (
      <div className="admin-login-wrap">
        <div className="admin-login-card">
          <div>
            <div className="nabil-wordmark">NABIL</div>
            <div style={{ fontSize: 13, color: "var(--muted)", marginTop: 2 }}>Admin Console</div>
            <div className="nabil-gold-rule" />
          </div>
          {loginError && <div className="admin-login-error">{loginError}</div>}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <input
              placeholder="Username"
              value={username}
              autoComplete="username"
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleLogin(); }}
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              autoComplete="current-password"
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleLogin(); }}
            />
          </div>
          <button type="button" className="admin-login-btn" onClick={handleLogin} disabled={busy || !username.trim() || !password}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
          <p style={{ fontSize: 11, color: "var(--muted)", textAlign: "center", margin: 0 }}>
            Restricted area — authorised Nabil Bank staff only.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      <div style={{ position: "absolute", top: 22, right: 24, zIndex: 5, display: "flex", alignItems: "center", gap: 8 }}>
        {adminUser && <span style={{ fontSize: 11, color: "var(--muted)" }}>Signed in as <strong style={{ color: "var(--teal)" }}>{adminUser}</strong></span>}
        <button type="button" className="btn sm" onClick={handleSignOut} style={{ fontSize: 11, padding: "5px 12px" }}>
          Sign out
        </button>
      </div>
      {children}
    </div>
  );
}

const KYC_EXAMPLES = [
  "Customers with loans above 50 lakhs",
  "High risk customers",
  "Defaulters list",
  "Average balance by district"
];

function KycQueryPanel() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<KycQueryResult | null>(null);
  const [sqlOpen, setSqlOpen] = useState(false);
  const [dbStatus, setDbStatus] = useState<{ ok: boolean; connected: boolean; row_count: number } | null>(null);

  useEffect(() => {
    kycStatus().then(setDbStatus).catch(() => setDbStatus(null));
  }, []);

  const runQuery = async (q?: string) => {
    const text = (q ?? question).trim();
    if (!text || loading) return;
    if (q) setQuestion(q);
    setLoading(true);
    setResult(null);
    setSqlOpen(false);
    try {
      const res = await kycQuery(text);
      setResult(res);
    } catch (err) {
      setResult({ ok: false, error: err instanceof Error ? err.message : "KYC query failed." });
    } finally {
      setLoading(false);
    }
  };

  const rows = result?.rows ?? [];
  const visibleRows = rows.slice(0, 20);
  const columns = visibleRows.length ? Object.keys(visibleRows[0]) : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, maxWidth: 980 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <h3 style={{ margin: 0, fontSize: 16, fontWeight: 800 }}>🏦 KYC Data Console</h3>
        {dbStatus && (
          <span className={`pill ${dbStatus.connected ? "good" : "warn"}`} style={{ fontSize: 10 }}>
            {dbStatus.connected ? `● Connected · ${dbStatus.row_count.toLocaleString()} rows` : "○ Not connected"}
          </span>
        )}
      </div>
      <p style={{ margin: 0, fontSize: 12, color: "var(--muted)" }}>
        Ask questions about customer KYC data in plain English. The AI generates SQL, runs it read-only, and summarises the result.
      </p>

      <div style={{ display: "flex", gap: 8 }}>
        <input
          placeholder="Ask about customers in plain English…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") runQuery(); }}
          style={{ flex: 1, fontSize: 13 }}
        />
        <button type="button" className="rag-toolbar-btn primary" onClick={() => runQuery()} disabled={loading || !question.trim()}>
          {loading ? <RefreshCw size={13} style={{ animation: "spin 1s linear infinite" }} /> : <Search size={13} />}
          <span>{loading ? "Querying…" : "Ask"}</span>
        </button>
      </div>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {KYC_EXAMPLES.map((ex) => (
          <button key={ex} type="button" className="kyc-example-chip" onClick={() => runQuery(ex)} disabled={loading}>
            {ex}
          </button>
        ))}
      </div>

      {loading && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--muted)", fontSize: 13 }}>
          <RefreshCw size={16} style={{ animation: "spin 1.2s linear infinite", color: "var(--teal)" }} />
          Generating SQL and querying KYC database…
        </div>
      )}

      {result && !result.ok && (
        <div className="notice error" style={{ margin: 0 }}>
          <CircleAlert size={16} />
          <span>{result.error ?? "KYC query failed."}</span>
        </div>
      )}

      {result?.ok && (
        <>
          {result.answer && <div className="kyc-answer">{result.answer}</div>}

          {result.sql && (
            <div>
              <button type="button" className="link-btn" onClick={() => setSqlOpen((v) => !v)} style={{ fontSize: 12, cursor: "pointer", padding: 0 }}>
                {sqlOpen ? "▾ Hide generated SQL" : "▸ Show generated SQL"}
              </button>
              {sqlOpen && <div className="kyc-sql-block" style={{ marginTop: 8 }}>{result.sql}</div>}
            </div>
          )}

          {visibleRows.length > 0 ? (
            <div>
              <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6 }}>
                Showing {visibleRows.length} of {result.row_count ?? rows.length} rows
              </div>
              <div className="kyc-table-wrap">
                <table className="kyc-table">
                  <thead>
                    <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
                  </thead>
                  <tbody>
                    {visibleRows.map((row, i) => (
                      <tr key={i}>
                        {columns.map((c) => <td key={c}>{row[c] === null || row[c] === undefined ? "—" : String(row[c])}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            !result.answer && <p style={{ fontSize: 13, color: "var(--muted)", margin: 0 }}>No rows returned.</p>
          )}
        </>
      )}
    </div>
  );
}

function AdminView({
  providerStatus,
  voiceSocketStatus,
  systemMetrics,
  systemInfo,
  voices,
  galleryVoices,
  auditLogs,
  settings
}: {
  providerStatus: ProviderStatus[];
  voiceSocketStatus: VoiceSocketStatus | null;
  systemMetrics: SystemMetrics | null;
  systemInfo: SystemInfo | null;
  voices: VoicesResponse | null;
  galleryVoices: any[];
  auditLogs: any[];
  settings: BackendSettings;
}) {
  const [tab, setTab] = useState("monitor");

  const readiness = [
    { label: "Consent complete", ok: galleryVoices.every(v => v.consent_status === "completed"), fix: "Record consent for each voice profile" },
    { label: "Voice quality ≥70", ok: galleryVoices.some(v => Number(v.quality_score ?? 0) >= 70), fix: "Record more clean samples and retrain" },
    { label: "API keys secure", ok: true, fix: "" },
    { label: "Audio provenance", ok: true, fix: "" },
    { label: "Deletion policy set", ok: true, fix: "" },
    { label: "Backend reachable", ok: providerStatus.some(p => p.ok), fix: "Start the backend server" },
  ];

  const TABS = [
    { id: "monitor", label: "System Monitor", icon: "📊" },
    { id: "kyc", label: "KYC Data", icon: "🏦" },
    { id: "providers", label: "AI Providers", icon: "🤖" },
    { id: "voices", label: "Voice Models", icon: "🎙️" },
    { id: "rag", label: "RAG / Knowledge", icon: "📚" },
    { id: "audit", label: "Audit Logs", icon: "📋" },
    { id: "data", label: "Data & Storage", icon: "💾" },
    { id: "ready", label: "Readiness", icon: "✅" },
    { id: "debug", label: "Debug", icon: "🐛" },
  ];

  const MetricCard = ({ icon, label, value, sub, color }: { icon: string; label: string; value: string; sub?: string; color?: string }) => (
    <div style={{ padding: "18px 20px", background: "var(--panel)", borderRadius: 12, border: "1px solid var(--line)" }}>
      <div style={{ fontSize: 22, marginBottom: 8 }}>{icon}</div>
      <div style={{ fontSize: 28, fontWeight: 900, color: color ?? "var(--ink)", lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: 12, fontWeight: 600, color: "var(--muted)", marginTop: 4 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{sub}</div>}
    </div>
  );

  return (
    <section className="view-stack admin-view" style={{ padding: "0 0 40px" }}>
      {/* Header */}
      <div style={{ padding: "20px 24px 0", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>Admin Console</h2>
          <p style={{ margin: "4px 0 0", fontSize: 13, color: "var(--muted)" }}>System health, voice models, AI providers, logs, storage, and commercial readiness</p>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <span style={{ fontSize: 10, padding: "4px 10px", borderRadius: 20, background: "rgba(72,187,120,0.15)", color: "var(--green)", fontWeight: 800 }}>● LOCAL-FIRST</span>
        </div>
      </div>

      {/* Tab nav */}
      <div style={{ padding: "16px 24px 0", display: "flex", gap: 4, flexWrap: "wrap", borderBottom: "1px solid var(--line)" }}>
        {TABS.map(t => (
          <button key={t.id} type="button" onClick={() => setTab(t.id)}
            style={{ padding: "8px 14px", borderRadius: "8px 8px 0 0", border: "1px solid transparent", cursor: "pointer", fontSize: 12, fontWeight: 600, transition: "all 0.15s",
              background: tab === t.id ? "var(--panel)" : "transparent",
              color: tab === t.id ? "var(--teal)" : "var(--muted)",
              borderColor: tab === t.id ? "var(--line)" : "transparent",
              borderBottomColor: tab === t.id ? "var(--panel)" : "transparent",
              marginBottom: tab === t.id ? -1 : 0,
            }}>
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      <div style={{ padding: "24px" }}>

        {/* KYC DATA */}
        {tab === "kyc" && <KycQueryPanel />}

        {/* MONITOR */}
        {tab === "monitor" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 12, marginBottom: 24 }}>
              <MetricCard icon="💻" label="Operating System" value={systemInfo?.os ?? "—"} sub={systemInfo?.os_version} />
              <MetricCard icon="🔥" label="CPU Usage"
                value={typeof systemMetrics?.cpu_percent === "number" ? `${systemMetrics.cpu_percent.toFixed(0)}%` : "—"}
                color={typeof systemMetrics?.cpu_percent === "number" && systemMetrics.cpu_percent > 80 ? "var(--rose)" : "var(--ink)"}
                sub={typeof systemMetrics?.cpu_percent === "number" && systemMetrics.cpu_percent > 80 ? "High load" : "Normal"} />
              <MetricCard icon="🧠" label="RAM Used"
                value={typeof systemMetrics?.ram_used_gb === "number" ? `${systemMetrics.ram_used_gb.toFixed(1)}GB` : "—"}
                sub={typeof systemMetrics?.ram_total_gb === "number" ? `of ${systemMetrics.ram_total_gb.toFixed(0)}GB` : undefined} />
              <MetricCard icon="💾" label="Disk Free"
                value={typeof systemMetrics?.disk_free_gb === "number" ? `${systemMetrics.disk_free_gb.toFixed(0)}GB` : "—"}
                color={typeof systemMetrics?.disk_free_gb === "number" && systemMetrics.disk_free_gb < 5 ? "var(--rose)" : "var(--ink)"}
                sub={typeof systemMetrics?.disk_free_gb === "number" && systemMetrics.disk_free_gb < 5 ? "Low disk space" : "Available"} />
              <MetricCard icon="🎮" label="GPU / Accelerator"
                value={systemInfo?.gpu_mps_available ? "Apple MPS" : systemInfo?.cuda_available ? "CUDA" : "CPU only"}
                color={systemInfo?.gpu_mps_available || systemInfo?.cuda_available ? "var(--green)" : "var(--muted)"} />
              <MetricCard icon="🤖" label="Active Model" value={settings.local_model || settings.ollama_model || "—"} sub={`via ${settings.llm_provider ?? "local"}`} />
              <MetricCard icon="🔌" label="Voice Socket"
                value={voiceSocketStatus?.blocking_reasons?.length ? "Blocked" : "Ready"}
                color={voiceSocketStatus?.blocking_reasons?.length ? "var(--rose)" : "var(--green)"}
                sub={voiceSocketStatus?.blocking_reasons?.[0] ?? "All systems go"} />
              <MetricCard icon="📡" label="Recommendation"
                value={systemMetrics?.recommendation ? "✓" : "—"}
                sub={systemMetrics?.recommendation ?? "Run local defaults"} />
            </div>

            {/* Provider health grid */}
            <div style={{ fontSize: 12, fontWeight: 800, letterSpacing: 1.1, textTransform: "uppercase", color: "var(--teal)", marginBottom: 12, borderBottom: "1px solid var(--line)", paddingBottom: 6 }}>
              Provider Health
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 10 }}>
              {providerStatus.map(p => (
                <div key={p.name} style={{ padding: "12px 16px", background: "var(--panel)", borderRadius: 10, border: `1px solid ${p.ok ? "rgba(72,187,120,0.2)" : p.critical === false ? "rgba(245,158,11,0.2)" : "rgba(245,101,101,0.2)"}`, display: "flex", gap: 10, alignItems: "flex-start" }}>
                  <div style={{ width: 8, height: 8, borderRadius: "50%", marginTop: 5, flexShrink: 0, background: p.ok ? "var(--green)" : p.critical === false ? "#f59e0b" : "var(--rose)" }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 13, fontWeight: 700 }}>{p.name.replaceAll("_", " ")}</div>
                    <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>{p.detail}</div>
                    {!p.ok && p.fix && <code style={{ fontSize: 10, color: "var(--rose)", display: "block", marginTop: 4 }}>{p.fix}</code>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* PROVIDERS */}
        {tab === "providers" && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 14 }}>
            {[
              { name: "local", label: "Local Ollama", icon: "🦙", color: "var(--teal)", model: settings.ollama_model, url: settings.ollama_base_url, active: settings.llm_provider === "local" },
              { name: "openai", label: "OpenAI", icon: "🤖", color: "#10a37f", model: settings.openai_model, key: settings.openai_api_key, active: settings.llm_provider === "openai" },
              { name: "gemini", label: "Google Gemini", icon: "✨", color: "#4285f4", model: settings.gemini_model, key: settings.gemini_api_key, active: settings.llm_provider === "gemini" },
              { name: "elevenlabs", label: "ElevenLabs TTS", icon: "🎵", color: "#9b59b6", key: settings.elevenlabs_api_key, active: false },
            ].map(p => (
              <div key={p.name} style={{ padding: "18px", background: "var(--panel)", borderRadius: 12, border: `2px solid ${p.active ? p.color : "var(--line)"}`, transition: "border-color 0.2s" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                  <span style={{ fontSize: 24 }}>{p.icon}</span>
                  <div>
                    <div style={{ fontSize: 15, fontWeight: 700 }}>{p.label}</div>
                    {p.active && <span style={{ fontSize: 10, color: p.color, fontWeight: 800 }}>● ACTIVE PROVIDER</span>}
                  </div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 12 }}>
                  {p.model && <div style={{ display: "flex", gap: 8 }}><span style={{ color: "var(--muted)", width: 60 }}>Model</span><code style={{ fontSize: 11 }}>{p.model}</code></div>}
                  {p.url && <div style={{ display: "flex", gap: 8 }}><span style={{ color: "var(--muted)", width: 60 }}>URL</span><code style={{ fontSize: 11 }}>{p.url}</code></div>}
                  {"key" in p && <div style={{ display: "flex", gap: 8, alignItems: "center" }}><span style={{ color: "var(--muted)", width: 60 }}>API Key</span>
                    <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 10, background: p.key ? "rgba(72,187,120,0.15)" : "rgba(113,128,150,0.15)", color: p.key ? "var(--green)" : "var(--muted)", fontWeight: 700 }}>
                      {p.key ? "Configured ✓" : "Not set"}
                    </span>
                  </div>}
                </div>
                <div style={{ marginTop: 12, padding: "8px 12px", background: "var(--surface-2)", borderRadius: 8, fontSize: 11, color: "var(--muted)" }}>
                  {providerStatus.find(ps => ps.name.includes(p.name))?.detail ?? "Status not checked"}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* VOICES */}
        {tab === "voices" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 12 }}>
              {[...(voices?.voices ?? []), ...galleryVoices].map((voice, i) => {
                const ready = voice.status === "ready" || voice.publish_status === "published";
                return (
                  <div key={`${voice.id}-${i}`} style={{ padding: "16px 18px", background: "var(--panel)", borderRadius: 12, border: `1px solid ${ready ? "rgba(72,187,120,0.2)" : "var(--line)"}` }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                      <span style={{ fontSize: 18 }}>{voice.owner_name ? "👤" : "🔊"}</span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 14, fontWeight: 700 }}>{voice.name || voice.id}</div>
                        <div style={{ fontSize: 11, color: "var(--muted)" }}>{voice.owner_name ? `Owner: ${voice.owner_name}` : "Built-in voice"}</div>
                      </div>
                      <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: ready ? "rgba(72,187,120,0.15)" : "rgba(113,128,150,0.15)", color: ready ? "var(--green)" : "var(--muted)", fontWeight: 700 }}>
                        {voice.publish_status ?? voice.status ?? "Draft"}
                      </span>
                    </div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: "var(--surface-2)", color: "var(--muted)" }}>{voice.language ?? "auto"}</span>
                      <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: "var(--surface-2)", color: "var(--muted)" }}>{voice.engine ?? "piper"}</span>
                      <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: "var(--surface-2)", color: "var(--muted)" }}>Quality {Math.round(voice.quality_score ?? 100)}</span>
                    </div>
                    {voice.model_path && <code style={{ fontSize: 10, display: "block", marginTop: 8, color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{voice.model_path}</code>}
                  </div>
                );
              })}
              {(voices?.voices ?? []).length === 0 && galleryVoices.length === 0 && (
                <div style={{ textAlign: "center", padding: "32px", color: "var(--muted)", gridColumn: "1 / -1" }}>No voices found</div>
              )}
            </div>
          </div>
        )}

        {/* RAG */}
        {tab === "rag" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <div style={{ padding: "18px", background: "var(--panel)", borderRadius: 12, border: "1px solid var(--line)" }}>
                <div style={{ fontSize: 18, marginBottom: 10 }}>🗄️</div>
                <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 6 }}>Local ChromaDB</div>
                <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>Embedded vector database. No internet needed. Runs 100% offline.</div>
                <div style={{ display: "flex", gap: 6 }}><span style={{ fontSize: 11, padding: "3px 10px", borderRadius: 10, background: "rgba(72,187,120,0.15)", color: "var(--green)", fontWeight: 700 }}>Active ✓</span></div>
                <div style={{ marginTop: 12, fontSize: 12, color: "var(--muted)" }}>Path: <code style={{ fontSize: 11 }}>{settings.kb_chromadb_path}</code></div>
                <div style={{ marginTop: 6, fontSize: 12, color: "var(--muted)" }}>Embedding: <code style={{ fontSize: 11 }}>{settings.kb_embedding_provider} · {settings.kb_embedding_model}</code></div>
                <div style={{ marginTop: 6, fontSize: 12, color: "var(--muted)" }}>Search: <code style={{ fontSize: 11 }}>{settings.kb_search_mode}</code> · chunk size {settings.kb_chunk_size}</div>
              </div>
              <div style={{ padding: "18px", background: "var(--panel)", borderRadius: 12, border: "1px solid var(--line)" }}>
                <div style={{ fontSize: 18, marginBottom: 10 }}>🌐</div>
                <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 6 }}>Open WebUI RAG</div>
                <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>External RAG server. Requires Open WebUI running separately.</div>
                <div style={{ display: "flex", gap: 6 }}>
                  <span style={{ fontSize: 11, padding: "3px 10px", borderRadius: 10, background: settings.rag_enabled ? "rgba(72,187,120,0.15)" : "rgba(113,128,150,0.15)", color: settings.rag_enabled ? "var(--green)" : "var(--muted)", fontWeight: 700 }}>
                    {settings.rag_enabled ? "Enabled" : "Disabled"}
                  </span>
                </div>
                <div style={{ marginTop: 12, fontSize: 12, color: "var(--muted)" }}>URL: <code style={{ fontSize: 11 }}>{settings.open_webui_base_url}</code></div>
                <div style={{ marginTop: 6, fontSize: 12, color: "var(--muted)" }}>API Key: {settings.open_webui_api_key ? "Configured" : "Not set"}</div>
              </div>
            </div>
          </div>
        )}

        {/* AUDIT LOGS */}
        {tab === "audit" && (
          <div>
            <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
              <button className="rag-toolbar-btn" onClick={() => downloadBlob(new Blob([JSON.stringify(auditLogs, null, 2)], { type: "application/json" }), "audit-logs.json")}>
                <Download size={13} /><span>Export all logs</span>
              </button>
            </div>
            {auditLogs.length === 0 ? (
              <div style={{ textAlign: "center", padding: "48px", color: "var(--muted)" }}>
                <div style={{ fontSize: 32, marginBottom: 12 }}>📋</div>
                <div style={{ fontSize: 15, fontWeight: 600 }}>No audit events yet</div>
                <div style={{ fontSize: 13, marginTop: 6 }}>Actions like voice creation, consent, cloning, and publishing will appear here</div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {[...auditLogs].reverse().map((log, i) => {
                  const isGood = log.event?.includes("publish") || log.event?.includes("consent") || log.event?.includes("clone") || log.event?.includes("create");
                  const isWarn = log.event?.includes("delete") || log.event?.includes("error");
                  return (
                    <div key={i} style={{ padding: "12px 16px", background: "var(--panel)", borderRadius: 10, border: "1px solid var(--line)", display: "flex", gap: 12, alignItems: "flex-start" }}>
                      <div style={{ width: 8, height: 8, borderRadius: "50%", marginTop: 5, flexShrink: 0, background: isGood ? "var(--green)" : isWarn ? "var(--rose)" : "var(--teal)" }} />
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 13, fontWeight: 600 }}>{log.event}</div>
                        <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>{log.details}</div>
                      </div>
                      <div style={{ fontSize: 11, color: "var(--muted)", flexShrink: 0, fontFamily: "monospace" }}>{String(log.timestamp ?? "").slice(0, 16).replace("T", " ")}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* DATA STORAGE */}
        {tab === "data" && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 14 }}>
            {[
              { icon: "🗃️", title: "Voice recordings", desc: "Studio recordings stored as WAV files in .local/audio_work/. Used for training voice clones. Deletable per-profile.", status: "Local only" },
              { icon: "🔊", title: "Audio cache", desc: "Generated TTS audio cached to avoid re-synthesis. Stored in .local/audio_cache/. Safe to delete.", status: "Local only" },
              { icon: "📋", title: "Settings & config", desc: "Runtime settings stored in .local/settings.json (JSON fallback) or MySQL (if configured).", status: "Local + optional DB" },
              { icon: "🗄️", title: "ChromaDB vectors", desc: "Knowledge base embeddings stored in .local/chromadb/. Each collection is a folder.", status: "Local only" },
              { icon: "📝", title: "Conversation audio", desc: "Voice turn audio is NOT retained by default unless 'keep turn audio' is enabled.", status: "Not stored" },
              { icon: "🔐", title: "API keys", desc: "Stored in settings.json. Keys are masked in the UI. Never logged or sent to analytics.", status: "Encrypted at rest" },
            ].map(item => (
              <div key={item.title} style={{ padding: "18px", background: "var(--panel)", borderRadius: 12, border: "1px solid var(--line)" }}>
                <div style={{ fontSize: 24, marginBottom: 10 }}>{item.icon}</div>
                <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 6 }}>{item.title}</div>
                <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5, marginBottom: 10 }}>{item.desc}</div>
                <span style={{ fontSize: 10, padding: "3px 8px", borderRadius: 10, background: "rgba(0,166,81,0.15)", color: "var(--teal)", fontWeight: 700 }}>{item.status}</span>
              </div>
            ))}
          </div>
        )}

        {/* READINESS */}
        {tab === "ready" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 12, marginBottom: 24 }}>
              {readiness.map(({ label, ok, fix }) => (
                <div key={label} style={{ padding: "18px", background: "var(--panel)", borderRadius: 12, border: `1px solid ${ok ? "rgba(72,187,120,0.25)" : "rgba(245,101,101,0.25)"}` }}>
                  <div style={{ fontSize: 24, marginBottom: 8 }}>{ok ? "✅" : "❌"}</div>
                  <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 12, color: ok ? "var(--green)" : "var(--rose)", fontWeight: 600 }}>{ok ? "Ready" : "Needs attention"}</div>
                  {!ok && fix && <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 6 }}>Fix: {fix}</div>}
                </div>
              ))}
            </div>

            <div style={{ padding: "18px 20px", background: "var(--panel)", borderRadius: 12, border: "1px solid var(--line)" }}>
              <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Commercial Deployment Checklist</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12 }}>
                {[
                  "All voice profiles have signed consent",
                  "Voice clones quality-checked at ≥70/100",
                  "System prompt reviewed for your use case",
                  "API keys stored securely, not committed to git",
                  "Audio retention policy documented and set",
                  "Backend running behind HTTPS (not plain HTTP)",
                  "Regular backup of .local/ directory configured",
                ].map((item, i) => (
                  <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", padding: "6px 0", borderBottom: i < 6 ? "1px solid var(--line)" : "none" }}>
                    <CheckCircle2 size={14} style={{ color: "var(--teal)", flexShrink: 0 }} />
                    <span>{item}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* DEBUG */}
        {tab === "debug" && (
          <div>
            <div style={{ marginBottom: 12, display: "flex", gap: 8 }}>
              <button className="rag-toolbar-btn" onClick={() => downloadBlob(new Blob([JSON.stringify({ voiceSocketStatus, providerStatus, settings: { ...settings, openai_api_key: "***", gemini_api_key: "***" } }, null, 2)], { type: "application/json" }), "debug-dump.json")}>
                <Download size={13} /><span>Download debug dump</span>
              </button>
            </div>
            <div style={{ display: "grid", gap: 12 }}>
              {[
                { label: "Voice Socket Status", data: voiceSocketStatus },
                { label: "Provider Status", data: providerStatus },
                { label: "System Info", data: systemInfo },
                { label: "System Metrics", data: systemMetrics },
                { label: "Settings (keys masked)", data: { ...settings, openai_api_key: settings.openai_api_key ? "sk-****" : "", gemini_api_key: settings.gemini_api_key ? "AIza-****" : "", open_webui_api_key: settings.open_webui_api_key ? "****" : "", elevenlabs_api_key: settings.elevenlabs_api_key ? "****" : "" } },
              ].map(({ label, data }) => (
                <details key={label} style={{ background: "var(--panel)", borderRadius: 10, border: "1px solid var(--line)", overflow: "hidden" }}>
                  <summary style={{ padding: "12px 16px", cursor: "pointer", fontSize: 13, fontWeight: 600, userSelect: "none" }}>{label}</summary>
                  <pre style={{ margin: 0, padding: "12px 16px", fontSize: 11, overflowX: "auto", borderTop: "1px solid var(--line)", background: "var(--surface-2)", color: "var(--ink)" }}>
                    {JSON.stringify(data, null, 2)}
                  </pre>
                </details>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
function LogsView({
  appLogs,
  auditLogs,
  providerStatus,
  voiceSocketStatus,
  onClear
}: {
  appLogs: any[];
  auditLogs: any[];
  providerStatus: ProviderStatus[];
  voiceSocketStatus: VoiceSocketStatus | null;
  onClear: () => void;
}) {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const providerLogs = providerStatus.map((p) => ({
    id: `provider-${p.name}`, timestamp: new Date().toISOString(),
    level: p.ok ? "success" : p.critical === false ? "warning" : "error",
    event: `provider_${p.name}`, detail: p.ok ? p.detail : `${p.detail}${p.fix ? ` Fix: ${p.fix}` : ""}`, source: "runtime",
  }));
  const socketLogs = [
    ...(voiceSocketStatus?.blocking_reasons ?? []).map((detail, i) => ({ id: `socket-blocker-${i}`, timestamp: new Date().toISOString(), level: "error", event: "voice_socket_blocker", detail, source: "runtime" })),
    ...(voiceSocketStatus?.warnings ?? []).map((detail, i) => ({ id: `socket-warning-${i}`, timestamp: new Date().toISOString(), level: "warning", event: "voice_socket_warning", detail, source: "runtime" })),
  ];
  const backendAudit = auditLogs.map((log, i) => ({ id: log.id || `audit-${i}`, timestamp: log.timestamp, level: "info", event: log.event, detail: log.details || "", source: "backend audit" }));
  const merged = [...appLogs.map(l => ({ ...l, source: "browser app" })), ...backendAudit, ...socketLogs, ...providerLogs]
    .sort((a, b) => String(b.timestamp || "").localeCompare(String(a.timestamp || "")));

  const filtered = merged
    .filter(l => filter === "all" || l.level === filter)
    .filter(l => !search || l.event?.toLowerCase().includes(search.toLowerCase()) || l.detail?.toLowerCase().includes(search.toLowerCase()));

  const counts = { error: merged.filter(l => l.level === "error").length, warning: merged.filter(l => l.level === "warning").length, success: merged.filter(l => l.level === "success").length, info: merged.filter(l => l.level === "info").length };

  const LEVEL_COLOR: Record<string, string> = { error: "var(--rose)", warning: "#f59e0b", success: "var(--green)", info: "var(--teal)" };
  const LEVEL_BG: Record<string, string> = { error: "rgba(245,101,101,0.08)", warning: "rgba(245,158,11,0.08)", success: "rgba(72,187,120,0.08)", info: "rgba(0,166,81,0.08)" };

  return (
    <section className="view-stack logs-view" style={{ padding: "0 0 40px" }}>
      <div style={{ padding: "20px 24px 0", display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>System Logs</h2>
          <p style={{ margin: "4px 0 0", fontSize: 13, color: "var(--muted)" }}>Errors, voice events, provider health, socket status, and audit trail</p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="rag-toolbar-btn" onClick={() => downloadBlob(new Blob([JSON.stringify(merged, null, 2)], { type: "application/json" }), "swarlocal-logs.json")}>
            <Download size={13} /><span>Export all</span>
          </button>
          <button className="rag-toolbar-btn" onClick={onClear} style={{ color: "var(--rose)", borderColor: "var(--rose)" }}>
            <Trash2 size={13} /><span>Clear browser logs</span>
          </button>
        </div>
      </div>

      {/* Stats row */}
      <div style={{ padding: "16px 24px 0", display: "flex", gap: 10, flexWrap: "wrap" }}>
        {[
          { level: "error", label: "Errors", count: counts.error, icon: "🔴" },
          { level: "warning", label: "Warnings", count: counts.warning, icon: "🟡" },
          { level: "success", label: "Success", count: counts.success, icon: "🟢" },
          { level: "info", label: "Info", count: counts.info, icon: "🔵" },
        ].map(s => (
          <div key={s.level} style={{ padding: "12px 18px", background: "var(--panel)", borderRadius: 10, border: `1px solid ${filter === s.level ? LEVEL_COLOR[s.level] : "var(--line)"}`, cursor: "pointer", transition: "all 0.15s" }}
            onClick={() => setFilter(filter === s.level ? "all" : s.level)}>
            <div style={{ fontSize: 22, fontWeight: 900, color: LEVEL_COLOR[s.level] }}>{s.count}</div>
            <div style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600 }}>{s.icon} {s.label}</div>
          </div>
        ))}
        <div style={{ flex: 1, minWidth: 200, padding: "12px 18px", background: "var(--panel)", borderRadius: 10, border: "1px solid var(--line)" }}>
          <div style={{ fontSize: 22, fontWeight: 900 }}>{merged.length}</div>
          <div style={{ fontSize: 11, color: "var(--muted)", fontWeight: 600 }}>📊 Total events</div>
        </div>
      </div>

      {/* Filter + search bar */}
      <div style={{ padding: "14px 24px 0", display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
        <div style={{ flex: 1, minWidth: 200, display: "flex", gap: 0, border: "1px solid var(--line)", borderRadius: 8, overflow: "hidden" }}>
          <span style={{ padding: "8px 12px", background: "var(--panel)", color: "var(--muted)", fontSize: 13, display: "flex", alignItems: "center" }}>
            <Search size={14} />
          </span>
          <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search logs…"
            style={{ flex: 1, border: "none", background: "var(--panel)", padding: "8px 12px", fontSize: 13 }} />
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {["all", "error", "warning", "success", "info"].map(lvl => (
            <button key={lvl} type="button" onClick={() => setFilter(lvl)}
              style={{ padding: "7px 12px", borderRadius: 7, border: "1px solid", cursor: "pointer", fontSize: 12, fontWeight: 600, transition: "all 0.15s",
                background: filter === lvl ? (LEVEL_BG[lvl] ?? "var(--teal-soft)") : "transparent",
                color: filter === lvl ? (LEVEL_COLOR[lvl] ?? "var(--teal)") : "var(--muted)",
                borderColor: filter === lvl ? (LEVEL_COLOR[lvl] ?? "var(--teal)") : "var(--line)",
              }}>{lvl}</button>
          ))}
        </div>
      </div>

      {/* Log list */}
      <div style={{ padding: "16px 24px 0" }}>
        <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 10 }}>
          Showing {filtered.length} of {merged.length} events {search && `matching "${search}"`}
        </div>
        {filtered.length === 0 ? (
          <div style={{ textAlign: "center", padding: "48px", color: "var(--muted)" }}>
            <FileText size={28} style={{ marginBottom: 12, opacity: 0.3 }} />
            <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>No logs match this filter</div>
            <div style={{ fontSize: 12 }}>Try a different level or clear the search</div>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {filtered.map((log, i) => {
              const id = `${log.id ?? i}-${i}`;
              const expanded = expandedId === id;
              const lc = LEVEL_COLOR[log.level] ?? "var(--muted)";
              return (
                <div key={id} style={{ background: LEVEL_BG[log.level] ?? "var(--panel)", borderRadius: 10, border: `1px solid ${lc}33`, overflow: "hidden", cursor: "pointer" }}
                  onClick={() => setExpandedId(expanded ? null : id)}>
                  <div style={{ padding: "10px 16px", display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", flexShrink: 0, background: lc }} />
                    <span style={{ fontSize: 10, fontWeight: 800, color: lc, textTransform: "uppercase", width: 55, flexShrink: 0 }}>{log.level}</span>
                    <strong style={{ fontSize: 13, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {String(log.event || "event").replaceAll("_", " ")}
                    </strong>
                    <span style={{ fontSize: 10, color: "var(--muted)", flexShrink: 0 }}>{log.source}</span>
                    <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "monospace", flexShrink: 0 }}>
                      {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : "now"}
                    </span>
                    <span style={{ fontSize: 10, color: "var(--muted)" }}>{expanded ? "▲" : "▼"}</span>
                  </div>
                  {expanded && (
                    <div style={{ padding: "0 16px 12px", borderTop: `1px solid ${lc}22` }}>
                      <p style={{ fontSize: 12, color: "var(--ink)", margin: "10px 0 0", lineHeight: 1.6 }}>{log.detail}</p>
                      {log.meta && <code style={{ fontSize: 10, display: "block", marginTop: 8, color: "var(--muted)" }}>{JSON.stringify(log.meta)}</code>}
                      <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 8 }}>
                        Full timestamp: {log.timestamp ? new Date(log.timestamp).toLocaleString() : "unknown"}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
function SetupView({
  providerStatus,
  voiceSocketState,
  voiceSocketStatus,
  onRefresh,
  onTestVoiceSocket,
  onTtsTest
}: {
  providerStatus: ProviderStatus[];
  voiceSocketState: VoiceSocketConnectionState;
  voiceSocketStatus: VoiceSocketStatus | null;
  onRefresh: () => void;
  onTestVoiceSocket: () => void;
  onTtsTest: (language: "ne" | "en", voiceId?: string) => Promise<void> | void;
}) {
  const blockers = providerStatus.filter((provider) => !provider.ok && provider.critical !== false);
  const others = providerStatus.filter((provider) => provider.ok || provider.critical === false);
  return (
    <section className="view-stack">
      <div className="view-header">
        <div>
          <h2>First-Run Setup</h2>
          <p>Backend, Ollama, Piper, voices, microphone.</p>
        </div>
        <button className="icon-text" onClick={onRefresh} type="button">
          <Activity size={18} />
          <span>Refresh</span>
        </button>
      </div>
      {blockers.length ? (
        <div className="notice error setup-blockers">
          <CircleAlert size={18} />
          <span>{blockers.length} critical setup blocker{blockers.length === 1 ? "" : "s"} must be fixed before voice turns are ready.</span>
        </div>
      ) : (
        <div className="notice success">
          <CheckCircle2 size={18} />
          <span>Critical runtime checks passed.</span>
        </div>
      )}
      <article className={voiceSocketStatus?.blocking_reasons.length ? "status-card socket-card blocked" : "status-card socket-card"}>
        {voiceSocketStatus?.blocking_reasons.length ? <CircleAlert size={20} /> : <CheckCircle2 size={20} />}
        <div>
          <strong>Voice Socket</strong>
          <span>{voiceSocketState.replaceAll("_", " ")}</span>
          {voiceSocketStatus?.blocking_reasons.length ? <code>{voiceSocketStatus.blocking_reasons.join(", ")}</code> : <span>Handshake route is available.</span>}
          {voiceSocketStatus?.warnings.length ? <small>{voiceSocketStatus.warnings.join(" ")}</small> : null}
        </div>
        <button className="icon-text" onClick={onTestVoiceSocket} type="button">
          <Radio size={18} />
          <span>Test Voice Socket</span>
        </button>
      </article>
      <div className="status-grid">
        {[...blockers, ...others].map((provider) => (
          <article className="status-card" key={provider.name}>
            {provider.ok ? <CheckCircle2 size={20} /> : <CircleAlert size={20} />}
            <div>
              <strong>{provider.name.replaceAll("_", " ")}</strong>
              <span>{provider.detail}</span>
              {!provider.ok && provider.fix ? <code>{provider.fix}</code> : null}
            </div>
          </article>
        ))}
      </div>
      <div className="action-band">
        <button className="icon-text" onClick={() => { Promise.resolve(onTtsTest("ne")).catch(() => {}); }} type="button">
          <Volume2 size={18} />
          <span>Nepali TTS</span>
        </button>
        <button className="icon-text" onClick={() => { Promise.resolve(onTtsTest("en")).catch(() => {}); }} type="button">
          <Volume2 size={18} />
          <span>English TTS</span>
        </button>
        <button
          className="icon-text"
          onClick={() => navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => stream.getTracks().forEach((track) => track.stop()))}
          type="button"
        >
          <Mic size={18} />
          <span>Mic Test</span>
        </button>
      </div>
    </section>
  );
}

// ── Voice Studio: friendly guidance helpers ─────────────────────────────────
type CalloutTone = "info" | "tip" | "warn" | "guide";

function GuideCallout({
  tone = "info",
  icon,
  title,
  children,
}: {
  tone?: CalloutTone;
  icon?: React.ReactNode;
  title?: string;
  children: React.ReactNode;
}) {
  const tones: Record<CalloutTone, { bg: string; border: string; fg: string }> = {
    info: { bg: "rgba(96,165,250,0.10)", border: "rgba(96,165,250,0.35)", fg: "var(--blue)" },
    tip: { bg: "var(--teal-soft)", border: "rgba(0,166,81,0.35)", fg: "var(--teal)" },
    warn: { bg: "var(--amber-soft)", border: "rgba(201,162,39,0.4)", fg: "var(--amber)" },
    guide: { bg: "var(--purple-soft)", border: "rgba(167,139,250,0.35)", fg: "var(--purple)" },
  };
  const t = tones[tone];
  return (
    <div className="studio-callout" style={{ background: t.bg, border: `1px solid ${t.border}` }}>
      {icon && <span style={{ color: t.fg, flexShrink: 0, marginTop: 1, display: "inline-flex" }}>{icon}</span>}
      <div style={{ flex: 1, minWidth: 0 }}>
        {title && <div style={{ fontWeight: 700, fontSize: 13, color: t.fg, marginBottom: 4 }}>{title}</div>}
        <div className="studio-callout-body">{children}</div>
      </div>
    </div>
  );
}

function FieldHint({ children }: { children: React.ReactNode }) {
  return <div className="field-hint">{children}</div>;
}

function UseCaseCard({ icon, title, body, example }: { icon: React.ReactNode; title: string; body: string; example: string }) {
  return (
    <div className="usecase-card">
      <div className="usecase-icon">{icon}</div>
      <strong>{title}</strong>
      <span>{body}</span>
      <div className="usecase-example"><b>Example:</b> {example}</div>
    </div>
  );
}

// Friendly names + install hints for the cloning engines (keyed by engine id).
const ENGINE_LABELS: Record<string, string> = {
  chatterbox: "Chatterbox local clone",
  piper: "Piper fine-tune",
  elevenlabs: "ElevenLabs (cloud)",
  f5_tts: "F5-TTS",
  openvoice: "OpenVoice",
  voxcpm: "VoxCPM",
};
const ENGINE_INSTALL_HINT: Record<string, string> = {
  chatterbox: "pip install chatterbox-tts",
  f5_tts: "pip install f5-tts",
  openvoice: "pip install openvoice",
  voxcpm: "pip install voxcpm",
};
function fmtDuration(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

// Unified, friendly progress stepper for the Create Voice flow.
const WIZARD_STEPS = ["Details", "Consent", "Record", "Publish"];
function WizardStepper({ current }: { current: number }) {
  return (
    <div className="wizard-progress" role="list" aria-label="Voice creation progress">
      {WIZARD_STEPS.map((label, i) => {
        const step = i + 1;
        const state = step < current ? "done" : step === current ? "active" : "upcoming";
        return (
          <div
            className={`wizard-step ${state}`}
            role="listitem"
            aria-current={state === "active" ? "step" : undefined}
            key={label}
          >
            <span className="wizard-step-badge">{state === "done" ? <Check size={14} /> : step}</span>
            <span className="wizard-step-label">{label}</span>
            {i < WIZARD_STEPS.length - 1 && <span className="wizard-step-line" aria-hidden="true" />}
          </div>
        );
      })}
    </div>
  );
}

const STUDIO_HOWTO = [
  { n: 1, icon: "🎤", title: "Record", desc: "Read a few short sentences aloud in your normal voice." },
  { n: 2, icon: "✨", title: "Build", desc: "The studio cleans your audio and builds an AI voice from it." },
  { n: 3, icon: "🔊", title: "Listen", desc: "Preview a sample to check it sounds like you." },
  { n: 4, icon: "🚀", title: "Publish", desc: "Activate the voice so the assistant can speak with it." },
];

// Shared status banner for the Voice Studio (busy / ok / error feedback).
function StudioStatusBanner({ msg, onClose }: { msg: { kind: "ok" | "err" | "busy"; text: string } | null; onClose: () => void }) {
  if (!msg) return null;
  return (
    <div className={`studio-status ${msg.kind}`}>
      {msg.kind === "busy" ? <RefreshCw size={18} className="spin" />
        : msg.kind === "ok" ? <CheckCircle2 size={18} />
        : <CircleAlert size={18} />}
      <span className="studio-status-text">{msg.text}</span>
      {msg.kind !== "busy" && (
        <button className="studio-status-close" onClick={onClose} type="button" aria-label="Dismiss">✕</button>
      )}
    </div>
  );
}

// Pre-flight readiness — surfaces mic + engine problems BEFORE the user records,
// so nobody records a dozen takes only to hit an error at build time.
function RecordingPreflight({
  micState,
  engineId,
  engineInfo,
}: {
  micState: "unknown" | "granted" | "denied" | "unavailable";
  engineId: string;
  engineInfo: any | null;
}) {
  const engineLabel = ENGINE_LABELS[engineId] ?? engineId;
  const isCloud = engineId === "elevenlabs";
  // engineInfo === null → status not loaded yet; don't alarm the user.
  const engineMissing = engineInfo ? engineInfo.installed === false : false;

  const checks: { key: string; state: "ok" | "warn" | "bad" | "idle"; icon: React.ReactNode; label: string; detail: React.ReactNode }[] = [
    {
      key: "mic",
      state: micState === "granted" ? "ok" : micState === "denied" || micState === "unavailable" ? "bad" : "idle",
      icon: micState === "denied" || micState === "unavailable" ? <MicOff size={16} /> : <Mic size={16} />,
      label: "Microphone",
      detail:
        micState === "granted" ? "Ready — access granted."
        : micState === "denied" ? "Blocked. Click the 🔒/mic icon in your browser's address bar and allow the microphone, then reload."
        : micState === "unavailable" ? "No microphone detected, or your browser blocks recording on this page."
        : "We'll ask for access the first time you hit record — that's normal.",
    },
    {
      key: "engine",
      state: engineMissing ? "bad" : engineInfo ? "ok" : "idle",
      icon: engineMissing ? <AlertTriangle size={16} /> : <Cpu size={16} />,
      label: `Voice engine — ${engineLabel}`,
      detail: engineMissing ? (
        <>
          Not installed yet, so <b>building the voice will fail</b> after you record. Install it first:{" "}
          {ENGINE_INSTALL_HINT[engineId] ? <code>{ENGINE_INSTALL_HINT[engineId]}</code> : "see the project README"}.
          {" "}You can still record now and build once it's installed.
        </>
      ) : isCloud ? "Cloud engine — make sure your API key is set in Settings before building."
        : engineInfo ? "Installed and ready to build your voice."
        : "Checking engine availability…",
    },
  ];

  const hasBlocker = checks.some((c) => c.state === "bad");
  return (
    <div className={`preflight ${hasBlocker ? "has-issue" : "ready"}`}>
      <div className="preflight-head">
        {hasBlocker ? <AlertTriangle size={15} /> : <CheckCircle2 size={15} />}
        <strong>{hasBlocker ? "Before you record — a couple of things to fix" : "You're set up to record"}</strong>
      </div>
      <div className="preflight-rows">
        {checks.map((c) => (
          <div className={`preflight-row ${c.state}`} key={c.key}>
            <span className="preflight-icon">{c.icon}</span>
            <span className="preflight-label">{c.label}</span>
            <span className="preflight-detail">{c.detail}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function VoiceStudioView({
  voices,
  galleryVoices,
  auditLogs,
  onRefresh,
  onTtsTest,
  ttsTestingId
}: {
  voices: VoicesResponse | null;
  galleryVoices: any[];
  auditLogs: any[];
  onRefresh: () => void;
  onTtsTest: (language: "ne" | "en", voiceId?: string) => Promise<void> | void;
  ttsTestingId: string | null;
}) {
  const [selectedVoice, setSelectedVoice] = useState<any | null>(null);
  const [cleaningAll, setCleaningAll] = useState(false);
  const [cloningVoice, setCloningVoice] = useState(false);
  const [previewingVoice, setPreviewingVoice] = useState(false);
  const [wizardStep, setWizardStep] = useState<"gallery" | "identity" | "consent" | "recordings">("gallery");
  const [activeFilter, setActiveFilter] = useState<string>("All");

  // Wizard States
  const [name, setName] = useState("");
  const [ownerName, setOwnerName] = useState("");
  const [ownerEmail, setOwnerEmail] = useState("");
  const [organization, setOrganization] = useState("");
  const [language, setLanguage] = useState("ne");
  const [engine, setEngine] = useState("chatterbox");
  const [recordingGoal, setRecordingGoal] = useState("better");
  const [commercialAllowed, setCommercialAllowed] = useState(false);
  const [signature, setSignature] = useState("");
  const [isDownloading, setIsDownloading] = useState(false);

  // Recordings states
  const [prompts, setPrompts] = useState<any[]>([]);
  const [recordings, setRecordings] = useState<any[]>([]);
  const [activePrompt, setActivePrompt] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const selectedVoiceIdValue = selectedVoice?.id || selectedVoice?.voice_id;
  const selectedVoiceConsentComplete = selectedVoice?.consent_status === "completed";

  // Live recording feedback (waveform + timer) and pre-flight readiness
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const waveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [recordElapsed, setRecordElapsed] = useState(0);
  const [micState, setMicState] = useState<"unknown" | "granted" | "denied" | "unavailable">("unknown");
  const [engines, setEngines] = useState<Record<string, any> | null>(null);
  const [studioMsg, setStudioMsg] = useState<{ kind: "ok" | "err" | "busy"; text: string } | null>(null);

  // Load prompts & recordings if a custom voice is selected
  useEffect(() => {
    const voiceId = selectedVoice?.id || selectedVoice?.voice_id;
    if (voiceId) {
      getVoicesPrompts().then(setPrompts).catch(() => {});
      getVoiceRecordings(voiceId).then(setRecordings).catch(() => {});
    }
  }, [selectedVoice]);

  // Pre-flight: when the recording step opens, check engine availability and mic
  // permission up front so issues surface BEFORE the user records anything.
  useEffect(() => {
    if (wizardStep !== "recordings") return;
    getCloningEngines().then(setEngines).catch(() => setEngines(null));
    let cancelled = false;
    let permStatus: any = null;
    const apply = (state: string) =>
      !cancelled && setMicState(state === "granted" ? "granted" : state === "denied" ? "denied" : "unknown");
    (async () => {
      if (!navigator.mediaDevices?.getUserMedia) { if (!cancelled) setMicState("unavailable"); return; }
      try {
        const perms: any = navigator.permissions;
        if (perms?.query) {
          permStatus = await perms.query({ name: "microphone" as PermissionName });
          apply(permStatus.state);
          permStatus.onchange = () => apply(permStatus.state);
        }
      } catch { /* Safari/Firefox may not support the 'microphone' permission query */ }
    })();
    return () => { cancelled = true; if (permStatus) permStatus.onchange = null; };
  }, [wizardStep]);

  // Tear down the live waveform + timer + audio graph after a take.
  const stopRecordingInternals = () => {
    if (rafRef.current != null) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    if (timerRef.current != null) { clearInterval(timerRef.current); timerRef.current = null; }
    if (audioCtxRef.current) { audioCtxRef.current.close().catch(() => {}); audioCtxRef.current = null; }
    analyserRef.current = null;
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
  };

  // Draw an animated frequency-bar waveform from the live mic analyser.
  const drawWaveform = () => {
    const canvas = waveCanvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) { rafRef.current = requestAnimationFrame(drawWaveform); return; }
    const ctx = canvas.getContext("2d");
    if (!ctx) { rafRef.current = requestAnimationFrame(drawWaveform); return; }
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 320;
    const cssH = canvas.clientHeight || 64;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    const w = canvas.width, h = canvas.height;
    const bins = analyser.frequencyBinCount;
    const data = new Uint8Array(bins);
    analyser.getByteFrequencyData(data);
    ctx.clearRect(0, 0, w, h);
    const bars = 56;
    const usable = Math.floor(bins * 0.7); // skip the very-high empty bins
    const barW = w / bars;
    for (let i = 0; i < bars; i++) {
      const v = data[Math.floor((i / bars) * usable)] / 255;
      const bh = Math.max(2 * dpr, v * h * 0.95);
      const x = i * barW;
      const grad = ctx.createLinearGradient(0, h, 0, h - bh);
      grad.addColorStop(0, "rgba(0,166,81,0.45)");
      grad.addColorStop(1, "#00d36a");
      ctx.fillStyle = grad;
      const r = Math.min(barW / 2 - dpr, 3 * dpr);
      const bx = x + dpr, bw = barW - 2 * dpr, by = h - bh;
      ctx.beginPath();
      ctx.moveTo(bx + r, by);
      ctx.arcTo(bx + bw, by, bx + bw, by + r, r);
      ctx.lineTo(bx + bw, h);
      ctx.lineTo(bx, h);
      ctx.lineTo(bx, by + r);
      ctx.arcTo(bx, by, bx + r, by, r);
      ctx.closePath();
      ctx.fill();
    }
    rafRef.current = requestAnimationFrame(drawWaveform);
  };

  // Tear everything down if the component unmounts mid-recording.
  useEffect(() => () => stopRecordingInternals(), []);

  const [creatingVoice, setCreatingVoice] = useState(false);

  // Wraps onTtsTest with clear busy/done feedback so a slow first-run model
  // download never looks frozen (and re-clicks can't pile up more downloads).
  const handleTestVoice = async (language: "ne" | "en", voiceId?: string, isLocalClone = false) => {
    if (ttsTestingId) return; // a test is already running — ignore extra clicks
    setStudioMsg({
      kind: "busy",
      text: isLocalClone
        ? "Generating a preview with this voice… the first time, the local voice model (~1–2 GB) downloads, so this can take a few minutes. It only happens once — later previews are quick."
        : "Generating a quick voice sample…",
    });
    try {
      await onTtsTest(language, voiceId);
      setStudioMsg({ kind: "ok", text: "Preview ready — playing now. If you can't hear it, check your volume." });
    } catch (err: any) {
      setStudioMsg({ kind: "err", text: err?.message || "Couldn't generate a preview. Please try again in a moment." });
    }
  };

  const handleCreateVoice = async (e: React.FormEvent) => {
    e.preventDefault();
    if (creatingVoice) return; // guard against double-submit creating duplicate profiles
    setCreatingVoice(true);
    try {
      const resp = await createVoice({
        name,
        owner_name: ownerName,
        owner_email: ownerEmail || undefined,
        organization: organization || undefined,
        language,
        engine,
        commercial_allowed: commercialAllowed
      });
      // Auto-consent: local module, no consent prompt needed — but the BACKEND
      // must record it too, or clone/publish will be rejected later.
      try {
        await saveVoiceConsent(resp.voice_id, ownerName || name);
      } catch {
        // non-fatal: user can still sign consent from the gallery card
      }
      setSelectedVoice({ ...resp, id: resp.voice_id, consent_status: "completed", publish_status: "unpublished", language, owner_name: ownerName, engine });
      setWizardStep("recordings");
    } catch (err: any) {
      alert(err.message || "Failed to create voice profile");
    } finally {
      setCreatingVoice(false);
    }
  };

  const handleSaveConsent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!signature.trim()) {
      alert("Signature is required");
      return;
    }
    try {
      if (!selectedVoiceIdValue) {
        throw new Error("Voice profile ID is missing. Please go back to gallery and try again.");
      }
      await saveVoiceConsent(selectedVoiceIdValue, signature);
      setSelectedVoice((prev: any) => ({ ...prev, consent_status: "completed" }));
      setWizardStep("recordings");
    } catch (err: any) {
      alert(err.message || "Failed to save consent");
    }
  };

  const handleStartRecord = async (promptId: string) => {
    if (!selectedVoiceIdValue) {
      setStudioMsg({ kind: "err", text: "Voice profile ID is missing. Go back to the gallery and reopen this voice." });
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 44100 } });
    } catch (err: any) {
      // Surface the real reason instead of a scary generic alert.
      const denied = err?.name === "NotAllowedError" || err?.name === "SecurityError";
      setMicState(denied ? "denied" : "unavailable");
      setStudioMsg({
        kind: "err",
        text: denied
          ? "Microphone access was blocked. Click the mic / lock icon in your browser's address bar, allow the microphone, then try again."
          : "No microphone is available. Plug one in (or check your system settings) and try again.",
      });
      return;
    }
    try {
      setStudioMsg(null);
      setMicState("granted");
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      chunksRef.current = [];
      setActivePrompt(promptId);

      // Live waveform: tap the mic into an analyser node.
      try {
        const AC: typeof AudioContext = (window as any).AudioContext || (window as any).webkitAudioContext;
        const ctx = new AC();
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.7;
        source.connect(analyser);
        audioCtxRef.current = ctx;
        analyserRef.current = analyser;
        rafRef.current = requestAnimationFrame(drawWaveform);
      } catch { /* waveform is non-essential — recording still works without it */ }

      // Live timer.
      setRecordElapsed(0);
      const startedAt = Date.now();
      timerRef.current = window.setInterval(() => {
        setRecordElapsed(Math.floor((Date.now() - startedAt) / 1000));
      }, 250);

      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.onstop = async () => {
        stopRecordingInternals();
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        try {
          await uploadVoiceRecording(selectedVoiceIdValue, promptId, blob);
          const updated = await getVoiceRecordings(selectedVoiceIdValue);
          setRecordings(updated);
        } catch (err: any) {
          setStudioMsg({ kind: "err", text: err.message || "Couldn't save that recording. Please try again." });
        }
        setActivePrompt(null);
        setRecordElapsed(0);
      };
      recorder.start();
    } catch (err: any) {
      stopRecordingInternals();
      setActivePrompt(null);
      setStudioMsg({ kind: "err", text: err.message || "Couldn't start recording. Please try again." });
    }
  };

  const handleStopRecord = () => {
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop(); // onstop tears down internals + uploads
    } else {
      stopRecordingInternals();
      setActivePrompt(null);
    }
  };

  const handleDeleteRecord = async (promptId: string) => {
    try {
      if (!selectedVoiceIdValue) {
        throw new Error("Voice profile ID is missing.");
      }
      await deleteVoiceRecording(selectedVoiceIdValue, promptId);
      const updated = await getVoiceRecordings(selectedVoiceIdValue);
      setRecordings(updated);
    } catch (err: any) {
      setStudioMsg({ kind: "err", text: err.message || "Couldn't delete that recording." });
    }
  };

  const handleCleanAll = async () => {
    if (!selectedVoiceIdValue) return;
    try {
      setCleaningAll(true);
      const toClean = recordings.filter(r => r.exists);
      if (toClean.length === 0) {
        setStudioMsg({ kind: "err", text: "No recordings to clean yet — record a few samples first." });
        return;
      }
      setStudioMsg({ kind: "busy", text: `Cleaning ${toClean.length} recording${toClean.length > 1 ? "s" : ""} — removing noise, trimming silence, normalizing loudness…` });
      for (const rec of toClean) {
        await cleanRecording(selectedVoiceIdValue, rec.id);
      }
      const updated = await getVoiceRecordings(selectedVoiceIdValue);
      setRecordings(updated);
      setStudioMsg({ kind: "ok", text: "All recordings cleaned and normalized. They're ready to build a voice." });
    } catch (err: any) {
      setStudioMsg({ kind: "err", text: err.message || "Couldn't clean the recordings. Please try again." });
    } finally {
      setCleaningAll(false);
    }
  };

  const [publishing, setPublishing] = useState(false);

  const refreshSelectedVoice = async () => {
    const updatedList = await getVoicesGallery();
    const match = updatedList.find((v: any) => v.id === selectedVoiceIdValue);
    if (match) setSelectedVoice(match);
    return match;
  };

  const handleCloneVoice = async () => {
    if (!selectedVoiceIdValue) return;
    try {
      setCloningVoice(true);
      setStudioMsg({ kind: "busy", text: "Building your voice from the recordings — this can take a minute…" });
      await cloneVoice(selectedVoiceIdValue);
      await refreshSelectedVoice();
      setStudioMsg({ kind: "ok", text: "Your voice is built! Press Preview to hear it, then Publish to activate it." });
      onRefresh();
    } catch (err: any) {
      setStudioMsg({ kind: "err", text: err.message || "Failed to build the voice. Check that at least 3 recordings exist and try again." });
    } finally {
      setCloningVoice(false);
    }
  };

  const handlePreviewVoice = async () => {
    if (!selectedVoiceIdValue) return;
    try {
      setPreviewingVoice(true);
      setStudioMsg({ kind: "busy", text: "Generating a preview… the first run may download the voice model, so give it a moment." });
      // Chatterbox downloads ~1-2GB model on first use — no browser fetch timeout
      const controller = new AbortController();
      const resp = await fetch(`${API_HTTP}/voices/${selectedVoiceIdValue}/preview`, {
        method: "POST",
        signal: controller.signal,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => null);
        throw new Error(err?.detail ?? "Preview failed");
      }
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audio.play();
      audio.onended = () => URL.revokeObjectURL(url);
      setStudioMsg({ kind: "ok", text: "Playing a preview with your cloned voice." });
    } catch (err: any) {
      if (err.name === "AbortError") return;
      setStudioMsg({ kind: "err", text: err.message || "Couldn't generate a preview. Try building the voice again." });
    } finally {
      setPreviewingVoice(false);
    }
  };

  const handlePublish = async () => {
    if (!selectedVoiceIdValue) {
      setStudioMsg({ kind: "err", text: "Voice profile ID is missing. Go back to gallery and reopen this voice." });
      return;
    }
    setPublishing(true);
    try {
      // If the voice model hasn't been built yet, build it automatically first —
      // never block the user with a silent disabled button.
      let voice = selectedVoice;
      if (!voice?.model_exists) {
        setStudioMsg({ kind: "busy", text: "Step 1 of 2 — Building your voice model first (this can take a minute)…" });
        await cloneVoice(selectedVoiceIdValue);
        voice = await refreshSelectedVoice();
      }
      setStudioMsg({ kind: "busy", text: voice?.model_exists ? "Step 2 of 2 — Publishing your voice…" : "Publishing your voice…" });
      await publishVoice(selectedVoiceIdValue);
      setStudioMsg({ kind: "ok", text: "🎉 Voice published! It is now active and ready to use in conversations." });
      onRefresh();
      // Stay 2s so the user sees the success, then return to gallery
      setTimeout(() => { setSelectedVoice(null); setWizardStep("gallery"); setStudioMsg(null); }, 2000);
    } catch (err: any) {
      setStudioMsg({ kind: "err", text: err.message || "Failed to publish voice" });
    } finally {
      setPublishing(false);
    }
  };

  const handleDeleteVoiceProfile = async (voiceId: string) => {
    if (confirm("Are you sure you want to delete this voice profile and scrub all recorded files?")) {
      try {
        await deleteVoice(voiceId);
        onRefresh();
      } catch (err: any) {
        alert(err.message);
      }
    }
  };

  const triggerDownloadDefault = async () => {
    try {
      setIsDownloading(true);
      await downloadDefaultVoices();
      alert("Background downloader started successfully!");
    } catch (err: any) {
      alert(err.message);
    } finally {
      setIsDownloading(false);
    }
  };

  // ── Smart gallery filters ──────────────────────────────────────────
  const STUDIO_FILTERS = [
    { id: "All", label: "All voices", help: "Show every voice — built-in and your own." },
    { id: "Ready", label: "Ready to use", help: "Published voices the assistant can speak with right now." },
    { id: "Draft", label: "In progress", help: "Voices you started but haven't published yet." },
    { id: "NeedsTraining", label: "Needs building", help: "Voices with recordings that haven't been built into a model." },
    { id: "ne", label: "Nepali", help: "Voices set to speak Nepali." },
    { id: "en", label: "English", help: "Voices set to speak English." },
    { id: "mixed", label: "Mixed", help: "Voices set for natural Nepali-English mixing." },
    { id: "Mine", label: "My voices", help: "Only the custom voices you created." },
    { id: "Builtin", label: "Built-in voices", help: "Ready-made voices included with the app." },
  ];
  const matchesFilter = (cv: any): boolean => {
    switch (activeFilter) {
      case "All": case "Mine": return true;
      case "Ready": return cv.publish_status === "published";
      case "Draft": return cv.publish_status !== "published";
      case "NeedsTraining": return !cv.model_exists;
      case "ne": case "en": case "mixed": return cv.language === activeFilter;
      case "Builtin": return false;
      default: return true;
    }
  };
  const filteredGallery = galleryVoices.filter(matchesFilter);
  const showBuiltins =
    activeFilter === "All" || activeFilter === "Builtin" || activeFilter === "Ready" ||
    activeFilter === "ne" || activeFilter === "en";
  const builtinLangOk = (lang: string) => !["ne", "en", "mixed"].includes(activeFilter) || activeFilter === lang;

  const recordedCount = recordings.filter(r => r.exists).length;
  const languageCounts = recordings.reduce((counts: Record<string, number>, rec) => {
    if (rec.exists) {
      counts[rec.language] = (counts[rec.language] ?? 0) + 1;
    }
    return counts;
  }, {});

  return (
    <section className="view-stack voice-studio-view">
      {wizardStep === "gallery" && (
        <>
          <div className="view-header">
            <div>
              <h2>Voice Studio</h2>
              <p>Create, clean, build, test, and publish natural voices — no technical skills needed.</p>
            </div>
            <div className="button-row">
              <button className="icon-text" onClick={triggerDownloadDefault} disabled={isDownloading}
                title="Download the ready-made Nepali and English voices so you can use them offline without recording anything.">
                <Download size={18} />
                <span>{isDownloading ? "Starting..." : "Download Default Voices"}</span>
              </button>
              <button className="icon-text good" onClick={() => setWizardStep("identity")}>
                <Plus size={18} />
                <span>Create Voice</span>
              </button>
            </div>
          </div>

          {/* Test/preview feedback — never leave a click looking frozen */}
          <StudioStatusBanner msg={studioMsg} onClose={() => setStudioMsg(null)} />

          {/* Friendly intro + how it works */}
          <div className="studio-hero">
            <div className="studio-hero-main">
              <span className="studio-hero-eyebrow"><Sparkles size={13} /> Welcome to Voice Studio</span>
              <h3>Give the assistant a voice — in four guided steps</h3>
              <p>
                A "voice" is how the assistant sounds when it speaks. You can use a built-in voice instantly,
                or record a few sentences to build a custom one that sounds like you. The studio walks you
                through every step and never lets you get stuck.
              </p>
            </div>
            <div className="studio-howto" aria-label="How Voice Studio works">
              {STUDIO_HOWTO.map((s, i) => (
                <div className="howto-step" key={s.n}>
                  <div className="howto-badge">{s.icon}</div>
                  <div className="howto-text">
                    <strong>{s.n}. {s.title}</strong>
                    <span>{s.desc}</span>
                  </div>
                  {i < STUDIO_HOWTO.length - 1 && <ChevronRight className="howto-arrow" size={16} />}
                </div>
              ))}
            </div>
          </div>

          {/* Use cases — help people understand what this is for */}
          <div className="studio-section-label"><Lightbulb size={13} /> What can I use this for?</div>
          <div className="studio-usecases">
            <UseCaseCard
              icon={<User size={18} />}
              title="Your own assistant voice"
              body="Clone your voice so the assistant can speak as you — great for demos, narration, or a personal touch."
              example="Record 10 sentences and let the assistant read your notes back in your own voice."
            />
            <UseCaseCard
              icon={<Languages size={18} />}
              title="Preserve a Nepali dialect"
              body="Capture a regional accent and keep it as a reusable, exportable voice for archiving or research."
              example="Save a Chitwan or Kathmandu-valley accent as a named voice profile."
            />
            <UseCaseCard
              icon={<Building2 size={18} />}
              title="A consistent brand voice"
              body="Create one professional, on-brand voice for customer guidance, IVR, or announcements."
              example="Build a calm, clear voice for Nabil Bank customer messages."
            />
          </div>

          {/* Smart, working filters */}
          <div className="studio-section-label"><Search size={13} /> Browse your voices</div>
          <div className="filter-row" aria-label="Voice filters">
            {STUDIO_FILTERS.map((f) => (
              <button
                key={f.id}
                type="button"
                className={activeFilter === f.id ? "active" : ""}
                onClick={() => setActiveFilter(f.id)}
                title={f.help}
                aria-pressed={activeFilter === f.id}
              >
                {f.label}
              </button>
            ))}
          </div>

          <div className="voice-studio-grid">
            {/* Built-in Voices — ready instantly, no recording needed */}
            {showBuiltins && builtinLangOk("ne") && (
            <article className="voice-studio-card">
              <div className="voice-card-header">
                <ShieldCheck size={20} style={{ color: "var(--green)" }} />
                <strong>Nepali chitwan</strong>
                <span className="pill good">Ready</span>
              </div>
              <span className="voice-card-desc">
                A natural Nepali voice you can use right away — no recording or setup required.
              </span>
              <div className="pill-group">
                <span className="pill good">Nepali</span>
                <span className="pill">Piper · built-in</span>
                <span className="pill good">Commercial OK</span>
              </div>
              <div className="voice-card-actions">
                <button className="voice-action-secondary" onClick={() => handleTestVoice("ne")} type="button"
                  disabled={!!ttsTestingId}
                  title="Hear a short sample spoken in this Nepali voice.">
                  {ttsTestingId === "__ne" ? <RefreshCw size={16} className="spin" /> : <Volume2 size={16} />}
                  <span>{ttsTestingId === "__ne" ? "Playing…" : "Hear a sample"}</span>
                </button>
              </div>
            </article>
            )}

            {showBuiltins && builtinLangOk("en") && (
            <article className="voice-studio-card">
              <div className="voice-card-header">
                <ShieldCheck size={20} style={{ color: "var(--green)" }} />
                <strong>English lessac</strong>
                <span className="pill good">Ready</span>
              </div>
              <span className="voice-card-desc">
                A clear English voice ready to use instantly — perfect for a quick start.
              </span>
              <div className="pill-group">
                <span className="pill good">English</span>
                <span className="pill">Piper · built-in</span>
                <span className="pill good">Commercial OK</span>
              </div>
              <div className="voice-card-actions">
                <button className="voice-action-secondary" onClick={() => handleTestVoice("en")} type="button"
                  disabled={!!ttsTestingId}
                  title="Hear a short sample spoken in this English voice.">
                  {ttsTestingId === "__en" ? <RefreshCw size={16} className="spin" /> : <Volume2 size={16} />}
                  <span>{ttsTestingId === "__en" ? "Playing…" : "Hear a sample"}</span>
                </button>
              </div>
            </article>
            )}

            {/* Custom Profiles */}
            {filteredGallery.map(cv => (
              <article key={cv.id} className="voice-studio-card">
                <div className="voice-card-header">
                  <User size={20} style={{ color: "var(--blue)" }} />
                  <strong>{cv.name}</strong>
                </div>
                <span className="voice-card-desc">Owner: {cv.owner_name} ({cv.owner_org || "Personal"})</span>
                <div className="pill-group">
                  <span className="pill good">{cv.language}</span>
                  <span className="pill">{cv.engine}</span>
                  <span className={`pill ${cv.consent_status === 'completed' ? 'good' : 'warn'}`}>
                    Consent: {cv.consent_status}
                  </span>
                  <span className={`pill ${cv.publish_status === 'published' ? 'good' : 'warn'}`}>
                    {cv.publish_status === 'published' ? 'Ready to use' : 'Needs more samples'}
                  </span>
                  <span className="pill">Quality {Math.round(cv.quality_score ?? 0)}</span>
                </div>
                <div className="voice-card-actions">
                  <button className="voice-action-secondary"
                    onClick={() => handleTestVoice(cv.language === "en" ? "en" : "ne", cv.id, (cv.engine || "").includes("chatterbox") || (cv.engine || "").includes("f5") || (cv.engine || "").includes("openvoice") || (cv.engine || "").includes("voxcpm"))}
                    disabled={!!ttsTestingId} type="button">
                    {ttsTestingId === cv.id ? <RefreshCw size={16} className="spin" /> : <Volume2 size={16} />}
                    <span>{ttsTestingId === cv.id ? "Generating…" : "Test"}</span>
                  </button>
                  {cv.publish_status !== 'published' ? (
                    <button className="voice-action-primary" onClick={() => { setSelectedVoice(cv); setWizardStep(cv.consent_status === "completed" ? "recordings" : "consent"); }} type="button">
                      <Mic size={16} />
                      <span>{cv.consent_status === "completed" ? "Continue" : "Add consent"}</span>
                    </button>
                  ) : (
                    <span className="pill good voice-action-ready">
                      <Check size={14} /> Ready
                    </span>
                  )}
                  <button className="voice-action-danger" onClick={() => handleDeleteVoiceProfile(cv.id)} type="button" title="Delete this voice profile">
                    <Trash2 size={16} />
                  </button>
                </div>
              </article>
            ))}
            {(() => {
              const builtinVisible = showBuiltins && (builtinLangOk("ne") || builtinLangOk("en"));
              if (builtinVisible || filteredGallery.length > 0) return null;
              if (!galleryVoices.length) {
                return (
                  <article className="empty-state gallery-empty">
                    <User size={28} />
                    <strong>No custom voices yet</strong>
                    <span>Create your first voice in a few guided steps. You can start with a quick preview and improve it later.</span>
                    <button className="icon-text good" type="button" onClick={() => setWizardStep("identity")}>
                      <Plus size={18} />
                      <span>Create your first voice</span>
                    </button>
                  </article>
                );
              }
              return (
                <article className="empty-state gallery-empty">
                  <Search size={28} />
                  <strong>No voices match "{STUDIO_FILTERS.find(f => f.id === activeFilter)?.label ?? activeFilter}"</strong>
                  <span>Try a different filter, or clear it to see all of your voices.</span>
                  <button className="icon-text" type="button" onClick={() => setActiveFilter("All")}>
                    <RefreshCw size={16} />
                    <span>Show all voices</span>
                  </button>
                </article>
              );
            })()}
          </div>

          {/* Quick TTS Voices — no training needed */}
          <div className="studio-section-label"><Sparkles size={13} /> Cloud TTS Voices · OpenAI — instant, no training</div>
          <div className="voice-tts-grid">
            {[
              { id: "openai-alloy", label: "Alloy", desc: "Neutral, balanced" },
              { id: "openai-echo", label: "Echo", desc: "Clear & steady" },
              { id: "openai-fable", label: "Fable", desc: "Warm, storytelling" },
              { id: "openai-onyx", label: "Onyx", desc: "Deep, authoritative" },
              { id: "openai-nova", label: "Nova", desc: "Bright & upbeat" },
              { id: "openai-shimmer", label: "Shimmer", desc: "Gentle, soothing" },
            ].map(v => (
              <div key={v.id} className="voice-tts-card">
                <strong>{v.label}</strong>
                <span>{v.desc}</span>
                <button className="voice-action-secondary" onClick={() => handleTestVoice("en", v.id)} disabled={!!ttsTestingId} type="button">
                  {ttsTestingId === v.id ? <RefreshCw size={14} className="spin" /> : <Volume2 size={14} />}
                  <span>{ttsTestingId === v.id ? "Playing…" : "Preview"}</span>
                </button>
              </div>
            ))}
          </div>

          {/* Enhanced Audit Log */}
          <div className="audit-log-container">
            <div className="audit-log-header">
              <Activity size={15} />
              <strong>Consent &amp; Audit Log</strong>
              <span>{auditLogs.length} events</span>
            </div>
            {auditLogs.length === 0 ? (
              <div className="audit-log-empty">No audit events yet</div>
            ) : (
              <div className="audit-log-list">
                {[...auditLogs].reverse().map((log, i) => {
                  const isGood = log.event?.includes("publish") || log.event?.includes("consent") || log.event?.includes("clone");
                  const isWarn = log.event?.includes("delete") || log.event?.includes("error");
                  const tone = isGood ? "good" : isWarn ? "warn" : "info";
                  return (
                    <div key={i} className="audit-log-entry">
                      <div className={`audit-log-dot ${tone}`} />
                      <div className="audit-log-body">
                        <div className={`audit-log-event ${tone}`}>{log.event}</div>
                        <div className="audit-log-details">{log.details}</div>
                      </div>
                      <div className="audit-log-timestamp">{log.timestamp?.slice(0, 16)?.replace("T", " ") ?? ""}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </>
      )}

      {wizardStep === "identity" && (
        <form onSubmit={handleCreateVoice} className="voice-wizard-form">
          <WizardStepper current={1} />
          <h3>Create a voice</h3>
          <p className="wizard-subtitle">Name the voice and confirm who it belongs to. This takes about a minute — nothing is recorded yet.</p>
          <GuideCallout tone="info" icon={<Info size={16} />} title="What happens in this step">
            You're just naming the voice and saying who it belongs to — <b>nothing is recorded yet</b>.
            Next you'll read a few sentences aloud, and the studio turns them into a voice for you.
          </GuideCallout>
          <div className="form-group">
            <label>Voice name</label>
            <input type="text" value={name} onChange={e => setName(e.target.value)} required placeholder="e.g. Kushal English Voice" />
            <FieldHint>The label you'll see in the voice list. <b>Tip:</b> include the speaker and language — <i>e.g. "Kushal — English"</i>.</FieldHint>
          </div>
          <div className="form-group">
            <label>Owner full name</label>
            <input type="text" value={ownerName} onChange={e => setOwnerName(e.target.value)} required placeholder="e.g. Kushal Khadka" />
            <FieldHint>The person this voice belongs to. Their consent is recorded and required before the voice can be built.</FieldHint>
          </div>
          <div className="form-group">
            <label>Owner email <span className="field-optional">(optional)</span></label>
            <input type="email" value={ownerEmail} onChange={e => setOwnerEmail(e.target.value)} placeholder="e.g. kushal@example.com" />
            <FieldHint>Used only for the consent record and audit log. Leave blank for personal use.</FieldHint>
          </div>
          <div className="form-group">
            <label>Organization / client <span className="field-optional">(optional)</span></label>
            <input type="text" value={organization} onChange={e => setOrganization(e.target.value)} placeholder="e.g. Personal" />
            <FieldHint>Who this voice is for — a department, client, or "Personal". Defaults to Personal if left blank.</FieldHint>
          </div>
          <div className="form-group">
            <label>Language goal</label>
            <select value={language} onChange={e => setLanguage(e.target.value)}>
              <option value="ne">Nepali</option>
              <option value="en">English</option>
              <option value="mixed">Natural mixed Nepali-English</option>
            </select>
            <FieldHint>Which language this voice should speak best. <b>Mixed</b> is ideal if you switch between Nepali and English in one sentence.</FieldHint>
          </div>
          <div className="form-group">
            <label>Cloning engine</label>
            <select value={engine} onChange={e => setEngine(e.target.value)}>
              <option value="chatterbox">Chatterbox local clone (recommended)</option>
              <option value="piper">Piper fine-tune (.onnx export)</option>
              <option value="elevenlabs">ElevenLabs (cloud)</option>
              <option value="f5_tts">F5-TTS (experimental)</option>
              <option value="openvoice">OpenVoice (experimental)</option>
              <option value="voxcpm">VoxCPM (experimental)</option>
            </select>
            <FieldHint>How the voice is built. <b>Not sure? Keep Chatterbox</b> — it runs fully on this machine and needs no API keys.</FieldHint>
          </div>

          <GuideCallout tone="guide" icon={<Cpu size={16} />} title="Which engine should I choose?">
            <ul className="guide-list">
              <li><b>Chatterbox (recommended):</b> fast local clone from your recordings — private, offline, no setup. Best for most people.</li>
              <li><b>Piper fine-tune:</b> exports a portable <code>.onnx</code> file. Pick this for a permanent, shareable voice model.</li>
              <li><b>ElevenLabs:</b> highest cloud quality, but sends audio to the cloud and needs an API key.</li>
              <li><b>F5-TTS / OpenVoice / VoxCPM:</b> experimental — only for previews; results may vary.</li>
            </ul>
          </GuideCallout>

          <div className="form-group">
            <label>Recording goal</label>
            <select value={recordingGoal} onChange={e => setRecordingGoal(e.target.value)}>
              <option value="quick">Quick preview — 5 to 10 minutes</option>
              <option value="better">Better quality — 30 to 60 minutes</option>
              <option value="production">Production quality — 2+ hours</option>
            </select>
            <FieldHint>How much audio you plan to record. <b>More clean recordings → a more accurate voice.</b> You can start small and add more anytime.</FieldHint>
          </div>

          <GuideCallout tone="tip" icon={<Mic size={16} />} title="Recording tips for the best result">
            Speak naturally — don't whisper or shout. Use the same microphone every time, sit 10–20 cm away,
            and record in a quiet room with no fan, echo, or background chatter.
          </GuideCallout>

          <label className="checkbox-label">
            <input type="checkbox" checked={commercialAllowed} onChange={e => setCommercialAllowed(e.target.checked)} required />
            <span>I confirm I have permission to use this person's voice.</span>
          </label>
          <FieldHint>Required by law and ethics — cloning someone's voice without their consent is not allowed.</FieldHint>
          <div className="form-action-row">
            <button type="button" className="icon-text" onClick={() => setWizardStep("gallery")}>Cancel</button>
            <button type="submit" className="icon-text good" disabled={creatingVoice}>
              {creatingVoice ? "Creating…" : "Create & start recording →"}
            </button>
          </div>
        </form>
      )}

      {wizardStep === "consent" && (
        <form onSubmit={handleSaveConsent} className="voice-wizard-form">
          <WizardStepper current={2} />
          <h3>Consent &amp; ownership</h3>
          <p className="wizard-subtitle">A quick signature confirming the owner agreed to have their voice cloned.</p>
          <GuideCallout tone="info" icon={<ShieldCheck size={16} />} title="Why we ask for this">
            A voice is personal — like a fingerprint. This signature is a record that the owner agreed to have
            their voice cloned. It protects both you and them, and it's required before the voice can be built or published.
          </GuideCallout>
          <GuideCallout tone="tip" icon={<HardDrive size={16} />} title="What happens to my data">
            Everything stays <b>on this machine</b>. Recordings and the consent record live in a local folder and are
            never uploaded — unless you specifically choose a cloud engine like ElevenLabs. You can delete the voice and all its files anytime.
          </GuideCallout>
          <div className="form-group">
            <label>Type the owner's full name as a signature</label>
            <input type="text" value={signature} onChange={e => setSignature(e.target.value)} required placeholder="e.g. Kushal Khadka" />
            <FieldHint>Type the full name exactly. This is saved to the audit log with a timestamp as proof of consent.</FieldHint>
          </div>
          <div className="form-action-row">
            <button type="button" className="icon-text" onClick={() => setWizardStep("gallery")}>Cancel</button>
            <button type="submit" className="icon-text good">Sign and continue →</button>
          </div>
        </form>
      )}

      {wizardStep === "recordings" && selectedVoice && (
        <>
          <div className="view-header">
            <div>
              <h2 style={{ display: "flex", alignItems: "center", gap: 10 }}>
                🎙️ {selectedVoice.name}
                <span style={{ fontSize: 11, padding: "3px 10px", borderRadius: 20, fontWeight: 700, background: selectedVoice.publish_status === "published" ? "rgba(72,187,120,0.15)" : selectedVoice.model_exists ? "rgba(0,166,81,0.15)" : "rgba(245,158,11,0.15)", color: selectedVoice.publish_status === "published" ? "var(--green)" : selectedVoice.model_exists ? "var(--teal)" : "#f59e0b" }}>
                  {selectedVoice.publish_status === "published" ? "Published ✓" : selectedVoice.model_exists ? "Voice built — ready to publish" : "In training"}
                </span>
              </h2>
              <p>Follow the 4 steps below — record, build, listen, publish. You can't get stuck: every step tells you what to do next.</p>
            </div>
            <button className="icon-text" onClick={() => { setStudioMsg(null); setWizardStep("gallery"); }} type="button">← Back to Gallery</button>
          </div>

          {/* Pre-flight: catch mic / engine issues BEFORE recording, not after */}
          <RecordingPreflight
            micState={micState}
            engineId={selectedVoice?.engine || "chatterbox"}
            engineInfo={engines ? engines[selectedVoice?.engine || "chatterbox"] : null}
          />

          {/* Live status banner — always visible feedback, no silent failures */}
          <StudioStatusBanner msg={studioMsg} onClose={() => setStudioMsg(null)} />

          {/* ── 4-STEP PIPELINE ── */}
          {(() => {
            const isPublished = selectedVoice.publish_status === "published";
            const hasModel = !!selectedVoice.model_exists;
            const enoughTakes = recordedCount >= 3;
            const busy = cloningVoice || publishing || previewingVoice || cleaningAll;
            const steps = [
              {
                n: 1, icon: "🎤", title: "Record", done: enoughTakes,
                active: !enoughTakes,
                sub: enoughTakes ? `${recordedCount} recordings done — great!` : `${recordedCount}/3 minimum — record ${3 - recordedCount} more below`,
              },
              {
                n: 2, icon: "✨", title: "Build voice", done: hasModel,
                active: enoughTakes && !hasModel,
                sub: hasModel ? "Voice model built ✓" : enoughTakes ? "Click to turn recordings into your AI voice" : "Unlocks after 3 recordings",
                action: enoughTakes && !hasModel ? { label: cloningVoice ? "Building…" : "Build my voice", onClick: handleCloneVoice, primary: true } : hasModel ? { label: cloningVoice ? "Rebuilding…" : "Rebuild", onClick: handleCloneVoice, primary: false } : undefined,
              },
              {
                n: 3, icon: "🔊", title: "Listen", done: false,
                active: hasModel && !isPublished,
                sub: hasModel ? "Hear a sample with your new voice (optional)" : "Unlocks after building",
                action: hasModel ? { label: previewingVoice ? "Generating…" : "Preview voice", onClick: handlePreviewVoice, primary: false } : undefined,
              },
              {
                n: 4, icon: "🚀", title: "Publish", done: isPublished,
                active: enoughTakes && !isPublished,
                sub: isPublished ? "Live and ready to use ✓" : enoughTakes ? (hasModel ? "Make this voice active in conversations" : "Will build automatically, then publish") : "Unlocks after 3 recordings",
                action: enoughTakes && !isPublished ? { label: publishing ? "Publishing…" : hasModel ? "Publish voice" : "Build & Publish", onClick: handlePublish, primary: true } : undefined,
              },
            ];
            return (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 16 }}>
                {steps.map(s => (
                  <div key={s.n} style={{
                    padding: "16px", borderRadius: 12, display: "flex", flexDirection: "column", gap: 8, minHeight: 130,
                    background: s.done ? "rgba(72,187,120,0.08)" : s.active ? "var(--teal-soft)" : "var(--panel)",
                    border: `2px solid ${s.done ? "rgba(72,187,120,0.4)" : s.active ? "var(--teal)" : "var(--line)"}`,
                    opacity: s.done || s.active ? 1 : 0.55, transition: "all 0.2s",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{
                        width: 24, height: 24, borderRadius: "50%", display: "inline-flex", alignItems: "center", justifyContent: "center",
                        fontSize: 12, fontWeight: 800, flexShrink: 0,
                        background: s.done ? "var(--green)" : s.active ? "var(--teal)" : "var(--line)",
                        color: s.done || s.active ? "#0a1412" : "var(--muted)",
                      }}>{s.done ? "✓" : s.n}</span>
                      <span style={{ fontSize: 14, fontWeight: 700 }}>{s.icon} {s.title}</span>
                    </div>
                    <div style={{ fontSize: 11, color: s.done ? "var(--green)" : "var(--muted)", lineHeight: 1.5, flex: 1 }}>{s.sub}</div>
                    {s.action && (
                      <button type="button" className={s.action.primary ? "rag-toolbar-btn primary" : "rag-toolbar-btn"}
                        onClick={s.action.onClick} disabled={busy}
                        style={{ width: "100%", justifyContent: "center", fontSize: 12 }}>
                        {(s.n === 2 && cloningVoice) || (s.n === 3 && previewingVoice) || (s.n === 4 && publishing) ? <RefreshCw size={13} className="spin" /> : null}
                        <span>{s.action.label}</span>
                      </button>
                    )}
                  </div>
                ))}
              </div>
            );
          })()}

          {/* Samples summary + helper actions */}
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 16, padding: "12px 16px", background: "var(--panel)", borderRadius: 10, border: "1px solid var(--line)" }}>
            <div style={{ display: "flex", gap: 14, fontSize: 12, flex: 1, flexWrap: "wrap" }}>
              <span>🇳🇵 Nepali <strong>{languageCounts.ne ?? 0}</strong></span>
              <span>🇬🇧 English <strong>{languageCounts.en ?? 0}</strong></span>
              <span>🔀 Mixed <strong>{languageCounts.mixed ?? 0}</strong></span>
              <span style={{ color: "var(--teal)", fontWeight: 700 }}>Total {recordedCount}/{prompts.length || 8}</span>
            </div>
            <button className="rag-toolbar-btn" type="button" onClick={handleCleanAll} disabled={cleaningAll || recordedCount === 0}
              title="Removes background noise, trims silence, and normalizes loudness on every recording.">
              <Wand2 size={13} />
              <span>{cleaningAll ? "Cleaning…" : "Clean all recordings"}</span>
            </button>
          </div>

          {/* Step-by-step instructions — collapsed by default once recording has started */}
          <details open={recordedCount === 0} style={{ background: "var(--teal-soft)", border: "1px solid rgba(0,166,81,0.3)", borderRadius: 12, marginBottom: 16, overflow: "hidden" }}>
            <summary style={{ fontWeight: 700, fontSize: 14, color: "var(--teal)", padding: "14px 20px", cursor: "pointer", userSelect: "none" }}>
              📖 How it works — step by step guide (click to {recordedCount === 0 ? "hide" : "open"})
            </summary>
            <div style={{ padding: "0 20px 16px" }}>
              <ol style={{ margin: 0, padding: "0 0 0 18px", display: "flex", flexDirection: "column", gap: 8, fontSize: 13, lineHeight: 1.6 }}>
                <li><strong>Find a quiet place</strong> — Turn off fans, TV, and close windows. Your room should be as quiet as possible.</li>
                <li><strong>Sit close to your microphone</strong> — About 15–20cm (one hand-width) away. Not too close, not too far.</li>
                <li><strong>Read the sentence shown</strong> — Read it out loud in your normal, natural voice. Don't whisper or shout.</li>
                <li><strong>Click the red circle button ●</strong> — It starts recording. Speak the sentence, then click again to stop.</li>
                <li><strong>Listen back</strong> — Press the play button to hear yourself. If it sounds clear, great! If not, delete and try again.</li>
                <li><strong>Record at least 3 sentences</strong> — More recordings = better voice quality. Aim for 10+ if you can.</li>
                <li><strong>Click "Clean all recordings"</strong> — This removes background noise automatically (optional but recommended).</li>
                <li><strong>Click "Build my voice"</strong> (Step 2 card above) — Wait a minute while your AI voice is built.</li>
                <li><strong>Click "Preview voice"</strong> (Step 3 card) — Listen to make sure it sounds like you.</li>
                <li><strong>Click "Publish voice"</strong> (Step 4 card) — Your voice becomes active in conversations! If you skipped Step 2, publishing builds it for you automatically.</li>
              </ol>
              <div style={{ marginTop: 12, padding: "8px 12px", background: "rgba(0,166,81,0.15)", borderRadius: 8, fontSize: 12, color: "var(--muted)" }}>
                💡 <strong>Tip:</strong> Even 3 recordings work for a preview. The more you record, the more natural your voice sounds!
              </div>
            </div>
          </details>

          {/* Quick Do / Don't reference — keep it visible while recording */}
          <div className="dodont-grid">
            <div className="dodont do">
              <div className="dodont-head"><Check size={15} /> Do</div>
              <ul>
                <li>Read in your normal, relaxed voice</li>
                <li>Record in a quiet room, same mic each time</li>
                <li>Sit 10–20 cm from the microphone</li>
                <li>Pause briefly before and after each sentence</li>
                <li>Listen back — re-record anything unclear</li>
              </ul>
            </div>
            <div className="dodont dont">
              <div className="dodont-head"><CircleAlert size={15} /> Don't</div>
              <ul>
                <li>Whisper, shout, or rush the words</li>
                <li>Record with a fan, TV, or echo in the room</li>
                <li>Change your distance or mic between takes</li>
                <li>Eat, chew gum, or cover your mouth</li>
                <li>Skip the cleanup step before building</li>
              </ul>
            </div>
          </div>

          <GuideCallout tone="info" icon={<ListChecks size={16} />} title="How to record each sentence below">
            Each row is one short sentence to read aloud. Press the <b>microphone button</b> on the right to start,
            read the sentence naturally, then press <b>stop</b>. A quality score appears so you know if a take is good —
            aim for green. Record at least <b>3</b> to build a voice; more sentences make it sound more like you.
          </GuideCallout>

          <div className="prompt-list voice-studio-recordings" style={{ marginTop: '1rem' }}>
            {prompts.map((prompt) => {
              const rec = recordings.find(r => r.id === prompt.id);
              const active = activePrompt === prompt.id;
              const goodLength = recordElapsed >= 2 && recordElapsed <= 12;
              return (
                <article className={active ? "prompt-row recording" : "prompt-row"} key={prompt.id}>
                  <div className="prompt-row-top">
                    <div className="prompt-copy">
                      <span className="prompt-id">{prompt.id}</span>
                      <strong>{prompt.text}</strong>
                      <LanguageBadge language={prompt.language} />
                    </div>
                    <div className="quality-strip">
                      {active ? (
                        <span className="recording-live"><span className="rec-dot" /> Recording…</span>
                      ) : rec?.exists ? (
                        <>
                          <span
                            className={`quality ${rec.verdict || rec.quality?.verdict || 'review'}`}
                            title={rec.reason || rec.quality?.reason || ''}
                          >
                            {rec.score ?? rec.quality?.score ?? '—'}/100
                          </span>
                          <audio src={absoluteAudioUrl(rec.audio_url) || ""} controls />
                          <button className="icon-text danger record-small" onClick={() => handleDeleteRecord(prompt.id)} type="button" title="Delete and re-record">
                            <Trash2 size={14} />
                          </button>
                        </>
                      ) : (
                        <span className="muted">No recording yet</span>
                      )}
                      <button
                        className={active ? "record-small-circle active" : "record-small-circle"}
                        onClick={active ? handleStopRecord : () => handleStartRecord(prompt.id)}
                        type="button"
                        title={active ? "Stop recording" : "Start recording"}
                        aria-label={active ? "Stop recording" : "Start recording this sentence"}
                      >
                        {active ? <Square size={16} /> : <Mic size={16} />}
                      </button>
                    </div>
                  </div>
                  {active && (
                    <div className="recording-panel">
                      <canvas ref={waveCanvasRef} className="rec-wave" />
                      <div className="rec-meta">
                        <span className={`rec-timer ${goodLength ? "good" : ""}`}>{fmtDuration(recordElapsed)}</span>
                        <span className="rec-hint">
                          {recordElapsed < 2 ? "Read the sentence aloud in your normal voice…"
                            : goodLength ? "Great length — press stop when you finish the sentence."
                            : "That's plenty — press stop now."}
                        </span>
                        <button className="rec-stop-btn" onClick={handleStopRecord} type="button">
                          <Square size={14} /> Stop
                        </button>
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
          </div>

          <details className="advanced-panel">
            <summary>Advanced Voice Controls</summary>
            <div className="settings-grid no-shadow">
              <label title="Engine selection chooses which cloning engine creates this voice. Safe default: Piper stable. Example: try F5-TTS only for experimental previews."><span>Engine</span><select><option>Piper stable</option><option>F5-TTS experimental</option><option>Chatterbox experimental</option><option>OpenVoice experimental</option><option>VoxCPM experimental</option></select></label>
              <label title="Denoise strength controls how aggressively background noise is removed. Safe default: medium. Example: raise it for fan noise, lower it if speech sounds watery."><span>Denoise strength</span><input type="range" min="0" max="10" defaultValue="5" /></label>
              <label title="VAD sensitivity controls how easily the app detects speech. Higher sensitivity helps quiet speakers but may detect background noise as speech. Safe default: 5."><span>VAD sensitivity</span><input type="range" min="0" max="10" defaultValue="5" /></label>
              <label title="Speaker match threshold blocks clips that sound like the wrong speaker. Safe default: 70. Example: increase for commercial voice protection."><span>Speaker match threshold</span><input type="number" defaultValue="70" /></label>
              <label title="Train language split strategy chooses one multilingual model, separate Nepali/English models, or automatic routing. Safe default: auto."><span>Language split strategy</span><select><option>auto</option><option>one multilingual model</option><option>separate Nepali/English models</option></select></label>
              <label className="toggle-line" title="Delete raw recordings after training keeps only cleaned approved audio and model artifacts. Safe default: off until you verify the voice."><input type="checkbox" /><span>Delete raw recordings after training</span></label>
            </div>
          </details>
        </>
      )}
    </section>
  );
}

// ── System Map / Architecture Explorer ──────────────────────────────────────
const SYS_NODE: Record<string, { fill: string; stroke: string }> = {
  teal: { fill: "var(--teal-soft)", stroke: "var(--teal)" },
  blue: { fill: "rgba(96,165,250,0.14)", stroke: "var(--blue)" },
  amber: { fill: "var(--amber-soft)", stroke: "var(--amber)" },
  purple: { fill: "var(--purple-soft)", stroke: "var(--purple)" },
  green: { fill: "var(--green-soft)", stroke: "var(--green)" },
  rose: { fill: "var(--rose-soft)", stroke: "var(--rose)" },
  gray: { fill: "var(--panel-2)", stroke: "var(--line)" },
};

type SysCrit = "Critical" | "Optional";
interface SysComp { n: string; crit: SysCrit; what: string; why: string; recv: string; prod: string; deps: string; tech: string; }

const SYS_COMPONENTS: Record<string, SysComp> = {
  frontend: { n: "Web app (frontend)", crit: "Critical", what: "The browser interface where you talk, type, manage voices, and view logs.", why: "Gives every audience one friendly place to use the system.", recv: "Your clicks, typing, and microphone audio.", prod: "Requests to the gateway and audio playback on screen.", deps: "API gateway.", tech: "React 19, Vite, TypeScript." },
  gateway: { n: "API gateway", crit: "Critical", what: "The single backend entry point for all requests and live audio.", why: "Routes work to the right service and keeps the app and engines decoupled.", recv: "HTTP requests and WebSocket audio from the web app.", prod: "JSON responses, streamed audio, and status.", deps: "All services and providers.", tech: "FastAPI, Uvicorn, Pydantic; ~80 endpoints + /ws/voice." },
  conversation: { n: "Conversation orchestrator", crit: "Critical", what: "The brain that runs a turn end to end.", why: "Coordinates language routing, retrieval, prompt assembly, the model, and fallbacks.", recv: "Transcribed or typed text plus options (voice, knowledge, provider).", prod: "The answer text, routing metadata, and timings.", deps: "Language router, RAG, web retrieval, an LLM, TTS.", tech: "Python service module." },
  langrouter: { n: "Language router", crit: "Critical", what: "Detects whether text is Nepali, English, or mixed.", why: "Lets the assistant reply naturally in the right language and route speech correctly.", recv: "Text (and any Whisper language hint).", prod: "A language label used in the prompt and TTS routing.", deps: "None.", tech: "Unicode script heuristics (Devanagari vs Latin)." },
  stt: { n: "Speech-to-text", crit: "Critical", what: "Turns spoken audio into text.", why: "Without it, voice turns are impossible.", recv: "Microphone audio (WAV).", prod: "A transcript plus detected language and confidence.", deps: "Audio pipeline.", tech: "faster-whisper (local, default) or OpenAI Whisper (cloud)." },
  tts_piper: { n: "Piper text-to-speech", crit: "Critical", what: "The default local voice engine.", why: "Provides offline, free, natural Nepali and English speech.", recv: "Answer text and a chosen voice model.", prod: "Spoken audio (WAV).", deps: "Local ONNX voice files, ffmpeg.", tech: "Piper, .onnx voices (e.g. ne_NP-chitwan, en_US-lessac)." },
  tts_chatterbox: { n: "Chatterbox voice clone", crit: "Optional", what: "Local zero-shot voice cloning engine.", why: "Builds a custom voice from a few consented recordings, fully on-device.", recv: "A reference recording and text to speak.", prod: "Cloned-voice audio.", deps: "Voice Studio, local model weights.", tech: "Chatterbox (CPU by default)." },
  tts_openai: { n: "OpenAI text-to-speech", crit: "Optional", what: "Cloud voice option recommended for Nepali-English chat.", why: "High-quality multilingual voices without local setup.", recv: "Answer text and a voice id (alloy, nova, etc.).", prod: "Spoken audio.", deps: "OpenAI API key.", tech: "OpenAI TTS." },
  tts_eleven: { n: "ElevenLabs", crit: "Optional", what: "Premium cloud voice provider.", why: "Top-tier expressive voices when cloud is acceptable.", recv: "Text and a voice selection.", prod: "Spoken audio.", deps: "ElevenLabs API key.", tech: "ElevenLabs API." },
  llm_ollama: { n: "Local LLM (Ollama)", crit: "Critical", what: "The default reasoning engine, running transformer models on-device.", why: "Private, offline answering with no per-message cost; also the fallback for cloud.", recv: "An assembled prompt.", prod: "The answer text.", deps: "Ollama running locally with a model pulled.", tech: "Ollama (llama3, qwen2.5, gemma3). No vLLM." },
  llm_openai: { n: "OpenAI (cloud LLM)", crit: "Optional", what: "Cloud reasoning option.", why: "Higher quality when the internet and cloud use are allowed.", recv: "An assembled prompt.", prod: "The answer text.", deps: "OpenAI API key; falls back to Ollama on error.", tech: "OpenAI gpt-4o-mini." },
  llm_gemini: { n: "Google Gemini (cloud LLM)", crit: "Optional", what: "Alternative cloud reasoning option.", why: "Choice of cloud provider.", recv: "An assembled prompt.", prod: "The answer text.", deps: "Gemini API key with the Generative Language API enabled.", tech: "Gemini flash models via generateContent." },
  rag: { n: "RAG / knowledge service", crit: "Optional", what: "Finds relevant passages from your documents to ground answers.", why: "Lets the assistant answer from your PDFs, notes, and pages instead of guessing.", recv: "A question and chosen collections.", prod: "Ranked context passages with sources.", deps: "Embeddings, ChromaDB; falls back to direct chat.", tech: "ChromaDB + BM25, RRF fusion, optional cross-encoder rerank." },
  embeddings: { n: "Embeddings", crit: "Optional", what: "Turns text into vectors so similar meaning can be matched.", why: "The foundation of semantic search in RAG.", recv: "Document chunks and queries.", prod: "Numeric vectors.", deps: "Ollama, sentence-transformers, or OpenAI.", tech: "nomic-embed-text, all-MiniLM-L6-v2, or text-embedding-3-small." },
  chromadb: { n: "ChromaDB (vector store)", crit: "Optional", what: "The database that stores and searches embeddings.", why: "Fast nearest-neighbour search over your knowledge.", recv: "Vectors and metadata.", prod: "The closest matching chunks.", deps: "Embeddings provider.", tech: "ChromaDB, persistent client, cosine HNSW." },
  sqlite: { n: "SQLite (app database)", crit: "Critical", what: "The local relational store for application data.", why: "Persists settings, chat history, voices, RAG metadata, analytics, and audit.", recv: "Writes from services.", prod: "Stored rows on demand.", deps: "None (a local file).", tech: "SQLite (swarlocal.db)." },
  mysql: { n: "MySQL (voiceai)", crit: "Optional", what: "The banking database server holding KYC records and admin users.", why: "Source of customer data for KYC lookups and admin accounts.", recv: "Read-only SELECTs (KYC) and auth queries.", prod: "Customer rows and user records.", deps: "A running MySQL server.", tech: "MySQL 8, customer_kyc_v view, admin_users table." },
  kyc: { n: "KYC service", crit: "Optional", what: "Turns plain-language banking questions into safe SQL.", why: "Lets staff query KYC data conversationally without writing SQL.", recv: "A natural-language question.", prod: "Read-only results (up to 100 rows).", deps: "An LLM and MySQL.", tech: "NL to SQL with SELECT-only guardrails and one retry." },
  admin: { n: "Admin authentication", crit: "Optional", what: "Login and sessions for the admin surface.", why: "Protects sensitive configuration and tools.", recv: "Username and password.", prod: "A session token.", deps: "MySQL, with a built-in fallback if it is down.", tech: "PBKDF2-SHA256, salted, expiring sessions." },
  voicestudio: { n: "Voice Studio", crit: "Optional", what: "The consent-gated workflow for creating custom voices.", why: "Ensures voices are owned and consented before they exist.", recv: "Consent signature, recordings, and build requests.", prod: "A published, usable voice and audit records.", deps: "TTS engines, audio pipeline, SQLite.", tech: "Consent, DSP cleanup, Chatterbox/Piper build, publish." },
  vad: { n: "Turn detection (VAD)", crit: "Critical", what: "Detects when speech starts and stops in a voice turn.", why: "Knows when your turn is finished so it can respond.", recv: "Streaming audio.", prod: "Turn boundaries.", deps: "Audio stream.", tech: "Voice-activity detection / push-to-talk." },
  webretrieval: { n: "Web retrieval", crit: "Optional", what: "Fetches public web results for time-sensitive questions.", why: "Adds current information when explicitly enabled.", recv: "A search query.", prod: "A few cited results.", deps: "Internet access (off by default).", tech: "DuckDuckGo HTML, parsed locally." },
  sysmonitor: { n: "System monitor", crit: "Optional", what: "Reports machine resource usage.", why: "Lets operators see health at a glance.", recv: "Polls of the host.", prod: "CPU, memory, and disk metrics.", deps: "Host OS.", tech: "Endpoint + WebSocket stream." },
  audit: { n: "Audit log", crit: "Optional", what: "A timestamped record of sensitive actions.", why: "Supports compliance and traceability.", recv: "Events from services (consent, clone, publish, delete, settings).", prod: "Queryable audit history.", deps: "SQLite.", tech: "voice_audit_log table." },
};

const SYS_AUD: Record<string, string> = {
  general: "Pick a view below. Each one explains the same system at a different depth — start with the Executive overview, then click any coloured box to drill in.",
  exec: "Bottom line: a private, on-device voice assistant for Nabil Bank that answers in Nepali and English. It works offline, keeps customer data on the machine, and uses the cloud only if you switch it on.",
  business: "It removes the language barrier (mixed Nepali-English) and the privacy barrier (data stays local) for customer-facing and internal assistance, with consent-gated voice cloning.",
  ops: "Day to day it is one backend service plus a web app. Local models (Ollama, Whisper, Piper) do the work; cloud is optional. The Monitoring view shows health, audit, and failures.",
  compliance: "Consent is enforced before any voice is cloned, every sensitive action is written to an audit log, KYC queries are read-only by design, and data stays local unless a cloud provider is explicitly enabled.",
  support: "When a customer talks or types, the assistant transcribes, finds an answer (optionally from the knowledge base), and speaks back. If something fails it falls back automatically and shows what happened.",
  product: "The system is modular: each capability (speech, reasoning, knowledge, voice) is a swappable provider with a graceful fallback. Optional features degrade without breaking the core.",
  dev: "FastAPI gateway plus a React app. Providers (STT, TTS, LLM, embeddings) and services (conversation, RAG, KYC, voice studio) are lazy-loaded modules. Click any box for inputs, outputs, dependencies, and tech.",
  infra: "Runs on one machine: Ollama, faster-whisper, Piper, Chatterbox, ChromaDB, and SQLite locally; MySQL for KYC and admin. Cloud (OpenAI, Gemini, ElevenLabs) is opt-in egress only.",
  audit: "Trace any action: audit logs in SQLite, RAG query analytics, consent records with signatures, read-only SQL guardrails, and API-key masking. Nothing leaves the device unless cloud is enabled.",
  vendor: "Integration points are OpenAI, Google Gemini, ElevenLabs, Open WebUI, DuckDuckGo, and Ollama. All are optional and isolated behind provider interfaces with timeouts and fallbacks.",
  student: "A real-world AI voice pipeline: speech-to-text, language routing, retrieval-augmented generation (embeddings + vector search), a transformer LLM, then text-to-speech. Click boxes to learn each part.",
  nontech: "Think of it as a smart, polite bank helper you can talk to in Nepali or English. It listens, thinks, and talks back — and it does this on your own computer, privately.",
};

interface SysJStep { t: string; d: string; comp: string[]; }
const SYS_JOURNEYS: Record<string, { label: string; steps: SysJStep[] }> = {
  voice: { label: "Speaking a question (voice turn)", steps: [
    { t: "You speak", d: "You tap the mic in the web app and talk in Nepali, English, or a mix. The audio streams live to the backend over a WebSocket.", comp: ["frontend", "gateway"] },
    { t: "The app hears you", d: "A voice-activity detector notices when you start and stop talking, so the system knows your turn is finished.", comp: ["vad"] },
    { t: "Speech becomes text", d: "faster-whisper transcribes your audio into text on the machine (or OpenAI Whisper if cloud speech is enabled).", comp: ["stt"] },
    { t: "It understands and answers", d: "The language router detects Nepali or English, optional knowledge is retrieved, and a language model writes the reply.", comp: ["langrouter", "conversation", "rag", "llm_ollama"] },
    { t: "The answer becomes a voice", d: "Piper, your cloned voice, or a cloud voice turns the reply text into natural speech.", comp: ["tts_piper"] },
    { t: "You hear the reply", d: "Audio streams back to the browser and plays, with latency and routing details shown on screen.", comp: ["frontend"] },
  ] },
  text: { label: "Asking from your documents (text + knowledge)", steps: [
    { t: "You type a question", d: "You type in the chat box and optionally pick a knowledge collection to search.", comp: ["frontend", "gateway"] },
    { t: "Find relevant documents", d: "Your question is embedded into a vector and matched against the knowledge base with semantic + keyword search, fused and reranked.", comp: ["embeddings", "chromadb", "rag"] },
    { t: "Build the prompt", d: "The bank-wide instruction, system prompt, language guidance, and retrieved context are assembled together.", comp: ["conversation"] },
    { t: "The model answers", d: "A local model (Ollama) or a cloud model (OpenAI/Gemini) generates the answer, citing sources where relevant.", comp: ["llm_ollama"] },
    { t: "You read the answer", d: "The reply appears with citations, you can rate it, and the turn is saved.", comp: ["sqlite", "audit"] },
  ] },
  kyc: { label: "A KYC lookup in plain language", steps: [
    { t: "You ask a banking question", d: "For example: show customers with loans overdue above 80 lakhs in Kathmandu.", comp: ["frontend"] },
    { t: "Turn words into safe SQL", d: "A model converts your question into a read-only SQL SELECT against the KYC view, understanding terms like lakh and crore.", comp: ["kyc", "llm_ollama"] },
    { t: "Safety checks", d: "The SQL is validated: only SELECT is allowed, dangerous keywords are blocked, and only one statement may run.", comp: ["kyc"] },
    { t: "Run against the database", d: "The query runs against MySQL voiceai.customer_kyc_v and returns up to 100 rows.", comp: ["mysql"] },
    { t: "You see the results", d: "Rows are shown in the app; if the query fails, it retries once with the error fed back to the model.", comp: ["frontend"] },
  ] },
  clone: { label: "Creating a custom voice", steps: [
    { t: "Create a voice profile", d: "In Voice Studio you name the voice and record who owns it. Nothing is recorded yet.", comp: ["voicestudio"] },
    { t: "Give consent", d: "The owner signs by typing their name; consent is recorded with a timestamp before anything is built.", comp: ["voicestudio", "audit"] },
    { t: "Record samples", d: "You read a few sentences; each take is scored for quality and can be cleaned (denoise, trim, normalise).", comp: ["voicestudio"] },
    { t: "Build the voice", d: "Chatterbox builds a zero-shot clone locally, or Piper fine-tunes an exportable model.", comp: ["tts_chatterbox"] },
    { t: "Preview and publish", d: "You listen to a sample, then publish so the assistant can speak with the new voice.", comp: ["voicestudio"] },
  ] },
};

function SysFlow({ stages, onPick, mid }: { stages: { c: string; id: string; t: string; s: string }[]; onPick: (id: string) => void; mid: string }) {
  const h = 58, gap = 30, x = 120, w = 440, top = 10;
  const H = top + stages.length * (h + gap) - gap + 10;
  return (
    <svg className="sysmap-svg" width="100%" viewBox={`0 0 680 ${H}`} role="img" aria-label="Process flow">
      <defs>
        <marker id={`syarr-${mid}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M2 1L8 5L2 9" fill="none" stroke="var(--muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </marker>
      </defs>
      {stages.map((st, i) => {
        const y = top + i * (h + gap);
        const col = SYS_NODE[st.c] || SYS_NODE.gray;
        return (
          <g key={st.id + i} className="sysmap-node" onClick={() => onPick(st.id)}>
            {i > 0 && <line x1={x + w / 2} y1={y - gap} x2={x + w / 2} y2={y - 4} stroke="var(--muted)" strokeWidth="1.5" markerEnd={`url(#syarr-${mid})`} />}
            <rect x={x} y={y} width={w} height={h} rx="9" fill={col.fill} stroke={col.stroke} strokeWidth="1" />
            <text className="sysmap-th" x={x + w / 2} y={y + 22} textAnchor="middle" dominantBaseline="central">{st.t}</text>
            <text className="sysmap-ts" x={x + w / 2} y={y + 40} textAnchor="middle" dominantBaseline="central">{st.s}</text>
          </g>
        );
      })}
    </svg>
  );
}

const SYS_TABS: { id: string; label: string }[] = [
  { id: "exec", label: "1 · Executive" },
  { id: "business", label: "2 · Business" },
  { id: "tech", label: "3 · Technical" },
  { id: "infra", label: "4 · Infrastructure" },
  { id: "ai", label: "5 · AI" },
  { id: "voice", label: "6 · Voice" },
  { id: "sec", label: "7 · Security" },
  { id: "data", label: "8 · Data flow" },
  { id: "dep", label: "9 · Dependencies" },
  { id: "journey", label: "10 · User journey" },
  { id: "fail", label: "11 · Failure recovery" },
  { id: "mon", label: "12 · Monitoring" },
];

function SystemExplorerView() {
  const [tab, setTab] = useState<string>("exec");
  const [aud, setAud] = useState<string>("general");
  const [openComp, setOpenComp] = useState<string | null>(null);
  const [jKey, setJKey] = useState<string>("voice");
  const [jIdx, setJIdx] = useState<number>(0);

  const pick = (id: string) => setOpenComp(id);
  const chip = (id: string) => (
    <button key={id} className="sysmap-chip" type="button" onClick={() => pick(id)}>{SYS_COMPONENTS[id]?.n ?? id}</button>
  );
  const aiStages = [
    { c: "teal", id: "stt", t: "1 · Question in", s: "typed, or spoken then transcribed" },
    { c: "blue", id: "langrouter", t: "2 · Detect language", s: "Nepali / English / mixed" },
    { c: "purple", id: "rag", t: "3 · Find knowledge (RAG)", s: "embed → search → fuse → rerank" },
    { c: "gray", id: "conversation", t: "4 · Build the prompt", s: "bank rule + system + context" },
    { c: "amber", id: "llm_ollama", t: "5 · Model answers", s: "local Ollama or cloud" },
    { c: "rose", id: "tts_piper", t: "6 · Speak the answer", s: "Piper, clone, or cloud" },
  ];
  const voiceStages = [
    { c: "blue", id: "frontend", t: "1 · Microphone → WebSocket", s: "browser streams audio to /ws/voice" },
    { c: "teal", id: "vad", t: "2 · Detect speech & turn end", s: "voice-activity detection" },
    { c: "purple", id: "stt", t: "3 · Transcribe", s: "faster-whisper on-device" },
    { c: "gray", id: "conversation", t: "4 · Think & answer", s: "the AI pipeline from view 5" },
    { c: "amber", id: "tts_piper", t: "5 · Synthesise voice", s: "Piper, clone, or cloud" },
    { c: "rose", id: "frontend", t: "6 · Play back", s: "audio returns; latency shown" },
  ];

  function renderTab() {
    if (tab === "exec") {
      const N = (color: string, x: number, id: string, title: string, sub: string) => {
        const c = SYS_NODE[color];
        return (
          <g className="sysmap-node" onClick={() => pick(id)}>
            <rect x={x} y={78} width={184} height={58} rx="9" fill={c.fill} stroke={c.stroke} strokeWidth="1" />
            <text className="sysmap-th" x={x + 92} y={100} textAnchor="middle" dominantBaseline="central">{title}</text>
            <text className="sysmap-ts" x={x + 92} y={120} textAnchor="middle" dominantBaseline="central">{sub}</text>
          </g>
        );
      };
      return (
        <>
          <p className="sysmap-h">Executive overview</p>
          <p className="sysmap-d">SwarLocal lets a person talk or type to a bank assistant in Nepali, English, or a natural mix — and get a spoken answer. The core runs entirely on one computer, so customer data never has to leave the building. Cloud AI is available but optional.</p>
          <svg className="sysmap-svg" width="100%" viewBox="0 0 680 200" role="img" aria-label="How it works at a glance">
            <rect x="20" y="40" width="640" height="120" rx="18" fill="var(--surface-2)" stroke="var(--line)" />
            <text className="sysmap-ts" x="40" y="62">Runs on your computer — no internet needed for the core experience</text>
            {N("teal", 44, "stt", "Listen", "understands speech")}
            {N("blue", 248, "conversation", "Think", "finds the answer")}
            {N("amber", 452, "tts_piper", "Reply", "speaks back to you")}
            <line x1="228" y1="107" x2="246" y2="107" stroke="var(--muted)" strokeWidth="1.5" />
            <line x1="432" y1="107" x2="450" y2="107" stroke="var(--muted)" strokeWidth="1.5" />
            <text className="sysmap-th" x="20" y="26">You speak or type — Nepali, English, or both</text>
            <text className="sysmap-ts" x="20" y="186">Tap any box to see what it is. Optional cloud AI can be switched on for extra quality.</text>
          </svg>
          <div className="sysmap-grid" style={{ marginTop: 14 }}>
            <div className="sysmap-card"><h4>What it is</h4><p>A bilingual voice + chat assistant that runs locally, answers questions, reads documents aloud, and queries banking records safely.</p></div>
            <div className="sysmap-card"><h4>Why it matters</h4><p>Privacy by default, no language barrier, and no per-message cloud cost for the core experience.</p></div>
            <div className="sysmap-card"><h4>Confidence</h4><p>Consent-gated voice cloning, read-only banking queries, full audit logging, and automatic fallbacks.</p></div>
          </div>
        </>
      );
    }
    if (tab === "business") {
      return (
        <>
          <p className="sysmap-h">Business view</p>
          <p className="sysmap-d">The problem: customers and staff in Nepal naturally mix Nepali and English, and ordinary assistants mispronounce or misunderstand them — while sending sensitive data to the cloud. SwarLocal solves both.</p>
          <div className="sysmap-grid">
            <div className="sysmap-card"><h4>Problem solved</h4><p>Bilingual understanding and speech, plus privacy: sensitive data never leaves the machine unless explicitly allowed.</p></div>
            <div className="sysmap-card"><h4>Who uses it</h4><p>Customer support, staff querying KYC records, and anyone wanting hands-free spoken access to local knowledge.</p></div>
            <div className="sysmap-card"><h4>Value delivered</h4><p>Lower cloud cost, regulatory comfort, faster bilingual service, and a reusable branded voice.</p></div>
            <div className="sysmap-card"><h4>Where it runs</h4><p>On a single machine. Cloud is opt-in, so spend and data exposure stay controlled.</p></div>
          </div>
          <p className="sysmap-d" style={{ marginTop: 14 }}>Capabilities, as features:</p>
          <div className="sysmap-chips">{["conversation", "rag", "kyc", "voicestudio", "webretrieval"].map(chip)}</div>
        </>
      );
    }
    if (tab === "tech") {
      const layer = (color: string, y: number, id: string, title: string, sub: string) => {
        const c = SYS_NODE[color];
        return (
          <g className="sysmap-node" onClick={() => pick(id)}>
            <rect x="40" y={y} width="600" height="58" rx="10" fill={c.fill} stroke={c.stroke} strokeWidth="1" />
            <text className="sysmap-th" x="340" y={y + 22} textAnchor="middle" dominantBaseline="central">{title}</text>
            <text className="sysmap-ts" x="340" y={y + 40} textAnchor="middle" dominantBaseline="central">{sub}</text>
          </g>
        );
      };
      return (
        <>
          <p className="sysmap-h">Technical view</p>
          <p className="sysmap-d">Four layers: a React web app talks to a FastAPI gateway, which routes to swappable provider and service modules, which read and write a few local data stores.</p>
          <svg className="sysmap-svg" width="100%" viewBox="0 0 680 344" role="img" aria-label="Technical layers">
            {layer("blue", 10, "frontend", "Web app", "React · Vite · TypeScript — runs in the browser")}
            <line x1="340" y1="68" x2="340" y2="92" stroke="var(--muted)" strokeWidth="1.5" />
            {layer("teal", 96, "gateway", "API gateway", "FastAPI · Uvicorn — HTTP + WebSocket")}
            <line x1="340" y1="154" x2="340" y2="178" stroke="var(--muted)" strokeWidth="1.5" />
            {layer("purple", 182, "conversation", "Engines & services", "speech · reasoning · knowledge · voice · KYC")}
            <line x1="340" y1="240" x2="340" y2="264" stroke="var(--muted)" strokeWidth="1.5" />
            {layer("amber", 268, "sqlite", "Data stores", "SQLite · ChromaDB · MySQL · local files")}
          </svg>
          <p className="sysmap-d" style={{ marginTop: 14 }}>Engines &amp; services (each a swappable module):</p>
          <div className="sysmap-chips">{["stt", "langrouter", "llm_ollama", "rag", "embeddings", "tts_piper", "voicestudio", "kyc", "admin", "vad"].map(chip)}</div>
        </>
      );
    }
    if (tab === "infra") {
      return (
        <>
          <p className="sysmap-h">Infrastructure view</p>
          <p className="sysmap-d">Almost everything lives on one machine. Cloud services and the database server are separate, optional, and only contacted when enabled.</p>
          <div className="sysmap-grid">
            <div className="sysmap-card"><h4>🖥️ On your machine</h4><p>Ollama, faster-whisper, Piper, Chatterbox, ChromaDB, and SQLite — the workhorses, no internet required.</p><div className="sysmap-chips" style={{ marginTop: 10 }}>{["llm_ollama", "stt", "tts_piper", "chromadb", "sqlite"].map(chip)}</div></div>
            <div className="sysmap-card" style={{ borderLeftColor: "var(--purple)" }}><h4>☁️ Optional cloud</h4><p>OpenAI, Gemini, ElevenLabs, and DuckDuckGo — each opt-in with timeouts and key masking. Off by default.</p><div className="sysmap-chips" style={{ marginTop: 10 }}>{["llm_openai", "llm_gemini", "tts_eleven", "webretrieval"].map(chip)}</div></div>
            <div className="sysmap-card" style={{ borderLeftColor: "var(--amber)" }}><h4>🗄️ On your network</h4><p>MySQL (voiceai) holds KYC records and admin users; Open WebUI is an optional document source.</p><div className="sysmap-chips" style={{ marginTop: 10 }}>{["mysql", "kyc", "admin"].map(chip)}</div></div>
          </div>
        </>
      );
    }
    if (tab === "ai") {
      return (
        <>
          <p className="sysmap-h">AI architecture view</p>
          <p className="sysmap-d">Every question flows through the same pipeline. Retrieval-augmented generation adds your documents; the language model can be local or cloud. Tap a stage to learn it.</p>
          <SysFlow stages={aiStages} onPick={pick} mid="ai" />
          <div className="sysmap-grid" style={{ marginTop: 14 }}>
            <div className="sysmap-card"><h4>Embeddings &amp; vectors</h4><p>Text becomes numbers via nomic-embed-text, sentence-transformers, or OpenAI; stored in ChromaDB with cosine HNSW.</p><div className="sysmap-chips" style={{ marginTop: 10 }}>{chip("embeddings")}</div></div>
            <div className="sysmap-card"><h4>Retrieval &amp; reranking</h4><p>Hybrid search = semantic + BM25 keyword, fused with RRF, then an optional cross-encoder reranks the best chunks.</p><div className="sysmap-chips" style={{ marginTop: 10 }}>{chip("rag")}</div></div>
            <div className="sysmap-card"><h4>Local vs cloud</h4><p>Ollama runs transformer LLMs on-device by default; cloud falls back to local if it errors. No vLLM is used.</p><div className="sysmap-chips" style={{ marginTop: 10 }}>{["llm_ollama", "llm_openai"].map(chip)}</div></div>
          </div>
        </>
      );
    }
    if (tab === "voice") {
      return (
        <>
          <p className="sysmap-h">Voice architecture view</p>
          <p className="sysmap-d">For spoken turns, audio streams live both ways over a WebSocket. Voice Studio is a separate, consent-gated path for building custom voices.</p>
          <SysFlow stages={voiceStages} onPick={pick} mid="voice" />
          <p className="sysmap-d" style={{ marginTop: 14 }}>Voice Studio — build a custom voice (consent first):</p>
          <div className="sysmap-chips">{["voicestudio", "tts_chatterbox", "tts_piper"].map(chip)}</div>
        </>
      );
    }
    if (tab === "sec") {
      return (
        <>
          <p className="sysmap-h">Security view</p>
          <p className="sysmap-d">Defence in depth, with privacy as the default. The biggest control is simply that the core never needs the internet.</p>
          <div className="sysmap-grid">
            <div className="sysmap-card"><h4>Local-first by default</h4><p>Transcripts, recordings, and reasoning stay on the machine. Cloud is opt-in egress only, per provider.</p></div>
            <div className="sysmap-card"><h4>Consent gating</h4><p>No voice can be cloned without a recorded, timestamped consent signature.</p><div className="sysmap-chips" style={{ marginTop: 10 }}>{chip("voicestudio")}</div></div>
            <div className="sysmap-card"><h4>Read-only KYC</h4><p>Generated SQL is forced to a single SELECT; insert/update/delete/drop are blocked.</p><div className="sysmap-chips" style={{ marginTop: 10 }}>{chip("kyc")}</div></div>
            <div className="sysmap-card"><h4>Admin authentication</h4><p>PBKDF2-hashed, salted passwords with expiring session tokens.</p><div className="sysmap-chips" style={{ marginTop: 10 }}>{chip("admin")}</div></div>
            <div className="sysmap-card"><h4>Secret handling</h4><p>API keys are masked in responses and never overwritten by a masked value sent back from the UI.</p></div>
            <div className="sysmap-card"><h4>Audit trail</h4><p>Consent, publish, clone, delete, and settings changes are written to an audit log.</p><div className="sysmap-chips" style={{ marginTop: 10 }}>{chip("audit")}</div></div>
          </div>
        </>
      );
    }
    if (tab === "data") {
      const store = (color: string, title: string, badge: SysCrit, body: string, id: string) => (
        <div className="sysmap-card" style={{ borderLeftColor: `var(--${color})` }}>
          <h4>{title} <span className={"sysmap-badge " + (badge === "Critical" ? "crit" : "opt")}>{badge}</span></h4>
          <p>{body}</p>
          <div className="sysmap-chips" style={{ marginTop: 10 }}>{chip(id)}</div>
        </div>
      );
      return (
        <>
          <p className="sysmap-h">Data flow view</p>
          <p className="sysmap-d">Four kinds of storage, each with a clear owner. Click a store to see what depends on it.</p>
          <div className="sysmap-grid">
            {store("teal", "SQLite — app database", "Critical", "Settings, chat turns, voice profiles, RAG metadata, query analytics, and audit log.", "sqlite")}
            {store("blue", "ChromaDB — vectors", "Optional", "Document chunk embeddings for semantic search (cosine HNSW). Powers the knowledge base.", "chromadb")}
            {store("amber", "MySQL — voiceai", "Optional", "Customer KYC view and admin users. Used by the KYC and admin features only.", "mysql")}
            {store("purple", "Local files — .local/", "Critical", "Audio cache, recordings, voice datasets, and models. Stays out of version control.", "tts_piper")}
          </div>
          <p className="sysmap-d" style={{ marginTop: 14 }}>What flows: your words → text → retrieved context + assembled prompt → model answer → spoken audio. Each turn is persisted; recordings stay local; cloud only sees what a turn sends when a cloud provider is on.</p>
        </>
      );
    }
    if (tab === "dep") {
      return (
        <>
          <p className="sysmap-h">Service dependency view</p>
          <p className="sysmap-d">Critical services must run for the core. Optional services add capability and degrade gracefully — if one is missing, the system keeps working without it.</p>
          <p className="sysmap-d" style={{ margin: "0 0 6px" }}><span className="sysmap-badge crit">Critical</span> &nbsp;the core breaks without these</p>
          <div className="sysmap-chips">{["frontend", "gateway", "conversation", "langrouter", "stt", "llm_ollama", "tts_piper", "sqlite"].map(chip)}</div>
          <p className="sysmap-d" style={{ margin: "16px 0 6px" }}><span className="sysmap-badge opt">Optional</span> &nbsp;nice to have, degrade safely</p>
          <div className="sysmap-chips">{["rag", "chromadb", "embeddings", "llm_openai", "llm_gemini", "tts_chatterbox", "tts_eleven", "kyc", "mysql", "webretrieval", "voicestudio", "sysmonitor"].map(chip)}</div>
        </>
      );
    }
    if (tab === "journey") {
      const j = SYS_JOURNEYS[jKey];
      const s = j.steps[jIdx];
      return (
        <>
          <p className="sysmap-h">User journey view</p>
          <p className="sysmap-d">Step through what actually happens, one stage at a time. The components touched at each step are clickable.</p>
          <div className="sysmap-bar" style={{ marginTop: 0 }}>
            <select value={jKey} onChange={(e) => { setJKey(e.target.value); setJIdx(0); }}>
              {Object.entries(SYS_JOURNEYS).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </select>
            <button className="sysmap-tab" type="button" onClick={() => setJIdx((jIdx - 1 + j.steps.length) % j.steps.length)}>← Back</button>
            <button className="sysmap-tab" type="button" onClick={() => setJIdx((jIdx + 1) % j.steps.length)}>Next →</button>
          </div>
          <div className="sysmap-jstep">
            <div style={{ fontSize: 12, color: "var(--muted)" }}>Step {jIdx + 1} of {j.steps.length}</div>
            <div style={{ fontSize: 15, fontWeight: 700, color: "var(--ink)", margin: "6px 0 4px" }}>{s.t}</div>
            <div style={{ fontSize: 13, color: "var(--muted)", lineHeight: 1.6 }}>{s.d}</div>
            {s.comp.length > 0 && <div className="sysmap-chips" style={{ marginTop: 10 }}>{s.comp.map(chip)}</div>}
            <div className="sysmap-jdots">{j.steps.map((_, i) => <span key={i} className={"sysmap-jdot" + (i === jIdx ? " on" : "")} />)}</div>
          </div>
        </>
      );
    }
    if (tab === "fail") {
      const f = (title: string, body: string, id: string) => (
        <div className="sysmap-card" style={{ borderLeftColor: "var(--rose)" }}><h4>{title}</h4><p>{body}</p><div className="sysmap-chips" style={{ marginTop: 10 }}>{chip(id)}</div></div>
      );
      return (
        <>
          <p className="sysmap-h">Failure recovery view</p>
          <p className="sysmap-d">The system is built to keep talking. Each common failure has an automatic fallback, and nothing fails silently.</p>
          <div className="sysmap-grid">
            {f("Cloud model fails", "If OpenAI or Gemini errors, the turn automatically retries on the local Ollama model.", "llm_ollama")}
            {f("Knowledge prompt fails", "If a RAG-augmented prompt errors, it retries as a direct local chat without context.", "rag")}
            {f("Selected voice can't speak", "It falls back to a built-in voice — unless strict mode is on, which fails clearly instead.", "tts_piper")}
            {f("MySQL is down", "Admin login uses a built-in fallback so you are not locked out.", "admin")}
            {f("KYC query fails", "The bad SQL and its error are fed back to the model, which fixes it and retries once.", "kyc")}
            {f("Settings store hiccup", "Settings persist to SQLite with a JSON file fallback, so config survives storage errors.", "sqlite")}
          </div>
        </>
      );
    }
    if (tab === "mon") {
      const m = (title: string, body: string, id?: string) => (
        <div className="sysmap-card" style={{ borderLeftColor: "var(--blue)" }}><h4>{title}</h4><p>{body}</p>{id && <div className="sysmap-chips" style={{ marginTop: 10 }}>{chip(id)}</div>}</div>
      );
      return (
        <>
          <p className="sysmap-h">Monitoring &amp; observability view</p>
          <p className="sysmap-d">Health and history are visible in the app, not buried in logs only an engineer can read.</p>
          <div className="sysmap-grid">
            {m("System metrics", "Live CPU, memory, and disk via an endpoint and a WebSocket stream.", "sysmonitor")}
            {m("Audit log", "Consent, clone, publish, delete, and settings changes, timestamped.", "audit")}
            {m("RAG query analytics", "Query counts, latency, zero-result rate, and top questions over time.")}
            {m("Voice socket status", "Tells you which voice capabilities are ready and what is blocking the rest.")}
            {m("Provider tests", "One-click connection + latency checks for each AI provider, with clear errors.")}
            {m("Logs page", "Combines app events, backend audit, provider status, and warnings in one place.")}
          </div>
        </>
      );
    }
    return null;
  }

  const detail = openComp ? SYS_COMPONENTS[openComp] : null;

  return (
    <section className="view-stack sysmap">
      <div className="view-header">
        <div>
          <h2>System Map</h2>
          <p>An interactive explorer of the whole platform — for every audience.</p>
        </div>
      </div>
      <p className="sysmap-sub">A local-first Nepali + English voice AI platform. One web app, one backend, local AI models, optional cloud. Pick a lens and a view; click any coloured box or chip to drill in.</p>

      <div className="sysmap-bar">
        <span style={{ fontSize: 13, color: "var(--muted)" }}>Read it as:</span>
        <select value={aud} onChange={(e) => setAud(e.target.value)} aria-label="Choose audience lens">
          <option value="general">Everyone (start here)</option>
          <option value="exec">Executive management</option>
          <option value="business">Business team</option>
          <option value="ops">Operations</option>
          <option value="compliance">Compliance</option>
          <option value="support">Customer support</option>
          <option value="product">Product owner</option>
          <option value="dev">Developer</option>
          <option value="infra">Infrastructure engineer</option>
          <option value="audit">Auditor</option>
          <option value="vendor">Vendor / integrator</option>
          <option value="student">Student</option>
          <option value="nontech">Non-technical</option>
        </select>
      </div>
      <p className="sysmap-aud">{SYS_AUD[aud]}</p>

      <div className="sysmap-tabs">
        {SYS_TABS.map((t) => (
          <button key={t.id} type="button" className={"sysmap-tab" + (tab === t.id ? " active" : "")} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>

      {detail && (
        <div className="sysmap-detail">
          <h4>
            {detail.n}
            <span className={"sysmap-badge " + (detail.crit === "Critical" ? "crit" : "opt")}>{detail.crit}</span>
            <button type="button" className="sysmap-chip" style={{ marginLeft: "auto", padding: "2px 9px" }} onClick={() => setOpenComp(null)}>✕</button>
          </h4>
          <dl className="sysmap-dl">
            <dt>What it is</dt><dd>{detail.what}</dd>
            <dt>Why it exists</dt><dd>{detail.why}</dd>
            <dt>Receives</dt><dd>{detail.recv}</dd>
            <dt>Produces</dt><dd>{detail.prod}</dd>
            <dt>Depends on</dt><dd>{detail.deps}</dd>
            <dt>Technology</dt><dd>{detail.tech}</dd>
          </dl>
        </div>
      )}

      <div className="sysmap-view">{renderTab()}</div>
    </section>
  );
}

function EvaluationView({
  ratings,
  onChange,
  history
}: {
  ratings: Record<string, number>;
  onChange: (ratings: Record<string, number>) => void;
  history: ConversationTurn[];
}) {
  const [activeTab, setActiveTab] = useState<"quality" | "latency" | "history">("quality");
  const latency = history[0]?.timings;

  const METRICS = [
    { key: "naturalness", label: "Naturalness", desc: "Does the AI voice sound like a real human speaking?", emoji: "🎤" },
    { key: "voiceSimilarity", label: "Voice similarity", desc: "Does the cloned voice match the original speaker?", emoji: "🎭" },
    { key: "nepaliPronunciation", label: "Nepali pronunciation", desc: "Are Nepali words and letters spoken correctly?", emoji: "🇳🇵" },
    { key: "englishPronunciation", label: "English pronunciation", desc: "Are English words clear and understandable?", emoji: "🇬🇧" },
    { key: "responseRelevance", label: "Response relevance", desc: "Did the AI answer what you actually asked?", emoji: "🎯" },
    { key: "conversationFlow", label: "Conversation flow", desc: "Did the turn feel smooth and natural?", emoji: "💬" },
  ] as const;

  const STARS = [
    { v: 1, label: "Poor", color: "#f87171" },
    { v: 2, label: "Fair", color: "#fbbf24" },
    { v: 3, label: "OK", color: "#facc15" },
    { v: 4, label: "Good", color: "#86efac" },
    { v: 5, label: "Great", color: "#4ade80" },
  ];

  const avgScore = Object.values(ratings).length > 0
    ? Object.values(ratings).reduce((a, b) => a + b, 0) / Object.values(ratings).length
    : 0;

  const scoreColor = avgScore >= 4 ? "#4ade80" : avgScore >= 3 ? "#facc15" : avgScore >= 2 ? "#fbbf24" : "#f87171";
  const scoreLabel = avgScore >= 4.5 ? "Excellent" : avgScore >= 4 ? "Good" : avgScore >= 3 ? "Average" : avgScore >= 2 ? "Below average" : avgScore > 0 ? "Poor" : "Not rated";

  const latencyItems = [
    { label: "Speech to Text", value: latency?.audio_received_to_transcript_ms, color: "#38bdf8", icon: "🎙️", good: 500 },
    { label: "LLM First Token", value: latency?.llm_first_token_ms, color: "#a78bfa", icon: "🧠", good: 800 },
    { label: "TTS Generation", value: latency?.tts_generation_ms, color: "#34d399", icon: "🔊", good: 1000 },
    { label: "Total Turn", value: latency?.total_turn_ms, color: "#fb923c", icon: "⏱️", good: 2500 },
  ];

  const exportData = () => {
    downloadBlob(
      new Blob([JSON.stringify({ ratings, avgScore: avgScore.toFixed(2), scoreLabel, latency, history: history.slice(0, 5) }, null, 2)], { type: "application/json" }),
      "evaluation-report.json"
    );
  };

  return (
    <section className="view-stack" style={{ padding: "0 0 40px" }}>
      {/* Header */}
      <div style={{ padding: "20px 24px 0", display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>Quality Evaluation</h2>
          <p style={{ margin: "4px 0 0", fontSize: 13, color: "var(--muted)" }}>Rate voice quality, pronunciation accuracy, and response timing</p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="rag-toolbar-btn" onClick={() => onChange({})} disabled={Object.keys(ratings).length === 0}>
            <RefreshCw size={13} /><span>Reset ratings</span>
          </button>
          <button className="rag-toolbar-btn primary" onClick={exportData}>
            <Download size={13} /><span>Export report</span>
          </button>
        </div>
      </div>

      {/* Score summary card */}
      <div style={{ margin: "20px 24px 0", padding: "20px 24px", background: "var(--panel)", borderRadius: 14, border: "1px solid var(--line)", display: "flex", alignItems: "center", gap: 24, flexWrap: "wrap" }}>
        <div style={{ textAlign: "center", minWidth: 90 }}>
          <div style={{ fontSize: 48, fontWeight: 900, color: scoreColor, lineHeight: 1 }}>{avgScore > 0 ? avgScore.toFixed(1) : "—"}</div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>out of 5.0</div>
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div style={{ fontSize: 18, fontWeight: 700, color: scoreColor, marginBottom: 8 }}>{scoreLabel}</div>
          <div style={{ height: 8, background: "var(--line)", borderRadius: 4, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${(avgScore / 5) * 100}%`, background: scoreColor, borderRadius: 4, transition: "width 0.5s" }} />
          </div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 6 }}>{Object.keys(ratings).length} of {METRICS.length} dimensions rated</div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, minWidth: 200 }}>
          {latencyItems.slice(0, 2).map(item => (
            <div key={item.label} style={{ padding: "8px 12px", background: "var(--surface-2)", borderRadius: 8, textAlign: "center" }}>
              <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 2 }}>{item.icon} {item.label}</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: item.value != null && item.value <= item.good ? "var(--green)" : item.value != null ? "var(--rose)" : "var(--muted)" }}>
                {item.value != null ? `${Math.round(item.value)}ms` : "—"}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ padding: "16px 24px 0", display: "flex", gap: 4, borderBottom: "1px solid var(--line)" }}>
        {[{ id: "quality", label: "Voice Quality" }, { id: "latency", label: "Speed & Latency" }, { id: "history", label: "Turn History" }].map(t => (
          <button key={t.id} type="button" onClick={() => setActiveTab(t.id as any)}
            style={{ padding: "8px 16px", borderRadius: "8px 8px 0 0", border: "1px solid transparent", cursor: "pointer", fontSize: 13, fontWeight: 600, transition: "all 0.15s",
              background: activeTab === t.id ? "var(--panel)" : "transparent",
              color: activeTab === t.id ? "var(--teal)" : "var(--muted)",
              borderColor: activeTab === t.id ? "var(--line)" : "transparent",
              borderBottomColor: activeTab === t.id ? "var(--panel)" : "transparent",
              marginBottom: activeTab === t.id ? -1 : 0,
            }}>{t.label}</button>
        ))}
      </div>

      <div style={{ padding: "24px" }}>

        {activeTab === "quality" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {METRICS.map(({ key, label, desc, emoji }) => {
              const score = ratings[key] ?? 0;
              return (
                <div key={key} style={{ padding: "16px 20px", background: "var(--panel)", borderRadius: 12, border: `1px solid ${score > 0 ? "rgba(0,166,81,0.2)" : "var(--line)"}`, transition: "border-color 0.2s" }}>
                  <div style={{ display: "flex", alignItems: "flex-start", gap: 12, marginBottom: 12 }}>
                    <span style={{ fontSize: 22, flexShrink: 0 }}>{emoji}</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 14, fontWeight: 700 }}>{label}</div>
                      <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>{desc}</div>
                    </div>
                    {score > 0 && (
                      <div style={{ fontSize: 20, fontWeight: 900, color: STARS[score - 1]?.color }}>{score}/5</div>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 8 }}>
                    {STARS.map(({ v, label: starLabel, color }) => (
                      <button key={v} type="button" onClick={() => onChange({ ...ratings, [key]: v })}
                        style={{ flex: 1, padding: "10px 4px", borderRadius: 8, border: `2px solid ${score === v ? color : "var(--line)"}`, background: score === v ? `${color}22` : "transparent", cursor: "pointer", transition: "all 0.15s", display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
                        <span style={{ fontSize: 18 }}>{"★".repeat(v)}{"☆".repeat(5 - v)}</span>
                        <span style={{ fontSize: 10, color: score === v ? color : "var(--muted)", fontWeight: score === v ? 700 : 400 }}>{starLabel}</span>
                      </button>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {activeTab === "latency" && (
          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 16, marginBottom: 24 }}>
              {latencyItems.map(item => {
                const isGood = item.value != null && item.value <= item.good;
                const pct = item.value != null ? Math.min(100, (item.value / (item.good * 2)) * 100) : 0;
                return (
                  <div key={item.label} style={{ padding: "20px", background: "var(--panel)", borderRadius: 12, border: "1px solid var(--line)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                      <span style={{ fontSize: 20 }}>{item.icon}</span>
                      <span style={{ fontSize: 13, fontWeight: 600 }}>{item.label}</span>
                      <span style={{ marginLeft: "auto", fontSize: 10, padding: "2px 8px", borderRadius: 10, fontWeight: 700,
                        background: isGood ? "rgba(72,187,120,0.15)" : item.value != null ? "rgba(245,101,101,0.15)" : "transparent",
                        color: isGood ? "var(--green)" : item.value != null ? "var(--rose)" : "var(--muted)" }}>
                        {isGood ? "Fast" : item.value != null ? "Slow" : "No data"}
                      </span>
                    </div>
                    <div style={{ fontSize: 36, fontWeight: 900, color: isGood ? "var(--green)" : item.value != null ? "var(--rose)" : "var(--muted)", marginBottom: 8 }}>
                      {item.value != null ? `${Math.round(item.value)}` : "—"}<span style={{ fontSize: 16, fontWeight: 400, color: "var(--muted)" }}>{item.value != null ? "ms" : ""}</span>
                    </div>
                    <div style={{ height: 6, background: "var(--line)", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${pct}%`, background: isGood ? "var(--green)" : "var(--rose)", borderRadius: 3, transition: "width 0.5s" }} />
                    </div>
                    <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 6 }}>Target: &lt;{item.good}ms</div>
                  </div>
                );
              })}
            </div>

            <div style={{ padding: "16px 20px", background: "var(--panel)", borderRadius: 12, border: "1px solid var(--line)" }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Latency targets for real-time conversation</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12 }}>
                {[
                  { label: "Speech-to-text", target: "<500ms", good: "Under 500ms feels instant", note: "Whisper small gives best speed/accuracy" },
                  { label: "LLM first token", target: "<800ms", good: "Under 800ms feels responsive", note: "Local Ollama: 200-600ms · OpenAI: 400-900ms" },
                  { label: "TTS generation", target: "<1000ms", good: "Under 1s feels natural", note: "Piper is fastest · OpenAI TTS ~500ms" },
                  { label: "Total turn", target: "<2500ms", good: "Under 2.5s is comfortable", note: "Users notice latency above 3 seconds" },
                ].map(r => (
                  <div key={r.label} style={{ display: "flex", gap: 12, padding: "8px 0", borderBottom: "1px solid var(--line)" }}>
                    <span style={{ width: 130, fontWeight: 600 }}>{r.label}</span>
                    <span style={{ color: "var(--teal)", fontWeight: 700, width: 70 }}>{r.target}</span>
                    <span style={{ color: "var(--muted)", flex: 1 }}>{r.note}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === "history" && (
          <div>
            {history.length === 0 ? (
              <div style={{ textAlign: "center", padding: "48px 24px", color: "var(--muted)" }}>
                <Mic size={32} style={{ marginBottom: 12, opacity: 0.3 }} />
                <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>No conversation turns yet</div>
                <div style={{ fontSize: 13 }}>Complete a voice turn to see turn-by-turn history and timing breakdown here</div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {history.slice(0, 10).map((turn, i) => (
                  <div key={i} style={{ padding: "16px 20px", background: "var(--panel)", borderRadius: 12, border: "1px solid var(--line)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                      <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "monospace" }}>Turn #{history.length - i}</span>
                      <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--teal)", fontWeight: 700 }}>
                        {turn.timings?.total_turn_ms != null ? `${Math.round(turn.timings.total_turn_ms)}ms total` : ""}
                      </span>
                    </div>
                    <div style={{ fontSize: 13, marginBottom: 6, fontStyle: "italic", color: "var(--muted)" }}>"{turn.transcript}"</div>
                    <div style={{ fontSize: 12 }}>{turn.response}</div>
                    {turn.timings && (
                      <div style={{ display: "flex", gap: 12, marginTop: 10, fontSize: 11, color: "var(--muted)" }}>
                        <span>STT: {Math.round(turn.timings.audio_received_to_transcript_ms ?? 0)}ms</span>
                        <span>LLM: {Math.round(turn.timings.llm_first_token_ms ?? 0)}ms</span>
                        <span>TTS: {Math.round(turn.timings.tts_generation_ms ?? 0)}ms</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
const OPENAI_VOICES = [
  { id: "openai-alloy", name: "Alloy", desc: "Balanced, neutral" },
  { id: "openai-echo", name: "Echo", desc: "Crisp, clear" },
  { id: "openai-fable", name: "Fable", desc: "Expressive, warm" },
  { id: "openai-nova", name: "Nova", desc: "Friendly, bright" },
  { id: "openai-onyx", name: "Onyx", desc: "Deep, authoritative" },
  { id: "openai-shimmer", name: "Shimmer", desc: "Soft, gentle" },
];

// ─────────────────────────────────────────────────────────────────
// LLM Brain Selector — standalone card component
// ─────────────────────────────────────────────────────────────────
function LLMBrainSelector({
  settings,
  onSave,
}: {
  settings: BackendSettings;
  onSave: (s: BackendSettings) => void;
}) {
  const [active, setActive] = useState(settings.llm_provider || "local");
  const [apiKey, setApiKey] = useState(settings.openai_api_key || "");
  const [openaiModel, setOpenaiModel] = useState(settings.openai_model || "gpt-4o-mini");
  const [openaiModels, setOpenaiModels] = useState<string[]>([
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
    "gpt-3.5-turbo", "o1", "o1-mini", "o3-mini",
  ]);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; detail: string; latency_ms?: number } | null>(null);
  const [saving, setSaving] = useState(false);
  const [fetchingModels, setFetchingModels] = useState(false);

  useEffect(() => { setActive(settings.llm_provider || "local"); }, [settings.llm_provider]);

  const fetchOpenAIModels = async () => {
    if (!apiKey && !settings.openai_api_key) return;
    setFetchingModels(true);
    try {
      const res = await getOpenAIModels();
      if (res.models.length > 0) setOpenaiModels(res.models);
    } catch {}
    finally { setFetchingModels(false); }
  };

  const handleSetActive = async (provider: string) => {
    setSaving(true);
    try {
      await setActiveAIProvider(provider);
      setActive(provider);
      onSave({ ...settings, llm_provider: provider });
    } catch {}
    finally { setSaving(false); }
  };

  const handleSaveOpenAI = async () => {
    setSaving(true);
    try {
      // Must call updateSettings FIRST to save the key
      await updateSettings({ openai_api_key: apiKey, openai_model: openaiModel, llm_provider: "openai" });
      await setActiveAIProvider("openai");
      onSave({ ...settings, llm_provider: "openai", openai_api_key: apiKey, openai_model: openaiModel });
      setActive("openai");
      setTestResult({ ok: true, detail: "OpenAI settings saved!" });
    } catch (e: any) {
      setTestResult({ ok: false, detail: e.message || "Failed to save" });
    } finally { setSaving(false); }
  };

  const handleTest = async () => {
    setTesting(true); setTestResult(null);
    try {
      const res = await testProvider(active, active === "openai" ? { openai_api_key: apiKey, openai_model: openaiModel } : undefined);
      setTestResult(res);
    } catch (e: any) {
      setTestResult({ ok: false, detail: e.message });
    } finally { setTesting(false); }
  };

  const PROVIDERS = [
    {
      id: "local",
      label: "Local AI",
      sub: "Ollama · runs on your machine",
      icon: "🦙",
      color: "var(--teal)",
      bg: "var(--teal-soft)",
    },
    {
      id: "openai",
      label: "OpenAI",
      sub: "GPT-4o, o1, o3 · cloud API",
      icon: "🤖",
      color: "#10a37f",
      bg: "rgba(16,163,127,0.1)",
    },
    {
      id: "gemini",
      label: "Gemini",
      sub: "Google AI · cloud API",
      icon: "✨",
      color: "#4285f4",
      bg: "rgba(66,133,244,0.1)",
    },
  ];

  return (
    <div style={{ gridColumn: "1 / -1" }}>
      <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", letterSpacing: 0.6, fontWeight: 700, marginBottom: 10 }}>
        AI Brain — Active LLM Provider
      </div>

      {/* Provider cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginBottom: 14 }}>
        {PROVIDERS.map(p => (
          <button key={p.id} type="button" onClick={() => void handleSetActive(p.id)}
            style={{
              padding: "12px 14px", borderRadius: 10, cursor: "pointer",
              display: "flex", alignItems: "center", gap: 10,
              border: active === p.id ? `2px solid ${p.color}` : "2px solid var(--line)",
              background: active === p.id ? p.bg : "var(--panel)",
              transition: "all 0.15s", textAlign: "left",
            }}>
            <span style={{ fontSize: 22, flexShrink: 0 }}>{p.icon}</span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: active === p.id ? p.color : "var(--ink)" }}>
                {p.label}
                {active === p.id && <span style={{ marginLeft: 6, fontSize: 10, fontWeight: 800, color: p.color }}>● ACTIVE</span>}
              </div>
              <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 1 }}>{p.sub}</div>
            </div>
          </button>
        ))}
      </div>

      {/* Expanded config for active provider */}
      {active === "local" && (
        <div style={{ padding: "12px 16px", background: "var(--panel)", borderRadius: 8, border: "1px solid var(--teal)", borderLeft: "3px solid var(--teal)", display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 200 }}>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 2 }}>Ollama — Local model active</div>
            <div style={{ fontSize: 11, color: "var(--muted)" }}>Model: <strong>{settings.ollama_model}</strong> · URL: {settings.ollama_base_url}</div>
          </div>
          <button className="rag-toolbar-btn" onClick={handleTest} disabled={testing} style={{ flexShrink: 0 }}>
            {testing ? <RefreshCw size={13} className="spin" /> : <CheckCircle2 size={13} />}
            <span>{testing ? "Testing…" : "Test connection"}</span>
          </button>
          {testResult && (
            <span className={`rag-status-badge ${testResult.ok ? "ok" : "err"}`} style={{ fontSize: 11 }}>
              {testResult.ok ? `✓ ${testResult.latency_ms?.toFixed(0)}ms` : `✗ ${testResult.detail}`}
            </span>
          )}
        </div>
      )}

      {active === "openai" && (
        <div style={{ padding: "14px 16px", background: "var(--panel)", borderRadius: 8, border: "1px solid #10a37f33", borderLeft: "3px solid #10a37f" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 200px", gap: 10, marginBottom: 10 }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 11, color: "var(--muted)" }}>OpenAI API Key</span>
              <div style={{ display: "flex", gap: 6 }}>
                <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)}
                  placeholder="sk-proj-…" style={{ flex: 1, fontFamily: "monospace", fontSize: 12 }}
                  onBlur={fetchOpenAIModels} />
              </div>
              <span style={{ fontSize: 10, color: "var(--muted)" }}>
                Get key at <span style={{ color: "#10a37f" }}>platform.openai.com/api-keys</span>
              </span>
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 11, color: "var(--muted)", display: "flex", alignItems: "center", gap: 4 }}>
                Model
                {fetchingModels && <RefreshCw size={10} className="spin" />}
              </span>
              <select value={openaiModel} onChange={e => setOpenaiModel(e.target.value)}>
                {openaiModels.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
              <button type="button" style={{ fontSize: 10, background: "none", border: "none", color: "var(--muted)", cursor: "pointer", textAlign: "left", padding: 0 }}
                onClick={fetchOpenAIModels} disabled={fetchingModels}>
                ↻ Fetch available models
              </button>
            </label>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button className="rag-toolbar-btn primary" onClick={handleSaveOpenAI} disabled={saving} style={{ background: "#10a37f", borderColor: "#10a37f" }}>
              <Save size={13} /><span>{saving ? "Saving…" : "Save & Activate OpenAI"}</span>
            </button>
            <button className="rag-toolbar-btn" onClick={handleTest} disabled={testing}>
              {testing ? <RefreshCw size={13} className="spin" /> : <CheckCircle2 size={13} />}
              <span>{testing ? "Testing…" : "Test connection"}</span>
            </button>
            {testResult && (
              <span className={`rag-status-badge ${testResult.ok ? "ok" : "err"}`} style={{ fontSize: 11, padding: "6px 10px" }}>
                {testResult.ok ? `✓ Connected · ${testResult.latency_ms?.toFixed(0)}ms` : `✗ ${testResult.detail.slice(0, 80)}`}
              </span>
            )}
          </div>
        </div>
      )}

      {active === "gemini" && (
        <div style={{ padding: "14px 16px", background: "var(--panel)", borderRadius: 8, border: "1px solid #4285f433", borderLeft: "3px solid #4285f4" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 200px", gap: 10, marginBottom: 10 }}>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 11, color: "var(--muted)" }}>Gemini API Key</span>
              <input type="password" value={settings.gemini_api_key || ""}
                onChange={e => onSave({ ...settings, gemini_api_key: e.target.value })}
                placeholder="AIza…" style={{ fontFamily: "monospace", fontSize: 12 }} />
              <span style={{ fontSize: 10, color: "var(--muted)" }}>Get key at aistudio.google.com/apikey</span>
            </label>
            <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 11, color: "var(--muted)" }}>Model</span>
              <select value={settings.gemini_model || "gemini-1.5-flash"}
                onChange={e => onSave({ ...settings, gemini_model: e.target.value })}>
                <option value="gemini-2.0-flash">gemini-2.0-flash</option>
                <option value="gemini-1.5-flash">gemini-1.5-flash</option>
                <option value="gemini-1.5-pro">gemini-1.5-pro</option>
                <option value="gemini-2.0-flash-thinking-exp">gemini-2.0-flash-thinking-exp</option>
              </select>
            </label>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="rag-toolbar-btn" style={{ background: "#4285f4", borderColor: "#4285f4", color: "#fff" }}
              onClick={async () => { await setActiveAIProvider("gemini"); onSave({ ...settings, llm_provider: "gemini" }); }}>
              <Save size={13} /><span>Save & Activate Gemini</span>
            </button>
            <button className="rag-toolbar-btn" onClick={handleTest} disabled={testing}>
              {testing ? <RefreshCw size={13} className="spin" /> : <CheckCircle2 size={13} />}
              <span>{testing ? "Testing…" : "Test connection"}</span>
            </button>
            {testResult && (
              <span className={`rag-status-badge ${testResult.ok ? "ok" : "err"}`} style={{ fontSize: 11, padding: "6px 10px" }}>
                {testResult.ok ? `✓ Connected · ${testResult.latency_ms?.toFixed(0)}ms` : `✗ ${testResult.detail.slice(0, 80)}`}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// SETTINGS VIEW REPLACEMENT — paste between markers
function SettingsView({
  settings,
  voices,
  onSave,
  onDelete,
  onTtsTest
}: {
  settings: BackendSettings;
  voices: VoicesResponse | null;
  onSave: (settings: BackendSettings) => void;
  onDelete: () => void;
  onTtsTest: (language: "ne" | "en", voiceId?: string) => Promise<void> | void;
}) {
  const [draft, setDraft] = useState(settings);
  useEffect(() => setDraft(settings), [settings]);
  const [tab, setTab] = useState<"ai" | "voice" | "speech" | "system" | "advanced">("ai");
  const [saving, setSaving] = useState(false);
  const [saveOk, setSaveOk] = useState(false);
  const [voicePreviewLoading, setVoicePreviewLoading] = useState<string | null>(null);
  const [voicePreviewError, setVoicePreviewError] = useState<string | null>(null);
  const [providerTests, setProviderTests] = useState<Record<string, { ok: boolean; detail: string; latency_ms?: number; time?: string }>>({});

  const handleVoicePreview = async (voiceId: string) => {
    setVoicePreviewLoading(voiceId);
    setVoicePreviewError(null);
    try {
      const res = await previewTts("Hello, this is a preview of my voice. How can I help you today?", voiceId, "en");
      if (res.ok && res.audio_url) {
        const audio = new Audio(`${API_HTTP}${res.audio_url}`);
        audio.play();
      } else {
        setVoicePreviewError(res.detail ?? "Preview failed");
      }
    } catch (e: any) {
      setVoicePreviewError(e.message);
    } finally {
      setVoicePreviewLoading(null);
    }
  };

  const handleTestConnection = async (provider: string) => {
    try {
      const payload = provider === "local" ? undefined : {
        [`${provider}_api_key`]: draft[`${provider}_api_key` as keyof BackendSettings],
        [`${provider}_model`]: draft[`${provider}_model` as keyof BackendSettings],
      };
      const res = await testProvider(provider, payload);
      setProviderTests(prev => ({ ...prev, [provider]: { ok: res.ok, detail: res.detail, latency_ms: res.latency_ms, time: new Date().toLocaleTimeString() } }));
    } catch (err: any) {
      setProviderTests(prev => ({ ...prev, [provider]: { ok: false, detail: err.message || "Connection test failed", time: new Date().toLocaleTimeString() } }));
    }
  };

  const handleDeleteKey = async (provider: "openai" | "gemini" | "elevenlabs") => {
    const name = provider === 'openai' ? 'OpenAI' : provider === 'gemini' ? 'Gemini' : 'ElevenLabs';
    if (!confirm(`Delete the ${name} API key?`)) return;
    try {
      let updated: BackendSettings;
      if (provider === "openai") updated = await deleteOpenAIKey();
      else if (provider === "gemini") updated = await deleteGeminiKey();
      else updated = await deleteElevenLabsKey();
      setDraft(updated); onSave(updated);
      setProviderTests(prev => { const n = { ...prev }; delete n[provider]; return n; });
    } catch (err: any) { alert(`Failed to delete key: ${err.message}`); }
  };

  const handleSave = async () => {
    setSaving(true);
    try { await onSave(draft); setSaveOk(true); setTimeout(() => setSaveOk(false), 2500); }
    finally { setSaving(false); }
  };

  const TABS = [
    { id: "ai", label: "AI Brain", icon: "🧠" },
    { id: "voice", label: "Voice & TTS", icon: "🎙️" },
    { id: "speech", label: "Speech & STT", icon: "🔊" },
    { id: "system", label: "System", icon: "⚙️" },
    { id: "advanced", label: "Advanced", icon: "🔧" },
  ] as const;

  const SectionLabel = ({ children }: { children: React.ReactNode }) => (
    <div style={{ fontSize: 10, fontWeight: 800, letterSpacing: 1.2, textTransform: "uppercase", color: "var(--teal)", marginBottom: 12, marginTop: 4, borderBottom: "1px solid var(--line)", paddingBottom: 6 }}>{children}</div>
  );

  const FieldRow = ({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) => (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 14 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--ink)" }}>{label}</span>
        {hint && <span style={{ fontSize: 10, color: "var(--muted)" }}>{hint}</span>}
      </div>
      {children}
    </div>
  );

  const Toggle = ({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) => (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 14px", background: "var(--panel)", borderRadius: 8, border: "1px solid var(--line)", marginBottom: 8, cursor: "pointer" }}
      onClick={() => onChange(!checked)}>
      <div style={{ width: 36, height: 20, borderRadius: 10, background: checked ? "var(--teal)" : "var(--line)", position: "relative", transition: "background 0.2s", flexShrink: 0 }}>
        <div style={{ position: "absolute", top: 2, left: checked ? 18 : 2, width: 16, height: 16, borderRadius: "50%", background: "#fff", transition: "left 0.2s", boxShadow: "0 1px 3px rgba(0,0,0,0.3)" }} />
      </div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>{label}</div>
        {hint && <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 1 }}>{hint}</div>}
      </div>
      <span style={{ fontSize: 10, fontWeight: 700, color: checked ? "var(--teal)" : "var(--muted)" }}>{checked ? "ON" : "OFF"}</span>
    </div>
  );

  const providerBadge = (p: string, test?: { ok: boolean; latency_ms?: number }) => {
    if (!test) return null;
    return (
      <span style={{ fontSize: 10, padding: "3px 8px", borderRadius: 20, background: test.ok ? "rgba(72,187,120,0.15)" : "rgba(245,101,101,0.15)", color: test.ok ? "var(--green)" : "var(--rose)", fontWeight: 700 }}>
        {test.ok ? `✓ ${test.latency_ms?.toFixed(0) ?? "?"}ms` : "✗ Failed"}
      </span>
    );
  };

  return (
    <section className="view-stack" style={{ padding: "0 0 40px" }}>
      {/* Header */}
      <div style={{ padding: "20px 24px 0", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>Settings</h2>
          <p style={{ margin: "4px 0 0", fontSize: 13, color: "var(--muted)" }}>Configure AI providers, voices, audio pipeline, and system behavior</p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {saveOk && <span style={{ fontSize: 12, color: "var(--green)", fontWeight: 700, display: "flex", alignItems: "center", gap: 4 }}><CheckCircle2 size={14} /> Saved</span>}
          <button className="rag-toolbar-btn primary" onClick={handleSave} disabled={saving}>
            <Save size={13} /><span>{saving ? "Saving…" : "Save changes"}</span>
          </button>
        </div>
      </div>

      {/* Tab nav */}
      <div style={{ padding: "16px 24px 0", display: "flex", gap: 4, borderBottom: "1px solid var(--line)", marginBottom: 0 }}>
        {TABS.map(t => (
          <button key={t.id} type="button" onClick={() => setTab(t.id as any)}
            style={{ padding: "8px 16px", borderRadius: "8px 8px 0 0", border: "1px solid transparent", borderBottom: "none", cursor: "pointer", fontSize: 13, fontWeight: 600, transition: "all 0.15s",
              background: tab === t.id ? "var(--panel)" : "transparent",
              color: tab === t.id ? "var(--teal)" : "var(--muted)",
              borderColor: tab === t.id ? "var(--line)" : "transparent",
              borderBottomColor: tab === t.id ? "var(--panel)" : "transparent",
              marginBottom: tab === t.id ? -1 : 0,
            }}>
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ padding: "24px 24px" }}>

        {/* ── AI BRAIN TAB ── */}
        {tab === "ai" && (
          <div>
            <LLMBrainSelector settings={draft} onSave={(s) => { setDraft(s); onSave(s); }} />

            <div style={{ marginTop: 28 }}>
              <SectionLabel>Local Ollama Settings</SectionLabel>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <FieldRow label="Ollama server URL" hint="Where Ollama is running">
                  <input value={draft.ollama_base_url} onChange={e => setDraft({ ...draft, ollama_base_url: e.target.value })} style={{ fontSize: 13 }} />
                </FieldRow>
                <FieldRow label="Local model" hint="Pulled via ollama pull">
                  <select value={draft.ollama_model} onChange={e => setDraft({ ...draft, ollama_model: e.target.value })}>
                    <option value="qwen2.5:7b">qwen2.5:7b — Best overall</option>
                    <option value="gemma3:4b">gemma3:4b — Google</option>
                    <option value="qwen3:1.7b">qwen3:1.7b — Fast & light</option>
                    <option value="qwen3:4b">qwen3:4b — Balanced</option>
                    <option value="llama3:latest">llama3 — Meta</option>
                    <option value="mistral:latest">mistral — Multilingual</option>
                    <option value="llama3.2:3b">llama3.2:3b — Compact</option>
                    <option value="llama3.2:1b">llama3.2:1b — Tiny</option>
                  </select>
                </FieldRow>
                <FieldRow label="Temperature" hint="0 = precise, 1 = creative">
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input type="range" min="0" max="1" step="0.05" value={draft.ollama_temperature}
                      onChange={e => setDraft({ ...draft, ollama_temperature: Number(e.target.value) })} style={{ flex: 1 }} />
                    <span style={{ fontSize: 13, fontWeight: 700, minWidth: 32, textAlign: "right" }}>{draft.ollama_temperature}</span>
                  </div>
                </FieldRow>
                <FieldRow label="Max tokens" hint="Response length limit">
                  <input type="number" min="32" max="2048" step="16" value={draft.ollama_num_predict}
                    onChange={e => setDraft({ ...draft, ollama_num_predict: Number(e.target.value) })} style={{ fontSize: 13 }} />
                </FieldRow>
                <FieldRow label="Keep alive" hint="How long model stays in RAM">
                  <select value={draft.ollama_keep_alive} onChange={e => setDraft({ ...draft, ollama_keep_alive: e.target.value })}>
                    <option value="5m">5 minutes</option>
                    <option value="10m">10 minutes</option>
                    <option value="30m">30 minutes</option>
                    <option value="1h">1 hour</option>
                    <option value="-1">Forever (keep loaded)</option>
                  </select>
                </FieldRow>
                <FieldRow label="Fallback model" hint="Used if primary fails">
                  <select value={draft.local_fallback_model} onChange={e => setDraft({ ...draft, local_fallback_model: e.target.value })}>
                    <option value="qwen3:4b">qwen3:4b</option>
                    <option value="llama3.2:3b">llama3.2:3b</option>
                    <option value="llama3.2:1b">llama3.2:1b</option>
                    <option value="gemma3:4b">gemma3:4b</option>
                  </select>
                </FieldRow>
              </div>

              <button className="rag-toolbar-btn" onClick={() => handleTestConnection("local")} style={{ marginTop: 4 }}>
                <Activity size={13} /><span>Test Local AI</span>
                {providerBadge("local", providerTests["local"])}
              </button>
              {providerTests["local"] && !providerTests["local"].ok && (
                <div style={{ marginTop: 8, fontSize: 12, color: "var(--rose)", background: "rgba(245,101,101,0.1)", padding: "8px 12px", borderRadius: 6 }}>
                  {providerTests["local"].detail}
                </div>
              )}
            </div>

            <div style={{ marginTop: 28 }}>
              <SectionLabel>Cloud Provider Keys</SectionLabel>
              <div style={{ display: "grid", gap: 12 }}>
                {/* OpenAI */}
                <div style={{ padding: "16px", background: "var(--panel)", borderRadius: 10, border: "1px solid var(--line)", borderLeft: "3px solid #10a37f" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                    <span style={{ fontSize: 18 }}>🤖</span>
                    <strong style={{ fontSize: 14 }}>OpenAI</strong>
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>GPT-4o, o1, o3 · Chat + TTS</span>
                    <span style={{ marginLeft: "auto", fontSize: 10, padding: "2px 8px", borderRadius: 10, background: draft.openai_api_key ? "rgba(72,187,120,0.15)" : "rgba(113,128,150,0.15)", color: draft.openai_api_key ? "var(--green)" : "var(--muted)", fontWeight: 700 }}>
                      {draft.openai_api_key ? "Configured" : "Not set"}
                    </span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8, marginBottom: 8 }}>
                    <input type="password" placeholder={draft.openai_api_key ? "sk-proj-•••• (saved)" : "sk-proj-…"}
                      value={draft.openai_api_key} onChange={e => setDraft({ ...draft, openai_api_key: e.target.value })}
                      style={{ fontFamily: "monospace", fontSize: 12 }} />
                    {draft.openai_api_key && (
                      <button className="rag-toolbar-btn" onClick={() => handleDeleteKey("openai")} style={{ color: "var(--rose)", borderColor: "var(--rose)", padding: "6px 10px" }}>
                        <Trash2 size={12} />
                      </button>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                    <select value={draft.openai_model} onChange={e => setDraft({ ...draft, openai_model: e.target.value })} style={{ fontSize: 12, flex: 1, minWidth: 140 }}>
                      <option value="gpt-4o">gpt-4o — Best</option>
                      <option value="gpt-4o-mini">gpt-4o-mini — Fast & cheap</option>
                      <option value="gpt-4-turbo">gpt-4-turbo</option>
                      <option value="o1">o1 — Reasoning</option>
                      <option value="o1-mini">o1-mini</option>
                      <option value="o3-mini">o3-mini</option>
                    </select>
                    <button className="rag-toolbar-btn" onClick={() => handleTestConnection("openai")}>
                      <Activity size={12} /><span>Test</span>
                    </button>
                    {providerBadge("openai", providerTests["openai"])}
                  </div>
                  {providerTests["openai"] && (
                    <div style={{ marginTop: 8, fontSize: 11, color: providerTests["openai"].ok ? "var(--green)" : "var(--rose)" }}>
                      {providerTests["openai"].detail} · {providerTests["openai"].time}
                    </div>
                  )}
                </div>

                {/* Gemini */}
                <div style={{ padding: "16px", background: "var(--panel)", borderRadius: 10, border: "1px solid var(--line)", borderLeft: "3px solid #4285f4" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                    <span style={{ fontSize: 18 }}>✨</span>
                    <strong style={{ fontSize: 14 }}>Google Gemini</strong>
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>gemini-2.0-flash, gemini-1.5-pro</span>
                    <span style={{ marginLeft: "auto", fontSize: 10, padding: "2px 8px", borderRadius: 10, background: draft.gemini_api_key ? "rgba(72,187,120,0.15)" : "rgba(113,128,150,0.15)", color: draft.gemini_api_key ? "var(--green)" : "var(--muted)", fontWeight: 700 }}>
                      {draft.gemini_api_key ? "Configured" : "Not set"}
                    </span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8, marginBottom: 8 }}>
                    <input type="password" placeholder={draft.gemini_api_key ? "AIza•••• (saved)" : "AIza…"}
                      value={draft.gemini_api_key} onChange={e => setDraft({ ...draft, gemini_api_key: e.target.value })}
                      style={{ fontFamily: "monospace", fontSize: 12 }} />
                    {draft.gemini_api_key && (
                      <button className="rag-toolbar-btn" onClick={() => handleDeleteKey("gemini")} style={{ color: "var(--rose)", borderColor: "var(--rose)", padding: "6px 10px" }}>
                        <Trash2 size={12} />
                      </button>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                    <select value={draft.gemini_model} onChange={e => setDraft({ ...draft, gemini_model: e.target.value })} style={{ fontSize: 12, flex: 1, minWidth: 140 }}>
                      <option value="gemini-2.0-flash">gemini-2.0-flash — Fast</option>
                      <option value="gemini-1.5-flash">gemini-1.5-flash</option>
                      <option value="gemini-1.5-pro">gemini-1.5-pro — Smart</option>
                      <option value="gemini-2.0-flash-thinking-exp">gemini-2.0-flash-thinking-exp</option>
                    </select>
                    <button className="rag-toolbar-btn" onClick={() => handleTestConnection("gemini")}>
                      <Activity size={12} /><span>Test</span>
                    </button>
                    {providerBadge("gemini", providerTests["gemini"])}
                  </div>
                </div>

                {/* ElevenLabs */}
                <div style={{ padding: "16px", background: "var(--panel)", borderRadius: 10, border: "1px solid var(--line)", borderLeft: "3px solid #9b59b6" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                    <span style={{ fontSize: 18 }}>🎵</span>
                    <strong style={{ fontSize: 14 }}>ElevenLabs</strong>
                    <span style={{ fontSize: 11, color: "var(--muted)" }}>Premium TTS & voice cloning</span>
                    <span style={{ marginLeft: "auto", fontSize: 10, padding: "2px 8px", borderRadius: 10, background: draft.elevenlabs_api_key ? "rgba(72,187,120,0.15)" : "rgba(113,128,150,0.15)", color: draft.elevenlabs_api_key ? "var(--green)" : "var(--muted)", fontWeight: 700 }}>
                      {draft.elevenlabs_api_key ? "Configured" : "Not set"}
                    </span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8 }}>
                    <input type="password" placeholder={draft.elevenlabs_api_key ? "sk_•••• (saved)" : "sk_live_…"}
                      value={draft.elevenlabs_api_key} onChange={e => setDraft({ ...draft, elevenlabs_api_key: e.target.value })}
                      style={{ fontFamily: "monospace", fontSize: 12 }} />
                    {draft.elevenlabs_api_key && (
                      <button className="rag-toolbar-btn" onClick={() => handleDeleteKey("elevenlabs")} style={{ color: "var(--rose)", borderColor: "var(--rose)", padding: "6px 10px" }}>
                        <Trash2 size={12} />
                      </button>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                    <button className="rag-toolbar-btn" onClick={() => handleTestConnection("elevenlabs")}>
                      <Activity size={12} /><span>Test ElevenLabs</span>
                    </button>
                    {providerBadge("elevenlabs", providerTests["elevenlabs"])}
                  </div>
                </div>
              </div>
            </div>

            <div style={{ marginTop: 24 }}>
              <SectionLabel>Cloud Fallback & Routing</SectionLabel>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                <FieldRow label="Cloud temperature" hint="0–1">
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input type="range" min="0" max="1" step="0.05" value={draft.cloud_temperature}
                      onChange={e => setDraft({ ...draft, cloud_temperature: Number(e.target.value) })} style={{ flex: 1 }} />
                    <span style={{ fontSize: 13, fontWeight: 700 }}>{draft.cloud_temperature}</span>
                  </div>
                </FieldRow>
                <FieldRow label="Cloud max tokens">
                  <input type="number" min="32" max="2048" step="16" value={draft.cloud_max_tokens}
                    onChange={e => setDraft({ ...draft, cloud_max_tokens: Number(e.target.value) })} />
                </FieldRow>
                <FieldRow label="Cloud timeout (seconds)">
                  <input type="number" min="5" max="120" step="1" value={draft.cloud_timeout_seconds}
                    onChange={e => setDraft({ ...draft, cloud_timeout_seconds: Number(e.target.value) })} />
                </FieldRow>
              </div>
              <Toggle checked={draft.cloud_fallback_to_local} onChange={v => setDraft({ ...draft, cloud_fallback_to_local: v })}
                label="Fallback to local Ollama if cloud fails"
                hint="Automatically uses your local model when OpenAI or Gemini returns an error" />
            </div>

            <div style={{ marginTop: 24 }}>
              <SectionLabel>Bank Instruction — applied before every answer</SectionLabel>
              <FieldRow label="Bank-wide instruction" hint="Injected ahead of the system prompt on every turn">
                <textarea
                  value={draft.bank_instruction ?? ""}
                  onChange={e => setDraft({ ...draft, bank_instruction: e.target.value })}
                  placeholder="e.g. You represent Nabil Bank. Never disclose internal data. Always follow NRB compliance guidelines…"
                  style={{ minHeight: 100, fontSize: 13, lineHeight: 1.6, fontFamily: "inherit" }}
                />
              </FieldRow>
              <p style={{ fontSize: 11, color: "var(--muted)", margin: "0 0 8px" }}>
                This instruction is injected before the system prompt on every chat and voice turn. Use it for compliance rules, tone, product guidance.
              </p>
              <span className="pill info" style={{ fontSize: 10 }}>Applies to text + voice</span>
            </div>

            <div style={{ marginTop: 24 }}>
              <SectionLabel>System Prompt</SectionLabel>
              <FieldRow label="Assistant personality & rules" hint="The AI reads this before every conversation">
                <textarea value={draft.system_prompt} onChange={e => setDraft({ ...draft, system_prompt: e.target.value })}
                  style={{ minHeight: 120, fontSize: 13, lineHeight: 1.6, fontFamily: "inherit" }} />
              </FieldRow>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="rag-toolbar-btn" onClick={() => setDraft({ ...draft, system_prompt: "You are Nabil Voice AI, a helpful Nepali-English voice agent. Reply naturally in the user's language. Keep spoken answers concise." })}>
                  <RefreshCw size={12} /><span>Reset to default</span>
                </button>
                <button className="rag-toolbar-btn" onClick={() => setDraft({ ...draft, system_prompt: "तपाईं Nabil Voice AI हुनुहुन्छ, एक सहायक नेपाली-अंग्रेजी भ्वाइस एजेन्ट। प्रयोगकर्ताको भाषामा स्वाभाविक रूपमा जवाफ दिनुहोस्। छोटो र स्पष्ट राख्नुहोस्।" })}>
                  नेपाली
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── VOICE & TTS TAB ── */}
        {tab === "voice" && (
          <div>
            <SectionLabel>Local Piper Voices</SectionLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 20 }}>
              {voices ? (
                <>
                  <VoiceCard label="Nepali voice" voice={voices.selected.nepali} onTest={() => { Promise.resolve(onTtsTest("ne", voices.selected.nepali.id)).catch(() => {}); }} />
                  <VoiceCard label="English voice" voice={voices.selected.english} onTest={() => { Promise.resolve(onTtsTest("en", voices.selected.english.id)).catch(() => {}); }} />
                </>
              ) : (
                <p style={{ color: "var(--muted)", fontSize: 13 }}>Voice registry unavailable</p>
              )}
            </div>

            <FieldRow label="Nepali voice model path" hint=".onnx file path">
              <input value={String(draft.piper_nepali_voice)} onChange={e => setDraft({ ...draft, piper_nepali_voice: e.target.value })} style={{ fontFamily: "monospace", fontSize: 12 }} />
            </FieldRow>
            <FieldRow label="English voice model path" hint=".onnx file path">
              <input value={String(draft.piper_english_voice)} onChange={e => setDraft({ ...draft, piper_english_voice: e.target.value })} style={{ fontFamily: "monospace", fontSize: 12 }} />
            </FieldRow>

            <div style={{ marginTop: 24 }}>
              <SectionLabel>Voice Behavior</SectionLabel>
              <FieldRow label="Text-to-Speech engine" hint="OpenAI is recommended for Nepali-English voice chat">
                <select value={draft.tts_provider || "openai"} onChange={e => setDraft({ ...draft, tts_provider: e.target.value })}>
                  <option value="openai">OpenAI Cloud</option>
                  <option value="piper">Local Piper</option>
                </select>
              </FieldRow>
              <Toggle checked={draft.single_tts_voice_model} onChange={v => setDraft({ ...draft, single_tts_voice_model: v })}
                label="Use one voice model for both Nepali and English"
                hint="Instead of switching between separate language voices, use one multilingual model" />
              <Toggle checked={draft.force_selected_voice} onChange={v => setDraft({ ...draft, force_selected_voice: v })}
                label="Force selected cloned voice only"
                hint="If a cloned voice can't speak a segment, the turn fails instead of falling back" />
              <Toggle checked={draft.fallback_allowed} onChange={v => setDraft({ ...draft, fallback_allowed: v })}
                label="Allow automatic voice fallback"
                hint="When the selected voice fails, fall back to a built-in voice automatically" />
            </div>

            <div style={{ marginTop: 24 }}>
              <SectionLabel>Chatterbox Hyperparameters</SectionLabel>
              <p style={{ fontSize: 11, color: "var(--muted)", margin: "0 0 12px" }}>
                Fine-tune zero-shot local voice cloning. Increasing exaggeration and guidance can match your voice more closely but may affect stability.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <FieldRow label="Voice Exaggeration" hint="0.0 - 1.0 (Default 0.5)">
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input type="range" min="0" max="1" step="0.05" value={draft.chatterbox_exaggeration ?? 0.5}
                      onChange={e => setDraft({ ...draft, chatterbox_exaggeration: Number(e.target.value) })} style={{ flex: 1 }} />
                    <span style={{ fontSize: 13, fontWeight: 700, minWidth: 32 }}>{draft.chatterbox_exaggeration ?? 0.5}</span>
                  </div>
                </FieldRow>
                <FieldRow label="Guidance (CFG) Weight" hint="0.0 - 1.5 (Default 0.5)">
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input type="range" min="0" max="1.5" step="0.05" value={draft.chatterbox_cfg_weight ?? 0.5}
                      onChange={e => setDraft({ ...draft, chatterbox_cfg_weight: Number(e.target.value) })} style={{ flex: 1 }} />
                    <span style={{ fontSize: 13, fontWeight: 700, minWidth: 32 }}>{draft.chatterbox_cfg_weight ?? 0.5}</span>
                  </div>
                </FieldRow>
                <FieldRow label="Temperature" hint="0.1 - 1.5 (Default 0.8)">
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input type="range" min="0.1" max="1.5" step="0.05" value={draft.chatterbox_temperature ?? 0.8}
                      onChange={e => setDraft({ ...draft, chatterbox_temperature: Number(e.target.value) })} style={{ flex: 1 }} />
                    <span style={{ fontSize: 13, fontWeight: 700, minWidth: 32 }}>{draft.chatterbox_temperature ?? 0.8}</span>
                  </div>
                </FieldRow>
                <FieldRow label="Repetition Penalty" hint="1.0 - 3.0 (Default 1.2)">
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input type="range" min="1" max="3" step="0.1" value={draft.chatterbox_repetition_penalty ?? 1.2}
                      onChange={e => setDraft({ ...draft, chatterbox_repetition_penalty: Number(e.target.value) })} style={{ flex: 1 }} />
                    <span style={{ fontSize: 13, fontWeight: 700, minWidth: 32 }}>{draft.chatterbox_repetition_penalty ?? 1.2}</span>
                  </div>
                </FieldRow>
              </div>
            </div>

            <div style={{ marginTop: 24 }}>
              <SectionLabel>OpenAI Cloud Voices</SectionLabel>
              {voicePreviewError && (
                <div style={{ marginBottom: 12, padding: "8px 12px", background: "rgba(245,101,101,0.1)", borderRadius: 6, fontSize: 12, color: "var(--rose)", display: "flex", alignItems: "center", gap: 8 }}>
                  <CircleAlert size={14} />{voicePreviewError}
                  <button style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer", color: "var(--rose)" }} onClick={() => setVoicePreviewError(null)}>✕</button>
                </div>
              )}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 10 }}>
                {OPENAI_VOICES.map(v => {
                  const isSelected = (draft as any).openai_tts_voice === v.id.replace("openai-", "") || (!((draft as any).openai_tts_voice) && v.id === "openai-alloy");
                  const isLoading = voicePreviewLoading === v.id;
                  return (
                    <div key={v.id} style={{ padding: "14px", borderRadius: 10, border: `2px solid ${isSelected ? "var(--teal)" : "var(--line)"}`, background: isSelected ? "var(--teal-soft)" : "var(--panel)", display: "flex", flexDirection: "column", gap: 8, cursor: "pointer" }}
                      onClick={() => setDraft({ ...draft, openai_tts_voice: v.id.replace("openai-", "") } as any)}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <strong style={{ fontSize: 14 }}>{v.name}</strong>
                        {isSelected && <span style={{ fontSize: 10, color: "var(--teal)", fontWeight: 800 }}>● ACTIVE</span>}
                      </div>
                      <span style={{ fontSize: 11, color: "var(--muted)", flex: 1 }}>{v.desc}</span>
                      <button type="button" style={{ fontSize: 11, padding: "5px 10px", borderRadius: 6, border: "1px solid var(--line)", background: "transparent", cursor: "pointer", display: "flex", alignItems: "center", gap: 4, color: "var(--ink)" }}
                        disabled={!draft.openai_api_key || isLoading} onClick={e => { e.stopPropagation(); handleVoicePreview(v.id); }}>
                        {isLoading ? <RefreshCw size={12} className="spin" /> : <Play size={12} />}
                        {isLoading ? "Loading…" : "Preview"}
                      </button>
                    </div>
                  );
                })}
              </div>
              {!draft.openai_api_key && (
                <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 8 }}>Add an OpenAI API key in the AI Brain tab to preview cloud voices</p>
              )}
            </div>
          </div>
        )}

        {/* ── SPEECH & STT TAB ── */}
        {tab === "speech" && (
          <div>
            <SectionLabel>Speech Recognition (Whisper)</SectionLabel>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginBottom: 20 }}>
              <FieldRow label="Speech-to-Text engine" hint="Local model or Cloud API">
                <select value={draft.stt_provider || "openai"} onChange={e => setDraft({ ...draft, stt_provider: e.target.value })}>
                  <option value="openai">OpenAI Cloud (Whisper API)</option>
                  <option value="local">Local (Faster Whisper)</option>
                </select>
              </FieldRow>
              <FieldRow label="Whisper model size" hint={draft.stt_provider === "openai" ? "Not applicable for OpenAI Cloud" : "Larger = more accurate but slower"}>
                <select value={draft.whisper_model_size} onChange={e => setDraft({ ...draft, whisper_model_size: e.target.value })} disabled={draft.stt_provider === "openai"}>
                  <option value="tiny">tiny — Fastest, lowest accuracy</option>
                  <option value="base">base — Very fast</option>
                  <option value="small">small — Good balance ✓ Recommended</option>
                  <option value="medium">medium — High accuracy</option>
                  <option value="large-v3">large-v3 — Best accuracy (slow)</option>
                  <option value="large-v3-turbo">large-v3-turbo — Best + faster</option>
                </select>
              </FieldRow>
              <FieldRow label="Max recording time" hint="Seconds before auto-stop">
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <input type="range" min="5" max="120" step="5" value={draft.max_recording_seconds}
                    onChange={e => setDraft({ ...draft, max_recording_seconds: Number(e.target.value) })} style={{ flex: 1 }} />
                  <span style={{ fontSize: 13, fontWeight: 700, minWidth: 40 }}>{draft.max_recording_seconds}s</span>
                </div>
              </FieldRow>
            </div>

            <div style={{ padding: "14px 16px", background: "var(--panel)", borderRadius: 10, border: "1px solid var(--line)", marginBottom: 20 }}>
              <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8, color: "var(--teal)" }}>Whisper Model Guide</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, fontSize: 11 }}>
                {[
                  { m: "tiny", ram: "~1GB", speed: "~10x", acc: "Basic" },
                  { m: "small", ram: "~2GB", speed: "~4x", acc: "Good" },
                  { m: "medium", ram: "~5GB", speed: "~2x", acc: "Great" },
                  { m: "large-v3", ram: "~10GB", speed: "1x", acc: "Best" },
                ].map(r => (
                  <div key={r.m} style={{ padding: "8px 10px", background: "var(--surface-2)", borderRadius: 6, border: draft.whisper_model_size === r.m ? "1px solid var(--teal)" : "1px solid transparent" }}>
                    <div style={{ fontWeight: 700 }}>{r.m}</div>
                    <div style={{ color: "var(--muted)", marginTop: 2 }}>RAM: {r.ram}</div>
                    <div style={{ color: "var(--muted)" }}>Speed: {r.speed}</div>
                    <div style={{ color: "var(--green)" }}>Accuracy: {r.acc}</div>
                  </div>
                ))}
              </div>
            </div>

            <SectionLabel>Latency & Quality Mode</SectionLabel>
            <Toggle checked={draft.low_latency_mode} onChange={v => setDraft({ ...draft, low_latency_mode: v })}
              label="Low-latency mode"
              hint="Prioritizes speed — starts speaking before the full response is ready. Best for real-time conversation." />
            <Toggle checked={draft.quality_mode} onChange={v => setDraft({ ...draft, quality_mode: v })}
              label="Quality mode"
              hint="Waits for complete generation before playing audio. Better for longer, nuanced responses." />
          </div>
        )}

        {/* ── SYSTEM TAB ── */}
        {tab === "system" && (
          <div>
            <SectionLabel>Connection</SectionLabel>
            <FieldRow label="Backend URL" hint="Read-only — set via VITE_API_HTTP env var">
              <input value={API_HTTP} disabled style={{ fontFamily: "monospace", fontSize: 12, opacity: 0.7 }} />
            </FieldRow>

            <div style={{ marginTop: 24 }}>
              <SectionLabel>Open WebUI Integration</SectionLabel>
              <div style={{ padding: "12px 14px", background: "rgba(245,101,101,0.08)", border: "1px solid rgba(245,101,101,0.2)", borderRadius: 8, marginBottom: 14, fontSize: 12, color: "var(--muted)" }}>
                Open WebUI is optional — only needed for Open WebUI RAG mode. Local ChromaDB RAG is enabled by default.
              </div>
              <FieldRow label="Open WebUI server URL">
                <input value={draft.open_webui_base_url} onChange={e => setDraft({ ...draft, open_webui_base_url: e.target.value })} style={{ fontFamily: "monospace", fontSize: 12 }} />
              </FieldRow>
              <FieldRow label="Open WebUI API key">
                <input type="password" value={draft.open_webui_api_key} onChange={e => setDraft({ ...draft, open_webui_api_key: e.target.value })} placeholder="Optional" style={{ fontFamily: "monospace", fontSize: 12 }} />
              </FieldRow>
            </div>

            <div style={{ marginTop: 24 }}>
              <SectionLabel>Knowledge & RAG</SectionLabel>
              <Toggle checked={draft.rag_enabled} onChange={v => setDraft({ ...draft, rag_enabled: v })}
                label="Use Open WebUI RAG (external)"
                hint="Routes knowledge queries to your Open WebUI server instead of local ChromaDB" />
              <Toggle checked={draft.rag_fallback_to_ollama} onChange={v => setDraft({ ...draft, rag_fallback_to_ollama: v })}
                label="Fallback to local Ollama when RAG fails"
                hint="If the RAG service is unavailable, answers directly from the LLM" />
              <FieldRow label="Default knowledge collection" hint="Collection ID used when no collection is specified">
                <input value={draft.rag_default_collection} onChange={e => setDraft({ ...draft, rag_default_collection: e.target.value })} placeholder="Leave blank to use none by default" />
              </FieldRow>
            </div>

            <div style={{ marginTop: 24 }}>
              <SectionLabel>Internet Search</SectionLabel>
              <Toggle checked={draft.internet_retrieval_enabled} onChange={v => setDraft({ ...draft, internet_retrieval_enabled: v })}
                label="Enable internet retrieval"
                hint="Searches the web when local knowledge can't answer the question" />
              <Toggle checked={draft.internet_require_citation} onChange={v => setDraft({ ...draft, internet_require_citation: v })}
                label="Require citations for internet answers"
                hint="Every answer from the internet must include the source URL" />
              <FieldRow label="Max internet sources" hint="How many web results to fetch per query">
                <input type="number" min="1" max="10" value={draft.internet_max_sources}
                  onChange={e => setDraft({ ...draft, internet_max_sources: Number(e.target.value) })} style={{ maxWidth: 100 }} />
              </FieldRow>
            </div>
          </div>
        )}

        {/* ── ADVANCED TAB ── */}
        {tab === "advanced" && (
          <div>
            <SectionLabel>Piper Train Command</SectionLabel>
            <FieldRow label="Custom piper train command" hint="Used when cloning voice with piper engine">
              <input value={draft.piper_train_command} onChange={e => setDraft({ ...draft, piper_train_command: e.target.value })} placeholder="Leave blank for default" style={{ fontFamily: "monospace", fontSize: 12 }} />
            </FieldRow>

            <div style={{ marginTop: 20 }}>
              <SectionLabel>ffmpeg</SectionLabel>
              <FieldRow label="ffmpeg binary path" hint="Leave blank to auto-detect from PATH">
                <input value={draft.ffmpeg_binary ?? ""} onChange={e => setDraft({ ...draft, ffmpeg_binary: e.target.value } as any)} placeholder="Auto-detect" style={{ fontFamily: "monospace", fontSize: 12 }} />
              </FieldRow>
            </div>

            <div style={{ marginTop: 20 }}>
              <SectionLabel>Danger Zone</SectionLabel>
              <div style={{ padding: "16px", background: "rgba(245,101,101,0.06)", border: "1px solid rgba(245,101,101,0.25)", borderRadius: 10 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Delete all local data</div>
                <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
                  Removes voice recordings, audio cache, and local settings. This action cannot be undone.
                </div>
                <button className="rag-toolbar-btn" onClick={onDelete} style={{ color: "var(--rose)", borderColor: "var(--rose)" }}>
                  <Trash2 size={13} /><span>Delete local data</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sticky save bar */}
      <div style={{ position: "sticky", bottom: 0, background: "var(--bg)", borderTop: "1px solid var(--line)", padding: "12px 24px", display: "flex", justifyContent: "flex-end", gap: 10, zIndex: 10 }}>
        {saveOk && <span style={{ fontSize: 12, color: "var(--green)", fontWeight: 700, display: "flex", alignItems: "center", gap: 4 }}><CheckCircle2 size={14} /> All changes saved</span>}
        <button className="rag-toolbar-btn" onClick={() => setDraft(settings)} disabled={saving}>Reset</button>
        <button className="rag-toolbar-btn primary" onClick={handleSave} disabled={saving}>
          <Save size={13} /><span>{saving ? "Saving…" : "Save changes"}</span>
        </button>
      </div>
    </section>
  );
}


function VoiceCard({ label, voice, onTest }: { label: string; voice: VoicesResponse["selected"]["nepali"]; onTest: () => void }) {
  const ready = voice.status === "ready";
  return (
    <article className={ready ? "voice-card ready" : "voice-card missing"}>
      <div className="panel-heading">
        <h3>{label}</h3>
        <span className={ready ? "pill good" : "pill"}>{ready ? "Ready" : "Missing files"}</span>
      </div>
      <code>{voice.model_path}</code>
      {voice.missing_files.length ? (
        <div className="missing-list">
          {voice.missing_files.map((file) => (
            <span key={file}>{file}</span>
          ))}
        </div>
      ) : null}
      <p className="muted">{voice.license_path ? `License: ${voice.license_path}` : "License/model card not detected. Verify license before use."}</p>
      <button className="icon-text" disabled={!ready} onClick={onTest} type="button">
        <Play size={18} />
        <span>Test synth</span>
      </button>
    </article>
  );
}

function StatusDot({ status }: { status: AssistantStatus }) {
  return <span className={`status-dot ${status}`} aria-hidden="true" />;
}

function LanguageBadge({ language }: { language: LanguageCode }) {
  return <span className={`language-badge ${language}`}>{languageLabels[language]}</span>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatMs(value?: number | null): string {
  if (value == null) {
    return "--";
  }
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)}s`;
}

function latencyColor(value?: number | null): string {
  if (value == null) return "var(--muted)";
  if (value <= 3000) return "var(--green)";
  if (value <= 8000) return "var(--amber)";
  return "var(--rose)";
}

function useLocalStorage<T>(key: string, initialValue: T): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => {
    const existing = window.localStorage.getItem(key);
    return existing ? (JSON.parse(existing) as T) : initialValue;
  });
  useEffect(() => {
    window.localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);
  return [value, setValue];
}

export default App;
