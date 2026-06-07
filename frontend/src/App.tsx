import {
  Activity,
  CheckCircle2,
  CircleAlert,
  Database,
  Download,
  Mic,
  Play,
  Radio,
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
  Wrench
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
  getAuditLogs,
  getRagStatus,
  getSystemInfo,
  getSystemMetrics,
  downloadDefaultVoices,
  getRagCollections,
  getDatasetRecordings,
  uploadDatasetRecording,
  deleteDatasetRecording,
  deriveDatasetExportUrl,
  testProvider,
  testRagQuestion,
  deleteOpenAIKey,
  deleteGeminiKey,
  deleteElevenLabsKey
} from "./api";
import { blobToBase64, downloadBlob, scoreRecording } from "./audio";
import type {
  AssistantStatus,
  BackendSettings,
  ConversationTurn,
  DatasetRecording,
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
  { id: "knowledge", label: "Knowledge", icon: BookOpen },
  { id: "setup", label: "Setup", icon: CheckCircle2 },
  { id: "evaluation", label: "Eval", icon: Activity },
  { id: "admin", label: "Admin", icon: Wrench },
  { id: "logs", label: "Logs", icon: FileText },
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
    "You are a helpful bilingual Nepali-English voice assistant. Reply naturally, concisely, and in the same language the user used unless they ask otherwise. For mixed Nepali-English input, answer naturally in mixed Nepali-English.",
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
  llm_provider: "local",
  local_model: "qwen2.5:7b",
  local_fallback_model: "gemma3:4b",
  openai_api_key: "",
  openai_model: "gpt-4o-mini",
  gemini_api_key: "",
  gemini_model: "gemini-1.5-flash",
  elevenlabs_api_key: "",
  cloud_fallback_to_local: true,
  cloud_timeout_seconds: 30.0,
  cloud_temperature: 0.35,
  cloud_max_tokens: 180,
  force_selected_voice: false,
  fallback_allowed: true,
  single_tts_voice_model: true
};

function App() {
  const [activeView, setActiveView] = useState<ViewId>("conversation");
  const [status, setStatus] = useState<AssistantStatus>("idle");
  const [autoVad, setAutoVad] = useLocalStorage("swarlocal.autoVad", true);
  const [history, setHistory] = useLocalStorage<ConversationTurn[]>("swarlocal.history", []);
  const [settings, setSettings] = useState<BackendSettings>(defaultSettings);
  const [providerStatus, setProviderStatus] = useState<ProviderStatus[]>([]);
  const [voiceSocketStatus, setVoiceSocketStatus] = useState<VoiceSocketStatus | null>(null);
  const [voiceSocketState, setVoiceSocketState] = useState<VoiceSocketConnectionState>("untested");
  const [voices, setVoices] = useState<VoicesResponse | null>(null);
  const [manualText, setManualText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [lastAudioUrl, setLastAudioUrl] = useState<string | null>(null);
  const [datasetRecordings, setDatasetRecordings] = useState<DatasetRecording[]>([]);
  const [datasetActivePrompt, setDatasetActivePrompt] = useState<string | null>(null);
  const [ratings, setRatings] = useLocalStorage<Record<string, number>>("swarlocal.eval", {
    naturalness: 3,
    voiceSimilarity: 3,
    nepaliPronunciation: 3,
    englishPronunciation: 3
  });

  const [selectedVoiceId, setSelectedVoiceId] = useLocalStorage("swarlocal.selectedVoiceId", "auto");
  const [selectedKnowledgeId, setSelectedKnowledgeId] = useLocalStorage("swarlocal.selectedKnowledgeId", "none");
  const [useInternet, setUseInternet] = useLocalStorage("swarlocal.useInternet", false);
  const [selectedBrain, setSelectedBrain] = useLocalStorage("swarlocal.selectedBrain", "local");
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
        ragStatusPayload
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
        getRagStatus().catch(() => null)
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
      setHistory((items) => [stamped, ...items].slice(0, 80));
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
        return "backend unavailable: SwarLocal FastAPI server is not running or unreachable.";
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
      return "backend unavailable: Cannot connect to SwarLocal local server.";
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
            selected_brain: selectedBrain
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
  }, [handleTurn, selectedVoiceId, selectedKnowledgeId, useInternet, selectedBrain]);

  useEffect(() => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.close();
    }
  }, [selectedVoiceId, selectedKnowledgeId, useInternet, selectedBrain]);

  useEffect(() => {
    if (!voices || selectedVoiceId === "auto") {
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
    try {
      const socket = await ensureSocket();
      socket.send(JSON.stringify({ type: "text", text }));
    } catch {
      try {
        await handleTurn(await sendTextTurn(text, {
          voice_id: selectedVoiceId === "auto" ? undefined : selectedVoiceId,
          knowledge_id: selectedKnowledgeId === "none" ? undefined : selectedKnowledgeId,
          use_internet: useInternet,
          llm_provider_id: selectedBrain,
        }));
      } catch (chatError) {
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
      const socketStatus = voiceSocketStatus ?? (await getVoiceSocketStatus());
      setVoiceSocketStatus(socketStatus);
      if (!socketStatus.capabilities.audio_turns) {
        setVoiceSocketState("setup_required");
        setStatus("error");
        setError(`Setup required before recording: ${socketStatus.blocking_reasons.join(", ")}`);
        logEvent("warning", "recording_blocked", `Setup required before recording: ${socketStatus.blocking_reasons.join(", ")}`);
        return;
      }
      await ensureSocket();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
      });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm";
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
      recorder.start();
      setStatus("listening");
      logEvent("info", "recording_started", "Microphone recording started.");
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
    setStatus("thinking");
    try {
      const socket = await ensureSocket();
      logEvent("info", "audio_sent", "Audio sent to the voice pipeline.", { bytes: blob.size, mimeType });
      socket.send(
        JSON.stringify({
          type: "audio",
          mimeType,
          audioBase64: await blobToBase64(blob)
        })
      );
    } catch (audioError) {
      setStatus("error");
      setError(audioError instanceof Error ? audioError.message : "Audio turn failed.");
      logEvent("error", "audio_send_failed", audioError instanceof Error ? audioError.message : "Audio turn failed.");
    }
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

  async function runTtsTest(language: "ne" | "en") {
    setError(null);
    const text = language === "ne" ? "नमस्ते, Piper आवाज परीक्षण सफल भयो।" : "Hello, Piper voice test was successful.";
    try {
      const result = await testTts(text, language);
      const audioUrl = absoluteAudioUrl(result.audio_url);
      setLastAudioUrl(audioUrl);
      if (audioUrl) {
        await new Audio(audioUrl).play();
      }
    } catch (ttsError) {
      setError(ttsError instanceof Error ? ttsError.message : "TTS test failed.");
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
      setSelectedBrain("local");
      setAutoVad(false);
      void saveSettings({ ...settings, low_latency_mode: true, quality_mode: false });
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
        return <EvaluationView ratings={ratings} onChange={setRatings} history={history} />;
      case "admin":
        return (
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
            systemMetrics={systemMetrics}
            onApplyPreset={applyPreset}
            showSettings={showSettings}
            onToggleSettings={() => setShowSettings((v) => !v)}
            playingAudioUrl={playingAudioUrl}
            onPlayAudio={handlePlayTurnAudio}
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
    ratings,
    refreshStatus,
    settings,
    setAutoVad,
    setHistory,
    setRatings,
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
    ragCollections,
    ragStatus,
    galleryVoices,
    auditLogs,
    systemInfo,
    appLogs,
    setAppLogs
  ]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">स्व</div>
          <div>
            <h1>SwarLocal</h1>
            <span>macOS voice AI</span>
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
  systemMetrics,
  onApplyPreset,
  showSettings,
  onToggleSettings,
  playingAudioUrl,
  onPlayAudio
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
  systemMetrics: SystemMetrics | null;
  onApplyPreset: (preset: "fast" | "quality" | "noisy" | "cloned" | "local") => void;
  showSettings: boolean;
  onToggleSettings: () => void;
  playingAudioUrl: string | null;
  onPlayAudio: (url: string | null | undefined) => void;
}) {
  const recording = status === "listening";
  const socketBlocked = voiceSocketStatus ? !voiceSocketStatus.capabilities.audio_turns : false;
  const builtInVoiceIds = ["ne_NP-chitwan-medium", "ne_NP-google-medium", "en_US-lessac-medium", "en_US-ryan-medium"];
  const customVoices = voices?.voices?.filter((voice) => !builtInVoiceIds.includes(voice.id) && !voice.id.startsWith("openai-")) ?? [];
  const openaiVoices = voices?.voices?.filter((voice) => voice.id.startsWith("openai-")) ?? [];
  const selectedVoice = voices?.voices?.find((voice) => voice.id === selectedVoiceId);
  const selectedKnowledge = ragCollections.find((collection) => collection.id === selectedKnowledgeId);
  const activeVoiceName = latestTurn?.actual_voice_name || 
    (selectedVoiceId.startsWith("openai-") ? `OpenAI ${selectedVoiceId.split("-")[1].charAt(0).toUpperCase() + selectedVoiceId.split("-")[1].slice(1)}` : "") ||
    selectedVoice?.name || 
    selectedVoice?.id || 
    "Auto bilingual";
  const orbStateLabel = status === "listening" && autoVad ? "Listening for your voice" : statusLabels[status];
  return (
    <section className={`voice-layout ${showSettings ? "" : "no-settings"}`}>
      {showSettings && (
        <div className="voice-control voice-left">
        <div className={socketBlocked ? "socket-summary blocked" : "socket-summary"}>
          <span>Voice Socket</span>
          <strong>{voiceSocketState.replaceAll("_", " ")}</strong>
          {voiceSocketStatus?.blocking_reasons.length ? (
            <small>{voiceSocketStatus.blocking_reasons.join(", ")}</small>
          ) : (
            <small>{voiceSocketStatus ? "Ready for local voice turns" : "Not tested yet"}</small>
          )}
        </div>
        
        <div className="preset-grid" aria-label="Voice presets">
          <button type="button" onClick={() => onApplyPreset("fast")} title="Fast response: uses Local AI, turns off internet search, and prefers lower latency. Example: quick daily voice chat.">
            <Sparkles size={16} />
            <span>Fast response</span>
          </button>
          <button type="button" onClick={() => onApplyPreset("quality")} title="Best quality: keeps more processing time for clearer voice output. Example: recording a polished answer.">
            <Gauge size={16} />
            <span>Best quality</span>
          </button>
          <button type="button" onClick={() => onApplyPreset("noisy")} title="Noisy room: prefers Push to Talk so fan or street noise does not keep the assistant waiting.">
            <Activity size={16} />
            <span>Noisy room</span>
          </button>
          <button type="button" onClick={() => onApplyPreset("cloned")} title="My cloned voice only: blocks fallback if the selected cloned voice cannot be used. Example: commercial voice QA.">
            <ShieldCheck size={16} />
            <span>My cloned voice only</span>
          </button>
          <button type="button" onClick={() => onApplyPreset("local")} title="Local only: keeps reasoning on this computer and disables internet search for private conversations.">
            <Cpu size={16} />
            <span>Local only</span>
          </button>
        </div>

        <div className="dropdown-control">
          <label title="Local AI is private and runs on your computer. OpenAI/Gemini are cloud reasoning options and send text only when selected.">Brain</label>
          <select value={selectedBrain} onChange={(e) => onSelectBrain(e.target.value)}>
            <option value="local">Local AI - private on this Mac</option>
            <option value="openai">OpenAI - cloud reasoning</option>
            <option value="gemini">Gemini - cloud reasoning</option>
            <option value="auto">Auto - choose best available</option>
          </select>
          <small>Local AI is private. Cloud brains send text, not voice audio.</small>
        </div>

        <div className="dropdown-control">
          <label title="Choose how SwarLocal speaks. Custom voices show owner, language, quality score, and engine when available.">Assistant voice</label>
          <select value={selectedVoiceId} onChange={(e) => onSelectVoice(e.target.value)}>
            <optgroup label="Recommended">
              <option value="auto">Auto - natural bilingual routing</option>
            </optgroup>
            {customVoices.length ? (
              <optgroup label="My cloned voices">
                {customVoices.map((voice) => (
                  <option key={voice.id} value={voice.id} disabled={!voice.model_exists}>
                    {voice.name || voice.id} - {voice.owner_name || "Owner unknown"} - {voice.language} - quality {Math.round(voice.quality_score ?? 0)} - {voice.engine || "piper"}{voice.model_exists ? "" : " - not trained yet"}
                  </option>
                ))}
              </optgroup>
            ) : null}
            <optgroup label="Built-in voices">
              <option value="ne_NP-chitwan-medium">Chitwan Nepali - Piper - ready</option>
              <option value="ne_NP-google-medium">Google Nepali - Piper - optional</option>
              <option value="en_US-lessac-medium">Lessac English - Piper - ready</option>
              <option value="en_US-ryan-medium">Ryan English - Piper - optional</option>
            </optgroup>
            {openaiVoices.length ? (
              <optgroup label="OpenAI Cloud voices">
                {openaiVoices.map((voice) => (
                  <option key={voice.id} value={voice.id} disabled={voice.disabled_reason ? true : false} title={voice.disabled_reason || undefined}>
                    {voice.name || voice.id} - OpenAI Cloud{voice.disabled_reason ? " - API key missing" : " - ready"}
                  </option>
                ))}
              </optgroup>
            ) : null}
            <optgroup label="Experimental voices">
              <option value="experimental-disabled" disabled title="Experimental engines appear here after an admin imports an engine artifact.">
                Add F5-TTS, Chatterbox, OpenVoice, or VoxCPM in Admin
              </option>
            </optgroup>
          </select>
          <div className="selector-meta">
            <span className="pill good">Speaking as {activeVoiceName}</span>
            {selectedVoice?.quality_score ? <span className="pill">Quality {Math.round(selectedVoice.quality_score)}</span> : null}
            {selectedVoice?.disabled_reason ? <span className="pill warn" title={selectedVoice.disabled_reason}>This voice is not ready yet</span> : null}
            <button type="button" className="link-button" onClick={() => onReplay()} disabled={!lastAudioUrl} title={lastAudioUrl ? "Replay the last generated answer with the current audio player." : "No generated answer yet. Send a message first."}>
              Test this voice
            </button>
          </div>
        </div>

        <div className="dropdown-control">
          <label title="Knowledge helps the assistant answer from uploaded documents managed in Open WebUI. Example: company handbook or product FAQ.">Knowledge</label>
          <select value={selectedKnowledgeId} onChange={(e) => onSelectKnowledge(e.target.value)}>
            <option value="none">No Knowledge - normal conversation</option>
            {ragCollections.map(col => (
              <option key={col.id} value={col.id}>
                {col.name || col.id} - Open WebUI
              </option>
            ))}
          </select>
          <small>{ragCollections.length ? "Choose a document collection for grounded answers." : "Open WebUI API key is needed to list collections."}</small>
        </div>

        <div className="toggle-control">
          <label className="checkbox-label" title="Internet search is off unless you enable it here or ask a current/latest question. Example: recent news or weather.">
            <input type="checkbox" checked={useInternet} onChange={onToggleInternet} />
            <Globe size={16} />
            <span>Use internet when needed</span>
          </label>
        </div>
        </div>
      )}
      <div className="voice-stage">
        <div className="voice-badge-row" style={{ display: 'flex', width: '100%', alignItems: 'center', gap: '8px' }}>
          <span className="pill good">Speaking as {activeVoiceName}</span>
          <span className="pill">Brain: {selectedBrain === "local" ? "Local AI" : selectedBrain === "openai" ? "OpenAI" : selectedBrain === "gemini" ? "Gemini" : "Auto"}</span>
          <span className="pill">Knowledge: {selectedKnowledge?.name || (selectedKnowledgeId === "none" ? "Off" : selectedKnowledgeId)}</span>
          {latestTurn?.internet_used ? <span className="pill good">Internet used</span> : null}
          {latestTurn?.fallback_used ? <span className="pill warn" title={latestTurn.fallback_reason ?? undefined}>Using fallback voice</span> : null}
          <button 
            type="button" 
            className={`settings-toggle-btn ${showSettings ? "active" : ""}`}
            onClick={onToggleSettings}
            title={showSettings ? "Hide settings panel" : "Show settings panel"}
            style={{ 
              marginLeft: 'auto', 
              background: showSettings ? 'rgba(79,209,197,0.15)' : 'rgba(255,255,255,0.08)', 
              border: showSettings ? '1px solid #4fd1c5' : '1px solid rgba(255,255,255,0.15)',
              color: showSettings ? '#4fd1c5' : '#e2e8f0',
              padding: '4px 10px',
              borderRadius: '6px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.75rem',
              fontWeight: 500,
              transition: 'all 0.2s ease-in-out'
            }}
          >
            <SlidersHorizontal size={12} />
            <span>{showSettings ? "Hide Settings" : "Configure Settings"}</span>
          </button>
        </div>

        <div className={`orb ${status}`} data-orb-state={orbStateLabel}>
          <div className="orb-ring" aria-hidden="true" />
          <div className="orb-waveform" aria-hidden="true">
            <span /><span /><span /><span /><span /><span /><span />
          </div>
          <button
            className="mic-button"
            onClick={recording ? onStop : onRecord}
            disabled={!recording && socketBlocked}
            type="button"
            title={socketBlocked ? `Setup required before recording: ${voiceSocketStatus?.blocking_reasons.join(", ") || "audio route unavailable"}` : recording ? "Stop recording" : "Push to Talk"}
          >
            {recording ? <Square size={42} /> : <Mic size={46} />}
          </button>
        </div>

        <div className="status-row">
          <StatusDot status={status} />
          <strong>{orbStateLabel}</strong>
          <LanguageBadge language={latestTurn?.input_language ?? "unknown"} />
          {latestTurn?.llm_provider && (
            <span className="pill good" style={{ marginLeft: '4px' }}>
              Brain: {latestTurn.llm_provider === 'local' ? 'Local AI' : latestTurn.llm_provider === 'openai' ? 'OpenAI' : 'Gemini'}
            </span>
          )}
          {latestTurn?.rag_used && <span className="pill good" style={{ marginLeft: '4px' }}>RAG</span>}
          {latestTurn?.internet_used && <span className="pill good" style={{ marginLeft: '4px' }}>Internet</span>}
          {latestTurn?.rag_fallback_used && <span className="pill" style={{ marginLeft: '4px' }}>Fallback</span>}
          {latestTurn?.fallback_reason && (
            <span className="pill warn" style={{ marginLeft: '4px' }} title={latestTurn.fallback_reason}>
              Fallback Active
            </span>
          )}
        </div>

        <div className="mode-row">
          <button className={autoVad ? "toggle on" : "toggle"} onClick={onToggleVad} type="button">
            <Activity size={16} />
            <span>{autoVad ? "Auto Detect" : "Push to Talk"}</span>
          </button>
          <button className="toggle" onClick={recording ? onStop : onRecord} disabled={!recording && socketBlocked} type="button" title={socketBlocked ? "Fix setup blockers before recording." : recording ? "Stop recording now." : "Hold the conversation by recording one turn."}>
            {recording ? <Square size={16} /> : <Mic size={16} />}
            <span>{recording ? "Stop" : "Push to Talk"}</span>
          </button>
          <button className="toggle" onClick={onInterrupt} disabled={status !== "speaking"} type="button" title={status === "speaking" ? "Stop the current spoken answer." : "Nothing is speaking right now."}>
            <Square size={16} />
            <span>Interrupt</span>
          </button>
          <button className="toggle" onClick={onReplay} disabled={!lastAudioUrl} type="button" title={lastAudioUrl ? "Replay the last spoken answer." : "No answer has been generated yet."}>
            <Play size={16} />
            <span>Replay</span>
          </button>
          <span className={settings.low_latency_mode ? "pill good" : "pill"}>Low latency</span>
          <span className={settings.quality_mode ? "pill good" : "pill"}>Quality</span>
        </div>

        <div className="live-hint" role="status">
          {recording ? "Waiting for you to finish..." : status === "transcribing" ? "Turning your voice into text..." : status === "thinking" ? (selectedKnowledgeId !== "none" ? "Searching knowledge..." : "Thinking...") : status === "speaking" ? "Speaking now..." : socketBlocked ? "Too noisy or setup incomplete. Try Push to Talk after setup is fixed." : "Ready when you are."}
        </div>

        {latestTurn?.tts_route?.length ? (
          <div className="route-row">
            <span>TTS route</span>
            {latestTurn.tts_route.map((part, index) => (
              <LanguageBadge language={part.language} key={`${part.language}-${index}`} />
            ))}
          </div>
        ) : null}
        {latestTurn ? (
          <div className="route-row" title={latestTurn.actual_model_path ?? undefined}>
            <span>Voice</span>
            <span className={latestTurn.fallback_used ? "pill" : "pill good"}>
              {latestTurn.actual_voice_name || latestTurn.actual_voice_id || "auto"}
            </span>
            <span className="pill">{latestTurn.actual_engine || "piper"}</span>
            {latestTurn.requested_voice_id && latestTurn.requested_voice_id !== latestTurn.actual_voice_id ? (
              <span className="pill">requested {latestTurn.requested_voice_name || latestTurn.requested_voice_id}</span>
            ) : null}
          </div>
        ) : null}
        {systemMetrics ? (
          <div className="mini-metrics" title={systemMetrics.recommendation ?? (systemMetrics.recommendations as string[] | undefined)?.join(" ")}>
            <Metric label="CPU" value={typeof systemMetrics.cpu_percent === "number" ? `${systemMetrics.cpu_percent.toFixed(0)}%` : "..."} />
            <Metric label="RAM" value={typeof systemMetrics.ram_used_gb === "number" ? `${systemMetrics.ram_used_gb.toFixed(1)} GB` : "..."} />
            <Metric label="Disk" value={typeof systemMetrics.disk_free_gb === "number" ? `${systemMetrics.disk_free_gb.toFixed(1)} GB` : "..."} />
          </div>
        ) : null}
        {latestTurn ? (
          <div className="mini-metrics">
            <Metric label="LLM" value={formatMs(latestTurn.timings.llm_total_ms ?? latestTurn.timings.llm_first_token_ms)} />
            <Metric label="TTS" value={formatMs(latestTurn.timings.tts_generation_ms)} />
            <Metric label="Total" value={formatMs(latestTurn.timings.total_turn_ms)} />
          </div>
        ) : null}
        <form
          className="text-turn"
          onSubmit={(event) => {
            event.preventDefault();
            onSendText();
          }}
        >
          <input value={manualText} onChange={(event) => onManualText(event.target.value)} placeholder="Type a test turn" />
          <button type="submit" title="Send">
            <Send size={18} />
          </button>
        </form>
        {lastAudioUrl ? (
          <audio className="audio-player" controls src={lastAudioUrl}>
            <track kind="captions" />
          </audio>
        ) : null}
      </div>

      <div className="conversation-panels">
        <section className="panel chat-panel">
          <div className="panel-heading">
            <h2>Conversation</h2>
            <span className="muted">{history.length ? `${history.length} turn${history.length === 1 ? "" : "s"}` : "No messages yet"}</span>
          </div>
          {!history.length ? (
            <div className="empty-state">
              <Radio size={24} />
              <strong>Start a voice conversation</strong>
              <span>Press Push to Talk or type a quick test message. I’ll answer in Nepali, English, or natural mixed language.</span>
            </div>
          ) : (
            <div className="chat-bubbles">
              {history.map((turn, index) => (
                <article className="chat-turn" key={`${turn.created_at}-${index}`}>
                  <div className="bubble user-bubble">
                    <div className="bubble-meta" style={{ display: 'flex', width: '100%', alignItems: 'center', gap: '6px' }}>
                      <span>You</span>
                      <LanguageBadge language={turn.input_language} />
                      
                      {turn.user_audio_url && (
                        <button 
                          type="button" 
                          className={`play-turn-btn ${playingAudioUrl === turn.user_audio_url ? "playing" : ""}`}
                          onClick={() => onPlayAudio(turn.user_audio_url)}
                          title={playingAudioUrl === turn.user_audio_url ? "Pause audio" : "Play your recording"}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: playingAudioUrl === turn.user_audio_url ? '#4fd1c5' : '#a0aec0',
                            cursor: 'pointer',
                            marginLeft: 'auto',
                            padding: '4px',
                            display: 'flex',
                            alignItems: 'center',
                            borderRadius: '4px'
                          }}
                        >
                          {playingAudioUrl === turn.user_audio_url ? <Square size={14} fill="currentColor" /> : <Play size={14} fill="currentColor" />}
                        </button>
                      )}
                    </div>
                    <p>{turn.transcript}</p>
                  </div>
                  <div className="bubble assistant-bubble">
                    <div className="bubble-meta" style={{ display: 'flex', width: '100%', alignItems: 'center', gap: '6px' }}>
                      <span>SwarLocal</span>
                      <LanguageBadge language={turn.response_language} />
                      {turn.llm_provider ? <span className="pill">Brain: {turn.llm_provider}</span> : null}
                      {turn.rag_used ? <span className="pill good">Knowledge</span> : null}
                      {turn.internet_used ? <span className="pill good">Internet</span> : null}
                      
                      {turn.audio_url && (
                        <button 
                          type="button" 
                          className={`play-turn-btn ${playingAudioUrl === absoluteAudioUrl(turn.audio_url) ? "playing" : ""}`}
                          onClick={() => onPlayAudio(turn.audio_url)}
                          title={playingAudioUrl === absoluteAudioUrl(turn.audio_url) ? "Pause audio" : "Play generated audio"}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: playingAudioUrl === absoluteAudioUrl(turn.audio_url) ? '#4fd1c5' : '#a0aec0',
                            cursor: 'pointer',
                            marginLeft: 'auto',
                            padding: '4px',
                            display: 'flex',
                            alignItems: 'center',
                            borderRadius: '4px'
                          }}
                        >
                          {playingAudioUrl === absoluteAudioUrl(turn.audio_url) ? <Square size={14} fill="currentColor" /> : <Play size={14} fill="currentColor" />}
                        </button>
                      )}
                    </div>
                    <p>{turn.response}</p>
                    
                    <div className="turn-metadata-footer" style={{
                      marginTop: '8px',
                      paddingTop: '6px',
                      borderTop: '1px solid rgba(255,255,255,0.08)',
                      fontSize: '0.7rem',
                      color: '#718096',
                      display: 'flex',
                      flexWrap: 'wrap',
                      gap: '8px',
                      alignItems: 'center'
                    }}>
                      {turn.actual_voice_name && (
                        <span title={`Actual voice: ${turn.actual_voice_id}`}>
                          🗣️ {turn.actual_voice_name}
                        </span>
                      )}
                      {turn.timings && (
                        <span>
                          ⏱️ LLM: {formatMs(turn.timings.llm_total_ms ?? turn.timings.llm_first_token_ms)} | TTS: {formatMs(turn.timings.tts_generation_ms)}
                        </span>
                      )}
                      {turn.rag_path && (
                        <span title={`RAG pathway: ${turn.rag_collection_id || 'default'}`}>
                          📚 {turn.rag_path}
                        </span>
                      )}
                      {turn.fallback_used && (
                        <span className="pill warn" style={{ fontSize: '0.65rem', padding: '1px 4px' }} title={turn.fallback_reason || "Using fallback voice"}>
                          Fallback Active
                        </span>
                      )}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
        <section className="panel live-panel">
          <div className="panel-heading">
            <h2>Live transcript</h2>
            <LanguageBadge language={latestTurn?.input_language ?? "unknown"} />
          </div>
          <p>{recording ? "Listening..." : latestTurn?.transcript ?? "Your words will appear here while we process the turn."}</p>
        </section>
        {latestTurn?.citations?.length ? (
          <section className="panel citations-panel">
            <div className="panel-heading">
              <h2 style={{ fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Globe size={16} color="#63b3ed" />
                <span>Internet Citations ({latestTurn.citations.length})</span>
              </h2>
            </div>
            <div className="citations-list">
              {latestTurn.citations.map((cite, i) => (
                <div key={i} className="citation-item">
                  <a href={cite.url} target="_blank" rel="noreferrer">
                    {cite.title}
                  </a>
                  <span className="muted">{cite.url}</span>
                  <p className="muted">{cite.snippet}</p>
                </div>
              ))}
            </div>
          </section>
        ) : null}
        <section className="timeline">
          <div className="panel-heading">
            <h2>Latency</h2>
            <span className="muted">latest turn</span>
          </div>
          <div className="latency-grid">
            <Metric label="STT" value={formatMs(latestTurn?.timings.audio_received_to_transcript_ms)} />
            <Metric label="RAG" value={latestTurn?.rag_used ? "Used" : "Off"} />
            <Metric label="LLM" value={formatMs(latestTurn?.timings.llm_total_ms ?? latestTurn?.timings.llm_first_token_ms)} />
            <Metric label="TTS" value={formatMs(latestTurn?.timings.tts_generation_ms)} />
            <Metric label="Total" value={formatMs(latestTurn?.timings.total_turn_ms)} />
          </div>
        </section>
      </div>
    </section>
  );
}

function KnowledgeView({
  settings,
  ragStatus,
  ragCollections,
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
  const [question, setQuestion] = useState("What should the assistant know from my documents?");
  const [testResult, setTestResult] = useState<any | null>(null);
  const [testing, setTesting] = useState(false);

  async function runKnowledgeTest() {
    setTesting(true);
    try {
      setTestResult(await testRagQuestion(question, selectedKnowledgeId === "none" ? undefined : selectedKnowledgeId));
    } catch (error: any) {
      setTestResult({ ok: false, detail: error.message || "Knowledge test failed." });
    } finally {
      setTesting(false);
    }
  }

  return (
    <section className="view-stack">
      <div className="view-header">
        <div>
          <h2>Knowledge</h2>
          <p>Upload documents in Open WebUI, then choose them here for voice answers.</p>
        </div>
        <button className="icon-text" type="button" onClick={() => window.open(settings.open_webui_base_url, "_blank")}>
          <BookOpen size={18} />
          <span>Open WebUI</span>
        </button>
      </div>

      <div className="knowledge-hero">
        <article className={ragStatus?.api_key_configured ? "status-card" : "status-card blocked"}>
          {ragStatus?.api_key_configured ? <CheckCircle2 size={20} /> : <KeyRound size={20} />}
          <div>
            <strong>{ragStatus?.api_key_configured ? "Knowledge is connected" : "Open WebUI API key needed"}</strong>
            <span>{ragStatus?.api_key_configured ? `${ragCollections.length} collection(s) available.` : "Create an API key in Open WebUI, then paste it in Settings."}</span>
            <code>{settings.open_webui_base_url}</code>
          </div>
        </article>
        <article className="status-card">
          <BookOpen size={20} />
          <div>
            <strong>How knowledge works</strong>
            <span>Knowledge helps the assistant answer from uploaded documents, such as a handbook, FAQ, policy, or product guide.</span>
          </div>
        </article>
      </div>

      <section className="panel">
        <div className="panel-heading">
          <h2>Collections</h2>
          <span className="muted">{ragCollections.length ? "Available from Open WebUI" : "No collections visible yet"}</span>
        </div>
        <div className="settings-grid no-shadow">
          <label className="wide">
            <span>Default knowledge source</span>
            <select value={selectedKnowledgeId} onChange={(event) => onSelectKnowledge(event.target.value)}>
              <option value="none">No Knowledge</option>
              {ragCollections.map((collection) => (
                <option value={collection.id} key={collection.id}>{collection.name || collection.id}</option>
              ))}
            </select>
          </label>
          <label className="wide">
            <span>Save as backend default</span>
            <button className="icon-text" type="button" onClick={() => onSave({ ...settings, rag_default_collection: selectedKnowledgeId === "none" ? "" : selectedKnowledgeId, rag_enabled: selectedKnowledgeId !== "none" })}>
              <Save size={18} />
              <span>Save knowledge choice</span>
            </button>
          </label>
        </div>
        {!ragCollections.length ? (
          <div className="empty-state">
            <Database size={24} />
            <strong>No knowledge collections yet</strong>
            <span>Finish Open WebUI onboarding, add an API key, and upload documents there. Collections will appear here automatically.</span>
          </div>
        ) : (
          <div className="collection-grid">
            {ragCollections.map((collection) => (
              <article className="voice-card" key={collection.id}>
                <div className="panel-heading">
                  <h3>{collection.name || collection.id}</h3>
                  <span className={selectedKnowledgeId === collection.id ? "pill good" : "pill"}>{selectedKnowledgeId === collection.id ? "Selected" : "Available"}</span>
                </div>
                <code>{collection.id}</code>
                <button type="button" className="icon-text" onClick={() => onSelectKnowledge(collection.id)}>
                  <Check size={18} />
                  <span>Use for voice answers</span>
                </button>
              </article>
            ))}
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>Test knowledge</h2>
          <span className="muted">Ask one grounded question</span>
        </div>
        <form className="text-turn wide-turn" onSubmit={(event) => { event.preventDefault(); void runKnowledgeTest(); }}>
          <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question from your uploaded documents" />
          <button type="submit" title={ragStatus?.api_key_configured ? "Test selected knowledge" : "Add Open WebUI API key first"} disabled={!ragStatus?.api_key_configured || testing}>
            <Search size={18} />
          </button>
        </form>
        {testResult ? (
          <div className={testResult.ok ? "notice success" : "notice error"}>
            {testResult.ok ? <CheckCircle2 size={18} /> : <CircleAlert size={18} />}
            <span>{testResult.response || testResult.detail || "No response returned."}</span>
          </div>
        ) : null}
      </section>
    </section>
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
    ["Consent complete", galleryVoices.every((voice) => voice.consent_status === "completed")],
    ["Voice quality passed", galleryVoices.some((voice) => Number(voice.quality_score ?? 0) >= 70)],
    ["Deletion ready", true],
    ["API keys secure", true],
    ["Generated audio provenance", true]
  ] as const;

  return (
    <section className="view-stack admin-view">
      <div className="view-header">
        <div>
          <h2>Admin</h2>
          <p>Deep controls for system health, voice models, providers, storage, readiness, and debug work.</p>
        </div>
        <span className="pill good">Local-first</span>
      </div>

      <div className="admin-tabs" role="tablist" aria-label="Admin sections">
        {[
          ["monitor", "System Monitor"],
          ["voices", "Voice Models"],
          ["providers", "AI Providers"],
          ["rag", "RAG / Knowledge"],
          ["audit", "Audit Logs"],
          ["data", "Data Storage"],
          ["ready", "Commercial Readiness"],
          ["debug", "Debug Tools"]
        ].map(([id, label]) => (
          <button key={id} type="button" className={tab === id ? "active" : ""} onClick={() => setTab(id)}>{label}</button>
        ))}
      </div>

      {tab === "monitor" ? (
        <div className="metric-grid admin-metrics">
          <Metric label="OS" value={`${systemInfo?.os || "macOS"} ${systemInfo?.os_version || ""}`.trim()} />
          <Metric label="CPU" value={typeof systemMetrics?.cpu_percent === "number" ? `${systemMetrics.cpu_percent.toFixed(0)}%` : "..."} />
          <Metric label="RAM used" value={typeof systemMetrics?.ram_used_gb === "number" ? `${systemMetrics.ram_used_gb.toFixed(1)} GB` : "..."} />
          <Metric label="Disk free" value={typeof systemMetrics?.disk_free_gb === "number" ? `${systemMetrics.disk_free_gb.toFixed(1)} GB` : "..."} />
          <Metric label="GPU" value={systemInfo?.gpu_mps_available ? "Apple MPS" : systemInfo?.cuda_available ? "CUDA" : "Not detected"} />
          <Metric label="Active model" value={settings.local_model || settings.ollama_model} />
          <Metric label="Voice socket" value={voiceSocketStatus?.capabilities.audio_turns ? "Ready" : "Needs setup"} />
          <Metric label="Recommendation" value={systemMetrics?.recommendation || "Run local defaults"} />
        </div>
      ) : null}

      {tab === "voices" ? (
        <div className="voice-studio-grid">
          {[...(voices?.voices ?? []), ...galleryVoices].map((voice) => (
            <article className="voice-card" key={`${voice.id}-${voice.name || voice.model_path}`}>
              <div className="panel-heading">
                <h3>{voice.name || voice.id}</h3>
                <span className={voice.status === "ready" || voice.publish_status === "published" ? "pill good" : "pill"}>{voice.publish_status || voice.status || "Draft"}</span>
              </div>
              <span className="muted">{voice.owner_name ? `Owner: ${voice.owner_name}` : "Built-in voice"}</span>
              <div className="pill-group">
                <span className="pill">{voice.language || "auto"}</span>
                <span className="pill">{voice.engine || "piper"}</span>
                <span className="pill">Quality {Math.round(voice.quality_score ?? 100)}</span>
              </div>
              <code>{voice.model_path || voice.id}</code>
            </article>
          ))}
        </div>
      ) : null}

      {tab === "providers" ? (
        <div className="status-grid">
          {providerStatus.map((provider) => (
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
      ) : null}

      {tab === "rag" ? (
        <section className="panel">
          <h2>RAG / Knowledge</h2>
          <p className="muted">Open WebUI: {settings.open_webui_base_url}. API key {settings.open_webui_api_key ? "configured" : "missing"}.</p>
        </section>
      ) : null}

      {tab === "audit" ? (
        <section className="panel">
          <div className="panel-heading">
            <h2>Audit Logs</h2>
            <button className="icon-text" type="button" onClick={() => downloadBlob(new Blob([JSON.stringify(auditLogs, null, 2)], { type: "application/json" }), "audit-logs.json")}>
              <Download size={18} />
              <span>Export logs</span>
            </button>
          </div>
          <div className="audit-table">
            {auditLogs.map((log, index) => (
              <div key={`${log.id || index}`}>
                <span>{log.timestamp}</span>
                <strong>{log.event}</strong>
                <span>{log.user_id || "system"}</span>
                <span>{log.details}</span>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {tab === "data" ? (
        <div className="status-grid">
          <article className="status-card"><HardDrive size={20} /><div><strong>SQLite database</strong><span>Voice profiles, consent, samples, jobs, artifacts, permissions, usage, and audit logs are local.</span></div></article>
          <article className="status-card"><Database size={20} /><div><strong>Audio policy</strong><span>Voice Studio recordings are saved as training data. Conversation audio is not kept unless enabled.</span></div></article>
        </div>
      ) : null}

      {tab === "ready" ? (
        <div className="readiness-grid">
          {readiness.map(([label, ok]) => (
            <article className={ok ? "status-card" : "status-card blocked"} key={label}>
              {ok ? <CheckCircle2 size={20} /> : <CircleAlert size={20} />}
              <div><strong>{label}</strong><span>{ok ? "Ready" : "Needs attention"}</span></div>
            </article>
          ))}
        </div>
      ) : null}

      {tab === "debug" ? (
        <details className="advanced-panel" open>
          <summary>Debug trace</summary>
          <pre>{JSON.stringify({ voiceSocketStatus, providerStatus, settings: { ...settings, openai_api_key: "***", gemini_api_key: "***", open_webui_api_key: "***" } }, null, 2)}</pre>
        </details>
      ) : null}
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
  const providerLogs = providerStatus.map((provider) => ({
    id: `provider-${provider.name}`,
    timestamp: new Date().toISOString(),
    level: provider.ok ? "success" : provider.critical === false ? "warning" : "error",
    event: `provider_${provider.name}`,
    detail: provider.ok ? provider.detail : `${provider.detail}${provider.fix ? ` Fix: ${provider.fix}` : ""}`,
    source: "runtime",
  }));
  const socketLogs = [
    ...(voiceSocketStatus?.blocking_reasons ?? []).map((detail, index) => ({
      id: `socket-blocker-${index}`,
      timestamp: new Date().toISOString(),
      level: "error",
      event: "voice_socket_blocker",
      detail,
      source: "runtime",
    })),
    ...(voiceSocketStatus?.warnings ?? []).map((detail, index) => ({
      id: `socket-warning-${index}`,
      timestamp: new Date().toISOString(),
      level: "warning",
      event: "voice_socket_warning",
      detail,
      source: "runtime",
    })),
  ];
  const backendAudit = auditLogs.map((log, index) => ({
    id: log.id || `audit-${index}`,
    timestamp: log.timestamp,
    level: "info",
    event: log.event,
    detail: log.details || "",
    source: "backend audit",
  }));
  const merged = [
    ...appLogs.map((log) => ({ ...log, source: "browser app" })),
    ...backendAudit,
    ...socketLogs,
    ...providerLogs,
  ].sort((a, b) => String(b.timestamp || "").localeCompare(String(a.timestamp || "")));
  const visible = filter === "all" ? merged : merged.filter((log) => log.level === filter);

  return (
    <section className="view-stack logs-view">
      <div className="view-header">
        <div>
          <h2>Logs</h2>
          <p>Errors, voice events, provider status, setup warnings, audit details, and debugging context.</p>
        </div>
        <div className="button-row">
          <button className="icon-text" type="button" onClick={() => downloadBlob(new Blob([JSON.stringify(merged, null, 2)], { type: "application/json" }), "swarlocal-logs.json")}>
            <Download size={18} />
            <span>Export logs</span>
          </button>
          <button className="icon-text danger" type="button" onClick={onClear} title="Clears browser app logs only. Backend audit logs remain in SQLite.">
            <Trash2 size={18} />
            <span>Clear app logs</span>
          </button>
        </div>
      </div>

      <div className="metric-grid">
        <Metric label="Errors" value={`${merged.filter((log) => log.level === "error").length}`} />
        <Metric label="Warnings" value={`${merged.filter((log) => log.level === "warning").length}`} />
        <Metric label="App logs" value={`${appLogs.length}`} />
        <Metric label="Audit logs" value={`${auditLogs.length}`} />
      </div>

      <div className="admin-tabs" role="tablist" aria-label="Log filters">
        {["all", "error", "warning", "success", "info"].map((level) => (
          <button key={level} type="button" className={filter === level ? "active" : ""} onClick={() => setFilter(level)}>
            {level}
          </button>
        ))}
      </div>

      <section className="panel">
        <div className="panel-heading">
          <h2>Log details</h2>
          <span className="muted">{visible.length} shown</span>
        </div>
        {visible.length ? (
          <div className="log-table">
            {visible.map((log, index) => (
              <article className={`log-row ${log.level}`} key={`${log.id || index}-${index}`}>
                <span className={`pill ${log.level === "success" ? "good" : log.level === "warning" ? "warn" : ""}`}>{log.level}</span>
                <time>{log.timestamp ? new Date(log.timestamp).toLocaleString() : "now"}</time>
                <strong>{String(log.event || "event").replaceAll("_", " ")}</strong>
                <span>{log.source}</span>
                <p>{log.detail}</p>
                {log.meta ? <code>{JSON.stringify(log.meta)}</code> : null}
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <FileText size={24} />
            <strong>No logs for this filter</strong>
            <span>Run a voice turn, test a provider, or refresh status to add entries.</span>
          </div>
        )}
      </section>
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
  onTtsTest: (language: "ne" | "en") => void;
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
        <button className="icon-text" onClick={() => onTtsTest("ne")} type="button">
          <Volume2 size={18} />
          <span>Nepali TTS</span>
        </button>
        <button className="icon-text" onClick={() => onTtsTest("en")} type="button">
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

function VoiceStudioView({
  voices,
  galleryVoices,
  auditLogs,
  onRefresh,
  onTtsTest
}: {
  voices: VoicesResponse | null;
  galleryVoices: any[];
  auditLogs: any[];
  onRefresh: () => void;
  onTtsTest: (language: "ne" | "en") => void;
}) {
  const [selectedVoice, setSelectedVoice] = useState<any | null>(null);
  const [cleaningAll, setCleaningAll] = useState(false);
  const [cloningVoice, setCloningVoice] = useState(false);
  const [wizardStep, setWizardStep] = useState<"gallery" | "identity" | "consent" | "recordings">("gallery");
  
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

  // Load prompts & recordings if a custom voice is selected
  useEffect(() => {
    const voiceId = selectedVoice?.id || selectedVoice?.voice_id;
    if (voiceId) {
      getVoicesPrompts().then(setPrompts).catch(() => {});
      getVoiceRecordings(voiceId).then(setRecordings).catch(() => {});
    }
  }, [selectedVoice]);

  const handleCreateVoice = async (e: React.FormEvent) => {
    e.preventDefault();
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
      setSelectedVoice({ ...resp, id: resp.voice_id, consent_status: "pending", publish_status: "unpublished", language, owner_name: ownerName, engine });
      setWizardStep("consent");
    } catch (err: any) {
      alert(err.message || "Failed to create voice profile");
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
    try {
      if (!selectedVoiceIdValue) {
        throw new Error("Voice profile ID is missing. Please go back to gallery and try again.");
      }
      if (!selectedVoiceConsentComplete) {
        setWizardStep("consent");
        throw new Error("Recording is blocked: please complete voice consent first.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      chunksRef.current = [];
      setActivePrompt(promptId);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        try {
          await uploadVoiceRecording(selectedVoiceIdValue, promptId, blob);
          const updated = await getVoiceRecordings(selectedVoiceIdValue);
          setRecordings(updated);
        } catch (err: any) {
          alert(err.message || "Failed to upload recording");
        }
        setActivePrompt(null);
      };
      recorder.start();
    } catch (err: any) {
      alert("Microphone permission denied or unavailable");
    }
  };

  const handleStopRecord = () => {
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
      recorderRef.current.stream.getTracks().forEach(track => track.stop());
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
      alert(err.message);
    }
  };

  const handleCleanAll = async () => {
    if (!selectedVoiceIdValue) return;
    try {
      setCleaningAll(true);
      const toClean = recordings.filter(r => r.exists);
      if (toClean.length === 0) {
        alert("No recordings to clean yet. Record some samples first.");
        return;
      }
      for (const rec of toClean) {
        await cleanRecording(selectedVoiceIdValue, rec.id);
      }
      alert("All recordings have been cleaned and normalized successfully!");
      const updated = await getVoiceRecordings(selectedVoiceIdValue);
      setRecordings(updated);
    } catch (err: any) {
      alert(err.message || "Failed to clean recordings.");
    } finally {
      setCleaningAll(false);
    }
  };

  const handleCloneVoice = async () => {
    if (!selectedVoiceIdValue) return;
    try {
      setCloningVoice(true);
      await cloneVoice(selectedVoiceIdValue);
      alert("Voice cloned successfully! You can now publish it.");
      // Refresh custom voice details
      const updatedList = await getVoicesGallery();
      const match = updatedList.find((v: any) => v.id === selectedVoiceIdValue);
      if (match) {
        setSelectedVoice(match);
      }
      onRefresh();
    } catch (err: any) {
      alert(err.message || "Failed to clone voice.");
    } finally {
      setCloningVoice(false);
    }
  };

  const handlePublish = async () => {
    try {
      if (!selectedVoiceIdValue) {
        throw new Error("Voice profile ID is missing.");
      }
      await publishVoice(selectedVoiceIdValue);
      alert("Voice model published successfully!");
      setSelectedVoice(null);
      setWizardStep("gallery");
      onRefresh();
    } catch (err: any) {
      alert(err.message || "Failed to publish voice");
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

  const recordedCount = recordings.filter(r => r.exists).length;
  const languageCounts = recordings.reduce((counts: Record<string, number>, rec) => {
    if (rec.exists) {
      counts[rec.language] = (counts[rec.language] ?? 0) + 1;
    }
    return counts;
  }, {});

  return (
    <section className="view-stack">
      {wizardStep === "gallery" && (
        <>
          <div className="view-header">
            <div>
              <h2>Voice Studio</h2>
              <p>Create, clean, train, test, and publish friendly voices for conversation.</p>
            </div>
            <div className="button-row">
              <button className="icon-text" onClick={triggerDownloadDefault} disabled={isDownloading}>
                <Download size={18} />
                <span>{isDownloading ? "Starting..." : "Download Default Voices"}</span>
              </button>
              <button className="icon-text good" onClick={() => setWizardStep("identity")}>
                <Plus size={18} />
                <span>Create Voice</span>
              </button>
            </div>
          </div>

          <div className="studio-steps" aria-label="Voice Studio workflow">
            {["Voice Gallery", "Create Voice", "Record & Clean", "Train / Clone", "Test Voice", "Publish", "Quality Report", "Audit & Data"].map((step, index) => (
              <span key={step} className={index === 0 ? "active" : ""}>{step}</span>
            ))}
          </div>

          <div className="filter-row" aria-label="Voice filters">
            {["Ready voices", "Draft voices", "Needs training", "Nepali", "English", "Mixed", "My voices", "Built-in voices"].map((filter) => (
              <button key={filter} type="button" title={`${filter}: narrows the gallery view. Example: show only ${filter.toLowerCase()}.`}>
                {filter}
              </button>
            ))}
          </div>

          <div className="voice-studio-grid">
            {/* Built-in Voices */}
            <article className="status-card voice-studio-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <ShieldCheck size={20} color="#48bb78" />
                <strong>Nepali chitwan</strong>
              </div>
              <span className="muted">rhasspy/piper-voices</span>
              <div className="pill-group">
                <span className="pill good">ne</span>
                <span className="pill">piper</span>
                <span className="pill good">commercial</span>
              </div>
              <button className="icon-text" onClick={() => onTtsTest("ne")} type="button">
                <Volume2 size={16} />
                <span>Test Synthesize</span>
              </button>
            </article>

            <article className="status-card voice-studio-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <ShieldCheck size={20} color="#48bb78" />
                <strong>English lessac</strong>
              </div>
              <span className="muted">rhasspy/piper-voices</span>
              <div className="pill-group">
                <span className="pill good">en</span>
                <span className="pill">piper</span>
                <span className="pill good">commercial</span>
              </div>
              <button className="icon-text" onClick={() => onTtsTest("en")} type="button">
                <Volume2 size={16} />
                <span>Test Synthesize</span>
              </button>
            </article>

            {/* Custom Profiles */}
            {galleryVoices.map(cv => (
              <article key={cv.id} className="status-card voice-studio-card">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <User size={20} color="#4299e1" />
                  <strong>{cv.name}</strong>
                </div>
                <span className="muted">Owner: {cv.owner_name} ({cv.owner_org || "Personal"})</span>
                <div className="pill-group">
                  <span className="pill good">{cv.language}</span>
                  <span className="pill">{cv.engine}</span>
                  <span className={`pill ${cv.consent_status === 'completed' ? 'good' : 'warn'}`}>
                    Consent: {cv.consent_status}
                  </span>
                  <span className={`pill ${cv.publish_status === 'published' ? 'good' : 'warn'}`}>
                    {cv.publish_status === 'published' ? 'Ready to Use' : 'Needs More Samples'}
                  </span>
                  <span className="pill">Voice Match pending</span>
                  <span className="pill">Quality {Math.round(cv.quality_score ?? 0)}</span>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', width: '100%', marginTop: 'auto', paddingTop: '0.5rem' }}>
                  <button className="icon-text" onClick={() => onTtsTest(cv.language === "en" ? "en" : "ne")} type="button">
                    <Volume2 size={16} />
                    <span>Test</span>
                  </button>
                  {cv.publish_status !== 'published' ? (
                    <button className="icon-text" onClick={() => { setSelectedVoice(cv); setWizardStep(cv.consent_status === "completed" ? "recordings" : "consent"); }} type="button">
                      <Mic size={16} />
                      <span>{cv.consent_status === "completed" ? "Continue setup" : "Add consent"}</span>
                    </button>
                  ) : (
                    <span className="pill good" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                      <Check size={14} /> Ready
                    </span>
                  )}
                  <button className="icon-text danger" onClick={() => handleDeleteVoiceProfile(cv.id)} style={{ marginLeft: 'auto', padding: '0.25rem 0.5rem' }} type="button">
                    <Trash2 size={16} />
                  </button>
                </div>
              </article>
            ))}
            {!galleryVoices.length ? (
              <article className="empty-state gallery-empty">
                <User size={28} />
                <strong>No custom voices yet</strong>
                <span>Create your first voice in a few guided steps. You can start with a quick preview and improve it later.</span>
                <button className="icon-text good" type="button" onClick={() => setWizardStep("identity")}>
                  <Plus size={18} />
                  <span>Create your first voice</span>
                </button>
              </article>
            ) : null}
          </div>

          {/* Audit Logs */}
          <div className="audit-logs-card">
            <h3>Commercial Consent & Audit logs</h3>
            <div className="audit-logs-list">
              {auditLogs.map((log, i) => (
                <div key={i} className="audit-logs-item">
                  <span>[{log.timestamp}]</span> <strong>{log.event}</strong>: {log.details}
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {wizardStep === "identity" && (
        <form onSubmit={handleCreateVoice} className="status-card voice-wizard-form">
          <div className="wizard-progress"><span className="active">1 Name</span><span>2 Goal</span><span>3 Consent</span><span>4 Record</span><span>5 Publish</span></div>
          <h3>Create Voice</h3>
          <p className="muted">Choose how this voice should speak in conversations.</p>
          <div className="form-group">
            <label>Voice name</label>
            <input type="text" value={name} onChange={e => setName(e.target.value)} required placeholder="e.g. Kushal English Voice" />
          </div>
          <div className="form-group">
            <label>Owner Full Name</label>
            <input type="text" value={ownerName} onChange={e => setOwnerName(e.target.value)} required placeholder="e.g. Kushal Khadka" />
          </div>
          <div className="form-group">
            <label>Owner Email</label>
            <input type="email" value={ownerEmail} onChange={e => setOwnerEmail(e.target.value)} placeholder="e.g. kushal@example.com" />
          </div>
          <div className="form-group">
            <label>Organization / Client</label>
            <input type="text" value={organization} onChange={e => setOrganization(e.target.value)} placeholder="e.g. Personal" />
          </div>
          <div className="form-group">
            <label>Language goal</label>
            <select value={language} onChange={e => setLanguage(e.target.value)}>
              <option value="ne">Nepali</option>
              <option value="en">English</option>
              <option value="mixed">Natural mixed Nepali-English</option>
            </select>
          </div>
          <div className="form-group">
            <label>Cloning Engine</label>
            <select value={engine} onChange={e => setEngine(e.target.value)}>
              <option value="chatterbox">Chatterbox local clone</option>
              <option value="piper">Piper fine-tune (.onnx export)</option>
              <option value="elevenlabs">ElevenLabs (cloud-trusted)</option>
              <option value="f5_tts">F5-TTS (experimental)</option>
              <option value="openvoice">OpenVoice (experimental)</option>
              <option value="voxcpm">VoxCPM (experimental)</option>
            </select>
          </div>
          <div className="form-group">
            <label>Recording goal</label>
            <select value={recordingGoal} onChange={e => setRecordingGoal(e.target.value)} title="More clean recordings make the cloned voice sound more like you.">
              <option value="quick">Quick preview - 5 to 10 minutes</option>
              <option value="better">Better quality - 30 to 60 minutes</option>
              <option value="production">Production quality - 2+ hours</option>
            </select>
          </div>
          <div className="tips-card">
            <strong>Recording tips</strong>
            <span>Speak naturally. Use the same microphone. Avoid fan noise. Keep 10-20 cm from the microphone.</span>
          </div>
          <label className="checkbox-label">
            <input type="checkbox" checked={commercialAllowed} onChange={e => setCommercialAllowed(e.target.checked)} />
            <span>I have permission to use this voice.</span>
          </label>
          <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
            <button type="button" className="icon-text danger" onClick={() => setWizardStep("gallery")}>Cancel</button>
            <button type="submit" className="icon-text good" style={{ marginLeft: 'auto' }}>Next: Consent Signature</button>
          </div>
        </form>
      )}

      {wizardStep === "consent" && (
        <form onSubmit={handleSaveConsent} className="status-card voice-wizard-form">
          <div className="wizard-progress"><span>1 Name</span><span className="active">2 Consent</span><span>3 Record</span><span>4 Test</span><span>5 Publish</span></div>
          <h3>Consent and ownership</h3>
          <p style={{ fontSize: '0.85rem', color: '#a0aec0', lineHeight: 1.5 }}>
            Keep it simple: confirm you have permission to use this voice. Admins can review the detailed audit later.
          </p>
          <div className="form-group">
            <label>Type full name signature</label>
            <input type="text" value={signature} onChange={e => setSignature(e.target.value)} required placeholder="Type signature exactly" />
          </div>
          <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
            <button type="button" className="icon-text danger" onClick={() => setWizardStep("gallery")}>Cancel</button>
            <button type="submit" className="icon-text good" style={{ marginLeft: 'auto' }}>Sign and Continue</button>
          </div>
        </form>
      )}

      {wizardStep === "recordings" && selectedVoice && (
        <>
          <div className="view-header">
            <div>
              <h2>Record and Clean: {selectedVoice.name}</h2>
              <p>Record one sentence at a time. Weak recordings can be retried before training.</p>
            </div>
            <div className="button-row">
              <button className="icon-text danger" onClick={() => setWizardStep("gallery")} type="button">Back to Gallery</button>
              <button className="icon-text good" onClick={handlePublish} disabled={recordedCount < 3 || !selectedVoice?.model_exists} type="button" title={!selectedVoice?.model_exists ? "Please click 'Create my voice' to clone/build the model before publishing." : "Publish this voice profile."}>
                <CheckCircle2 size={18} />
                <span>Publish Voice ({recordedCount} takes)</span>
              </button>
            </div>
          </div>

          <div className="quality-overview">
            <Metric label="Nepali samples" value={`${languageCounts.ne ?? 0}`} />
            <Metric label="English samples" value={`${languageCounts.en ?? 0}`} />
            <Metric label="Mixed samples" value={`${languageCounts.mixed ?? 0}`} />
            <Metric label="Progress" value={`${recordedCount}/${prompts.length || 8}`} />
          </div>

          <div className="action-band studio-actions">
            <button className="icon-text" type="button" onClick={handleCleanAll} disabled={cleaningAll || recordedCount === 0} title="Clean Recording removes background noise, trims silence, and normalizes loudness. Safe default: automatic. Example: fan noise behind a clean voice.">
              <Wand2 size={18} />
              <span>{cleaningAll ? "Cleaning..." : "Clean all recordings"}</span>
            </button>
            <button className="icon-text" type="button" title="Retry weak recordings shows clips with low quality scores so you can replace them. Example: clipped or too quiet audio.">
              <Mic size={18} />
              <span>Retry weak recordings</span>
            </button>
            <button className="icon-text good" type="button" onClick={handleCloneVoice} disabled={cloningVoice || recordedCount < 3} title="Create my voice prepares the dataset, trains or imports a voice artifact, tests it, and makes it ready for preview.">
              <Sparkles size={18} />
              <span>{cloningVoice ? "Cloning..." : "Create my voice"}</span>
            </button>
          </div>

          <div className="prompt-list voice-studio-recordings" style={{ marginTop: '1rem' }}>
            {prompts.filter(p => p.language === selectedVoice.language || selectedVoice.language === 'mixed').map((prompt) => {
              const rec = recordings.find(r => r.id === prompt.id);
              const active = activePrompt === prompt.id;
              return (
                <article className="prompt-row" key={prompt.id}>
                  <div className="prompt-copy">
                    <span style={{ fontSize: '0.75rem', color: '#718096', marginRight: '0.5rem' }}>{prompt.id}</span>
                    <strong>{prompt.text}</strong>
                    <LanguageBadge language={prompt.language} />
                  </div>
                  <div className="quality-strip">
                    {rec?.exists ? (
                      <>
                        <span className={`quality ${rec.quality?.verdict || 'review'}`}>
                          {rec.quality?.score ?? 'N/A'}
                        </span>
                        <audio src={absoluteAudioUrl(rec.audio_url) || ""} controls />
                        <button className="icon-text danger record-small" onClick={() => handleDeleteRecord(prompt.id)} type="button">
                          <Trash2 size={14} />
                        </button>
                      </>
                    ) : (
                      <span className="muted">No recording yet</span>
                    )}
                  </div>
                  <button
                    className={active ? "record-small-circle active" : "record-small-circle"}
                    onClick={active ? handleStopRecord : () => handleStartRecord(prompt.id)}
                    type="button"
                  >
                    {active ? <Square size={16} /> : <Mic size={16} />}
                  </button>
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

function EvaluationView({
  ratings,
  onChange,
  history
}: {
  ratings: Record<string, number>;
  onChange: (ratings: Record<string, number>) => void;
  history: ConversationTurn[];
}) {
  const metrics = [
    ["naturalness", "Naturalness"],
    ["voiceSimilarity", "Voice similarity"],
    ["nepaliPronunciation", "Nepali pronunciation"],
    ["englishPronunciation", "English pronunciation"]
  ] as const;
  const latency = history[0]?.timings;
  return (
    <section className="view-stack">
      <div className="view-header">
        <div>
          <h2>Manual Evaluation</h2>
          <p>Voice quality, pronunciation, and turn latency.</p>
        </div>
        <button
          className="icon-text"
          onClick={() => downloadBlob(new Blob([JSON.stringify({ ratings, latestLatency: latency }, null, 2)], { type: "application/json" }), "manual-evaluation.json")}
          type="button"
        >
          <Download size={18} />
          <span>Export</span>
        </button>
      </div>
      <div className="rating-grid">
        {metrics.map(([key, label]) => (
          <div className="rating-row" key={key}>
            <strong>{label}</strong>
            <div className="segmented">
              {[1, 2, 3, 4, 5].map((value) => (
                <button
                  className={ratings[key] === value ? "active" : ""}
                  key={value}
                  onClick={() => onChange({ ...ratings, [key]: value })}
                  type="button"
                >
                  {value}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="metric-grid">
        <Metric label="Transcript" value={formatMs(latency?.audio_received_to_transcript_ms)} />
        <Metric label="LLM first token" value={formatMs(latency?.llm_first_token_ms)} />
        <Metric label="TTS" value={formatMs(latency?.tts_generation_ms)} />
        <Metric label="Total" value={formatMs(latency?.total_turn_ms)} />
      </div>
    </section>
  );
}

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
  onTtsTest: (language: "ne" | "en") => void;
}) {
  const [draft, setDraft] = useState(settings);
  useEffect(() => setDraft(settings), [settings]);

  const [providerTests, setProviderTests] = useState<Record<string, { ok: boolean; detail: string; latency_ms?: number; time?: string }>>({});

  const handleTestConnection = async (provider: string) => {
    try {
      const payload = provider === "local" ? undefined : {
        [`${provider}_api_key`]: draft[`${provider}_api_key` as keyof BackendSettings],
        [`${provider}_model`]: draft[`${provider}_model` as keyof BackendSettings],
      };
      const res = await testProvider(provider, payload);
      setProviderTests(prev => ({
        ...prev,
        [provider]: {
          ok: res.ok,
          detail: res.detail,
          latency_ms: res.latency_ms,
          time: new Date().toLocaleTimeString()
        }
      }));
    } catch (err: any) {
      setProviderTests(prev => ({
        ...prev,
        [provider]: {
          ok: false,
          detail: err.message || "Connection test failed",
          time: new Date().toLocaleTimeString()
        }
      }));
    }
  };

  const handleDeleteKey = async (provider: "openai" | "gemini" | "elevenlabs") => {
    const providerName = provider === 'openai' ? 'OpenAI' : provider === 'gemini' ? 'Gemini' : 'ElevenLabs';
    if (!confirm(`Are you sure you want to delete the ${providerName} API key?`)) {
      return;
    }
    try {
      let updated: BackendSettings;
      if (provider === "openai") {
        updated = await deleteOpenAIKey();
      } else if (provider === "gemini") {
        updated = await deleteGeminiKey();
      } else {
        updated = await deleteElevenLabsKey();
      }
      setDraft(updated);
      onSave(updated); // Save to parent state immediately
      setProviderTests(prev => {
        const next = { ...prev };
        delete next[provider];
        return next;
      });
      alert(`${providerName} API key deleted successfully.`);
    } catch (err: any) {
      alert(`Failed to delete key: ${err.message}`);
    }
  };

  return (
    <section className="view-stack">
      <div className="view-header">
        <div>
          <h2>Settings</h2>
          <p>Models, voices, latency, local data.</p>
        </div>
        <button className="icon-text" onClick={() => onSave(draft)} type="button">
          <Save size={18} />
          <span>Save</span>
        </button>
      </div>
      <div className="settings-grid">
        <label className="wide">
          <span>Backend URL</span>
          <input value={API_HTTP} disabled />
        </label>
        <label className="wide">
          <span>Ollama URL</span>
          <input value={draft.ollama_base_url} onChange={(event) => setDraft({ ...draft, ollama_base_url: event.target.value })} />
        </label>
        <label>
          <span>STT model</span>
          <select value={draft.whisper_model_size} onChange={(event) => setDraft({ ...draft, whisper_model_size: event.target.value })}>
            <option value="small">small multilingual</option>
            <option value="medium">medium multilingual</option>
            <option value="large-v3">large-v3 multilingual</option>
          </select>
        </label>
        <label>
          <span>LLM model</span>
          <select value={draft.ollama_model} onChange={(event) => setDraft({ ...draft, ollama_model: event.target.value })}>
            <option value="qwen2.5:7b">qwen2.5:7b</option>
            <option value="gemma3:4b">gemma3:4b</option>
            <option value="qwen3:1.7b">qwen3:1.7b</option>
            <option value="qwen3:4b">qwen3:4b</option>
            <option value="llama3:latest">llama3:latest</option>
            <option value="mistral:latest">mistral:latest</option>
            <option value="llama3.2:3b">llama3.2:3b</option>
            <option value="llama3.2:1b">llama3.2:1b</option>
          </select>
        </label>
        <label>
          <span>Temperature</span>
          <input
            type="number"
            min="0"
            max="1"
            step="0.05"
            value={draft.ollama_temperature}
            onChange={(event) => setDraft({ ...draft, ollama_temperature: Number(event.target.value) })}
          />
        </label>
        <label>
          <span>Max tokens</span>
          <input
            type="number"
            min="32"
            max="2048"
            step="16"
            value={draft.ollama_num_predict}
            onChange={(event) => setDraft({ ...draft, ollama_num_predict: Number(event.target.value) })}
          />
        </label>
        <label>
          <span>Keep alive</span>
          <input value={draft.ollama_keep_alive} onChange={(event) => setDraft({ ...draft, ollama_keep_alive: event.target.value })} />
        </label>
        <label>
          <span>Max recording seconds</span>
          <input
            type="number"
            min="1"
            max="120"
            step="1"
            value={draft.max_recording_seconds}
            onChange={(event) => setDraft({ ...draft, max_recording_seconds: Number(event.target.value) })}
          />
        </label>
        <label>
          <span>Internet max sources</span>
          <input
            type="number"
            min="1"
            max="10"
            step="1"
            value={draft.internet_max_sources}
            onChange={(event) => setDraft({ ...draft, internet_max_sources: Number(event.target.value) })}
          />
        </label>
        <label className="wide">
          <span>Nepali voice</span>
          <input value={draft.piper_nepali_voice} onChange={(event) => setDraft({ ...draft, piper_nepali_voice: event.target.value })} />
        </label>
        <label className="wide">
          <span>English voice</span>
          <input value={draft.piper_english_voice} onChange={(event) => setDraft({ ...draft, piper_english_voice: event.target.value })} />
        </label>
        <label className="wide">
          <span>Open WebUI URL</span>
          <input value={draft.open_webui_base_url} onChange={(event) => setDraft({ ...draft, open_webui_base_url: event.target.value })} />
        </label>
        <label className="wide">
          <span>Open WebUI API key</span>
          <input
            value={draft.open_webui_api_key}
            onChange={(event) => setDraft({ ...draft, open_webui_api_key: event.target.value })}
            placeholder="Optional until RAG API use"
            type="password"
          />
        </label>
        <label className="wide">
          <span>Default knowledge collection</span>
          <input value={draft.rag_default_collection} onChange={(event) => setDraft({ ...draft, rag_default_collection: event.target.value })} />
        </label>
        <label className="wide">
          <span>System prompt</span>
          <textarea value={draft.system_prompt} onChange={(event) => setDraft({ ...draft, system_prompt: event.target.value })} />
        </label>
        <label className="toggle-line">
          <input
            checked={draft.low_latency_mode}
            onChange={(event) => setDraft({ ...draft, low_latency_mode: event.target.checked })}
            type="checkbox"
          />
          <span>Low-latency mode</span>
        </label>
        <label className="toggle-line">
          <input checked={draft.quality_mode} onChange={(event) => setDraft({ ...draft, quality_mode: event.target.checked })} type="checkbox" />
          <span>Quality mode</span>
        </label>
        <label className="toggle-line">
          <input checked={draft.rag_enabled} onChange={(event) => setDraft({ ...draft, rag_enabled: event.target.checked })} type="checkbox" />
          <span>Use Open WebUI RAG</span>
        </label>
        <label className="toggle-line">
          <input checked={draft.rag_fallback_to_ollama} onChange={(event) => setDraft({ ...draft, rag_fallback_to_ollama: event.target.checked })} type="checkbox" />
          <span>Fallback to Ollama direct</span>
        </label>
        <label className="toggle-line">
          <input
            checked={draft.internet_retrieval_enabled}
            onChange={(event) => setDraft({ ...draft, internet_retrieval_enabled: event.target.checked })}
            type="checkbox"
          />
          <span>Use internet when needed</span>
        </label>
        <label className="toggle-line">
          <input
            checked={draft.internet_require_citation}
            onChange={(event) => setDraft({ ...draft, internet_require_citation: event.target.checked })}
            type="checkbox"
          />
          <span>Require internet citations</span>
        </label>
        <label className="toggle-line" title="When enabled, a selected cloned/custom voice must be used exactly. If its model file is missing or does not support the output language, the turn fails with a clear reason.">
          <input
            checked={draft.force_selected_voice}
            onChange={(event) => setDraft({ ...draft, force_selected_voice: event.target.checked })}
            type="checkbox"
          />
          <span>Force selected cloned voice only</span>
        </label>
        <label className="toggle-line" title="When enabled, SwarLocal may fall back to a built-in voice if a selected voice cannot speak a segment. Turn this off for strict commercial voice QA.">
          <input
            checked={draft.fallback_allowed}
            onChange={(event) => setDraft({ ...draft, fallback_allowed: event.target.checked })}
            type="checkbox"
          />
          <span>Allow voice fallback</span>
        </label>
        <label className="toggle-line" title="When enabled, one selected Piper voice model is used for Nepali and English chunks. This avoids switching between separate Nepali and English voices.">
          <input
            checked={draft.single_tts_voice_model}
            onChange={(event) => setDraft({ ...draft, single_tts_voice_model: event.target.checked })}
            type="checkbox"
          />
          <span>Use one voice model for Nepali and English</span>
        </label>
      </div>

      <section className="ai-providers-section" style={{ marginTop: '2rem', padding: '1.25rem', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', backgroundColor: 'rgba(255,255,255,0.01)' }}>
        <div className="panel-heading">
          <h2>AI Provider Configuration</h2>
          <span className="muted" style={{ fontSize: '0.85rem' }}>Configure active brain for transcripts & reasoning</span>
        </div>
        
        <div className="dropdown-control" style={{ margin: '1rem 0', maxWidth: '300px' }}>
          <label style={{ fontSize: '0.85rem', fontWeight: 600, color: '#a0aec0', display: 'block', marginBottom: '0.5rem' }}>Active LLM Provider</label>
          <select value={draft.llm_provider} onChange={(e) => setDraft({ ...draft, llm_provider: e.target.value })}>
            <option value="local">Local AI (Ollama)</option>
            <option value="openai">OpenAI</option>
            <option value="gemini">Google Gemini</option>
          </select>
        </div>

        <div className="provider-cards-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem', marginTop: '1rem' }}>
          
          {/* Local AI (Ollama) Card */}
          <article className="voice-card ready" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', padding: '1.25rem' }}>
            <div className="panel-heading">
              <h3>1. Local AI</h3>
              <span className="pill good">Ready</span>
            </div>
            <p className="muted" style={{ fontSize: '0.8rem' }}>Run open-source models completely locally on your Mac. Transcripts and audio remain offline.</p>
            
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a0aec0' }}>Ollama URL</span>
              <input value={draft.ollama_base_url} onChange={(e) => setDraft({ ...draft, ollama_base_url: e.target.value })} />
            </label>

            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a0aec0' }}>Model</span>
              <select value={draft.local_model} onChange={(e) => setDraft({ ...draft, local_model: e.target.value })}>
                <option value="qwen2.5:7b">qwen2.5:7b (Default)</option>
                <option value="gemma3:4b">gemma3:4b (Fallback)</option>
                <option value="qwen3:1.7b">qwen3:1.7b</option>
                <option value="llama3.2:3b">llama3.2:3b</option>
              </select>
            </label>

            {providerTests["local"] && (
              <div style={{ fontSize: '0.8rem', padding: '0.5rem', borderRadius: '4px', backgroundColor: providerTests["local"].ok ? 'rgba(72,187,120,0.1)' : 'rgba(245,101,101,0.1)' }}>
                <strong>Status:</strong> {providerTests["local"].ok ? "Success" : "Failed"}<br/>
                {providerTests["local"].latency_ms && <><strong>Latency:</strong> {providerTests["local"].latency_ms.toFixed(0)} ms<br/></>}
                <strong>Checked:</strong> {providerTests["local"].time}<br/>
                <small className="muted">{providerTests["local"].detail}</small>
              </div>
            )}

            <button className="icon-text" onClick={() => handleTestConnection("local")} type="button" style={{ marginTop: 'auto' }}>
              <Activity size={18} />
              <span>Test Local AI</span>
            </button>
          </article>

          {/* OpenAI Card */}
          <article className={`voice-card ${draft.openai_api_key ? 'ready' : 'missing'}`} style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', padding: '1.25rem' }}>
            <div className="panel-heading">
              <h3>2. OpenAI</h3>
              <span className={`pill ${draft.openai_api_key ? 'good' : ''}`}>{draft.openai_api_key ? 'Configured' : 'Needs setup'}</span>
            </div>
            <p className="muted" style={{ fontSize: '0.8rem' }}>Utilize OpenAI cloud LLM. Transcripts are sent to OpenAI (audio remains local).</p>
            
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a0aec0' }}>API Key</span>
              <input
                type="password"
                placeholder={draft.openai_api_key ? "••••••••••••••••" : "Enter OpenAI API Key"}
                value={draft.openai_api_key}
                onChange={(e) => setDraft({ ...draft, openai_api_key: e.target.value })}
              />
            </label>

            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a0aec0' }}>Model</span>
              <select value={draft.openai_model} onChange={(e) => setDraft({ ...draft, openai_model: e.target.value })}>
                <option value="gpt-4o-mini">gpt-4o-mini (Default)</option>
                <option value="gpt-4o">gpt-4o</option>
                <option value="gpt-4">gpt-4</option>
              </select>
            </label>

            {providerTests["openai"] && (
              <div style={{ fontSize: '0.8rem', padding: '0.5rem', borderRadius: '4px', backgroundColor: providerTests["openai"].ok ? 'rgba(72,187,120,0.1)' : 'rgba(245,101,101,0.1)' }}>
                <strong>Status:</strong> {providerTests["openai"].ok ? "Success" : "Failed"}<br/>
                {providerTests["openai"].latency_ms && <><strong>Latency:</strong> {providerTests["openai"].latency_ms.toFixed(0)} ms<br/></>}
                <strong>Checked:</strong> {providerTests["openai"].time}<br/>
                <small className="muted">{providerTests["openai"].detail}</small>
              </div>
            )}

            <div style={{ display: 'flex', gap: '0.5rem', marginTop: 'auto' }}>
              <button className="icon-text" onClick={() => handleTestConnection("openai")} type="button" style={{ flex: 1 }}>
                <Activity size={18} />
                <span>Test OpenAI</span>
              </button>
              {draft.openai_api_key && (
                <button className="icon-text danger" onClick={() => handleDeleteKey("openai")} type="button" title="Delete API Key" style={{ padding: '0.5rem' }}>
                  <Trash2 size={18} />
                </button>
              )}
            </div>
          </article>

          {/* Gemini Card */}
          <article className={`voice-card ${draft.gemini_api_key ? 'ready' : 'missing'}`} style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', padding: '1.25rem' }}>
            <div className="panel-heading">
              <h3>3. Gemini</h3>
              <span className={`pill ${draft.gemini_api_key ? 'good' : ''}`}>{draft.gemini_api_key ? 'Configured' : 'Needs setup'}</span>
            </div>
            <p className="muted" style={{ fontSize: '0.8rem' }}>Utilize Google Gemini cloud LLM. Transcripts are sent to Google (audio remains local).</p>
            
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a0aec0' }}>API Key</span>
              <input
                type="password"
                placeholder={draft.gemini_api_key ? "••••••••••••••••" : "Enter Gemini API Key"}
                value={draft.gemini_api_key}
                onChange={(e) => setDraft({ ...draft, gemini_api_key: e.target.value })}
              />
            </label>

            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a0aec0' }}>Model</span>
              <select value={draft.gemini_model} onChange={(e) => setDraft({ ...draft, gemini_model: e.target.value })}>
                <option value="gemini-1.5-flash">gemini-1.5-flash (Default)</option>
                <option value="gemini-1.5-pro">gemini-1.5-pro</option>
                <option value="gemini-2.0-flash">gemini-2.0-flash</option>
              </select>
            </label>

            {providerTests["gemini"] && (
              <div style={{ fontSize: '0.8rem', padding: '0.5rem', borderRadius: '4px', backgroundColor: providerTests["gemini"].ok ? 'rgba(72,187,120,0.1)' : 'rgba(245,101,101,0.1)' }}>
                <strong>Status:</strong> {providerTests["gemini"].ok ? "Success" : "Failed"}<br/>
                {providerTests["gemini"].latency_ms && <><strong>Latency:</strong> {providerTests["gemini"].latency_ms.toFixed(0)} ms<br/></>}
                <strong>Checked:</strong> {providerTests["gemini"].time}<br/>
                <small className="muted">{providerTests["gemini"].detail}</small>
              </div>
            )}

            <div style={{ display: 'flex', gap: '0.5rem', marginTop: 'auto' }}>
              <button className="icon-text" onClick={() => handleTestConnection("gemini")} type="button" style={{ flex: 1 }}>
                <Activity size={18} />
                <span>Test Gemini</span>
              </button>
              {draft.gemini_api_key && (
                <button className="icon-text danger" onClick={() => handleDeleteKey("gemini")} type="button" title="Delete API Key" style={{ padding: '0.5rem' }}>
                  <Trash2 size={18} />
                </button>
              )}
            </div>
          </article>

          {/* ElevenLabs Card */}
          <article className={`voice-card ${draft.elevenlabs_api_key ? 'ready' : 'missing'}`} style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', padding: '1.25rem' }}>
            <div className="panel-heading">
              <h3>4. ElevenLabs</h3>
              <span className={`pill ${draft.elevenlabs_api_key ? 'good' : ''}`}>{draft.elevenlabs_api_key ? 'Configured' : 'Needs setup'}</span>
            </div>
            <p className="muted" style={{ fontSize: '0.8rem' }}>Utilize ElevenLabs cloud API for premium voice cloning and bilingual English/Nepali synthesis.</p>
            
            <label style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a0aec0' }}>API Key</span>
              <input
                type="password"
                placeholder={draft.elevenlabs_api_key ? "••••••••••••••••" : "Enter ElevenLabs API Key"}
                value={draft.elevenlabs_api_key}
                onChange={(e) => setDraft({ ...draft, elevenlabs_api_key: e.target.value })}
              />
            </label>

            {providerTests["elevenlabs"] && (
              <div style={{ fontSize: '0.8rem', padding: '0.5rem', borderRadius: '4px', backgroundColor: providerTests["elevenlabs"].ok ? 'rgba(72,187,120,0.1)' : 'rgba(245,101,101,0.1)' }}>
                <strong>Status:</strong> {providerTests["elevenlabs"].ok ? "Success" : "Failed"}<br/>
                {providerTests["elevenlabs"].latency_ms && <><strong>Latency:</strong> {providerTests["elevenlabs"].latency_ms.toFixed(0)} ms<br/></>}
                <strong>Checked:</strong> {providerTests["elevenlabs"].time}<br/>
                <small className="muted">{providerTests["elevenlabs"].detail}</small>
              </div>
            )}

            <div style={{ display: 'flex', gap: '0.5rem', marginTop: 'auto' }}>
              <button className="icon-text" onClick={() => handleTestConnection("elevenlabs")} type="button" style={{ flex: 1 }}>
                <Activity size={18} />
                <span>Test ElevenLabs</span>
              </button>
              {draft.elevenlabs_api_key && (
                <button className="icon-text danger" onClick={() => handleDeleteKey("elevenlabs")} type="button" title="Delete API Key" style={{ padding: '0.5rem' }}>
                  <Trash2 size={18} />
                </button>
              )}
            </div>
          </article>
        </div>

        <div className="settings-grid" style={{ marginTop: '1.5rem', padding: '1.25rem', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', backgroundColor: 'rgba(255,255,255,0.02)' }}>
          <h4 style={{ gridColumn: '1 / -1', margin: 0, fontSize: '0.9rem', color: '#fff' }}>Cloud Provider Routing & Fallback</h4>
          
          <label className="toggle-line" style={{ gridColumn: '1 / -1' }}>
            <input
              checked={draft.cloud_fallback_to_local}
              onChange={(e) => setDraft({ ...draft, cloud_fallback_to_local: e.target.checked })}
              type="checkbox"
            />
            <span>Fallback to Local Ollama if Cloud fails</span>
          </label>

          <label>
            <span>Cloud Temperature</span>
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={draft.cloud_temperature}
              onChange={(e) => setDraft({ ...draft, cloud_temperature: Number(e.target.value) })}
            />
          </label>

          <label>
            <span>Cloud Max Tokens</span>
            <input
              type="number"
              min="32"
              max="2048"
              step="16"
              value={draft.cloud_max_tokens}
              onChange={(e) => setDraft({ ...draft, cloud_max_tokens: Number(e.target.value) })}
            />
          </label>

          <label>
            <span>Cloud Timeout (Seconds)</span>
            <input
              type="number"
              min="1"
              max="120"
              step="1"
              value={draft.cloud_timeout_seconds}
              onChange={(e) => setDraft({ ...draft, cloud_timeout_seconds: Number(e.target.value) })}
            />
          </label>
        </div>
      </section>
      <section className="voice-registry">
        <div className="panel-heading">
          <h2>Voices</h2>
          <span className="muted">No silent downloads</span>
        </div>
        <div className="voice-card-grid">
          {voices ? (
            <>
              <VoiceCard label="Nepali voice" voice={voices.selected.nepali} onTest={() => onTtsTest("ne")} />
              <VoiceCard label="English voice" voice={voices.selected.english} onTest={() => onTtsTest("en")} />
            </>
          ) : (
            <p className="muted">Voice registry unavailable until the backend responds.</p>
          )}
        </div>
      </section>
      <div className="danger-band">
        <button className="icon-text danger" onClick={onDelete} type="button">
          <Trash2 size={18} />
          <span>Delete local data</span>
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
  return `${Math.round(value)} ms`;
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
