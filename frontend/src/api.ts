import type { BackendSettings, ConversationTurn, ProviderStatus, RagStatus, SystemInfo, SystemMetrics, VoiceSocketStatus, VoicesResponse, DatasetRecording } from "./types";

export const API_HTTP = import.meta.env.VITE_API_HTTP ?? "http://localhost:8000";

export function deriveVoiceSocketUrl(httpUrl = API_HTTP): string {
  if (import.meta.env.VITE_API_WS) {
    return import.meta.env.VITE_API_WS;
  }
  try {
    const url = new URL(httpUrl);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = "/ws/voice";
    url.search = "";
    url.hash = "";
    return url.toString();
  } catch {
    return "ws://localhost:8000/ws/voice";
  }
}

export const API_WS = deriveVoiceSocketUrl();

export async function getHealth(): Promise<boolean> {
  const response = await fetch(`${API_HTTP}/health`);
  return response.ok;
}

export async function getModelStatus(): Promise<ProviderStatus[]> {
  const response = await fetch(`${API_HTTP}/models/status`);
  if (!response.ok) {
    throw new Error("Unable to read model status.");
  }
  const payload = await response.json();
  return payload.providers;
}

export async function getDoctor(): Promise<{ ready: boolean; checks: ProviderStatus[] }> {
  const response = await fetch(`${API_HTTP}/doctor`);
  if (!response.ok) {
    throw new Error("Unable to run doctor.");
  }
  return response.json();
}

export async function getVoiceSocketStatus(): Promise<VoiceSocketStatus> {
  const response = await fetch(`${API_HTTP}/ws/voice/status`);
  if (!response.ok) {
    throw new Error("Unable to read voice socket status.");
  }
  return response.json();
}

export async function getSystemMetrics(): Promise<SystemMetrics | null> {
  const response = await fetch(`${API_HTTP}/system/metrics`);
  if (!response.ok) {
    return null;
  }
  return response.json();
}

export async function getSystemInfo(): Promise<SystemInfo | null> {
  const response = await fetch(`${API_HTTP}/system/info`);
  if (!response.ok) {
    return null;
  }
  return response.json();
}

export async function getVoices(): Promise<VoicesResponse> {
  const response = await fetch(`${API_HTTP}/voices`);
  if (!response.ok) {
    throw new Error("Unable to read voices.");
  }
  return response.json();
}

export async function getSettings(): Promise<BackendSettings> {
  const response = await fetch(`${API_HTTP}/settings`);
  if (!response.ok) {
    throw new Error("Unable to read settings.");
  }
  const payload = await response.json();
  return payload.settings;
}

export async function updateSettings(settings: Partial<BackendSettings>): Promise<BackendSettings> {
  const response = await fetch(`${API_HTTP}/settings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings)
  });
  if (!response.ok) {
    throw new Error("Unable to save settings.");
  }
  const payload = await response.json();
  return payload.settings;
}

export async function sendTextTurn(
  text: string,
  options?: {
    voice_id?: string;
    knowledge_id?: string;
    use_internet?: boolean;
    llm_provider_id?: string;
  }
): Promise<ConversationTurn> {
  const response = await fetch(`${API_HTTP}/chat/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, ...options })
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "Chat test failed.");
  }
  return response.json();
}

export async function testTts(text: string, language: "ne" | "en") {
  const response = await fetch(`${API_HTTP}/tts/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language })
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "TTS test failed.");
  }
  return response.json() as Promise<{ language: string; audio_url: string }>;
}

export async function deleteLocalData(): Promise<void> {
  const response = await fetch(`${API_HTTP}/local-data`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error("Unable to delete local data.");
  }
}

export function absoluteAudioUrl(audioUrl?: string | null): string | null {
  if (!audioUrl) {
    return null;
  }
  if (audioUrl.startsWith("http")) {
    return audioUrl;
  }
  return `${API_HTTP}${audioUrl}`;
}

export async function getDatasetRecordings(): Promise<DatasetRecording[]> {
  const response = await fetch(`${API_HTTP}/dataset/recordings`);
  if (!response.ok) {
    throw new Error("Unable to read dataset recordings.");
  }
  const payload = await response.json();
  return payload.map((item: any) => ({
    id: item.id,
    text: item.text,
    language: item.language,
    url: absoluteAudioUrl(item.audio_url)!,
    quality: item.quality ? {
      score: item.quality.score,
      verdict: item.quality.verdict,
      durationSeconds: item.quality.duration_seconds,
      peak: item.quality.peak,
      rms: item.quality.rms,
      reason: item.quality.reason,
    } : null
  })).filter((item: any) => item.quality !== null);
}

export async function uploadDatasetRecording(promptId: string, blob: Blob): Promise<DatasetRecording> {
  const formData = new FormData();
  formData.append("upload", blob, `${promptId}.wav`);
  const response = await fetch(`${API_HTTP}/dataset/recordings/${promptId}`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "Unable to upload dataset recording.");
  }
  const item = await response.json();
  return {
    id: item.id,
    text: item.text,
    language: item.language,
    url: absoluteAudioUrl(item.audio_url)!,
    quality: {
      score: item.quality.score,
      verdict: item.quality.verdict,
      durationSeconds: item.quality.duration_seconds,
      peak: item.quality.peak,
      rms: item.quality.rms,
      reason: item.quality.reason,
    }
  };
}

export async function deleteDatasetRecording(promptId: string): Promise<void> {
  const response = await fetch(`${API_HTTP}/dataset/recordings/${promptId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error("Unable to delete dataset recording.");
  }
}

export function deriveDatasetExportUrl(): string {
  return `${API_HTTP}/dataset/export`;
}


// Custom upgrade Voice Studio APIs
export async function getVoicesGallery(): Promise<any[]> {
  const response = await fetch(`${API_HTTP}/voices/gallery`);
  if (!response.ok) {
    throw new Error("Unable to read voices gallery.");
  }
  return response.json();
}

export async function createVoice(payload: {
  name: string;
  owner_name: string;
  owner_email?: string;
  organization?: string;
  language: string;
  engine: string;
  commercial_allowed: boolean;
}): Promise<any> {
  const response = await fetch(`${API_HTTP}/voices/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error("Unable to create voice profile.");
  }
  return response.json();
}

export async function saveVoiceConsent(voiceId: string, signature: string, spokenConsentBlob?: Blob | null): Promise<void> {
  const formData = new FormData();
  formData.append("signature", signature);
  if (spokenConsentBlob) {
    formData.append("spoken_consent", spokenConsentBlob, "consent.wav");
  }
  const response = await fetch(`${API_HTTP}/voices/${voiceId}/consent`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw new Error("Unable to save voice consent.");
  }
}

export async function getVoiceRecordings(voiceId: string): Promise<any[]> {
  const response = await fetch(`${API_HTTP}/voices/${voiceId}/recordings`);
  if (!response.ok) {
    throw new Error("Unable to get voice recordings.");
  }
  return response.json();
}

export async function uploadVoiceRecording(voiceId: string, promptId: string, blob: Blob): Promise<any> {
  const formData = new FormData();
  formData.append("upload", blob, `${promptId}.wav`);
  const response = await fetch(`${API_HTTP}/voices/${voiceId}/recordings/${promptId}`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "Unable to upload recording.");
  }
  return response.json();
}

export async function deleteVoiceRecording(voiceId: string, promptId: string): Promise<void> {
  const response = await fetch(`${API_HTTP}/voices/${voiceId}/recordings/${promptId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error("Unable to delete recording.");
  }
}

export async function cleanRecording(voiceId: string, promptId: string): Promise<any> {
  const response = await fetch(`${API_HTTP}/voices/${voiceId}/recordings/${promptId}/clean`, {
    method: "POST"
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "Unable to clean recording.");
  }
  return response.json();
}

export async function cloneVoice(voiceId: string): Promise<any> {
  const response = await fetch(`${API_HTTP}/voices/${voiceId}/clone`, {
    method: "POST"
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "Unable to clone voice.");
  }
  return response.json();
}

export async function publishVoice(voiceId: string): Promise<any> {
  const response = await fetch(`${API_HTTP}/voices/${voiceId}/publish`, {
    method: "POST"
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "Unable to publish voice.");
  }
  return response.json();
}

export async function deleteVoice(voiceId: string): Promise<void> {
  const response = await fetch(`${API_HTTP}/voices/${voiceId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error("Unable to delete voice profile.");
  }
}

export async function getVoicesPrompts(): Promise<any[]> {
  const response = await fetch(`${API_HTTP}/voices/prompts`);
  if (!response.ok) {
    throw new Error("Unable to get voice prompts.");
  }
  return response.json();
}

export async function getAuditLogs(): Promise<any[]> {
  const response = await fetch(`${API_HTTP}/audit/logs`);
  if (!response.ok) {
    throw new Error("Unable to read audit logs.");
  }
  return response.json();
}

export async function downloadDefaultVoices(): Promise<void> {
  const response = await fetch(`${API_HTTP}/voices/download`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error("Unable to start downloading default voices.");
  }
}

export async function getRagCollections(): Promise<any[]> {
  const response = await fetch(`${API_HTTP}/rag/collections`);
  if (!response.ok) {
    throw new Error("Unable to read RAG collections.");
  }
  const payload = await response.json();
  return payload.collections ?? [];
}

export async function getRagStatus(): Promise<RagStatus | null> {
  const response = await fetch(`${API_HTTP}/rag/status`);
  if (!response.ok) {
    return null;
  }
  return response.json();
}

export async function testRagQuestion(query: string, collectionId?: string): Promise<any> {
  const response = await fetch(`${API_HTTP}/rag/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, collection_id: collectionId || null })
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "Knowledge test failed.");
  }
  return response.json();
}

export async function getAiProviders(): Promise<{ providers: any[]; active_provider: string }> {
  const response = await fetch(`${API_HTTP}/ai-providers`);
  if (!response.ok) {
    throw new Error("Unable to read AI providers.");
  }
  return response.json();
}

export async function getAiProvidersStatus(): Promise<Record<string, { available: boolean; models: string[] }>> {
  const response = await fetch(`${API_HTTP}/ai-providers/status`);
  if (!response.ok) {
    throw new Error("Unable to read AI providers status.");
  }
  return response.json();
}

export async function testProvider(provider: string, payload?: any): Promise<{ ok: boolean; detail: string; model?: string; latency_ms?: number }> {
  const response = await fetch(`${API_HTTP}/ai-providers/test/${provider}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload ? JSON.stringify(payload) : undefined
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.detail ?? "Connection test failed.");
  }
  return response.json();
}

export async function deleteOpenAIKey(): Promise<BackendSettings> {
  const response = await fetch(`${API_HTTP}/settings/openai-key`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error("Unable to delete OpenAI API key.");
  }
  const payload = await response.json();
  return payload.settings;
}

export async function deleteGeminiKey(): Promise<BackendSettings> {
  const response = await fetch(`${API_HTTP}/settings/gemini-key`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error("Unable to delete Gemini API key.");
  }
  const payload = await response.json();
  return payload.settings;
}

export async function deleteElevenLabsKey(): Promise<BackendSettings> {
  const response = await fetch(`${API_HTTP}/settings/elevenlabs-key`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error("Unable to delete ElevenLabs API key.");
  }
  const payload = await response.json();
  return payload.settings;
}

export async function getChatHistory(limit = 50): Promise<ConversationTurn[]> {
  const response = await fetch(`${API_HTTP}/chat/history?limit=${limit}`);
  if (!response.ok) {
    throw new Error("Unable to load chat history.");
  }
  const payload = await response.json();
  // Ensure audio urls are absolute
  return payload.map((turn: any) => ({
    ...turn,
    audio_url: absoluteAudioUrl(turn.audio_url)
  }));
}

export async function rateChatTurn(
  turnId: string,
  ratings: {
    naturalness?: number;
    voiceSimilarity?: number;
    nepaliPronunciation?: number;
    englishPronunciation?: number;
  }
): Promise<void> {
  // Map camelCase to snake_case for the backend
  const payload = {
    naturalness: ratings.naturalness,
    voice_similarity: ratings.voiceSimilarity,
    nepali_pronunciation: ratings.nepaliPronunciation,
    english_pronunciation: ratings.englishPronunciation
  };
  const response = await fetch(`${API_HTTP}/chat/turns/${turnId}/rate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error("Unable to save rating.");
  }
}

