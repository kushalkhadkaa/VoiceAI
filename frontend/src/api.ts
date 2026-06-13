import type { BackendSettings, ConversationTurn, KBCollection, KBDocument, KBSearchResult, KBStatus, ProviderStatus, RagStatus, SystemInfo, SystemMetrics, VoiceSocketStatus, VoicesResponse, DatasetRecording } from "./types";

export const API_HTTP = import.meta.env.VITE_API_HTTP ?? "http://localhost:8001";

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
    return "ws://localhost:8001/ws/voice";
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
    stt_language?: string;
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

export async function voiceTurnRest(
  audioBlob: Blob,
  options?: { voice_id?: string; knowledge_id?: string; use_internet?: boolean; llm_provider_id?: string; stt_provider_id?: string; stt_language?: string }
): Promise<ConversationTurn> {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  if (options?.voice_id) form.append("voice_id", options.voice_id);
  if (options?.knowledge_id) form.append("knowledge_id", options.knowledge_id);
  form.append("use_internet", String(options?.use_internet ?? false));
  if (options?.llm_provider_id) form.append("llm_provider_id", options.llm_provider_id);
  if (options?.stt_provider_id) form.append("stt_provider_id", options.stt_provider_id);
  if (options?.stt_language) form.append("stt_language", options.stt_language);
  const response = await fetch(`${API_HTTP}/voice/turn`, { method: "POST", body: form });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "Voice turn failed.");
  }
  return response.json();
}



export async function previewTts(text: string, voiceId: string, language = "en"): Promise<{ ok: boolean; audio_url?: string; detail?: string; engine?: string }> {
  const response = await fetch(`${API_HTTP}/tts/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice_id: voiceId, language })
  });
  return response.json();
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

export async function getCloningEngines(): Promise<Record<string, any>> {
  const response = await fetch(`${API_HTTP}/voices/cloning-engines`);
  if (!response.ok) {
    throw new Error("Unable to read voice-cloning engine status.");
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

// ============================================================
// Local Knowledge Base API  (/kb/*)
// ============================================================

export async function getKBStatus(): Promise<KBStatus | null> {
  const r = await fetch(`${API_HTTP}/kb/status`);
  return r.ok ? r.json() : null;
}

export async function getKBCollections(): Promise<KBCollection[]> {
  const r = await fetch(`${API_HTTP}/kb/collections`);
  if (!r.ok) return [];
  const data = await r.json();
  return data.collections ?? [];
}

export async function createKBCollection(name: string, description: string): Promise<KBCollection> {
  const r = await fetch(`${API_HTTP}/kb/collections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description }),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => null);
    throw new Error(d?.detail ?? "Failed to create collection.");
  }
  return r.json();
}

export async function deleteKBCollection(collectionId: string): Promise<void> {
  const r = await fetch(`${API_HTTP}/kb/collections/${collectionId}`, { method: "DELETE" });
  if (!r.ok) throw new Error("Failed to delete collection.");
}

export async function getKBDocuments(collectionId: string): Promise<KBDocument[]> {
  const r = await fetch(`${API_HTTP}/kb/collections/${collectionId}/documents`);
  if (!r.ok) return [];
  const data = await r.json();
  return data.documents ?? [];
}

export async function ingestKBFile(collectionId: string, file: File): Promise<KBDocument> {
  const form = new FormData();
  form.append("file", file, file.name);
  const r = await fetch(`${API_HTTP}/kb/collections/${collectionId}/ingest`, {
    method: "POST",
    body: form,
  });
  const data = await r.json();
  if (!r.ok || !data.ok) throw new Error(data?.detail ?? "Ingest failed.");
  return data.document;
}

export async function ingestKBUrl(collectionId: string, url: string): Promise<KBDocument> {
  const r = await fetch(`${API_HTTP}/kb/collections/${collectionId}/ingest-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  const data = await r.json();
  if (!r.ok || !data.ok) throw new Error(data?.detail ?? "URL ingest failed.");
  return data.document;
}

export async function deleteKBDocument(collectionId: string, docId: string): Promise<void> {
  const r = await fetch(`${API_HTTP}/kb/collections/${collectionId}/documents/${docId}`, { method: "DELETE" });
  if (!r.ok) throw new Error("Failed to delete document.");
}

export async function renameKBDocument(collectionId: string, docId: string, filename: string): Promise<{ ok: boolean }> {
  const res = await fetch(`${API_HTTP}/kb/collections/${collectionId}/documents/${docId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename })
  });
  return res.json();
}

export async function getKBDocumentChunks(collectionId: string, docId: string, limit = 10): Promise<{ ok: boolean; chunks: Array<{ text: string; chunk_index: number }> }> {
  const res = await fetch(`${API_HTTP}/kb/collections/${collectionId}/documents/${docId}/chunks?limit=${limit}`);
  return res.json();
}

export async function queryKB(query: string, collectionIds?: string[], nResults?: number): Promise<{ ok: boolean; results: KBSearchResult[]; context: string }> {
  const r = await fetch(`${API_HTTP}/kb/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, collection_ids: collectionIds ?? null, n_results: nResults ?? 5 }),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => null);
    throw new Error(d?.detail ?? "Query failed.");
  }
  return r.json();
}

export async function getKBEmbeddingStatus(): Promise<any> {
  const r = await fetch(`${API_HTTP}/kb/embedding/status`);
  return r.ok ? r.json() : null;
}

export interface AdvancedQueryOptions {
  query: string;
  collection_ids?: string[];
  n_results?: number;
  mode?: "semantic" | "keyword" | "hybrid";
  source_type_filter?: "file" | "url";
  rerank?: boolean;
  min_score?: number;
  doc_id_filter?: string;
}

export async function queryKBAdvanced(opts: AdvancedQueryOptions): Promise<{
  ok: boolean; query: string; mode: string; reranked: boolean;
  elapsed_ms: number; result_count: number; results: KBSearchResult[]; context: string;
}> {
  const r = await fetch(`${API_HTTP}/kb/query/advanced`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => null);
    throw new Error(d?.detail ?? "Advanced query failed.");
  }
  return r.json();
}

export async function getKBAnalytics(limit = 100): Promise<{
  ok: boolean; total_queries: number; avg_results: number; avg_latency_ms: number;
  zero_result_queries: number; top_queries: Array<{ query: string; count: number }>;
  recent: any[];
}> {
  const r = await fetch(`${API_HTTP}/kb/analytics?limit=${limit}`);
  return r.ok ? r.json() : { ok: false, total_queries: 0, avg_results: 0, avg_latency_ms: 0, zero_result_queries: 0, top_queries: [], recent: [] };
}

export async function clearKBAnalytics(): Promise<void> {
  await fetch(`${API_HTTP}/kb/analytics`, { method: "DELETE" });
}

export async function getKBCollectionStats(collectionId: string): Promise<any> {
  const r = await fetch(`${API_HTTP}/kb/collections/${collectionId}/stats`);
  return r.ok ? r.json() : null;
}

export async function exportKBCollection(collectionId: string): Promise<any> {
  const r = await fetch(`${API_HTTP}/kb/collections/${collectionId}/export`);
  return r.ok ? r.json() : null;
}

// ── RAG Evaluation ──
export interface EvalQA { q: string; a: string }
export async function kbEvalGenerate(opts: {
  collection_id: string; document_id?: string | null; n: number; gen_model?: string;
}): Promise<{ ok: boolean; count: number; questions: EvalQA[]; gen_model: string }> {
  const r = await fetch(`${API_HTTP}/kb/eval/generate`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(opts),
  });
  if (!r.ok) { const d = await r.json().catch(() => null); throw new Error(d?.detail ?? "Question generation failed."); }
  return r.json();
}

export interface EvalRow {
  question: string; reference: string; answer: string; rag_used: boolean;
  verdict: "correct" | "partial" | "incorrect" | "error"; reason: string; latency_s: number; audio_url: string | null;
}
export async function kbEvalRun(opts: {
  collection_id: string; document_id?: string | null; questions: EvalQA[];
  answer_model: string; verify_model: string; temperature: number; voice_id?: string | null;
}): Promise<{ ok: boolean; answer_model: string; verify_model: string; total: number; accuracy: number;
  counts: Record<string, number>; results: EvalRow[] }> {
  const r = await fetch(`${API_HTTP}/kb/eval/run`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(opts),
  });
  if (!r.ok) { const d = await r.json().catch(() => null); throw new Error(d?.detail ?? "Evaluation failed."); }
  return r.json();
}

// Chat scoped to a specific document (or whole collection) with model + temperature control.
export async function chatWithDocument(opts: {
  text: string; collection_id: string; document_id?: string | null;
  llm_provider_id: string; temperature: number; voice_id?: string | null;
}): Promise<any> {
  const r = await fetch(`${API_HTTP}/chat/test`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: opts.text, knowledge_id: opts.collection_id, document_id: opts.document_id || undefined,
      llm_provider_id: opts.llm_provider_id, temperature: opts.temperature, voice_id: opts.voice_id || undefined,
    }),
  });
  if (!r.ok) { const d = await r.json().catch(() => null); throw new Error(d?.detail ?? "Chat failed."); }
  return r.json();
}

// ============================================================
// Legacy RAG (Open WebUI) API
// ============================================================

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

export async function setActiveAIProvider(provider: string): Promise<{ ok: boolean; active_provider: string }> {
  const r = await fetch(`${API_HTTP}/ai-providers/set-active`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  if (!r.ok) {
    const d = await r.json().catch(() => null);
    throw new Error(d?.detail ?? "Failed to set active provider.");
  }
  return r.json();
}

export async function getOpenAIModels(): Promise<{ ok: boolean; models: string[]; current: string }> {
  const r = await fetch(`${API_HTTP}/ai-providers/openai/models`);
  return r.ok ? r.json() : { ok: false, models: [], current: "" };
}

export interface CrawlSiteOptions {
  url: string;
  max_pages?: number;
  same_domain_only?: boolean;
  delay_ms?: number;
}

export type CrawlEvent =
  | { status: "crawling"; url: string; page: number; total_queued: number; ingested: number; failed: number }
  | { status: "saved"; url: string; doc_id: string; chunks: number; page: number; total_queued: number; ingested: number; failed: number }
  | { status: "skipped"; url: string; reason: string; ingested: number; failed: number }
  | { status: "error"; url: string; error: string; ingested: number; failed: number }
  | { status: "done"; start_url: string; ingested: number; failed: number; total_visited: number }
  | { status: "fatal"; error: string };

export function crawlKBSite(
  collectionId: string,
  opts: CrawlSiteOptions,
  onEvent: (evt: CrawlEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return fetch(`${API_HTTP}/kb/collections/${collectionId}/crawl-site`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
    signal,
  }).then(async (resp) => {
    if (!resp.ok) {
      const d = await resp.json().catch(() => null);
      throw new Error(d?.detail ?? "Crawl failed to start.");
    }
    const reader = resp.body!.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ")) {
          try { onEvent(JSON.parse(trimmed.slice(6))); } catch {}
        }
      }
    }
  });
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


// ── Admin authentication ────────────────────────────────────────────────────
export interface AdminLoginResult {
  ok: boolean;
  token?: string;
  username?: string;
  role?: string;
  detail?: string;
}

export async function adminLogin(username: string, password: string): Promise<AdminLoginResult> {
  const response = await fetch(`${API_HTTP}/admin/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    return { ok: false, detail: payload?.detail ?? "Invalid username or password." };
  }
  return payload as AdminLoginResult;
}

export async function adminLogout(token: string): Promise<{ ok: boolean }> {
  const response = await fetch(`${API_HTTP}/admin/logout`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token })
  });
  if (!response.ok) {
    return { ok: false };
  }
  return response.json();
}

export async function adminVerify(token: string): Promise<boolean> {
  const response = await fetch(`${API_HTTP}/admin/verify?token=${encodeURIComponent(token)}`);
  if (!response.ok) {
    return false;
  }
  const payload = await response.json().catch(() => null);
  return Boolean(payload?.ok);
}

// ── KYC natural-language query ──────────────────────────────────────────────
export interface KycQueryResult {
  ok: boolean;
  sql?: string;
  rows?: Record<string, unknown>[];
  row_count?: number;
  answer?: string;
  error?: string;
}

export async function kycQuery(question: string): Promise<KycQueryResult> {
  const response = await fetch(`${API_HTTP}/kyc/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    return { ok: false, error: payload?.error ?? payload?.detail ?? `KYC query failed (${response.status}).` };
  }
  return payload as KycQueryResult;
}

export async function kycStatus(): Promise<{ ok: boolean; connected: boolean; row_count: number }> {
  const response = await fetch(`${API_HTTP}/kyc/status`);
  if (!response.ok) {
    throw new Error("Unable to read KYC status.");
  }
  return response.json();
}
