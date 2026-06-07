import type { QualityScore } from "./types";

export async function blobToBase64(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return window.btoa(binary);
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export async function scoreRecording(blob: Blob): Promise<{ wavBlob: Blob; quality: QualityScore }> {
  const AudioContextClass = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("AudioContext is not available in this browser.");
  }
  const context = new AudioContextClass();
  const buffer = await context.decodeAudioData(await blob.arrayBuffer());
  const samples = buffer.getChannelData(0);
  let peak = 0;
  let sumSquares = 0;
  let clipped = 0;
  for (const sample of samples) {
    const abs = Math.abs(sample);
    peak = Math.max(peak, abs);
    sumSquares += sample * sample;
    if (abs >= 0.98) {
      clipped += 1;
    }
  }
  const durationSeconds = buffer.duration;
  const rms = Math.sqrt(sumSquares / Math.max(samples.length, 1));
  let score = 100;
  const reasons: string[] = [];
  if (durationSeconds < 1.4) {
    score -= 38;
    reasons.push("too short");
  }
  if (durationSeconds > 12) {
    score -= 24;
    reasons.push("too long");
  }
  if (rms < 0.015) {
    score -= 30;
    reasons.push("quiet");
  }
  if (peak > 0.98 || clipped / Math.max(samples.length, 1) > 0.002) {
    score -= 35;
    reasons.push("clipped");
  }
  score = Math.max(0, Math.min(100, Math.round(score)));
  const verdict = score >= 82 ? "good" : score >= 62 ? "review" : "reject";
  const wavBlob = audioBufferToWav(buffer);
  await context.close();
  return {
    wavBlob,
    quality: {
      score,
      verdict,
      durationSeconds,
      peak,
      rms,
      reason: reasons.length ? reasons.join(", ") : "clean"
    }
  };
}

export function audioBufferToWav(buffer: AudioBuffer): Blob {
  const numChannels = 1;
  const sampleRate = buffer.sampleRate;
  const samples = buffer.getChannelData(0);
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = samples.length * bytesPerSample;
  const wav = new ArrayBuffer(44 + dataSize);
  const view = new DataView(wav);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, value: string): void {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}
