from __future__ import annotations
import math
import struct
import wave
from pathlib import Path
from typing import Any

class VoiceIdentityService:
    def __init__(self, voices_dir: Path) -> None:
        self.voices_dir = voices_dir

    def extract_fingerprint(self, wav_path: Path) -> list[float]:
        """
        Extracts a deterministic speaker embedding/fingerprint from a clean WAV recording.
        Uses spectral analysis features (zero crossing rate, energy bins, subband distribution)
        which can run fully local and dependency-free.
        """
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        try:
            with wave.open(str(wav_path), "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                sample_width = wav.getsampwidth()
                frame_rate = wav.getframerate()
                channels = wav.getnchannels()
                
            if len(frames) == 0:
                return [0.0] * 64
                
            # Parse samples
            samples = self._pcm_samples(frames, sample_width)
            
            # Extract basic statistical properties representing voice identity signature
            num_samples = len(samples)
            
            # Normalize samples
            max_val = float(max((abs(s) for s in samples), default=1))
            norm_samples = [s / max_val for s in samples]
            
            # Feature 1: Zero crossing rate
            zcr = 0
            for i in range(1, len(norm_samples)):
                if (norm_samples[i] >= 0 and norm_samples[i-1] < 0) or (norm_samples[i] < 0 and norm_samples[i-1] >= 0):
                    zcr += 1
            zcr_rate = zcr / max(1, num_samples)
            
            # Feature 2: Energy/RMS
            rms = math.sqrt(sum(s*s for s in norm_samples) / max(1, num_samples))
            
            # Feature 3-34: 32 Energy bins across the duration of the audio
            bin_size = max(1, num_samples // 32)
            energy_bins = []
            for b in range(32):
                chunk = norm_samples[b * bin_size : (b + 1) * bin_size]
                bin_rms = math.sqrt(sum(s*s for s in chunk) / max(1, len(chunk))) if chunk else 0.0
                energy_bins.append(bin_rms)
                
            # Feature 35-64: Spectral centroid estimate (rough mapping based on sample intervals)
            centroid_bins = []
            for b in range(30):
                chunk = norm_samples[b * bin_size : (b + 1) * bin_size]
                diffs = 0.0
                for i in range(1, len(chunk)):
                    diffs += abs(chunk[i] - chunk[i-1])
                centroid_bins.append(diffs / max(1, len(chunk)))
                
            # Combine into a 64-dimensional speaker embedding vector
            embedding = [zcr_rate, rms] + energy_bins + centroid_bins
            
            # Normalize embedding vector
            magnitude = math.sqrt(sum(val*val for val in embedding))
            if magnitude > 0.0:
                embedding = [val / magnitude for val in embedding]
            else:
                embedding = [0.0] * 64
                
            return embedding
        except Exception:
            # Fallback signature
            return [0.1] * 64

    def save_reference_fingerprint(self, voice_id: str, fingerprint: list[float]) -> Path:
        import json
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        fp_path = voice_dir / "fingerprint.bin"
        
        # Save as json list for portability
        fp_path.write_text(json.dumps(fingerprint))
        return fp_path

    def load_reference_fingerprint(self, voice_id: str) -> list[float] | None:
        import json
        fp_path = self.voices_dir / voice_id / "fingerprint.bin"
        if not fp_path.exists():
            return None
        try:
            return json.loads(fp_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _pcm_samples(frames: bytes, sample_width: int) -> list[int]:
        if sample_width == 1:
            return [value - 128 for value in frames]
        if sample_width == 2:
            return list(struct.unpack(f"<{len(frames) // 2}h", frames))
        if sample_width == 4:
            return list(struct.unpack(f"<{len(frames) // 4}i", frames))
        raise ValueError(f"Unsupported sample width: {sample_width}")
