from __future__ import annotations
import math
import struct
import wave
from pathlib import Path
from typing import Any
from app.services.voice_identity import VoiceIdentityService

class SpeakerIsolationService:
    def __init__(self, identity_service: VoiceIdentityService) -> None:
        self.identity_service = identity_service

    def verify_speaker_match(self, voice_id: str, sample_wav_path: Path, lock_if_empty: bool = True) -> dict[str, Any]:
        """
        Verifies if the sample sounds like the locked main speaker.
        """
        # Load reference speaker embedding
        ref_fp = self.identity_service.load_reference_fingerprint(voice_id)
        current_fp = self.identity_service.extract_fingerprint(sample_wav_path)
        
        if not ref_fp:
            if lock_if_empty:
                # Lock this speaker
                self.identity_service.save_reference_fingerprint(voice_id, current_fp)
                return {
                    "matched": True,
                    "score": 100.0,
                    "reason": "Main voice lock established from first sample.",
                    "multiple_speakers": False,
                    "background_speech": False
                }
            return {
                "matched": False,
                "score": 0.0,
                "reason": "No main voice lock established.",
                "multiple_speakers": False,
                "background_speech": False
            }

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(ref_fp, current_fp))
        mag1 = math.sqrt(sum(a * a for a in ref_fp))
        mag2 = math.sqrt(sum(b * b for b in current_fp))
        
        similarity = dot_product / (mag1 * mag2) if (mag1 > 0 and mag2 > 0) else 0.0
        
        # Check background speech and multiple speakers based on spectral properties
        multiple_speakers = False
        background_speech = False
        reasons = []
        
        # Parse wav properties
        try:
            with wave.open(str(sample_wav_path), "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                sample_width = wav.getsampwidth()
            
            # Simple multiple speaker heuristic: unexpected high frequency variance in energy bins
            if len(frames) > 0:
                samples = self.identity_service._pcm_samples(frames, sample_width)
                max_val = max((abs(s) for s in samples), default=1)
                norm_samples = [s / max_val for s in samples]
                
                # Compute variance of segment envelopes
                segment_len = max(1, len(norm_samples) // 8)
                envelopes = []
                for idx in range(8):
                    segment = norm_samples[idx * segment_len : (idx + 1) * segment_len]
                    env_val = sum(abs(s) for s in segment) / max(1, len(segment))
                    envelopes.append(env_val)
                    
                # High peaks variance could mean overlapping or alternating voices
                mean_env = sum(envelopes) / len(envelopes)
                variance = sum((e - mean_env)**2 for e in envelopes) / len(envelopes)
                
                if similarity < 0.65:
                    multiple_speakers = True
                    reasons.append("This does not sound like the selected owner")
                elif similarity < 0.80:
                    background_speech = True
                    reasons.append("Background speech detected")
                
                if variance > 0.08:
                    multiple_speakers = True
                    reasons.append("Multiple voices detected")
        except Exception:
            pass

        matched = similarity >= 0.78 and not multiple_speakers
        if matched:
            reason = "Speaker verified"
        else:
            reason = "; ".join(reasons) if reasons else "Please retry in a quieter place"

        return {
            "matched": matched,
            "score": round(similarity * 100, 1),
            "reason": reason,
            "multiple_speakers": multiple_speakers,
            "background_speech": background_speech
        }
