from __future__ import annotations
import math
import struct
from typing import Literal

InputMode = Literal["ptt", "auto", "hybrid"]
NoiseMode = Literal["quiet", "normal", "noisy", "custom"]

class TurnDetector:
    def __init__(
        self,
        input_mode: InputMode = "auto",
        silence_timeout_ms: int = 700,
        max_duration_seconds: float = 30.0,
        min_speech_duration_ms: int = 250,
        noise_gate_threshold: float = 300.0,
        vad_aggressiveness: int = 2,
        barge_in: bool = True
    ) -> None:
        self.input_mode = input_mode
        self.silence_timeout_ms = silence_timeout_ms
        self.max_duration_seconds = max_duration_seconds
        self.min_speech_duration_ms = min_speech_duration_ms
        self.noise_gate_threshold = noise_gate_threshold
        self.vad_aggressiveness = vad_aggressiveness
        self.barge_in = barge_in
        
        # Internal states for tracking active speech
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.last_speech_time = 0.0
        self.accumulated_duration = 0.0
        self.noise_level_calibrated = 200.0

    def calibrate_noise_floor(self, raw_audio_samples: list[int]) -> float:
        """
        Calibrates the noise gate threshold based on ambient background noise frames.
        """
        if not raw_audio_samples:
            return self.noise_level_calibrated
            
        # Compute RMS of ambient sound
        rms = int(math.sqrt(sum(s * s for s in raw_audio_samples) / max(len(raw_audio_samples), 1)))
        # Set threshold slightly above calibrated noise floor
        self.noise_level_calibrated = max(150.0, float(rms * 1.5))
        self.noise_gate_threshold = self.noise_level_calibrated
        return self.noise_gate_threshold

    def process_chunk(self, chunk_bytes: bytes, current_time: float) -> dict[str, Any]:
        """
        Inspects incoming PCM chunk bytes and determines speech status.
        Returns visual indicators and state transitions.
        """
        # Parse 16-bit mono samples
        if len(chunk_bytes) < 2:
            return {"status": "Listening", "turn_complete": False}

        samples = list(struct.unpack(f"<{len(chunk_bytes) // 2}h", chunk_bytes[:len(chunk_bytes) & ~1]))
        rms = int(math.sqrt(sum(s * s for s in samples) / max(len(samples), 1))) if samples else 0
        
        # Enforce PTT bypass
        if self.input_mode == "ptt":
            return {
                "status": "Listening to PTT",
                "turn_complete": False,
                "rms": rms,
                "speaking": True
            }

        # Check noise gate
        is_frame_speech = rms > self.noise_gate_threshold
        
        # Visual indicators mapping
        if is_frame_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = current_time
            self.last_speech_time = current_time
            status_str = "Speech detected"
        else:
            if self.is_speaking:
                silence_duration_ms = (current_time - self.last_speech_time) * 1000
                if silence_duration_ms >= self.silence_timeout_ms:
                    self.is_speaking = False
                    # Speech segment finished
                    if (self.last_speech_time - self.speech_start_time) * 1000 >= self.min_speech_duration_ms:
                        return {
                            "status": "Sending",
                            "turn_complete": True,
                            "rms": rms,
                            "speaking": False
                        }
                status_str = "Waiting for pause"
            else:
                status_str = "Background noise ignored"

        # Max duration safety cutoff
        if self.is_speaking and (current_time - self.speech_start_time) >= self.max_duration_seconds:
            self.is_speaking = False
            return {
                "status": "Sending (max duration cutoff)",
                "turn_complete": True,
                "rms": rms,
                "speaking": False
            }

        return {
            "status": status_str,
            "turn_complete": False,
            "rms": rms,
            "speaking": self.is_speaking
        }
