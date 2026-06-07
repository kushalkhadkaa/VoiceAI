from __future__ import annotations
import time
from typing import Any
from app.services.turn_detector import TurnDetector

class VADController:
    def __init__(self, turn_detector: TurnDetector) -> None:
        self.turn_detector = turn_detector
        self.session_started = False
        self.start_timestamp = 0.0

    def start_session(self) -> None:
        self.session_started = True
        self.start_timestamp = time.time()
        self.turn_detector.is_speaking = False
        self.turn_detector.speech_start_time = 0.0
        self.turn_detector.last_speech_time = 0.0

    def feed_chunk(self, chunk: bytes) -> dict[str, Any]:
        if not self.session_started:
            self.start_session()
            
        elapsed = time.time() - self.start_timestamp
        result = self.turn_detector.process_chunk(chunk, time.time())
        result["elapsed"] = elapsed
        return result

    def stop_session(self) -> None:
        self.session_started = False
