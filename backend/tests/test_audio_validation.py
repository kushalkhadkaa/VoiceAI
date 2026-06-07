import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from app.services.audio_validation import AudioValidationError, AudioValidator


class AudioValidationTest(unittest.TestCase):
    def test_rejects_empty_upload(self) -> None:
        validator = AudioValidator(max_bytes=1024, max_seconds=1)
        with self.assertRaises(AudioValidationError):
            validator.validate_upload(b"", "audio/wav")

    def test_rejects_unsupported_mime(self) -> None:
        validator = AudioValidator(max_bytes=1024, max_seconds=1)
        with self.assertRaises(AudioValidationError):
            validator.validate_upload(b"abc", "application/octet-stream")

    def test_accepts_short_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.wav"
            _write_wav(path, seconds=0.25)
            result = AudioValidator(max_bytes=1024 * 1024, max_seconds=1).validate_wav_duration(path)
            self.assertLess(result.duration_seconds, 1)

    def test_rejects_too_long_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long.wav"
            _write_wav(path, seconds=1.2)
            with self.assertRaises(AudioValidationError):
                AudioValidator(max_bytes=1024 * 1024, max_seconds=1).validate_wav_duration(path)


def _write_wav(path: Path, seconds: float, rate: int = 16000) -> None:
    frames = []
    for index in range(int(seconds * rate)):
        value = int(math.sin(index / 20) * 1200)
        frames.append(struct.pack("<h", value))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(frames))


if __name__ == "__main__":
    unittest.main()
