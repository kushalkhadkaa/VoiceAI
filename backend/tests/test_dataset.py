import subprocess
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.audio_validation import AudioValidator, AudioValidationError
from app.services.dataset import DatasetService, PROMPTS


class DatasetServiceTest(unittest.TestCase):
    def test_list_prompts(self) -> None:
        validator = MagicMock()
        service = DatasetService(Path("/tmp/fake-dataset"), validator)
        prompts = service.list_prompts()
        self.assertEqual(len(prompts), 6)
        self.assertEqual(prompts[0]["id"], "000001")

    def test_save_and_evaluate_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            validator = AudioValidator(max_bytes=1000000, max_seconds=30)
            service = DatasetService(dataset_dir, validator)

            def fake_run(command, stdout, stderr, check):
                output_path = Path(command[-1])  # Output path is the last argument
                # Write a tiny valid mono WAV file (22050Hz, 1 channel, s16 format)
                with wave.open(str(output_path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(22050)
                    # 2 seconds of samples with moderate amplitude to satisfy RMS checks
                    samples = struct_write_pcm(22050 * 2)
                    wav.writeframes(samples)
                return subprocess.CompletedProcess(command, 0)

            with patch("app.services.dataset.shutil.which", return_value="/usr/bin/ffmpeg"), \
                 patch("app.services.dataset.subprocess.run", side_effect=fake_run):
                rec = service.save_recording("000001", b"fake audio", "audio/wav")

            self.assertTrue(rec.exists)
            self.assertEqual(rec.id, "000001")
            self.assertEqual(rec.quality.verdict, "good")
            self.assertGreaterEqual(rec.quality.score, 82)
            self.assertTrue((dataset_dir / "wav" / "000001.wav").exists())
            self.assertTrue(service.metadata_path.exists())
            
            # Verify metadata file content
            meta_content = service.metadata_path.read_text("utf-8")
            self.assertIn("000001|नमस्ते", meta_content)

            # Test list recordings shows exists=True and quality loaded
            recordings = service.list_recordings()
            rec_check = next(r for r in recordings if r.id == "000001")
            self.assertTrue(rec_check.exists)
            self.assertIsNotNone(rec_check.quality)

            # Test zip export
            zip_bytes = service.get_zip_bytes()
            self.assertGreater(len(zip_bytes), 0)
            with zipfile.ZipFile(io_bytes_wrapper(zip_bytes)) as zf:
                self.assertIn("metadata.csv", zf.namelist())
                self.assertIn("wav/000001.wav", zf.namelist())

            # Test deletion
            service.delete_recording("000001")
            self.assertFalse((dataset_dir / "wav" / "000001.wav").exists())
            self.assertEqual(len(service.metadata_path.read_text("utf-8").strip()), 0)


def struct_write_pcm(count: int) -> bytes:
    import struct
    # Generate moderate amplitude sine wave or dummy values that pass RMS > 0.015 checks
    # RMS limit is 0.015, which translates to a moderate range of values.
    import math
    frames = []
    for i in range(count):
        val = int(12000 * math.sin(2 * math.pi * 440 * i / 22050))
        frames.append(struct.pack("<h", val))
    return b"".join(frames)


def io_bytes_wrapper(data: bytes):
    import io
    return io.BytesIO(data)


if __name__ == "__main__":
    unittest.main()
