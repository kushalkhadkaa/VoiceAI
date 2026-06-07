import tempfile
import unittest
from pathlib import Path

from app.services.voice_registry import VoiceRegistry


class VoiceRegistryTest(unittest.TestCase):
    def test_pairs_model_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "en_US-test-medium.onnx"
            model.write_bytes(b"model")
            Path(f"{model}.json").write_text("{}", encoding="utf-8")
            voices = VoiceRegistry(root).list_voices()
            test_voice = next((v for v in voices if v.id == "en_US-test-medium"), None)
            self.assertIsNotNone(test_voice)
            self.assertEqual(test_voice.language, "en")
            self.assertEqual(test_voice.status, "ready")

    def test_detects_missing_config_for_configured_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "ne_NP-test-medium.onnx"
            voice = VoiceRegistry(Path(tmp)).inspect(model)
            self.assertEqual(voice.language, "ne")
            self.assertIn(str(model), voice.missing_files)
            self.assertIn(f"{model}.json", voice.missing_files)


if __name__ == "__main__":
    unittest.main()
