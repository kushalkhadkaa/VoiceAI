import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.providers.llm import OllamaLLMProvider
from app.providers.stt import ProviderUnavailableError
from app.providers.tts import PiperTTSProvider, TTSPart, OpenAITTSProvider, TTSRouter


class ProvidersTest(unittest.TestCase):
    def test_ollama_chat_parses_api_chat_response(self) -> None:
        payload = {
            "message": {"content": "Hello"},
            "load_duration": 1_000_000,
            "prompt_eval_duration": 2_000_000,
            "eval_duration": 3_000_000,
        }
        fake_response = MagicMock()
        fake_response.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
        with patch("app.providers.llm.urllib.request.urlopen", return_value=fake_response):
            result = OllamaLLMProvider("http://localhost:11434", "qwen3:1.7b", system_prompt="x").chat("hi")
        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.load_ms, 1)
        self.assertEqual(result.prompt_eval_ms, 2)
        self.assertEqual(result.generation_ms, 3)

    def test_piper_command_generation_with_mock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "en_US-test.onnx"
            model.write_bytes(b"model")
            Path(f"{model}.json").write_text("{}", encoding="utf-8")
            provider = PiperTTSProvider("piper", model, model, root)

            fake_voice = MagicMock()
            def fake_synth(text, wav_file, **kwargs):
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 160)
            fake_voice.synthesize_wav.side_effect = fake_synth

            with patch("app.providers.tts.shutil.which", return_value="/usr/bin/piper"), \
                 patch("piper.voice.PiperVoice.load", return_value=fake_voice):
                result = provider.synthesize([TTSPart("Hello", "en")])
        self.assertTrue(result.audio_path.name.endswith(".wav"))

    def test_single_tts_voice_model_routes_english_and_nepali_to_one_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nepali_model = root / "ne_NP-test.onnx"
            english_model = root / "en_US-test.onnx"
            nepali_model.write_bytes(b"ne")
            english_model.write_bytes(b"en")
            Path(f"{nepali_model}.json").write_text("{}", encoding="utf-8")
            Path(f"{english_model}.json").write_text("{}", encoding="utf-8")
            provider = PiperTTSProvider("piper", nepali_model, english_model, root, single_voice_model=True)

            self.assertEqual(provider._voice_for("en")[0], nepali_model)
            self.assertEqual(provider._voice_for("ne")[0], nepali_model)
            self.assertFalse(provider._voice_for("en")[3])

    def test_stt_lazy_import_error_is_actionable(self) -> None:
        from app.providers.stt import FasterWhisperSTTProvider

        provider = FasterWhisperSTTProvider("small")
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with self.assertRaises(ProviderUnavailableError):
                _ = provider.model

    def test_openai_tts_missing_key(self) -> None:
        mock_settings = MagicMock()
        mock_settings.openai_api_key = ""
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAITTSProvider(mock_settings, Path(tmp))
            with self.assertRaises(ValueError):
                provider.synthesize([TTSPart("Hello", "en")])

    def test_openai_tts_synthesize_mocked(self) -> None:
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "fake_key"
        
        # We need a dummy wav file content that urllib.request.urlopen will return
        # A tiny WAV header and frame
        fake_wav_data = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        
        mock_response = MagicMock()
        mock_response.__enter__.return_value.read.return_value = fake_wav_data
        
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAITTSProvider(mock_settings, Path(tmp))
            with patch("urllib.request.urlopen", return_value=mock_response):
                result = provider.synthesize([TTSPart("Hello", "en")], voice_id="openai-alloy")
                
            self.assertTrue(result.audio_path.exists())
            self.assertEqual(result.actual_tts_engine, "openai")
            self.assertEqual(result.requested_voice_name, "OpenAI Alloy")

    def test_tts_router_delegates(self) -> None:
        mock_piper = MagicMock()
        mock_openai = MagicMock()
        router = TTSRouter(mock_piper, mock_openai)
        
        parts = [TTSPart("Hello", "en")]
        
        # Route to OpenAI if voice starts with openai-
        router.synthesize(parts, voice_id="openai-shimmer")
        mock_openai.synthesize.assert_called_once_with(parts, voice_id="openai-shimmer", fallback_allowed=True)
        mock_piper.synthesize.assert_not_called()
        
        mock_openai.reset_mock()
        # Route to Piper otherwise
        router.synthesize(parts, voice_id="ne_NP-chitwan-medium")
        mock_piper.synthesize.assert_called_once_with(parts, voice_id="ne_NP-chitwan-medium", fallback_allowed=True)
        mock_openai.synthesize.assert_not_called()


def _write_tiny_wav(path: Path) -> None:
    import wave

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)


if __name__ == "__main__":
    unittest.main()
