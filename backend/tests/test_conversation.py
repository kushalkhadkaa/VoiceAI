import tempfile
import unittest
from pathlib import Path

from app.providers.llm import LLMResult
from app.providers.stt import STTResult
from app.providers.tts import FallbackBlockedError, TTSResult
from app.services.conversation import ConversationService
from app.services.language_router import LanguageRouter


class FakeSTT:
    def transcribe_file(self, audio_path: Path) -> STTResult:
        return STTResult(text="Hello", language="en", confidence=0.9, duration_ms=12)


class FakeLLM:
    def chat(self, text: str, system_prompt: str = "", options: dict | None = None) -> LLMResult:
        return LLMResult(text=f"Reply to {text}", first_token_ms=5, total_ms=5, model="fake")


class FakeOpenWebUILLM:
    def __init__(self) -> None:
        class FakeSettings:
            keep_turn_audio = False
            llm_provider = "local"
            local_model = "qwen2.5:7b"
            openai_api_key = ""
            openai_model = "gpt-4o-mini"
            gemini_api_key = ""
            gemini_model = "gemini-1.5-flash"
            cloud_fallback_to_local = True
            cloud_timeout_seconds = 30.0
            cloud_temperature = 0.35
            cloud_max_tokens = 180
            system_prompt = "You are SwarLocal, a helpful Nepali-English voice agent."
            rag_fallback_to_ollama = True
            ollama_base_url = "http://localhost:11434"
            ollama_timeout_seconds = 60
            ollama_retries = 1
            ollama_temperature = 0.35
            ollama_num_predict = 180
            ollama_keep_alive = "10m"
        self.settings = FakeSettings()

    def chat(self, text: str, collection_id: str | None = None) -> LLMResult:
        return LLMResult(text=f"OpenWebUI RAG reply to {text}", first_token_ms=5, total_ms=5, model="fake")


class FakeTTS:
    def __init__(self, tmpdir: Path) -> None:
        self.path = tmpdir / "reply.wav"
        self.path.write_bytes(b"fake")

    def synthesize(self, parts, voice_id=None, fallback_allowed=True):
        return TTSResult(
            audio_path=self.path,
            generation_ms=3,
            cached=False,
            requested_voice_id=voice_id,
            requested_voice_name=voice_id,
            actual_voice_id=voice_id or "default",
            actual_voice_name=voice_id or "default",
            actual_tts_engine="piper",
            model_artifact_path=str(self.path),
            language="en",
            fallback_used=False,
            fallback_reason=None,
            generated_audio_path=str(self.path),
        )


class FallbackNeedingTTS(FakeTTS):
    def synthesize(self, parts, voice_id=None, fallback_allowed=True):
        if not fallback_allowed:
            raise FallbackBlockedError("Voice 'missing custom voice' artifact not found on disk.")
        result = super().synthesize(parts, voice_id=voice_id, fallback_allowed=fallback_allowed)
        return TTSResult(
            audio_path=result.audio_path,
            generation_ms=result.generation_ms,
            cached=result.cached,
            requested_voice_id=voice_id,
            requested_voice_name="missing custom voice",
            actual_voice_id="en_US-lessac-medium",
            actual_voice_name="Lessac English (Medium)",
            actual_tts_engine="piper",
            model_artifact_path="/models/piper/en_US-lessac-medium.onnx",
            language=result.language,
            fallback_used=True,
            fallback_reason="Voice 'missing custom voice' artifact not found on disk.",
            generated_audio_path=result.generated_audio_path,
        )


class FakeWebRetrieval:
    def __init__(self) -> None:
        self.enabled = False

    def retrieve(self, query: str) -> list[dict]:
        return []


class ConversationServiceTest(unittest.TestCase):
    def test_text_turn_has_audio_url_and_timings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ConversationService(
                stt_provider=FakeSTT(),
                llm_provider=FakeLLM(),
                openwebui_llm_provider=FakeOpenWebUILLM(),
                tts_provider=FakeTTS(Path(tmp)),
                language_router=LanguageRouter(),
                web_retrieval_provider=FakeWebRetrieval(),
            )
            response = service.handle_text("Hello")
            self.assertEqual(response.input_language, "en")
            self.assertEqual(response.response, "Reply to User question (en): Hello")
            self.assertEqual(response.audio_url, "/audio/reply.wav")
            self.assertEqual(response.requested_voice_id, None)
            self.assertEqual(response.actual_engine, "piper")
            self.assertEqual(response.actual_model_path, str(Path(tmp) / "reply.wav"))
            self.assertFalse(response.fallback_used)
            self.assertGreater(response.timings.total_turn_ms, 0)

    def test_voice_fallback_metadata_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ConversationService(
                stt_provider=FakeSTT(),
                llm_provider=FakeLLM(),
                openwebui_llm_provider=FakeOpenWebUILLM(),
                tts_provider=FallbackNeedingTTS(Path(tmp)),
                language_router=LanguageRouter(),
                web_retrieval_provider=FakeWebRetrieval(),
            )
            response = service.handle_text("Hello", voice_id="custom-missing")
            self.assertEqual(response.requested_voice_id, "custom-missing")
            self.assertEqual(response.actual_voice_id, "en_US-lessac-medium")
            self.assertEqual(response.actual_engine, "piper")
            self.assertEqual(response.actual_model_path, "/models/piper/en_US-lessac-medium.onnx")
            self.assertTrue(response.fallback_used)
            self.assertIn("artifact not found", response.fallback_reason)

    def test_force_selected_voice_blocks_silent_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ConversationService(
                stt_provider=FakeSTT(),
                llm_provider=FakeLLM(),
                openwebui_llm_provider=FakeOpenWebUILLM(),
                tts_provider=FallbackNeedingTTS(Path(tmp)),
                language_router=LanguageRouter(),
                web_retrieval_provider=FakeWebRetrieval(),
            )
            service.settings.force_selected_voice = True
            service.settings.fallback_allowed = True
            with self.assertRaisesRegex(ValueError, "fallback is disabled"):
                service.handle_text("Hello", voice_id="custom-missing")


if __name__ == "__main__":
    unittest.main()
