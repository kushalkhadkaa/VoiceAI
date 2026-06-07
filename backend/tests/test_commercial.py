import unittest
import tempfile
import sqlite3
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.main import app, settings
from app.database import init_db, get_db_connection, DB_FILE
from app.providers.web_retrieval import DDGParser


class CommercialFeaturesTest(unittest.TestCase):
    def test_database_initialization(self) -> None:
        # Check if database has our tables
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row["name"] for row in cursor.fetchall()}
            self.assertIn("voice_owners", tables)
            self.assertIn("voices", tables)
            self.assertIn("voice_consents", tables)
            self.assertIn("voice_samples", tables)
            self.assertIn("voice_training_jobs", tables)
            self.assertIn("voice_model_artifacts", tables)
            self.assertIn("voice_permissions", tables)
            self.assertIn("voice_usage_events", tables)
            self.assertIn("voice_audit_log", tables)
        finally:
            conn.close()

    def test_duckduckgo_parser(self) -> None:
        html = """
        <div class="web-result">
            <a class="result__a" href="http://example.com/item1">Example item 1</a>
            <a class="result__snippet" href="http://example.com/item1">This is a snippet for item 1</a>
        </div>
        <div class="web-result">
            <a class="result__a" href="http://example.com/item2">Example item 2</a>
            <a class="result__snippet" href="http://example.com/item2">This is a snippet for item 2</a>
        </div>
        """
        parser = DDGParser()
        parser.feed(html)
        parser.close()
        self.assertEqual(len(parser.results), 2)
        self.assertEqual(parser.results[0]["title"], "Example item 1")
        self.assertEqual(parser.results[0]["url"], "http://example.com/item1")
        self.assertEqual(parser.results[0]["snippet"], "This is a snippet for item 1")

    def test_voice_studio_crud_endpoints(self) -> None:
        client = TestClient(app)
        
        # 1. Create a voice
        payload = {
            "name": "Kushal Voice",
            "owner_name": "Kushal",
            "owner_email": "kushal@example.com",
            "organization": "SwarLocal",
            "language": "ne",
            "engine": "piper",
            "commercial_allowed": True
        }
        resp = client.post("/voices/create", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("voice_id", data)
        voice_id = data["voice_id"]
        
        # 2. Try to record sample before signing consent (should fail)
        resp = client.post(
            f"/voices/{voice_id}/recordings/ne_001",
            files={"upload": ("sample.wav", b"fake audio data", "audio/wav")}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("consent is required", resp.json()["detail"].lower())

        # 3. Submit consent
        resp = client.post(
            f"/voices/{voice_id}/consent",
            data={"signature": "Kushal Khadka"}
        )
        self.assertEqual(resp.status_code, 200)
        
        # 4. Now record sample (mock ffmpeg validation)
        def fake_run(command, stdout, stderr, check):
            import wave
            import struct
            import math
            output_path = Path(command[-1])
            with wave.open(str(output_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(22050)
                frames = [struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / 22050))) for i in range(22050 * 2)]
                wav.writeframes(b"".join(frames))
            class FakeSubprocess:
                returncode = 0
                stdout = b""
                stderr = b""
            return FakeSubprocess()
            
        with patch("app.services.voice_studio.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("app.services.voice_studio.subprocess.run", side_effect=fake_run):
            resp = client.post(
                f"/voices/{voice_id}/recordings/ne_001",
                files={"upload": ("sample.wav", b"fake audio data", "audio/wav")}
            )
        if resp.status_code != 200:
            print("ERROR RESP:", resp.json())
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["exists"])
        self.assertEqual(resp.json()["verdict"], "good")

        # 5. List voice recordings
        resp = client.get(f"/voices/{voice_id}/recordings")
        self.assertEqual(resp.status_code, 200)
        recs = resp.json()
        self.assertTrue(any(r["id"] == "ne_001" and r["exists"] for r in recs))
        
        # 6. Audit logs
        resp = client.get("/audit/logs")
        self.assertEqual(resp.status_code, 200)
        logs = resp.json()
        self.assertTrue(any(l["event"] == "voice_created" for l in logs))

        # 7. Clean up voice profile
        resp = client.delete(f"/voices/{voice_id}")
        self.assertEqual(resp.status_code, 200)

    def test_rag_blocking_reasons_only_when_enabled(self) -> None:
        from app.services.voice_socket_status import build_voice_socket_status
        from app.config import Settings
        
        # 1. When rag_enabled is False
        settings_test = Settings(rag_enabled=False, open_webui_api_key="")
        status = build_voice_socket_status(settings_test)
        self.assertNotIn("Open WebUI API key missing", status["blocking_reasons"])
        self.assertNotIn("Open WebUI unavailable", status["blocking_reasons"])
        
        # 2. When rag_enabled is True
        settings_test_2 = Settings(
            rag_enabled=True,
            open_webui_api_key="",
            rag_default_collection="",
            ollama_model=settings.ollama_model,
        )
        status_2 = build_voice_socket_status(settings_test_2)
        self.assertNotIn("Open WebUI API key missing", status_2["blocking_reasons"])
        self.assertTrue(any("Open WebUI API key missing" in item for item in status_2["warnings"]))
        self.assertTrue(status_2["capabilities"]["text_turns"])
        self.assertTrue(status_2["capabilities"]["audio_turns"])

    def test_ai_provider_settings_masking_and_deletion(self) -> None:
        from app.config import Settings
        # Test key masking
        s = Settings(openai_api_key="sk-abcdefghijklmnopqrstuvwxyz123456", gemini_api_key="AIzaSy1234567890abcdefghijklmnopqrstuv")
        pdict = s.public_dict()
        self.assertEqual(pdict["openai_api_key"], "sk-a...3456")
        self.assertEqual(pdict["gemini_api_key"], "AIza...stuv")

        # Test settings update without overriding with masked key
        s.update_from_payload({"openai_api_key": "sk-a...3456"})
        # Should remain the original full key
        self.assertEqual(s.openai_api_key, "sk-abcdefghijklmnopqrstuvwxyz123456")

        # Test settings endpoint
        client = TestClient(app)
        # Update key
        resp = client.post("/settings", json={"openai_api_key": "sk-test-key-1234567890"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["settings"]["openai_api_key"], "sk-t...7890")

        # Delete key
        resp = client.delete("/settings/openai-key")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["settings"]["openai_api_key"], "")

        # Delete Gemini key
        resp = client.delete("/settings/gemini-key")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["settings"]["gemini_api_key"], "")

    @patch("urllib.request.urlopen")
    def test_local_provider_connection_test(self, mock_urlopen) -> None:
        from unittest.mock import MagicMock
        # Mock /api/tags response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "models": [{"name": "qwen2.5:7b"}, {"name": "gemma3:4b"}]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        # Mock the chat call
        from app.providers.llm_ollama import OllamaLLMProvider
        provider = OllamaLLMProvider(
            base_url="http://localhost:11434",
            model="qwen2.5:7b",
        )
        
        # Patch the actual chat call to not call real Ollama
        from app.providers.llm import LLMResult
        with patch.object(provider, "chat", return_value=LLMResult(text="OK", first_token_ms=10, total_ms=10, model="qwen2.5:7b")):
            res = provider.test_connection()
            self.assertTrue(res["ok"])
            self.assertEqual(res["model"], "qwen2.5:7b")

    @patch("urllib.request.urlopen")
    def test_openai_provider_mock_success_failure(self, mock_urlopen) -> None:
        from unittest.mock import MagicMock
        # Mock success
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "OK"}}]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        from app.providers.llm_openai import OpenAILLMProvider
        provider = OpenAILLMProvider(api_key="sk-test", model="gpt-4o-mini")
        res = provider.test_connection()
        self.assertTrue(res["ok"])
        self.assertEqual(res["model"], "gpt-4o-mini")

        # Mock failure (HTTPError)
        import urllib.error
        from io import BytesIO
        err_fp = BytesIO(json.dumps({"error": {"message": "Invalid API key"}}).encode("utf-8"))
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=err_fp
        )
        res_fail = provider.test_connection()
        self.assertFalse(res_fail["ok"])
        self.assertIn("Invalid API key", res_fail["detail"])

    @patch("urllib.request.urlopen")
    def test_gemini_provider_mock_success_failure(self, mock_urlopen) -> None:
        from unittest.mock import MagicMock
        # Mock success
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "candidates": [{"content": {"parts": [{"text": "OK"}]}}]
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        from app.providers.llm_gemini import GeminiLLMProvider
        provider = GeminiLLMProvider(api_key="AIzaSyTest", model="gemini-1.5-flash")
        res = provider.test_connection()
        self.assertTrue(res["ok"])
        self.assertEqual(res["model"], "gemini-1.5-flash")

        # Mock failure (HTTPError)
        import urllib.error
        from io import BytesIO
        err_fp = BytesIO(json.dumps({"error": {"message": "API Key not valid"}}).encode("utf-8"))
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=err_fp
        )
        res_fail = provider.test_connection()
        self.assertFalse(res_fail["ok"])
        self.assertIn("API Key not valid", res_fail["detail"])

    def test_dynamic_conversational_routing_and_fallbacks(self) -> None:
        # Set up a Mock service and test routing/fallbacks
        from app.services.conversation import ConversationService
        from app.providers.llm import LLMResult
        
        # Mock classes
        class MockOllama:
            def __init__(self):
                self.system_prompt = ""
            def chat(self, text, system_prompt="", options=None):
                return LLMResult(text="Local Reply", first_token_ms=5, total_ms=5, model="local")
                
        class MockOpenAI:
            def chat(self, text, system_prompt="", options=None):
                raise RuntimeError("OpenAI failed")

        with patch("app.services.conversation.ConversationService._instantiate_provider") as mock_instantiate:
            # We mock the return of instantiating provider
            def side_effect(provider_id):
                if provider_id == "openai":
                    return MockOpenAI()
                return MockOllama()
            mock_instantiate.side_effect = side_effect

            # Setup Fake objects
            from test_conversation import FakeSTT, FakeTTS, FakeWebRetrieval, FakeOpenWebUILLM
            from app.services.language_router import LanguageRouter
            
            service = ConversationService(
                stt_provider=FakeSTT(),
                llm_provider=MockOllama(),
                openwebui_llm_provider=FakeOpenWebUILLM(),
                tts_provider=FakeTTS(Path(tempfile.mkdtemp())),
                language_router=LanguageRouter(),
                web_retrieval_provider=FakeWebRetrieval(),
            )

            # 1. Fallback enabled
            service.settings.cloud_fallback_to_local = True
            service.settings.llm_provider = "openai"
            
            # Test direct chat turn with OpenAI provider (which fails, triggering fallback to local)
            res = service.handle_text("Hello", llm_provider_id="openai")
            self.assertEqual(res.llm_provider, "local")
            self.assertTrue(res.rag_fallback_used)
            self.assertIn("OpenAI failed", res.fallback_reason)

            # 2. Fallback disabled
            service.settings.cloud_fallback_to_local = False
            with self.assertRaises(RuntimeError):
                service.handle_text("Hello", llm_provider_id="openai")

    def test_voice_cloning_endpoint(self) -> None:
        client = TestClient(app)
        
        # 1. Create a voice
        payload = {
            "name": "Cloning Test Voice",
            "owner_name": "TestOwner",
            "owner_email": "owner@example.com",
            "organization": "SwarLocal",
            "language": "ne",
            "engine": "chatterbox",
            "commercial_allowed": True
        }
        resp = client.post("/voices/create", json=payload)
        self.assertEqual(resp.status_code, 200)
        voice_id = resp.json()["voice_id"]
        
        # 2. Try to clone without consent (should fail with 400)
        resp = client.post(f"/voices/{voice_id}/clone")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("consent must be signed", resp.json()["detail"].lower())
        
        # Sign consent
        resp = client.post(f"/voices/{voice_id}/consent", data={"signature": "TestOwner"})
        self.assertEqual(resp.status_code, 200)
        
        # 3. Try to clone with less than 3 recordings (should fail with 400)
        resp = client.post(f"/voices/{voice_id}/clone")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("at least 3 clean recordings", resp.json()["detail"].lower())
        
        # 4. Now clone with real WAV files; Chatterbox creates a reference artifact.
        with tempfile.TemporaryDirectory() as tmp_voices_dir:
            from app.main import voice_clone_service
            voice_dir = Path(tmp_voices_dir) / voice_id
            normalized_dir = voice_dir / "normalized"
            normalized_dir.mkdir(parents=True, exist_ok=True)

            def write_wav(path: Path, frequency: int) -> None:
                import math
                import struct
                import wave
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(22050)
                    frames = [
                        struct.pack("<h", int(12000 * math.sin(2 * math.pi * frequency * i / 22050)))
                        for i in range(22050 * 2)
                    ]
                    wav.writeframes(b"".join(frames))

            conn = get_db_connection()
            try:
                for prompt_id, frequency in (("ne_001", 220), ("ne_002", 240), ("ne_003", 260)):
                    wav_path = normalized_dir / f"{prompt_id}.wav"
                    write_wav(wav_path, frequency)
                    conn.execute(
                        "INSERT INTO voice_samples (id, voice_id, prompt_id, wav_path, status, score, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                        (f"{voice_id}_{prompt_id}", voice_id, prompt_id, str(wav_path), "good", 90, "clean", "2026-06-06T00:00:00Z")
                    )
                conn.commit()
            finally:
                conn.close()

            with patch.object(voice_clone_service, "voices_base_dir", Path(tmp_voices_dir)):
                resp = client.post(f"/voices/{voice_id}/clone")
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp.json()["ok"])
                self.assertEqual(resp.json()["status"], "completed")
                self.assertTrue((voice_dir / "chatterbox_reference.wav").exists())

            conn = get_db_connection()
            try:
                artifact = conn.execute("SELECT * FROM voice_model_artifacts WHERE voice_id = ?;", (voice_id,)).fetchone()
                self.assertIsNotNone(artifact)
                self.assertTrue(artifact["onnx_path"].startswith("chatterbox://"))
            finally:
                conn.close()

    def test_piper_clone_without_training_command_is_not_faked(self) -> None:
        client = TestClient(app)
        payload = {
            "name": "Piper Clone Needs Training",
            "owner_name": "TestOwner",
            "owner_email": "owner@example.com",
            "organization": "SwarLocal",
            "language": "ne",
            "engine": "piper",
            "commercial_allowed": True
        }
        resp = client.post("/voices/create", json=payload)
        self.assertEqual(resp.status_code, 200)
        voice_id = resp.json()["voice_id"]
        resp = client.post(f"/voices/{voice_id}/consent", data={"signature": "TestOwner"})
        self.assertEqual(resp.status_code, 200)

        with tempfile.TemporaryDirectory() as tmp_voices_dir:
            from app.main import voice_clone_service, settings
            voice_dir = Path(tmp_voices_dir) / voice_id
            normalized_dir = voice_dir / "normalized"
            normalized_dir.mkdir(parents=True, exist_ok=True)
            wav_path = normalized_dir / "ne_001.wav"
            import math
            import struct
            import wave
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(22050)
                frames = [
                    struct.pack("<h", int(12000 * math.sin(2 * math.pi * 220 * i / 22050)))
                    for i in range(22050 * 2)
                ]
                wav.writeframes(b"".join(frames))

            conn = get_db_connection()
            try:
                for prompt_id in ("ne_001", "ne_002", "ne_003"):
                    sample_path = normalized_dir / f"{prompt_id}.wav"
                    if not sample_path.exists():
                        shutil.copy2(wav_path, sample_path)
                    conn.execute(
                        "INSERT INTO voice_samples (id, voice_id, prompt_id, wav_path, status, score, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                        (f"{voice_id}_{prompt_id}", voice_id, prompt_id, str(sample_path), "good", 90, "clean", "2026-06-06T00:00:00Z")
                    )
                conn.commit()
            finally:
                conn.close()

            original_command = settings.piper_train_command
            try:
                settings.piper_train_command = ""
                with patch.object(voice_clone_service, "voices_base_dir", Path(tmp_voices_dir)):
                    resp = client.post(f"/voices/{voice_id}/clone")
                self.assertEqual(resp.status_code, 400)
                self.assertIn("piper fine-tuning is not configured", resp.json()["detail"].lower())
                self.assertFalse(list(voice_dir.glob("*.onnx")))
            finally:
                settings.piper_train_command = original_command

    def test_clean_recording_endpoint(self) -> None:
        client = TestClient(app)
        
        # 1. Create a voice and sign consent
        payload = {
            "name": "Cleaning Test Voice",
            "owner_name": "TestOwner",
            "owner_email": "owner@example.com",
            "organization": "SwarLocal",
            "language": "ne",
            "engine": "piper",
            "commercial_allowed": True
        }
        resp = client.post("/voices/create", json=payload)
        self.assertEqual(resp.status_code, 200)
        voice_id = resp.json()["voice_id"]
        
        # Sign consent
        resp = client.post(f"/voices/{voice_id}/consent", data={"signature": "TestOwner"})
        self.assertEqual(resp.status_code, 200)
        
        # 2. Add raw sample file on disk
        from app.main import voices_base_dir
        with tempfile.TemporaryDirectory() as tmp_voices_dir:
            with patch("app.main.voices_base_dir", Path(tmp_voices_dir)):
                voice_dir = Path(tmp_voices_dir) / voice_id
                raw_dir = voice_dir / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / "ne_001.raw"
                raw_path.write_bytes(b"fake raw audio bytes")
                
                # Mock evaluation, ffmpeg run, noise reduction and audio enhancements
                q_eval = {"verdict": "good", "score": 95, "reason": "clean"}
                with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
                     patch("subprocess.run") as mock_run, \
                     patch("app.main.noise_reduction_service.denoise_audio") as mock_denoise, \
                     patch("app.main.audio_enhancement_service.normalize_loudness") as mock_loudness, \
                     patch("app.main.audio_enhancement_service.trim_silence") as mock_trim, \
                     patch("app.main.voice_studio_service._evaluate_wav", return_value=q_eval):
                     
                    resp = client.post(f"/voices/{voice_id}/recordings/ne_001/clean")
                    
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp.json()["ok"])
                self.assertEqual(resp.json()["verdict"], "good")
                self.assertEqual(resp.json()["score"], 95)

    def test_publish_voice_endpoint(self) -> None:
        client = TestClient(app)
        
        # 1. Create a voice
        payload = {
            "name": "Publish Test Voice",
            "owner_name": "TestOwner",
            "owner_email": "owner@example.com",
            "organization": "SwarLocal",
            "language": "ne",
            "engine": "piper",
            "commercial_allowed": True
        }
        resp = client.post("/voices/create", json=payload)
        self.assertEqual(resp.status_code, 200)
        voice_id = resp.json()["voice_id"]
        
        # 2. Try to publish before consent and recordings (should fail with 400)
        resp = client.post(f"/voices/{voice_id}/publish")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("consent is missing", resp.json()["detail"].lower())
        
        # Sign consent
        resp = client.post(f"/voices/{voice_id}/consent", data={"signature": "TestOwner"})
        self.assertEqual(resp.status_code, 200)
        
        # Try to publish with no recordings (should fail with 400)
        resp = client.post(f"/voices/{voice_id}/publish")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("at least 3 clean recordings", resp.json()["detail"].lower())
        
        # 3. Add 3 samples via direct SQL (to simulate mock inserts)
        conn = get_db_connection()
        try:
            for prompt_id, score in [("ne_001", 90), ("ne_002", 90), ("ne_003", 80)]:
                conn.execute(
                    "INSERT INTO voice_samples (id, voice_id, prompt_id, wav_path, status, score, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                    (f"{voice_id}_{prompt_id}", voice_id, prompt_id, f"dummy_{prompt_id}.wav", "good", score, "clean", "2026-06-06T00:00:00Z")
                )
            conn.commit()
        finally:
            conn.close()
            
        # 4. Now publish (should succeed and dynamically recalculate quality score)
        resp = client.post(f"/voices/{voice_id}/publish")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])
        
        # Verify status in database
        conn = get_db_connection()
        try:
            voice = conn.execute("SELECT * FROM voices WHERE id = ?;", (voice_id,)).fetchone()
            self.assertEqual(voice["publish_status"], "published")
            self.assertAlmostEqual(voice["quality_score"], 86.66666666666667)
        finally:
            conn.close()
            
        # 5. Clean up
        resp = client.delete(f"/voices/{voice_id}")
        self.assertEqual(resp.status_code, 200)

    def test_custom_voice_cloning_effect(self) -> None:
        from app.providers.tts import apply_voice_cloning_effect, estimate_pitch, PiperTTSProvider, TTSPart
        import wave
        import struct
        import math
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wav_path = tmp_path / "test.wav"
            
            # Write a simple WAV file containing a sine wave at 150Hz
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                # 0.5s of sine wave at 150Hz
                frames = []
                for i in range(8000):
                    val = int(32767 * math.sin(2 * math.pi * 150 * i / 16000))
                    frames.append(struct.pack("<h", val))
                wav.writeframes(b"".join(frames))
                
            pitch = estimate_pitch(wav_path)
            self.assertIsNotNone(pitch)
            self.assertGreater(pitch, 130)
            self.assertLess(pitch, 170)
            
            # Create the custom voice directory and pitch file in .local/voices/
            custom_voice_dir = Path(".local/voices") / "custom_voice_id"
            custom_voice_dir.mkdir(parents=True, exist_ok=True)
            pitch_file = custom_voice_dir / "pitch.txt"
            pitch_file.write_text("150.0", encoding="utf-8")
            
            try:
                # Test apply_voice_cloning_effect running ffmpeg with correct parameters
                with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
                     patch("subprocess.run") as mock_run:
                     
                    # Create a dummy temp_out file so the replace doesn't fail
                    temp_out = wav_path.with_name(f"vc_{wav_path.name}")
                    temp_out.write_bytes(b"dummy wav data")
                    
                    apply_voice_cloning_effect(wav_path, "custom_voice_id", "en")
                    
                    # Check subprocess was run with ffmpeg and alimiter filter
                    self.assertTrue(mock_run.called)
                    args = mock_run.call_args[0][0]
                    self.assertEqual(args[0], "ffmpeg")
                    
                    # Check filter graph includes peak limiter to prevent clipping
                    filter_arg = args[args.index("-af") + 1]
                    self.assertIn("asetrate=", filter_arg)
                    self.assertIn("atempo=", filter_arg)
                    self.assertIn("alimiter=limit=0.95", filter_arg)
            finally:
                if pitch_file.exists():
                    pitch_file.unlink()
                if custom_voice_dir.exists():
                    custom_voice_dir.rmdir()
                
            # Test that PiperTTSProvider.synthesize calls apply_voice_cloning_effect
            model = tmp_path / "ne_NP-test.onnx"
            model.write_bytes(b"model")
            Path(f"{model}.json").write_text("{}", encoding="utf-8")
            provider = PiperTTSProvider("piper", model, model, tmp_path)
            
            fake_voice = MagicMock()
            def fake_synth(text, wav_file, **kwargs):
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 160)
            fake_voice.synthesize_wav.side_effect = fake_synth
            
            with patch("app.providers.tts.shutil.which", return_value="/usr/bin/piper"), \
                 patch("piper.voice.PiperVoice.load", return_value=fake_voice), \
                 patch("app.providers.tts.apply_voice_cloning_effect") as mock_apply_effect:
                 
                # Request a custom voice
                result = provider.synthesize([TTSPart("नमस्ते", "ne")], voice_id="custom_voice_id")
                
                # Verify apply_voice_cloning_effect was called for the custom voice
                mock_apply_effect.assert_called_once()
                args_list, kwargs_list = mock_apply_effect.call_args
                self.assertEqual(args_list[1], "custom_voice_id")
                self.assertEqual(args_list[2], "ne")

    def test_elevenlabs_settings_endpoints(self) -> None:
        client = TestClient(app)
        
        # Reset settings
        settings.elevenlabs_api_key = ""
        
        # 1. Update elevenlabs_api_key
        payload = {"elevenlabs_api_key": "test-key-123"}
        resp = client.post("/settings", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(settings.elevenlabs_api_key, "test-key-123")
        
        # 2. Test ElevenLabs provider endpoint (Mock success)
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"voices": []}'
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            resp = client.post("/ai-providers/test/elevenlabs", json={"elevenlabs_api_key": "valid-key"})
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["ok"])
            
        # 3. Test ElevenLabs provider endpoint (Mock HTTP error)
        import urllib.error
        from io import BytesIO
        with patch("urllib.request.urlopen") as mock_urlopen:
            err = urllib.error.HTTPError(
                "https://api.elevenlabs.io/v1/voices",
                401,
                "Unauthorized",
                {},
                BytesIO(b"Invalid key")
            )
            mock_urlopen.side_effect = err
            
            resp = client.post("/ai-providers/test/elevenlabs", json={"elevenlabs_api_key": "invalid-key"})
            self.assertEqual(resp.status_code, 200)
            self.assertFalse(resp.json()["ok"])
            self.assertIn("API error (401)", resp.json()["detail"])
            
        # 4. Delete key
        resp = client.delete("/settings/elevenlabs-key")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(settings.elevenlabs_api_key, "")

    def test_elevenlabs_voice_cloning_and_synthesis(self) -> None:
        client = TestClient(app)
        
        # 1. Create ElevenLabs engine voice
        payload = {
            "name": "ElevenLabs Test",
            "owner_name": "TestOwner",
            "owner_email": "owner@example.com",
            "organization": "SwarLocal",
            "language": "ne",
            "engine": "elevenlabs",
            "commercial_allowed": True
        }
        resp = client.post("/voices/create", json=payload)
        self.assertEqual(resp.status_code, 200)
        voice_id = resp.json()["voice_id"]
        
        # Sign consent
        resp = client.post(f"/voices/{voice_id}/consent", data={"signature": "TestOwner"})
        self.assertEqual(resp.status_code, 200)
        
        # Set ElevenLabs API key
        settings.elevenlabs_api_key = "dummy-el-key"
        
        # Mock actual files on disk
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_voices_dir = Path(tmp_dir) / "voices"
            tmp_voices_dir.mkdir(parents=True, exist_ok=True)
            tmp_work_dir = Path(tmp_dir) / "work"
            tmp_work_dir.mkdir(parents=True, exist_ok=True)
            
            from app.main import voices_base_dir
            voice_dir = Path(tmp_voices_dir) / voice_id
            voice_dir.mkdir(parents=True, exist_ok=True)
            
            # Add 3 samples
            conn = get_db_connection()
            try:
                for prompt_id in ("ne_001", "ne_002", "ne_003"):
                    wav_file_path = voice_dir / f"dummy_{prompt_id}.wav"
                    wav_file_path.write_bytes(b"dummy wav bytes")
                    conn.execute(
                        "INSERT INTO voice_samples (id, voice_id, prompt_id, wav_path, status, score, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                        (f"{voice_id}_{prompt_id}", voice_id, prompt_id, str(wav_file_path), "good", 90, "clean", "2026-06-06T00:00:00Z")
                    )
                conn.commit()
            finally:
                conn.close()
                
            # Mock the ElevenLabs multipart upload API response and settings.audio_work_dir
            with patch("app.main.voices_base_dir", Path(tmp_voices_dir)), \
                 patch.object(settings, "audio_work_dir", tmp_work_dir), \
                 patch("urllib.request.urlopen") as mock_urlopen, \
                 patch("pathlib.Path.exists", return_value=True):
                 
                mock_response = MagicMock()
                mock_response.read.return_value = b'{"voice_id": "el-mock-voice-id-999"}'
                mock_urlopen.return_value.__enter__.return_value = mock_response
                
                # Clone voice
                resp = client.post(f"/voices/{voice_id}/clone")
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp.json()["ok"])
                self.assertEqual(resp.json()["status"], "completed")
                
                # Check that elevenlabs_id.txt was created
                id_file = voice_dir / "elevenlabs_id.txt"
                self.assertTrue(id_file.exists())
                self.assertEqual(id_file.read_text().strip(), "el-mock-voice-id-999")
                
            # Test TTS Synthesis Routing
            from app.providers.tts import TTSPart
            from app.main import tts_provider
            
            def fake_ffmpeg_run(cmd, *args, **kwargs):
                out_path = Path(cmd[-1])
                import wave
                with wave.open(str(out_path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(22050)
                    wav.writeframes(b"\x00\x00" * 160)
                class FakeSubprocess:
                    returncode = 0
                    stdout = b""
                    stderr = b""
                return FakeSubprocess()

            with patch("urllib.request.urlopen") as mock_urlopen, \
                 patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
                 patch.object(settings, "audio_work_dir", tmp_work_dir), \
                 patch("subprocess.run", side_effect=fake_ffmpeg_run) as mock_run:
                 
                # Mock ElevenLabs returning MP3 bytes
                mock_response = MagicMock()
                mock_response.read.return_value = b"mock mp3 bytes"
                mock_urlopen.return_value.__enter__.return_value = mock_response
                
                # Synthesize
                parts = [TTSPart(text="नमस्ते", language="ne")]
                res = tts_provider.synthesize(parts, voice_id=voice_id)
                
                # Verify routed to elevenlabs
                self.assertEqual(res.actual_tts_engine, "elevenlabs")
                self.assertEqual(res.model_artifact_path, "elevenlabs://el-mock-voice-id-999")
                
                # Verify transcode subprocess was run
                self.assertTrue(mock_run.called)
                args = mock_run.call_args[0][0]
                self.assertEqual(args[0], "ffmpeg")
                self.assertIn("-ac", args)
                self.assertIn("22050", args)


if __name__ == "__main__":
    unittest.main()
