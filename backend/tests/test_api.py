import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, settings


class ApiTest(unittest.TestCase):
    def test_health(self) -> None:
        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

    def test_voices_endpoint(self) -> None:
        client = TestClient(app)
        response = client.get("/voices")
        self.assertEqual(response.status_code, 200)
        self.assertIn("selected", response.json())

    def test_delete_local_data_only_deletes_local_dirs(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / ".local" / "audio_cache"
            work = Path(tmp) / ".local" / "audio_work"
            cache.mkdir(parents=True)
            work.mkdir(parents=True)
            (cache / "a.wav").write_bytes(b"a")
            (work / "b.wav").write_bytes(b"b")
            with patch.object(settings, "piper_audio_cache_dir", cache), patch.object(settings, "audio_work_dir", work):
                response = client.delete("/local-data")
            self.assertEqual(response.status_code, 200)
            self.assertFalse((cache / "a.wav").exists())
            self.assertFalse((work / "b.wav").exists())

    def test_stt_rejects_empty_upload(self) -> None:
        client = TestClient(app)
        response = client.post("/stt/test", files={"upload": ("empty.wav", b"", "audio/wav")})
        self.assertEqual(response.status_code, 400)

    def test_voice_socket_status(self) -> None:
        client = TestClient(app)
        response = client.get("/ws/voice/status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertIn("capabilities", payload)
        self.assertIn("blocking_reasons", payload)

    def test_rag_status_endpoint(self) -> None:
        client = TestClient(app)
        response = client.get("/rag/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("base_url", response.json())

    def test_settings_accept_strict_voice_routing_flags(self) -> None:
        client = TestClient(app)
        original_force = settings.force_selected_voice
        original_fallback = settings.fallback_allowed
        try:
            response = client.post("/settings", json={"force_selected_voice": True, "fallback_allowed": False})
            self.assertEqual(response.status_code, 200)
            payload = response.json()["settings"]
            self.assertTrue(payload["force_selected_voice"])
            self.assertFalse(payload["fallback_allowed"])
            self.assertTrue(settings.force_selected_voice)
            self.assertFalse(settings.fallback_allowed)
        finally:
            settings.force_selected_voice = original_force
            settings.fallback_allowed = original_fallback
            settings._write_runtime_overrides()

    def test_web_retrieval_off_by_default(self) -> None:
        client = TestClient(app)
        from app.main import web_retrieval_provider
        with patch.object(web_retrieval_provider, "enabled", False):
            response = client.get("/web-retrieval/status")
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()["enabled"])

    def test_voice_socket_handshake_connects_with_blockers(self) -> None:
        client = TestClient(app)
        with client.websocket_connect("/ws/voice") as websocket:
            websocket.send_json({"type": "hello", "client_version": "test"})
            payload = websocket.receive_json()
        self.assertEqual(payload["type"], "ready")
        self.assertIn("session_id", payload)
        self.assertIn("blocking_reasons", payload)

    def test_voice_socket_invalid_audio_message_is_safe(self) -> None:
        client = TestClient(app)
        with client.websocket_connect("/ws/voice") as websocket:
            websocket.send_json({"type": "hello", "client_version": "test"})
            websocket.receive_json()
            websocket.send_json({"type": "audio", "audioBase64": "not-base64", "mimeType": "audio/webm"})
            payload = websocket.receive_json()
            while payload["type"] == "status":
                payload = websocket.receive_json()
        self.assertIn(payload["type"], {"error", "setup_required"})
        if payload["type"] == "error":
            self.assertIn("Invalid audio message", payload["detail"])

    def test_dataset_endpoints(self) -> None:
        client = TestClient(app)
        # 1. GET /dataset/prompts
        response = client.get("/dataset/prompts")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 6)

        # 2. GET /dataset/recordings
        response = client.get("/dataset/recordings")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 6)

        # Mocking ffmpeg run for POST
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
            return subprocess.CompletedProcess(command, 0)

        # 3. POST /dataset/recordings/000001
        import subprocess
        with patch("app.services.dataset.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("app.services.dataset.subprocess.run", side_effect=fake_run):
            response = client.post("/dataset/recordings/000001", files={"upload": ("test.wav", b"fake audio data", "audio/wav")})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["exists"])
        self.assertEqual(response.json()["quality"]["verdict"], "good")

        # 4. GET /dataset/export
        response = client.get("/dataset/export")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")

        # 5. DELETE /dataset/recordings/000001
        response = client.delete("/dataset/recordings/000001")
        self.assertEqual(response.status_code, 200)
        
        # Verify it's gone
        response = client.get("/dataset/recordings")
        self.assertFalse(next(r for r in response.json() if r["id"] == "000001")["exists"])

    def test_chat_history_and_ratings(self) -> None:
        client = TestClient(app)
        
        # 1. Insert a mock turn into SQLite database
        from app.database import get_db_connection
        conn = get_db_connection()
        turn_id = "test-turn-123"
        try:
            conn.execute("DELETE FROM chat_turns WHERE id = ?;", (turn_id,))
            conn.execute(
                """
                INSERT INTO chat_turns (
                    id, session_id, timestamp, transcript, response, input_language, response_language,
                    audio_url, user_audio_url, tts_route, timings, rag_used, rag_collection_id,
                    rag_fallback_used, internet_used, citations, voice_id, requested_voice_id,
                    requested_voice_name, actual_voice_id, actual_voice_name, actual_engine,
                    actual_model_path, fallback_used, fallback_reason, llm_provider, rag_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    turn_id, "test-session", "2026-06-07T22:00:00Z", "hello", "hi there", "en", "en",
                    "/audio/test.wav", None, "[]", "{}", 0, None, 0, 0, "[]", "test_voice", "test_voice",
                    "Test Voice", "test_voice", "Test Voice", "piper", "path/to/model", 0, None, "local", None
                )
            )
            conn.commit()
        finally:
            conn.close()

        try:
            # 2. Get history and check if the inserted turn is in the list
            response = client.get("/chat/history")
            self.assertEqual(response.status_code, 200)
            history = response.json()
            turn = next((item for item in history if item["id"] == turn_id), None)
            self.assertIsNotNone(turn)
            self.assertEqual(turn["transcript"], "hello")
            self.assertEqual(turn["response"], "hi there")
            self.assertEqual(turn["ratings"], {}) # No ratings yet

            # 3. Update the rating via POST endpoint
            rating_payload = {
                "naturalness": 5,
                "voice_similarity": 4,
                "nepali_pronunciation": 3,
                "english_pronunciation": 4
            }
            rate_response = client.post(f"/chat/turns/{turn_id}/rate", json=rating_payload)
            self.assertEqual(rate_response.status_code, 200)
            self.assertTrue(rate_response.json()["ok"])

            # 4. Get history again and verify ratings are updated
            response2 = client.get("/chat/history")
            history2 = response2.json()
            turn2 = next((item for item in history2 if item["id"] == turn_id), None)
            self.assertIsNotNone(turn2)
            self.assertEqual(turn2["ratings"]["naturalness"], 5)
            self.assertEqual(turn2["ratings"]["voiceSimilarity"], 4)
            self.assertEqual(turn2["ratings"]["nepaliPronunciation"], 3)
            self.assertEqual(turn2["ratings"]["englishPronunciation"], 4)
            
            # 5. Rate with a non-existent turn_id and verify 404
            rate_404_res = client.post("/chat/turns/nonexistent-id/rate", json=rating_payload)
            self.assertEqual(rate_404_res.status_code, 404)
        finally:
            # Cleanup
            conn = get_db_connection()
            try:
                conn.execute("DELETE FROM chat_turns WHERE id = ?;", (turn_id,))
                conn.commit()
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()

