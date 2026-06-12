import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import Settings


class ConfigTest(unittest.TestCase):
    def test_loads_runtime_settings_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / "settings.json"
            with patch.dict(
                os.environ,
                {
                    "OLLAMA_MODEL": "llama3.2:1b",
                    "MAX_RECORDING_SECONDS": "12",
                    "PIPER_AUDIO_CACHE_DIR": str(Path(tmp) / "cache"),
                    "AUDIO_WORK_DIR": str(Path(tmp) / "work"),
                    "RUNTIME_SETTINGS_PATH": str(runtime),
                },
                clear=False,
            ), patch("app.db_settings.load_settings", return_value={}):
                settings = Settings.from_env()
                self.assertEqual(settings.ollama_model, "llama3.2:1b")
                self.assertEqual(settings.max_recording_seconds, 12)


if __name__ == "__main__":
    unittest.main()
