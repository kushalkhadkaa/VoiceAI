import unittest
from unittest.mock import patch

from app.config import Settings
from app.services.environment import run_environment_checks


class EnvironmentTest(unittest.TestCase):
    def test_report_not_ready_when_critical_missing(self) -> None:
        settings = Settings()
        with patch("app.services.audio_validation.get_ffmpeg_path", return_value=None), patch("app.services.environment._get_json", side_effect=OSError("down")):
            report = run_environment_checks(settings)
        self.assertFalse(report.ready)
        self.assertTrue(any(check.name == "ffmpeg" and not check.ok for check in report.checks))


if __name__ == "__main__":
    unittest.main()
