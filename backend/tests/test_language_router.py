import unittest

from app.services.language_router import LanguageRouter


class LanguageRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.router = LanguageRouter()

    def test_detects_nepali_from_devanagari(self) -> None:
        decision = self.router.detect("नमस्ते, आज मौसम कस्तो छ?")
        self.assertEqual(decision.language, "ne")

    def test_detects_english_from_latin(self) -> None:
        decision = self.router.detect("What is on my calendar today?")
        self.assertEqual(decision.language, "en")

    def test_detects_mixed_script(self) -> None:
        decision = self.router.detect("आज meeting कति बजे छ?")
        self.assertEqual(decision.language, "mixed")

    def test_romanized_nepali_is_unknown(self) -> None:
        decision = self.router.detect("namaste mero meeting kasto cha")
        self.assertEqual(decision.language, "unknown")

    def test_numbers_are_unknown(self) -> None:
        decision = self.router.detect("2026-06-06")
        self.assertEqual(decision.language, "unknown")

    def test_empty_string_is_unknown(self) -> None:
        decision = self.router.detect("")
        self.assertEqual(decision.language, "unknown")
        self.assertEqual(decision.confidence, 1.0)

    def test_uses_confident_whisper_language(self) -> None:
        decision = self.router.detect("namaste", whisper_language="ne", whisper_confidence=0.91)
        self.assertEqual(decision.language, "ne")

    def test_splits_mixed_text_for_tts(self) -> None:
        chunks = self.router.split_for_tts("ठीक छ. I will remind you.")
        self.assertEqual(chunks[0][1], "ne")
        self.assertEqual(chunks[1][1], "en")


if __name__ == "__main__":
    unittest.main()
