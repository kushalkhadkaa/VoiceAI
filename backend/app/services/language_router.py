from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import LanguageCode

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
NUMBER_RE = re.compile(r"\d")
SENTENCE_RE = re.compile(r"[^।.!?\n]+[।.!?]?")
ROMANIZED_NEPALI_WORDS = {
    "namaste",
    "mero",
    "timro",
    "tapai",
    "tapaiko",
    "kasto",
    "cha",
    "chha",
    "ho",
    "hoina",
    "dhanyabad",
    "kripaya",
    "aja",
    "bholi",
    "hijo",
    "malai",
    "garnu",
}


@dataclass(frozen=True, slots=True)
class LanguageDecision:
    language: LanguageCode
    confidence: float
    reason: str


class LanguageRouter:
    def detect(
        self,
        text: str,
        whisper_language: str | None = None,
        whisper_confidence: float | None = None,
    ) -> LanguageDecision:
        cleaned = text.strip()
        if not cleaned:
            return LanguageDecision("unknown", 1.0, "Empty text.")
        has_devanagari = bool(DEVANAGARI_RE.search(cleaned))
        has_latin = bool(LATIN_RE.search(cleaned))
        has_number = bool(NUMBER_RE.search(cleaned))

        if whisper_language in {"ne", "en"} and whisper_confidence is not None and whisper_confidence >= 0.72:
            if whisper_language == "ne" and has_latin and has_devanagari:
                return LanguageDecision("mixed", 0.72, "Whisper detected Nepali and text contains Latin script.")
            return LanguageDecision(whisper_language, whisper_confidence, "Whisper language detection.")

        if has_devanagari and has_latin:
            return LanguageDecision("mixed", 0.82, "Text contains both Devanagari and Latin scripts.")
        if has_devanagari:
            return LanguageDecision("ne", 0.9, "Text contains Devanagari script.")
        if has_latin:
            if self._looks_romanized_nepali(cleaned):
                return LanguageDecision("unknown", 0.48, "Text may be romanized Nepali; user preference is needed.")
            return LanguageDecision("en", 0.85, "Text is mostly Latin script.")
        if has_number:
            return LanguageDecision("unknown", 0.35, "Numbers or dates do not provide a strong language signal.")
        return LanguageDecision("unknown", 0.2, "No strong language signal.")

    def split_for_tts(self, text: str) -> list[tuple[str, LanguageCode]]:
        chunks: list[tuple[str, LanguageCode]] = []
        for match in SENTENCE_RE.finditer(text.strip()):
            chunk = match.group(0).strip()
            if not chunk:
                continue
            language = self.detect(chunk).language
            if language == "mixed":
                chunks.extend(self._split_mixed_clause(chunk))
            else:
                chunks.append((chunk, language if language != "unknown" else "en"))
        if not chunks and text.strip():
            chunks.append((text.strip(), "en"))
        return self._merge_adjacent(chunks)

    def _split_mixed_clause(self, text: str) -> list[tuple[str, LanguageCode]]:
        parts = re.split(r"(\s+)", text)
        chunks: list[tuple[str, LanguageCode]] = []
        buffer: list[str] = []
        current: LanguageCode | None = None
        for part in parts:
            if not part:
                continue
            if part.isspace():
                buffer.append(part)
                continue
            language = self.detect(part).language
            if language in {"mixed", "unknown"}:
                language = current or "en"
            if current is None:
                current = language
            if language != current and "".join(buffer).strip():
                chunks.append(("".join(buffer).strip(), current))
                buffer = []
                current = language
            buffer.append(part)
        if buffer:
            chunks.append(("".join(buffer).strip(), current or "en"))
        return chunks

    @staticmethod
    def _looks_romanized_nepali(text: str) -> bool:
        words = {word.lower().strip(".,!?;:()[]{}\"'") for word in text.split()}
        return len(words & ROMANIZED_NEPALI_WORDS) >= 2 or (
            len(words & ROMANIZED_NEPALI_WORDS) >= 1 and not any(word in {"the", "and", "with", "please"} for word in words)
        )

    @staticmethod
    def _merge_adjacent(chunks: list[tuple[str, LanguageCode]]) -> list[tuple[str, LanguageCode]]:
        merged: list[tuple[str, LanguageCode]] = []
        for text, language in chunks:
            if merged and merged[-1][1] == language:
                merged[-1] = (f"{merged[-1][0]} {text}".strip(), language)
            else:
                merged.append((text, language))
        return merged
