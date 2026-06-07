#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import Settings
from app.providers.llm_ollama import OllamaLLMProvider
from app.providers.llm_openwebui import OpenWebUILLMProvider
from app.providers.stt import FasterWhisperSTTProvider, ProviderUnavailableError
from app.providers.tts import PiperTTSProvider, TTSPart
from app.providers.web_retrieval import WebRetrievalProvider
from app.services.conversation import ConversationService
from app.services.language_router import LanguageRouter


def record(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main() -> int:
    settings = Settings.from_env()
    output_dir = Path(".local/eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    router = LanguageRouter()
    llm = OllamaLLMProvider(
        settings.ollama_base_url,
        settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        retries=settings.ollama_retries,
        temperature=settings.ollama_temperature,
        num_predict=settings.ollama_num_predict,
        keep_alive=settings.ollama_keep_alive,
        system_prompt=settings.system_prompt,
    )
    tts = PiperTTSProvider(settings.piper_binary, settings.piper_nepali_voice, settings.piper_english_voice, output_dir)
    failures = 0

    try:
        chat = llm.chat("Reply with one short English sentence.")
        record("ollama_chat", bool(chat.text), f"{settings.ollama_model}: {chat.text[:80]}")
    except Exception as exc:
        failures += 1
        record("ollama_chat", False, str(exc))

    for name, part in [
        ("piper_english_tts", TTSPart("Hello, this is an English Piper smoke test.", "en")),
        ("piper_nepali_tts", TTSPart("नमस्ते, यो नेपाली Piper परीक्षण हो।", "ne")),
    ]:
        try:
            result = tts.synthesize([part])
            record(name, result.audio_path.exists(), str(result.audio_path))
        except Exception as exc:
            failures += 1
            record(name, False, str(exc))

    sample_audio = Path("tests/fixtures/sample.wav")
    if sample_audio.exists():
        try:
            stt = FasterWhisperSTTProvider(settings.whisper_model_size, settings.whisper_device, settings.whisper_compute_type)
            result = stt.transcribe_file(sample_audio)
            record("faster_whisper_sample", bool(result.text), result.text[:120])
        except Exception as exc:
            failures += 1
            record("faster_whisper_sample", False, str(exc))
    else:
        record("faster_whisper_sample", True, "Skipped; tests/fixtures/sample.wav not present.")

    try:
        openwebui_llm = OpenWebUILLMProvider(settings, llm)
        web_retrieval = WebRetrievalProvider(
            settings.internet_retrieval_enabled,
            settings.internet_max_sources,
            settings.internet_require_citation,
            settings.internet_fallback_allowed,
        )
        service = ConversationService(
            stt_provider=FasterWhisperSTTProvider(settings.whisper_model_size, settings.whisper_device, settings.whisper_compute_type),
            llm_provider=llm,
            openwebui_llm_provider=openwebui_llm,
            tts_provider=tts,
            language_router=router,
            web_retrieval_provider=web_retrieval,
            audio_base_url=str(output_dir),
        )
        turn = service.handle_text("Say hello in one sentence.")
        record("full_text_to_tts", bool(turn.audio_url), f"{turn.response[:80]} -> {turn.audio_url}")
    except Exception as exc:
        failures += 1
        record("full_text_to_tts", False, str(exc))

    print(f"Outputs: {output_dir}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
