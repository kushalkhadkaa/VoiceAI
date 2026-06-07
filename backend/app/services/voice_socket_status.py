from __future__ import annotations

import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.environment import run_environment_checks


CHECK_BLOCKERS = {
    "ffmpeg": "ffmpeg missing",
    "ollama_api": "Ollama unavailable",
    "ollama_model": "selected Ollama model missing",
    "piper": "Piper unavailable",
    "piper_nepali_voice": "Piper voice missing",
    "piper_english_voice": "Piper voice missing",
    "stt": "STT unavailable",
    "open_webui": "Open WebUI unavailable",
    "open_webui_auth": "Open WebUI API key missing",
    "openai_configured": "OpenAI API key missing",
    "gemini_configured": "Gemini API key missing",
    "active_provider_ready": "selected active provider unavailable",
}


def build_voice_socket_status(settings: Settings) -> dict[str, Any]:
    report = run_environment_checks(settings)
    check_map = {check.name: check for check in report.checks}
    blocking_reasons: list[str] = []
    warnings: list[str] = []
    
    local_ready = _check_ok(check_map, "ollama_api") and _check_ok(check_map, "ollama_model")

    for check in report.checks:
        if check.ok:
            continue
        is_rag_check = check.name in {"open_webui", "open_webui_auth"}
        if is_rag_check:
            warnings.append(check.detail)
            continue
            
        if check.name == "openai_configured" and settings.llm_provider != "openai":
            warnings.append(check.detail)
            continue
        if check.name == "gemini_configured" and settings.llm_provider != "gemini":
            warnings.append(check.detail)
            continue
        if check.name in {"openai_tested", "gemini_tested"}:
            warnings.append(check.detail)
            continue
            
        if check.name == "openai_configured" and settings.llm_provider == "openai" and settings.cloud_fallback_to_local and local_ready:
            warnings.append(check.detail)
            continue
        if check.name == "gemini_configured" and settings.llm_provider == "gemini" and settings.cloud_fallback_to_local and local_ready:
            warnings.append(check.detail)
            continue

        reason = CHECK_BLOCKERS.get(check.name)
        if reason and reason not in blocking_reasons:
            blocking_reasons.append(reason)
        elif not check.critical:
            warnings.append(check.detail)

    if settings.rag_enabled:
        if not check_map.get("open_webui") or not check_map["open_webui"].ok:
            warnings.append("Open WebUI unavailable; direct local voice turns still work without RAG.")
        if not check_map.get("open_webui_auth") or not check_map["open_webui_auth"].ok:
            warnings.append("Open WebUI API key missing; selected RAG collections will be unavailable.")
        if not settings.rag_default_collection:
            warnings.append("No default RAG collection selected; choose a knowledge source per conversation.")

    for voice_path in (settings.piper_nepali_voice, settings.piper_english_voice):
        # If the path has custom voice folders but is missing, mark as selected custom voice unavailable
        is_custom = "custom" in str(voice_path).lower() or "myvoice" in str(voice_path).lower()
        missing = _missing_piper_parts(voice_path)
        if is_custom and (".onnx" in missing or ".onnx.json" in missing):
            if "selected custom voice unavailable" not in blocking_reasons:
                blocking_reasons.append("selected custom voice unavailable")
        if ".onnx" in missing and "Piper .onnx missing" not in blocking_reasons:
            blocking_reasons.append("Piper .onnx missing")
        if ".onnx.json" in missing and "Piper .onnx.json missing" not in blocking_reasons:
            blocking_reasons.append("Piper .onnx.json missing")

    active_provider = settings.llm_provider
    openai_ready = bool(settings.openai_api_key)
    gemini_ready = bool(settings.gemini_api_key)

    llm_ready = False
    if active_provider == "openai":
        llm_ready = openai_ready or (settings.cloud_fallback_to_local and local_ready)
    elif active_provider == "gemini":
        llm_ready = gemini_ready or (settings.cloud_fallback_to_local and local_ready)
    else:
        # local
        llm_ready = local_ready

    capabilities = {
        "backend": True,
        "websocket": True,
        "text_turns": llm_ready,
        "audio_turns": all(
            _check_ok(check_map, name)
            for name in ("ffmpeg", "piper", "piper_nepali_voice", "piper_english_voice", "stt")
        ) and llm_ready,
        "stt": _check_ok(check_map, "stt"),
        "tts": _check_ok(check_map, "piper") and _check_ok(check_map, "piper_nepali_voice") and _check_ok(check_map, "piper_english_voice"),
        "ollama": _check_ok(check_map, "ollama_api"),
        "open_webui": _check_ok(check_map, "open_webui"),
        "rag": _check_ok(check_map, "open_webui") and _check_ok(check_map, "open_webui_auth"),
        "internet_retrieval": True,
        "custom_voices": True,
    }
    return {
        "ok": True,
        "session_id": str(uuid.uuid4()),
        "capabilities": capabilities,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "checks": [asdict(check) for check in report.checks],
    }


def piper_binary_available(settings: Settings) -> bool:
    return shutil.which(settings.piper_binary) is not None


def _check_ok(check_map: dict[str, Any], name: str) -> bool:
    check = check_map.get(name)
    return bool(check and check.ok)


def _missing_piper_parts(model_path: Path) -> set[str]:
    missing: set[str] = set()
    if not model_path.exists():
        missing.add(".onnx")
    if not Path(f"{model_path}.json").exists():
        missing.add(".onnx.json")
    return missing
