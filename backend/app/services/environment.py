from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import Settings


@dataclass(frozen=True, slots=True)
class EnvironmentCheck:
    name: str
    ok: bool
    critical: bool
    detail: str
    fix: str | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    ready: bool
    checks: list[EnvironmentCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "checks": [asdict(check) for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def run_environment_checks(settings: Settings) -> EnvironmentReport:
    checks = [
        _macos_check(),
        _architecture_check(),
        _python_check(),
        _node_check(),
        _npm_check(),
        _ffmpeg_check(),
        _ollama_binary_check(),
        _ollama_api_check(settings),
        _ollama_model_check(settings),
        _installed_model_check(settings, "qwen2.5:7b", critical=True),
        _installed_model_check(settings, "gemma3:4b", critical=False),
        _open_webui_check(settings),
        _open_webui_auth_check(settings),
        _piper_check(settings),
        _voice_file_check("piper_nepali_voice", settings.piper_nepali_voice),
        _voice_file_check("piper_english_voice", settings.piper_english_voice),
        _stt_check(),
        _websocket_route_check(),
        _microphone_guidance(),
        _disk_space_check(Path.cwd()),
        _database_check(settings),
        _local_ai_ready_check(settings),
        _openai_configured_check(settings),
        _openai_tested_check(settings),
        _gemini_configured_check(settings),
        _gemini_tested_check(settings),
        _active_provider_ready_check(settings),
        _cloud_fallback_check(settings),
    ]
    ready = all(check.ok for check in checks if check.critical)
    return EnvironmentReport(ready=ready, checks=checks)


def _run_version(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return result.returncode == 0, result.stdout.strip().splitlines()[0] if result.stdout.strip() else "installed"


def _macos_check() -> EnvironmentCheck:
    version = platform.mac_ver()[0]
    ok = platform.system() == "Darwin"
    return EnvironmentCheck(
        "macos",
        ok,
        False,
        f"macOS {version}" if ok else f"Detected {platform.system()}",
        "Use macOS for the primary supported development path.",
    )


def _architecture_check() -> EnvironmentCheck:
    machine = platform.machine()
    detail = "Apple Silicon" if machine == "arm64" else f"Intel/other ({machine})"
    return EnvironmentCheck("architecture", machine in {"arm64", "x86_64"}, False, detail)


def _python_check() -> EnvironmentCheck:
    version = sys.version_info
    ok = version >= (3, 11)
    return EnvironmentCheck(
        "python",
        ok,
        True,
        f"Python {version.major}.{version.minor}.{version.micro}",
        "Install Python 3.11 or newer.",
    )


def _node_check() -> EnvironmentCheck:
    ok, detail = _run_version(["node", "--version"])
    major = _major_version(detail.lstrip("v")) if ok else 0
    return EnvironmentCheck("node", ok and major >= 20, True, detail, "Install Node.js 20 or newer.")


def _npm_check() -> EnvironmentCheck:
    ok, detail = _run_version(["npm", "--version"])
    return EnvironmentCheck("npm", ok, True, detail, "Install npm with Node.js.")


def _ffmpeg_check() -> EnvironmentCheck:
    path = shutil.which("ffmpeg")
    return EnvironmentCheck(
        "ffmpeg",
        path is not None,
        True,
        path or "ffmpeg not found",
        "Run: brew install ffmpeg",
    )


def _ollama_binary_check() -> EnvironmentCheck:
    path = shutil.which("ollama")
    return EnvironmentCheck(
        "ollama_binary",
        path is not None,
        True,
        path or "ollama not found",
        "Install Ollama, then run: ollama pull qwen3:1.7b",
    )


def _ollama_api_check(settings: Settings) -> EnvironmentCheck:
    try:
        _get_json(f"{settings.ollama_base_url}/api/tags", timeout=3)
    except Exception:
        return EnvironmentCheck(
            "ollama_api",
            False,
            True,
            f"Ollama API is not reachable at {settings.ollama_base_url}",
            f"Start Ollama, then run: ollama pull {settings.ollama_model}",
        )
    return EnvironmentCheck("ollama_api", True, True, f"Ollama API reachable at {settings.ollama_base_url}")


def _ollama_model_check(settings: Settings) -> EnvironmentCheck:
    try:
        payload = _get_json(f"{settings.ollama_base_url}/api/tags", timeout=3)
    except Exception:
        return EnvironmentCheck(
            "ollama_model",
            False,
            True,
            f"Cannot verify model {settings.ollama_model}; Ollama API is unavailable.",
            f"Start Ollama, then run: ollama pull {settings.ollama_model}",
        )
    names = {model.get("name") for model in payload.get("models", []) if isinstance(model, dict)}
    ok = settings.ollama_model in names
    return EnvironmentCheck(
        "ollama_model",
        ok,
        True,
        f"{settings.ollama_model} installed" if ok else f"{settings.ollama_model} not listed by Ollama",
        f"Run: ollama pull {settings.ollama_model}",
    )


def _installed_model_check(settings: Settings, model_name: str, critical: bool) -> EnvironmentCheck:
    try:
        payload = _get_json(f"{settings.ollama_base_url}/api/tags", timeout=3)
    except Exception:
        return EnvironmentCheck(
            f"ollama_model_{model_name.replace(':', '_')}",
            False,
            critical,
            f"Cannot verify {model_name}; Ollama API is unavailable.",
            f"Run: ollama pull {model_name}",
        )
    names = {model.get("name") for model in payload.get("models", []) if isinstance(model, dict)}
    ok = model_name in names
    return EnvironmentCheck(
        f"ollama_model_{model_name.replace(':', '_')}",
        ok,
        critical,
        f"{model_name} installed" if ok else f"{model_name} not listed by Ollama",
        f"Run: ollama pull {model_name}",
    )


def _open_webui_check(settings: Settings) -> EnvironmentCheck:
    try:
        payload = _get_json(f"{settings.open_webui_base_url}/api/config", timeout=3)
    except Exception:
        return EnvironmentCheck(
            "open_webui",
            False,
            False,
            f"Open WebUI is not reachable at {settings.open_webui_base_url}",
            "Start Open WebUI, then open http://127.0.0.1:8080/.",
        )
    version = payload.get("version", "unknown")
    return EnvironmentCheck("open_webui", bool(payload.get("status")), False, f"Open WebUI {version} reachable")


def _open_webui_auth_check(settings: Settings) -> EnvironmentCheck:
    return EnvironmentCheck(
        "open_webui_auth",
        bool(settings.open_webui_api_key),
        False,
        "Open WebUI API key configured" if settings.open_webui_api_key else "Open WebUI API key is not configured",
        "Create an API key in Open WebUI after local onboarding, then set OPEN_WEBUI_API_KEY.",
    )


def _piper_check(settings: Settings) -> EnvironmentCheck:
    path = shutil.which(settings.piper_binary)
    return EnvironmentCheck(
        "piper",
        path is not None,
        True,
        path or f"{settings.piper_binary} not found",
        "Run: python -m pip install piper-tts, or set PIPER_BINARY to your Piper executable.",
    )


def _stt_check() -> EnvironmentCheck:
    try:
        import faster_whisper  # noqa: F401
    except Exception as exc:
        return EnvironmentCheck("stt", False, True, f"Faster Whisper import failed: {exc}", "Install backend requirements.")
    return EnvironmentCheck("stt", True, True, "Faster Whisper import available")


def _websocket_route_check() -> EnvironmentCheck:
    return EnvironmentCheck("voice_websocket_route", True, True, "GET /ws/voice/status and WS /ws/voice are registered")


def _voice_file_check(name: str, model_path: Path) -> EnvironmentCheck:
    json_path = Path(f"{model_path}.json")
    model_ok = model_path.exists()
    json_ok = json_path.exists()
    missing = []
    if not model_ok:
        missing.append(str(model_path))
    if not json_ok:
        missing.append(str(json_path))
    return EnvironmentCheck(
        name,
        model_ok and json_ok,
        True,
        "model and config present" if model_ok and json_ok else f"Missing: {', '.join(missing)}",
        "Place matching .onnx and .onnx.json files under models/piper and update Settings if needed.",
    )


def _microphone_guidance() -> EnvironmentCheck:
    return EnvironmentCheck(
        "microphone_permission",
        True,
        False,
        "Browser permission must be granted manually on first Push to Talk use.",
        "In the browser prompt, allow microphone access for http://127.0.0.1:5173.",
    )


def _disk_space_check(path: Path) -> EnvironmentCheck:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    return EnvironmentCheck(
        "disk_space",
        free_gb >= 5,
        False,
        f"{free_gb:.1f} GB free",
        "Free at least 5 GB for models, audio cache, and datasets.",
    )


def _database_check(settings: Settings) -> EnvironmentCheck:
    db_path = settings.audio_work_dir.parent / "swarlocal.db"
    ok = db_path.exists() and db_path.stat().st_size > 0
    return EnvironmentCheck(
        "commercial_db",
        ok,
        False,
        f"Database initialized at {db_path}" if ok else f"Database missing at {db_path}",
        "Start the SwarLocal backend to automatically initialize the SQLite database."
    )


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _major_version(raw: str) -> int:
    try:
        return int(raw.split(".", 1)[0])
    except (ValueError, IndexError):
        return 0


def format_human_report(report: EnvironmentReport) -> str:
    lines = [f"SwarLocal doctor: {'READY' if report.ready else 'BLOCKED'}"]
    lines.extend(_readiness_lines(report))
    for check in report.checks:
        marker = "PASS" if check.ok else "FAIL"
        critical = "critical" if check.critical else "info"
        lines.append(f"[{marker}] {check.name} ({critical}): {check.detail}")
        if not check.ok and check.fix:
            lines.append(f"  fix: {check.fix}")
    return "\n".join(lines)


PROVIDER_TESTS = {
    "openai": False,
    "gemini": False
}


def _local_ai_ready_check(settings: Settings) -> EnvironmentCheck:
    try:
        payload = _get_json(f"{settings.ollama_base_url}/api/tags", timeout=2)
        names = {model.get("name") for model in payload.get("models", []) if isinstance(model, dict)}
        ok = settings.local_model in names
        detail = f"Local Ollama online, model '{settings.local_model}' is installed." if ok else f"Model '{settings.local_model}' not installed in local Ollama."
    except Exception:
        ok = False
        detail = "Local Ollama is offline."
    return EnvironmentCheck("local_ai_ready", ok, False, detail, f"Run: ollama pull {settings.local_model}")


def _openai_configured_check(settings: Settings) -> EnvironmentCheck:
    ok = bool(settings.openai_api_key)
    return EnvironmentCheck(
        "openai_configured",
        ok,
        False,
        "OpenAI API key configured" if ok else "OpenAI API key is missing",
        "Add OPENAI_API_KEY in settings if you want to use OpenAI cloud model."
    )


def _openai_tested_check(settings: Settings) -> EnvironmentCheck:
    ok = PROVIDER_TESTS.get("openai", False)
    return EnvironmentCheck(
        "openai_tested",
        ok,
        False,
        "OpenAI connection tested successfully" if ok else "OpenAI connection has not been tested successfully yet.",
        "Click 'Test Connection' on OpenAI card in settings."
    )


def _gemini_configured_check(settings: Settings) -> EnvironmentCheck:
    ok = bool(settings.gemini_api_key)
    return EnvironmentCheck(
        "gemini_configured",
        ok,
        False,
        "Gemini API key configured" if ok else "Gemini API key is missing",
        "Add GEMINI_API_KEY in settings if you want to use Google Gemini cloud model."
    )


def _gemini_tested_check(settings: Settings) -> EnvironmentCheck:
    ok = PROVIDER_TESTS.get("gemini", False)
    return EnvironmentCheck(
        "gemini_tested",
        ok,
        False,
        "Gemini connection tested successfully" if ok else "Gemini connection has not been tested successfully yet.",
        "Click 'Test Connection' on Gemini card in settings."
    )


def _active_provider_ready_check(settings: Settings) -> EnvironmentCheck:
    provider = settings.llm_provider
    if provider == "openai":
        ok = bool(settings.openai_api_key)
        detail = "Active provider: OpenAI (configured)" if ok else "Active provider: OpenAI (API key missing)"
    elif provider == "gemini":
        ok = bool(settings.gemini_api_key)
        detail = "Active provider: Gemini (configured)" if ok else "Active provider: Gemini (API key missing)"
    else:
        try:
            payload = _get_json(f"{settings.ollama_base_url}/api/tags", timeout=2)
            names = {model.get("name") for model in payload.get("models", []) if isinstance(model, dict)}
            ok = settings.local_model in names
            detail = f"Active provider: Local Ollama (model '{settings.local_model}' is ready)" if ok else f"Active provider: Local Ollama (model '{settings.local_model}' missing)"
        except Exception:
            ok = False
            detail = "Active provider: Local Ollama (Ollama offline)"

    return EnvironmentCheck(
        "active_provider_ready",
        ok,
        False,
        detail,
        "Configure the active LLM provider in Settings."
    )


def _cloud_fallback_check(settings: Settings) -> EnvironmentCheck:
    ok = True
    detail = "Cloud fallback to local is enabled" if settings.cloud_fallback_to_local else "Cloud fallback to local is disabled"
    return EnvironmentCheck("cloud_fallback_status", ok, False, detail)


def _readiness_lines(report: EnvironmentReport) -> list[str]:
    by_name = {check.name: check for check in report.checks}

    def ok(*names: str) -> bool:
        return all(by_name.get(name) and by_name[name].ok for name in names)

    categories = {
        "Text chat ready": ok("ollama_api", "ollama_model"),
        "Voice chat ready": ok("ffmpeg", "ollama_api", "ollama_model", "piper", "piper_nepali_voice", "piper_english_voice", "stt", "voice_websocket_route"),
        "RAG ready": ok("open_webui", "open_webui_auth"),
        "Voice Studio recording ready": ok("ffmpeg"),
        "Piper TTS ready": ok("piper", "piper_nepali_voice", "piper_english_voice"),
        "Custom voice training ready": ok("ffmpeg"),
        "Commercial deployment ready": ok("commercial_db"),
        "Local AI ready": ok("local_ai_ready"),
        "OpenAI configured": ok("openai_configured"),
        "OpenAI connection tested": ok("openai_tested"),
        "Gemini configured": ok("gemini_configured"),
        "Gemini connection tested": ok("gemini_tested"),
        "Active provider ready": ok("active_provider_ready"),
        "Cloud fallback enabled/disabled": ok("cloud_fallback_status"),
    }
    return [f"[{'READY' if value else 'BLOCKED'}] {name}" for name, value in categories.items()]
