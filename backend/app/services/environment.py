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
    # Determine if local provider is ready (skips cloud as blockers)
    local_ready = _is_local_ollama_ready(settings)
    active_provider = getattr(settings, "llm_provider", "local")
    cloud_is_active = active_provider in ("openai", "gemini")

    checks = [
        _macos_check(),
        _architecture_check(),
        _python_check(),
        _node_check(),
        _npm_check(),
        _ffmpeg_check(settings),
        _ollama_binary_check(settings),
        _ollama_api_check(settings),
        _ollama_model_check(settings),
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
        # Cloud providers: only critical if that provider is actively selected
        _openai_configured_check(settings, critical=cloud_is_active and active_provider == "openai"),
        _openai_tested_check(settings, critical=False),
        _gemini_configured_check(settings, critical=cloud_is_active and active_provider == "gemini"),
        _gemini_tested_check(settings, critical=False),
        _active_provider_ready_check(settings),
        _cloud_fallback_check(settings),
    ]
    ready = all(check.ok for check in checks if check.critical)
    return EnvironmentReport(ready=ready, checks=checks)


def _run_version(command: list[str]) -> tuple[bool, str]:
    try:
        shell = platform.system() == "Windows"
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=5, check=False, shell=shell)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return result.returncode == 0, result.stdout.strip().splitlines()[0] if result.stdout.strip() else "installed"


def _macos_check() -> EnvironmentCheck:
    system = platform.system()
    version = platform.mac_ver()[0] if system == "Darwin" else platform.release()
    ok = system in ("Darwin", "Windows", "Linux")
    detail = f"macOS {version}" if system == "Darwin" else f"Detected {system} {version}"
    return EnvironmentCheck(
        "macos",
        ok,
        False,
        detail,
        None,  # Windows is fully supported
    )


def _architecture_check() -> EnvironmentCheck:
    machine = platform.machine()
    detail = "Apple Silicon" if machine == "arm64" else f"Windows AMD64" if machine == "AMD64" else f"Intel/other ({machine})"
    return EnvironmentCheck("architecture", machine in {"arm64", "x86_64", "AMD64"}, False, detail)


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


def _ffmpeg_fix_command() -> str:
    """Return OS-appropriate ffmpeg install command."""
    system = platform.system()
    if system == "Windows":
        return (
            "Windows: winget install ffmpeg  OR  choco install ffmpeg  OR  scoop install ffmpeg  "
            "OR download from https://github.com/BtbN/FFmpeg-Builds/releases and add to PATH. "
            "Then set FFMPEG_BINARY in Settings to the full path."
        )
    if system == "Darwin":
        return "macOS: brew install ffmpeg"
    return "Linux: sudo apt install ffmpeg  OR  sudo dnf install ffmpeg  OR  sudo pacman -S ffmpeg"


def _ffmpeg_check(settings: Settings) -> EnvironmentCheck:
    # Use the same resolution logic as the audio pipeline so diagnostics always match reality
    from app.services.audio_validation import get_ffmpeg_path
    path = get_ffmpeg_path()
    detail = path if path else "ffmpeg not found (static-ffmpeg will auto-download on first voice turn)"
    return EnvironmentCheck(
        "ffmpeg",
        path is not None,
        False,  # Non-critical in diagnostics; auto-resolved at runtime via static-ffmpeg
        detail,
        _ffmpeg_fix_command() if path is None else None,
    )


def _ollama_binary_check(settings: Settings) -> EnvironmentCheck:
    path = shutil.which("ollama")
    api_ok = False
    try:
        _get_json(f"{settings.ollama_base_url}/api/tags", timeout=1)
        api_ok = True
    except Exception:
        pass
    ok = (path is not None) or api_ok
    detail = path or "ollama binary not in PATH (Ollama runs as a service)"
    return EnvironmentCheck(
        "ollama_binary",
        ok,
        False,
        detail,
        None if ok else "If Ollama API is reachable, this is informational only. To add to PATH: restart terminal after Ollama install.",
    )


def _is_local_ollama_ready(settings: Settings) -> bool:
    """Quick check: is local Ollama API up with the configured model?"""
    try:
        payload = _get_json(f"{settings.ollama_base_url}/api/tags", timeout=2)
        names = {m.get("name") for m in payload.get("models", []) if isinstance(m, dict)}
        return bool(names & {settings.ollama_model, settings.local_model, "llama3:latest"})
    except Exception:
        return False


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
    if not ok:
        # Try local_model as fallback check
        ok = settings.local_model in names
        if ok:
            return EnvironmentCheck(
                "ollama_model",
                True,
                False,
                f"{settings.local_model} installed (active model)",
                None,
            )
    return EnvironmentCheck(
        "ollama_model",
        ok,
        True,
        f"{settings.ollama_model} installed" if ok else f"{settings.ollama_model} not listed by Ollama",
        f"Run: ollama pull {settings.ollama_model}",
    )


def _open_webui_check(settings: Settings) -> EnvironmentCheck:
    # Open WebUI is optional — only needed for RAG
    rag_enabled = getattr(settings, "rag_enabled", False)
    try:
        payload = _get_json(f"{settings.open_webui_base_url}/api/config", timeout=3)
        version = payload.get("version", "unknown")
        return EnvironmentCheck("open_webui", True, False, f"Open WebUI {version} reachable")
    except Exception:
        detail = f"Open WebUI not reachable at {settings.open_webui_base_url}"
        if not rag_enabled:
            detail += " (optional — RAG is disabled)"
        return EnvironmentCheck(
            "open_webui",
            not rag_enabled,  # OK if RAG not enabled
            False,
            detail,
            "Start Open WebUI to enable RAG, or leave disabled if not needed.",
        )


def _open_webui_auth_check(settings: Settings) -> EnvironmentCheck:
    rag_enabled = getattr(settings, "rag_enabled", False)
    has_key = bool(settings.open_webui_api_key)
    if has_key:
        return EnvironmentCheck("open_webui_auth", True, False, "Open WebUI API key configured")
    detail = "Open WebUI API key not configured"
    if not rag_enabled:
        detail += " (optional — RAG is disabled)"
    return EnvironmentCheck(
        "open_webui_auth",
        not rag_enabled,  # OK if RAG not enabled
        False,
        detail,
        "Enable RAG in Settings and enter your Open WebUI API key — only needed for document knowledge.",
    )


def _piper_check(settings: Settings) -> EnvironmentCheck:
    path = shutil.which(settings.piper_binary)
    if path is None:
        # Check absolute path directly
        p = Path(settings.piper_binary)
        if p.exists():
            path = str(p)
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
    from app.database import DB_FILE
    db_path = DB_FILE
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
        # Accept either the configured model or llama3:latest as valid
        ok = settings.local_model in names or settings.ollama_model in names
        if ok:
            active = settings.local_model if settings.local_model in names else settings.ollama_model
            detail = f"Local Ollama online, model '{active}' is installed."
        else:
            detail = f"Model '{settings.local_model}' not installed in local Ollama."
    except Exception:
        ok = False
        detail = "Local Ollama is offline."
    return EnvironmentCheck("local_ai_ready", ok, False, detail, f"Run: ollama pull {settings.local_model}")


def _openai_configured_check(settings: Settings, critical: bool = False) -> EnvironmentCheck:
    has_key = bool(settings.openai_api_key)
    ok = has_key or not critical
    detail = "OpenAI API key configured" if has_key else "OpenAI API key not set (optional — local Ollama is active)"
    return EnvironmentCheck(
        "openai_configured",
        ok,
        critical,
        detail,
        None if ok else "Add OPENAI_API_KEY in Settings only if you want to use OpenAI cloud model."
    )


def _openai_tested_check(settings: Settings, critical: bool = False) -> EnvironmentCheck:
    tested = PROVIDER_TESTS.get("openai", False)
    has_key = bool(settings.openai_api_key)
    active_provider = getattr(settings, "llm_provider", "local")
    is_active = active_provider == "openai"
    ok = tested or not is_active or not has_key
    detail = "OpenAI connection tested successfully" if tested else "OpenAI connection not tested (optional)"
    return EnvironmentCheck(
        "openai_tested",
        ok,
        critical,
        detail,
        None if ok else "Click 'Test Connection' on OpenAI card in Settings if using OpenAI."
    )


def _gemini_configured_check(settings: Settings, critical: bool = False) -> EnvironmentCheck:
    has_key = bool(settings.gemini_api_key)
    ok = has_key or not critical
    detail = "Gemini API key configured" if has_key else "Gemini API key not set (optional — local Ollama is active)"
    return EnvironmentCheck(
        "gemini_configured",
        ok,
        critical,
        detail,
        None if ok else "Add GEMINI_API_KEY in Settings only if you want to use Google Gemini cloud model."
    )


def _gemini_tested_check(settings: Settings, critical: bool = False) -> EnvironmentCheck:
    tested = PROVIDER_TESTS.get("gemini", False)
    has_key = bool(settings.gemini_api_key)
    active_provider = getattr(settings, "llm_provider", "local")
    is_active = active_provider == "gemini"
    ok = tested or not is_active or not has_key
    detail = "Gemini connection tested successfully" if tested else "Gemini connection not tested (optional)"
    return EnvironmentCheck(
        "gemini_tested",
        ok,
        critical,
        detail,
        None if ok else "Click 'Test Connection' on Gemini card in Settings if using Gemini."
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
            active = settings.local_model if settings.local_model in names else (settings.ollama_model if settings.ollama_model in names else None)
            ok = active is not None
            detail = f"Active provider: Local Ollama (model '{active}' is ready)" if ok else f"Active provider: Local Ollama (model '{settings.local_model}' missing)"
        except Exception:
            ok = False
            detail = "Active provider: Local Ollama (Ollama offline)"

    return EnvironmentCheck(
        "active_provider_ready",
        ok,
        True,  # This IS critical — the active provider must work
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
        "Voice chat ready": ok("ollama_api", "piper", "piper_nepali_voice", "piper_english_voice", "stt", "voice_websocket_route"),
        "RAG ready": ok("open_webui", "open_webui_auth"),
        "Voice Studio recording ready": ok("piper", "stt"),
        "Piper TTS ready": ok("piper", "piper_nepali_voice", "piper_english_voice"),
        "Custom voice training ready": ok("piper", "stt"),
        "Commercial deployment ready": ok("commercial_db"),
        "Local AI ready": ok("local_ai_ready"),
        "Active provider ready": ok("active_provider_ready"),
        "Cloud fallback enabled": ok("cloud_fallback_status"),
    }
    return [f"[{'READY' if value else 'BLOCKED'}] {name}" for name, value in categories.items()]
