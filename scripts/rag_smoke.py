#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import Settings
from app.providers.rag_openwebui import OpenWebUIRagProvider


def record(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main() -> int:
    settings = Settings.from_env()
    provider = OpenWebUIRagProvider(settings)

    try:
        status = provider.status()
    except Exception as exc:
        status = {
            "ok": False,
            "enabled": settings.rag_enabled,
            "base_url": settings.open_webui_base_url,
            "detail": str(exc),
        }

    if not settings.rag_enabled:
        record("rag_config", True, f"RAG is off by default at {settings.open_webui_base_url}.")
        return 0

    if not status.get("ok"):
        detail = status.get("detail") or f"Open WebUI did not respond at {settings.open_webui_base_url}."
        record("openwebui_status", False, detail)
        return 2

    collections = provider.collections()
    if not collections.get("ok"):
        record("rag_collections", False, collections.get("detail", "Unable to list Open WebUI collections."))
        return 2

    record("rag_collections", True, f"{len(collections.get('collections', []))} collection(s) visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
