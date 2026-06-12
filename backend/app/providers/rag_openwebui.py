import logging
import json
import urllib.error
import urllib.request
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


class OpenWebUIRagProvider:
    def __init__(self, settings: Settings):
        self.settings = settings

    def status(self) -> dict[str, Any]:
        try:
            config = self._get_json("/api/config", require_key=False)
            ok = bool(config.get("status"))
            version = config.get("version")
        except Exception as exc:
            logger.warning("Failed to fetch Open WebUI RAG status: %s", exc)
            ok = False
            version = None

        return {
            "ok": ok,
            "base_url": self.settings.open_webui_base_url,
            "version": version,
            "api_key_configured": bool(self.settings.open_webui_api_key),
            "enabled": self.settings.rag_enabled,
            "default_collection": self.settings.rag_default_collection,
            "fallback_to_ollama": self.settings.rag_fallback_to_ollama,
        }

    def collections(self) -> dict[str, Any]:
        if not self.settings.open_webui_api_key:
            return {"ok": False, "collections": [], "detail": "Open WebUI API key missing"}
        # Open WebUI API surfaces differ by version; keep this conservative and nonfatal.
        for path in ("/api/v1/knowledge/", "/api/knowledge/"):
            try:
                payload = self._get_json(path, require_key=True)
            except Exception:
                continue
            items = payload if isinstance(payload, list) else payload.get("data", payload.get("items", []))
            return {"ok": True, "collections": items if isinstance(items, list) else []}
        return {"ok": False, "collections": [], "detail": "No supported Open WebUI knowledge endpoint responded"}

    def test(self, query: str, collection_id: str | None) -> dict[str, Any]:
        if not self.settings.open_webui_api_key:
            return {"ok": False, "detail": "RAG API key is required before SwarLocal can query Open WebUI collections."}
        try:
            payload = {
                "model": self.settings.ollama_model,
                "messages": [{"role": "user", "content": query}],
                "stream": False,
            }
            cid = collection_id or self.settings.rag_default_collection
            if cid:
                payload["files"] = [{"type": "collection", "id": cid}]
            
            url = f"{self.settings.open_webui_base_url}/api/v1/chat/completions"
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.settings.open_webui_api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                choices = resp_data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    return {"ok": True, "detail": "Success", "response": content}
                return {"ok": False, "detail": "No choices returned from Open WebUI"}
        except Exception as exc:
            return {"ok": False, "detail": f"RAG query failed: {exc}"}

    def sync(self) -> dict[str, Any]:
        return {"ok": True, "detail": "SwarLocal reads Open WebUI collections on demand; manage documents in Open WebUI."}

    def _get_json(self, path: str, *, require_key: bool) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.settings.open_webui_base_url}{path}")
        if require_key:
            request.add_header("Authorization", f"Bearer {self.settings.open_webui_api_key}")
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {"data": payload}
