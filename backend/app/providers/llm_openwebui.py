from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from app.config import Settings
from app.providers.llm import LLMResult, OllamaLLMProvider
from app.providers.stt import ProviderUnavailableError


class OpenWebUILLMProvider:
    def __init__(self, settings: Settings, fallback_provider: OllamaLLMProvider):
        self.settings = settings
        self.fallback_provider = fallback_provider

    def chat(self, text: str, collection_id: str | None = None) -> LLMResult:
        if not self.settings.open_webui_api_key:
            if self.settings.rag_fallback_to_ollama:
                return self.fallback_provider.chat(text)
            raise ProviderUnavailableError("Open WebUI API key missing and RAG fallback is disabled.")

        started = time.perf_counter()
        payload = {
            "model": self.settings.ollama_model,
            "messages": [
                {"role": "system", "content": self.settings.system_prompt},
                {"role": "user", "content": text.strip()},
            ],
            "stream": False,
        }
        if collection_id:
            payload["files"] = [{"type": "collection", "id": collection_id}]

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

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                total_ms = (time.perf_counter() - started) * 1000
                choices = resp_data.get("choices", [])
                if not choices:
                    raise ValueError("No choices returned from Open WebUI chat completions.")
                content = str(choices[0].get("message", {}).get("content", "")).strip()
                return LLMResult(
                    text=content,
                    first_token_ms=total_ms,
                    total_ms=total_ms,
                    model=self.settings.ollama_model,
                )
        except Exception as exc:
            if self.settings.rag_fallback_to_ollama:
                # Fallback to Ollama direct
                result = self.fallback_provider.chat(text)
                # Mark that fallback was used
                return result
            raise ProviderUnavailableError(f"Open WebUI chat completion failed: {exc}") from exc
