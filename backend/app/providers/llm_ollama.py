from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from app.providers.llm import LLMResult, OllamaLLMProvider as LegacyOllamaProvider
from app.providers.stt import ProviderUnavailableError


class OllamaLLMProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        retries: int = 1,
        temperature: float = 0.35,
        num_predict: int = 180,
        keep_alive: str = "10m",
        system_prompt: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.temperature = temperature
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.system_prompt = system_prompt
        self._legacy = LegacyOllamaProvider(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            retries=retries,
            temperature=temperature,
            num_predict=num_predict,
            keep_alive=keep_alive,
            system_prompt=system_prompt,
        )

    @property
    def name(self) -> str:
        return "local"

    @property
    def enabled(self) -> bool:
        return True

    @property
    def available(self) -> bool:
        ok, _ = self._legacy.probe()
        return ok

    def test_connection(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            # 1. Verify reachable
            ok, msg = self._legacy.probe()
            if not ok:
                return {"ok": False, "detail": msg, "latency_ms": 0.0}

            # 2. Verify model exists
            models = self.list_models()
            if self.model not in models:
                return {
                    "ok": False,
                    "detail": f"Model '{self.model}' is not installed in Ollama. Available: {', '.join(models[:5])}",
                    "latency_ms": 0.0,
                }

            # 3. Send test English turn
            self.chat("Hello, answer OK.", system_prompt="Answer only OK.")
            # 4. Send test Nepali turn
            self.chat("नमस्ते, OK भन्नुहोस्।", system_prompt="Answer only OK.")

            latency = (time.perf_counter() - started) * 1000
            return {
                "ok": True,
                "detail": "Local Ollama connection verified successfully.",
                "model": self.model,
                "latency_ms": latency,
            }
        except Exception as exc:
            return {"ok": False, "detail": f"Local Ollama test failed: {exc}", "latency_ms": 0.0}

    def chat(self, text: str, system_prompt: str = "", options: dict[str, Any] | None = None) -> LLMResult:
        if system_prompt:
            self._legacy.system_prompt = system_prompt
        if options:
            if "temperature" in options:
                self._legacy.temperature = options["temperature"]
            if "num_predict" in options:
                self._legacy.num_predict = options["num_predict"]
        return self._legacy.chat(text)

    def list_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
            return [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
        except Exception:
            return []
