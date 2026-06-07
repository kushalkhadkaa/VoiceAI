from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from app.providers.stt import ProviderUnavailableError

@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    first_token_ms: float
    total_ms: float
    model: str
    load_ms: float | None = None
    prompt_eval_ms: float | None = None
    generation_ms: float | None = None


class LLMProvider(Protocol):
    def chat(self, text: str) -> LLMResult:
        ...


class OllamaLLMProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 60,
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

    def chat(self, text: str) -> LLMResult:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text.strip()},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "top_p": 0.9,
            },
        }
        data = self._post_json("/api/chat", payload)
        total_ms = (time.perf_counter() - started) * 1000
        message = data.get("message") if isinstance(data.get("message"), dict) else {}
        response = self._strip_thinking(str(message.get("content", ""))).strip()
        if not response:
            response = "I heard you, but I could not generate a useful response."
        return LLMResult(
            text=response,
            first_token_ms=total_ms,
            total_ms=total_ms,
            model=self.model,
            load_ms=_ns_to_ms(data.get("load_duration")),
            prompt_eval_ms=_ns_to_ms(data.get("prompt_eval_duration")),
            generation_ms=_ns_to_ms(data.get("eval_duration")),
        )

    def probe(self) -> tuple[bool, str]:
        try:
            self._get_json("/api/tags")
        except ProviderUnavailableError as exc:
            return False, str(exc)
        return True, f"Ollama reachable at {self.base_url}"

    def _post_json(self, path: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for _ in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_exc = exc
        raise ProviderUnavailableError(
            f"Ollama is not running. Start Ollama, then run: ollama pull {self.model}"
        ) from last_exc

    def _get_json(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base_url}{path}", timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderUnavailableError(f"Ollama is not running. Start Ollama, then run: ollama pull {self.model}") from exc

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _ns_to_ms(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 1_000_000
    except (TypeError, ValueError):
        return None
