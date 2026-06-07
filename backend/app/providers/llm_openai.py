from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from app.providers.llm import LLMResult


class OpenAILLMProvider:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.35,
        max_tokens: int = 180,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "openai"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def test_connection(self) -> dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "detail": "OpenAI API key is missing.", "latency_ms": 0.0}

        started = time.perf_counter()
        try:
            self.chat("Hello, reply with OK.", system_prompt="Reply exactly with OK.")
            latency = (time.perf_counter() - started) * 1000
            return {
                "ok": True,
                "detail": "OpenAI connection verified successfully.",
                "model": self.model,
                "latency_ms": latency,
            }
        except Exception as exc:
            return {"ok": False, "detail": f"OpenAI test failed: {exc}", "latency_ms": 0.0}

    def chat(self, text: str, system_prompt: str = "", options: dict[str, Any] | None = None) -> LLMResult:
        if not self.api_key:
            raise ValueError("OpenAI API key is missing.")

        started = time.perf_counter()
        temp = options.get("temperature", self.temperature) if options else self.temperature
        tokens = options.get("max_tokens", self.max_tokens) if options else self.max_tokens

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": text.strip()},
            ],
            "temperature": temp,
            "max_tokens": tokens,
        }

        url = "https://api.openai.com/v1/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                resp_data = json.loads(response.read().decode("utf-8"))

            total_ms = (time.perf_counter() - started) * 1000
            choices = resp_data.get("choices", [])
            if not choices:
                raise ValueError("No choices returned from OpenAI chat completions.")

            content = str(choices[0].get("message", {}).get("content", "")).strip()
            return LLMResult(
                text=content,
                first_token_ms=total_ms,
                total_ms=total_ms,
                model=self.model,
            )
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8")
            try:
                err_json = json.loads(err_body)
                err_msg = err_json.get("error", {}).get("message", err_body)
            except Exception:
                err_msg = err_body
            raise RuntimeError(f"OpenAI API error ({exc.code}): {err_msg}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAI connection error: {exc}") from exc

    def list_models(self) -> list[str]:
        return ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"]
