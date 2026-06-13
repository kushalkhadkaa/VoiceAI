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

        # Retry transient failures (flaky TLS, network blips, 429/5xx) — these have
        # intermittently broken cloud calls; fail fast only on real client errors.
        last_exc: Exception | None = None
        for attempt in range(3):
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
                # 429 / 5xx are transient — retry; other 4xx are real errors — raise now.
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    last_exc = RuntimeError(f"OpenAI API error ({exc.code}): {err_msg}")
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAI API error ({exc.code}): {err_msg}") from exc
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAI connection error: {exc}") from exc
        raise RuntimeError(f"OpenAI connection error: {last_exc}")

    def list_models(self) -> list[str]:
        """Static fallback list — always available."""
        return [
            "gpt-4o",
            "o3-mini",
            "o1",
            "gpt-4o-mini",
            "o1-mini",
            "gpt-4-turbo",
            "gpt-4.5-preview",
        ]

    def list_models_live(self) -> list[str]:
        """Fetch model list from OpenAI API — requires valid key."""
        if not self.api_key:
            return self.list_models()
        url = "https://api.openai.com/v1/models"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.api_key}"}, method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            allowed = {
                "gpt-4o",
                "o3-mini",
                "o1",
                "gpt-4o-mini",
                "o1-mini",
                "gpt-4-turbo",
                "gpt-4.5-preview",
            }
            model_ids = []
            for m in data.get("data", []):
                mid = m["id"]
                if mid in allowed:
                    model_ids.append(mid)
            
            preferred = [
                "gpt-4o",
                "o3-mini",
                "o1",
                "gpt-4o-mini",
                "o1-mini",
                "gpt-4-turbo",
                "gpt-4.5-preview",
            ]
            model_ids = sorted(list(set(model_ids)), key=lambda m: preferred.index(m) if m in preferred else len(preferred))
            return model_ids or self.list_models()
        except Exception:
            return self.list_models()

