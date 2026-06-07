from __future__ import annotations

from typing import Any, Protocol
from app.providers.llm import LLMResult

class BaseLLMProvider(Protocol):
    @property
    def name(self) -> str:
        ...

    @property
    def enabled(self) -> bool:
        ...

    @property
    def available(self) -> bool:
        ...

    def test_connection(self) -> dict[str, Any]:
        ...

    def chat(self, text: str, system_prompt: str = "", options: dict[str, Any] | None = None) -> LLMResult:
        ...

    def list_models(self) -> list[str]:
        ...
