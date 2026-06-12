"""
Embedding providers for the local knowledge base.

Two backends supported:
  - "ollama"               → uses Ollama's /api/embeddings (nomic-embed-text recommended)
  - "sentence-transformers" → uses local sentence-transformers (no Ollama dependency)

Both expose the same interface: embed(texts) → list[list[float]]
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider call fails. Carries provider/model/reason."""

    def __init__(self, provider: str, model: str, reason: str) -> None:
        self.provider = provider
        self.model = model
        self.reason = reason
        super().__init__(f"Embedding failed [provider={provider}, model={model}]: {reason}")


# Known output dimensions per provider/model. Used for collection metadata and
# to detect dimension mismatches when switching providers.
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-MiniLM-L6-v2": 384,
    "nomic-embed-text": 768,
}


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_one(self, text: str) -> list[float]: ...
    def status(self) -> dict[str, Any]: ...
    @property
    def dimension(self) -> int | None: ...
    @property
    def provider_id(self) -> str: ...
    @property
    def model_id(self) -> str: ...


class OllamaEmbeddingProvider:
    """Embeddings via Ollama /api/embeddings endpoint."""

    def __init__(self, base_url: str, model: str = "nomic-embed-text") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimension: int | None = EMBEDDING_DIMENSIONS.get(model)

    @property
    def provider_id(self) -> str:
        return "ollama"

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            payload = json.dumps({"model": self.model, "prompt": text}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
            except Exception as exc:
                raise EmbeddingError("ollama", self.model, str(exc)) from exc
            vec = data.get("embedding", [])
            if vec:
                self._dimension = len(vec)
            results.append(vec)
        return results

    def status(self) -> dict[str, Any]:
        try:
            # Try a quick ping — just check if the tags endpoint responds
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as r:
                models = json.loads(r.read().decode()).get("models", [])
            model_names = [m.get("name", "") for m in models]
            available = any(self.model in n for n in model_names)
            return {
                "ok": True,
                "provider": "ollama",
                "model": self.model,
                "model_available": available,
                "detail": f"Ollama reachable; model {'present' if available else 'not pulled yet (run: ollama pull ' + self.model + ')'}",
            }
        except Exception as exc:
            return {"ok": False, "provider": "ollama", "model": self.model, "detail": str(exc)}


class SentenceTransformerEmbeddingProvider:
    """Embeddings via sentence-transformers (fully local, no Ollama)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dimension: int | None = EMBEDDING_DIMENSIONS.get(model_name)

    @property
    def provider_id(self) -> str:
        return "sentence-transformers"

    @property
    def model_id(self) -> str:
        return self.model_name

    @property
    def _st_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers", self.model_name,
                    "sentence-transformers not installed. Run: pip install sentence-transformers",
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise EmbeddingError("sentence-transformers", self.model_name, str(exc)) from exc
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            vecs = self._st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("sentence-transformers", self.model_name, str(exc)) from exc
        return [v.tolist() for v in vecs]

    def status(self) -> dict[str, Any]:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            return {"ok": True, "provider": "sentence-transformers", "model": self.model_name, "detail": "Available"}
        except ImportError:
            return {
                "ok": False,
                "provider": "sentence-transformers",
                "model": self.model_name,
                "detail": "sentence-transformers not installed. Run: pip install sentence-transformers",
            }


class OpenAIEmbeddingProvider:
    """Embeddings via OpenAI /v1/embeddings."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", timeout_seconds: float = 30.0) -> None:
        self.api_key = api_key
        self.model = model or "text-embedding-3-small"
        self.timeout_seconds = timeout_seconds
        self._dimension: int | None = EMBEDDING_DIMENSIONS.get(self.model)

    @property
    def provider_id(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise EmbeddingError("openai", self.model, "OpenAI API key is missing for KB embeddings.")
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise EmbeddingError("openai", self.model, f"HTTP {exc.code}: {body}") from exc
        except Exception as exc:
            raise EmbeddingError("openai", self.model, str(exc)) from exc
        vectors = [item.get("embedding", []) for item in data.get("data", [])]
        if vectors and vectors[0]:
            self._dimension = len(vectors[0])
        return vectors

    def status(self) -> dict[str, Any]:
        if not self.api_key:
            return {"ok": False, "provider": "openai", "model": self.model, "detail": "OpenAI API key is missing."}
        return {"ok": True, "provider": "openai", "model": self.model, "detail": "Configured"}


def get_embedding_provider(
    provider: str,
    ollama_base_url: str,
    model: str,
    openai_api_key: str = "",
    timeout_seconds: float = 30.0,
) -> OllamaEmbeddingProvider | SentenceTransformerEmbeddingProvider | OpenAIEmbeddingProvider:
    """Build an embedding provider.

    Selection policy (OpenAI-first with a fully-local fallback):
      - provider == "openai"               -> OpenAI text-embedding-3-small.
      - provider == "sentence-transformers" -> local all-MiniLM-L6-v2.
      - provider == "ollama"               -> Ollama nomic-embed-text (explicit override).
      - provider in ("", "auto") or unknown -> OpenAI when a key is configured,
        otherwise fall back to sentence-transformers (NOT ollama) so search always
        works locally even with no key and no Ollama running.
    """
    provider = (provider or "auto").strip().lower()

    if provider == "openai":
        selected = model if model and model.startswith("text-embedding-") else "text-embedding-3-small"
        return OpenAIEmbeddingProvider(api_key=openai_api_key, model=selected, timeout_seconds=timeout_seconds)
    if provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(model_name=model or "all-MiniLM-L6-v2")
    if provider == "ollama":
        return OllamaEmbeddingProvider(base_url=ollama_base_url, model=model or "nomic-embed-text")

    # auto / unknown: OpenAI-first, else local sentence-transformers.
    if openai_api_key:
        selected = model if model and model.startswith("text-embedding-") else "text-embedding-3-small"
        return OpenAIEmbeddingProvider(api_key=openai_api_key, model=selected, timeout_seconds=timeout_seconds)
    return SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
