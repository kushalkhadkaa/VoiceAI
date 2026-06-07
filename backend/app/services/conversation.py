from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.providers.llm import LLMProvider, LLMResult
from app.providers.llm_openwebui import OpenWebUILLMProvider
from app.providers.stt import STTProvider
from app.providers.tts import FallbackBlockedError, TTSPart, TTSProvider
from app.providers.web_retrieval import WebRetrievalProvider
from app.schemas import ConversationResponse, TimingMetrics
from app.services.language_router import LanguageRouter


class ConversationService:
    def __init__(
        self,
        stt_provider: STTProvider,
        llm_provider: LLMProvider,
        openwebui_llm_provider: OpenWebUILLMProvider,
        tts_provider: TTSProvider,
        language_router: LanguageRouter,
        web_retrieval_provider: WebRetrievalProvider,
        audio_base_url: str = "/audio",
    ) -> None:
        self.stt_provider = stt_provider
        self.llm_provider = llm_provider
        self.openwebui_llm_provider = openwebui_llm_provider
        self.tts_provider = tts_provider
        self.language_router = language_router
        self.web_retrieval_provider = web_retrieval_provider
        self.audio_base_url = audio_base_url.rstrip("/")
        self.settings = openwebui_llm_provider.settings

    def handle_audio(
        self,
        audio_path: Path,
        voice_id: str | None = None,
        knowledge_id: str | None = None,
        use_internet: bool = False,
        llm_provider_id: str | None = None,
    ) -> ConversationResponse:
        started = time.perf_counter()
        stt_result = self.stt_provider.transcribe_file(audio_path)
        transcript_ms = stt_result.duration_ms
        return self._complete_turn(
            text=stt_result.text,
            started=started,
            transcript_ms=transcript_ms,
            whisper_language=stt_result.language,
            whisper_confidence=stt_result.confidence,
            voice_id=voice_id,
            knowledge_id=knowledge_id,
            use_internet=use_internet,
            llm_provider_id=llm_provider_id,
        )

    def handle_text(
        self,
        text: str,
        voice_id: str | None = None,
        knowledge_id: str | None = None,
        use_internet: bool = False,
        llm_provider_id: str | None = None,
    ) -> ConversationResponse:
        started = time.perf_counter()
        return self._complete_turn(
            text=text,
            started=started,
            transcript_ms=None,
            voice_id=voice_id,
            knowledge_id=knowledge_id,
            use_internet=use_internet,
            llm_provider_id=llm_provider_id,
        )

    def _instantiate_provider(self, provider_id: str) -> Any:
        from app.providers.llm_ollama import OllamaLLMProvider
        from app.providers.llm_openai import OpenAILLMProvider
        from app.providers.llm_gemini import GeminiLLMProvider

        if provider_id == "openai":
            return OpenAILLMProvider(
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_model or "gpt-4o-mini",
                temperature=self.settings.cloud_temperature,
                max_tokens=self.settings.cloud_max_tokens,
                timeout_seconds=self.settings.cloud_timeout_seconds,
            )
        elif provider_id == "gemini":
            return GeminiLLMProvider(
                api_key=self.settings.gemini_api_key,
                model=self.settings.gemini_model or "gemini-1.5-flash",
                temperature=self.settings.cloud_temperature,
                max_tokens=self.settings.cloud_max_tokens,
                timeout_seconds=self.settings.cloud_timeout_seconds,
            )
        else:
            return OllamaLLMProvider(
                base_url=self.settings.ollama_base_url,
                model=self.settings.local_model or "qwen2.5:7b",
                timeout_seconds=self.settings.ollama_timeout_seconds,
                retries=self.settings.ollama_retries,
                temperature=self.settings.ollama_temperature,
                num_predict=self.settings.ollama_num_predict,
                keep_alive=self.settings.ollama_keep_alive,
                system_prompt=self.settings.system_prompt,
            )

    def _complete_turn(
        self,
        text: str,
        started: float,
        transcript_ms: float | None,
        whisper_language: str | None = None,
        whisper_confidence: float | None = None,
        voice_id: str | None = None,
        knowledge_id: str | None = None,
        use_internet: bool = False,
        llm_provider_id: str | None = None,
    ) -> ConversationResponse:
        input_language = self.language_router.detect(text, whisper_language, whisper_confidence)
        
        # 1. Check for web retrieval
        citations: list[dict[str, str]] = []
        internet_used = False
        
        should_search = use_internet
        if not should_search and self.web_retrieval_provider.enabled:
            lowered = text.lower()
            keywords = {"latest", "current", "news", "today", "recent", "now", "weather", "2026", "who is", "what is"}
            if any(kw in lowered for kw in keywords):
                should_search = True
                
        if should_search:
            search_results = self.web_retrieval_provider.retrieve(text)
            if search_results:
                internet_used = True
                citations = search_results
                
        # Prep prompt
        llm_input = text
        if internet_used and citations:
            context_str = "\n".join(
                f"- {c['title']}: {c['snippet']} (Source: {c['url']})" for c in citations
            )
            llm_input = (
                f"Use the following web search results to answer the query if relevant:\n"
                f"{context_str}\n\n"
                f"Query: {text}"
            )

        # 2. LLM Provider Routing and Fallbacks
        target_provider = llm_provider_id or self.settings.llm_provider or "local"
        if target_provider == "auto":
            # Auto defaults to local first
            actual_provider = "local"
        else:
            actual_provider = target_provider

        if actual_provider == "local":
            provider_instance = self.llm_provider
        else:
            provider_instance = self._instantiate_provider(actual_provider)
        
        rag_used = False
        rag_fallback_used = False
        fallback_used = False
        fallback_reason = None
        rag_path = None
        llm_result = None

        if knowledge_id and knowledge_id != "none":
            rag_used = True
            if actual_provider == "local":
                rag_path = "local_openwebui"
                try:
                    llm_result = self.openwebui_llm_provider.chat(llm_input, collection_id=knowledge_id)
                except Exception as e:
                    if self.settings.rag_fallback_to_ollama:
                        rag_fallback_used = True
                        fallback_reason = f"Local RAG failed: {e}. Falling back to direct Ollama."
                        llm_result = provider_instance.chat(llm_input, system_prompt=self.settings.system_prompt)
                    else:
                        raise
            else:
                # Cloud provider uses direct chat as RAG context adapter is not yet integrated
                rag_path = "cloud_direct_chat"
                try:
                    llm_result = provider_instance.chat(llm_input, system_prompt=self.settings.system_prompt)
                except Exception as e:
                    if self.settings.cloud_fallback_to_local:
                        fallback_used = True
                        fallback_reason = f"Cloud provider {actual_provider} failed: {e}. Falling back to local RAG."
                        actual_provider = "local"
                        local_provider = self.llm_provider
                        rag_path = "local_openwebui"
                        try:
                            llm_result = self.openwebui_llm_provider.chat(llm_input, collection_id=knowledge_id)
                        except Exception as local_rag_err:
                            if self.settings.rag_fallback_to_ollama:
                                rag_fallback_used = True
                                llm_result = local_provider.chat(llm_input, system_prompt=self.settings.system_prompt)
                            else:
                                raise local_rag_err
                    else:
                        raise
        else:
            # Direct chat without RAG
            try:
                llm_result = provider_instance.chat(llm_input, system_prompt=self.settings.system_prompt)
            except Exception as e:
                if actual_provider in ("openai", "gemini") and self.settings.cloud_fallback_to_local:
                    fallback_used = True
                    fallback_reason = f"Cloud provider {actual_provider} failed: {e}. Falling back to local Ollama."
                    actual_provider = "local"
                    local_provider = self.llm_provider
                    llm_result = local_provider.chat(llm_input, system_prompt=self.settings.system_prompt)
                else:
                    raise

        response_language = self.language_router.detect(llm_result.text)
        parts = [
            TTSPart(text=chunk, language=language)
            for chunk, language in self.language_router.split_for_tts(llm_result.text)
        ]
        
        # 3. Dynamic voice synthesis
        fallback_allowed = not getattr(self.settings, "force_selected_voice", False) and getattr(self.settings, "fallback_allowed", True)
        
        try:
            tts_result = self.tts_provider.synthesize(parts, voice_id=voice_id, fallback_allowed=fallback_allowed)
        except FallbackBlockedError as exc:
            raise ValueError(f"Voice synthesis blocked: selected voice cannot be used and fallback is disabled. {exc}") from exc

        # Combine fallback triggers and messages
        combined_fallback = (rag_fallback_used or fallback_used or tts_result.fallback_used)
        combined_fallback_reasons = []
        if fallback_reason:
            combined_fallback_reasons.append(fallback_reason)
        if tts_result.fallback_reason:
            combined_fallback_reasons.append(tts_result.fallback_reason)
        final_fallback_reason = "; ".join(combined_fallback_reasons) if combined_fallback_reasons else None

        total_ms = (time.perf_counter() - started) * 1000
        
        # Sidecar dictionary for commercial provenance
        audio_sidecar = {
            "generated_by": "SwarLocal",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "voice_id": voice_id or "auto",
            "requested_voice_id": tts_result.requested_voice_id,
            "requested_voice_name": tts_result.requested_voice_name,
            "actual_voice_id": tts_result.actual_voice_id,
            "actual_voice_name": tts_result.actual_voice_name,
            "actual_tts_engine": tts_result.actual_tts_engine,
            "model_artifact_path": tts_result.model_artifact_path,
            "language": tts_result.language,
            "fallback_used": tts_result.fallback_used,
            "fallback_reason": tts_result.fallback_reason,
            "generated_audio_path": tts_result.generated_audio_path,
            "llm_model": llm_result.model,
            "llm_provider": actual_provider,
            "rag_collection_id": knowledge_id if rag_used else None,
            "rag_path": rag_path,
            "internet_used": internet_used,
        }
        
        # Write sidecar to disk next to the audio file if keeping turn audio
        if self.settings.keep_turn_audio:
            try:
                sidecar_path = tts_result.audio_path.with_suffix(".json")
                sidecar_path.write_text(json.dumps(audio_sidecar, indent=2))
            except Exception:
                pass

        # Record usage event
        self._record_usage_event(audio_sidecar)

        return ConversationResponse(
            transcript=text,
            response=llm_result.text,
            input_language=input_language.language,
            response_language=response_language.language,
            audio_url=f"{self.audio_base_url}/{tts_result.audio_path.name}",
            tts_route=[{"text": part.text, "language": part.language} for part in parts],
            timings=TimingMetrics(
                audio_received_to_transcript_ms=transcript_ms,
                llm_first_token_ms=llm_result.first_token_ms,
                llm_total_ms=llm_result.total_ms,
                llm_load_ms=llm_result.load_ms,
                prompt_eval_ms=llm_result.prompt_eval_ms,
                generation_ms=llm_result.generation_ms,
                tts_generation_ms=tts_result.generation_ms,
                total_turn_ms=total_ms,
            ),
            rag_used=rag_used,
            rag_collection_id=knowledge_id if rag_used else None,
            rag_fallback_used=combined_fallback,
            internet_used=internet_used,
            citations=citations,
            voice_id=voice_id,
            requested_voice_id=tts_result.requested_voice_id,
            requested_voice_name=tts_result.requested_voice_name,
            actual_voice_id=tts_result.actual_voice_id,
            actual_voice_name=tts_result.actual_voice_name,
            actual_engine=tts_result.actual_tts_engine,
            actual_model_path=tts_result.model_artifact_path,
            fallback_used=combined_fallback,
            fallback_reason=final_fallback_reason,
            audio_sidecar=audio_sidecar,
            llm_provider=actual_provider,
            rag_path=rag_path,
        )

    def _record_usage_event(self, sidecar: dict[str, Any]) -> None:
        from app.database import get_db_connection
        import json
        import uuid
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO voice_usage_events (id, timestamp, event_type, details) VALUES (?, ?, ?, ?);",
                (str(uuid.uuid4()), sidecar["timestamp"], "audio_synthesis", json.dumps(sidecar))
            )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()
