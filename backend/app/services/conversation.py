from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.providers.llm import LLMProvider, LLMResult
from app.providers.llm_openwebui import OpenWebUILLMProvider
from app.providers.stt import STTProvider
from app.providers.tts import FallbackBlockedError, TTSPart, TTSProvider
from app.providers.web_retrieval import WebRetrievalProvider
from app.schemas import ConversationResponse, TimingMetrics
from app.services.language_router import LanguageRouter

logger = logging.getLogger(__name__)


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
        kb_service=None,
    ) -> None:
        self.stt_provider = stt_provider
        self.llm_provider = llm_provider
        self.openwebui_llm_provider = openwebui_llm_provider
        self.tts_provider = tts_provider
        self.language_router = language_router
        self.web_retrieval_provider = web_retrieval_provider
        self.audio_base_url = audio_base_url.rstrip("/")
        self.settings = openwebui_llm_provider.settings
        self.kb_service = kb_service

    def handle_audio(
        self,
        audio_path: Path,
        voice_id: str | None = None,
        knowledge_id: str | None = None,
        use_internet: bool = False,
        llm_provider_id: str | None = None,
        session_id: str | None = None,
        stt_provider_id: str | None = None,
        stt_language: str | None = None,
    ) -> ConversationResponse:
        started = time.perf_counter()
        stt_result = self.stt_provider.transcribe_file(audio_path, provider_name=stt_provider_id, language=stt_language)

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
            session_id=session_id,
            stt_provider_id=stt_provider_id,
        )

    def handle_text(
        self,
        text: str,
        voice_id: str | None = None,
        knowledge_id: str | None = None,
        use_internet: bool = False,
        llm_provider_id: str | None = None,
        session_id: str | None = None,
        stt_provider_id: str | None = None,
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
            session_id=session_id,
            stt_provider_id=stt_provider_id,
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

    def _effective_system_prompt(self, language_code: str | None = None, modality: str = "text") -> str:
        base = self.settings.system_prompt
        instruction = (getattr(self.settings, "bank_instruction", "") or "").strip()
        language_instruction = self._language_instruction(language_code, modality)
        sections = []
        if instruction:
            sections.append(f"[ADMIN INSTRUCTION - follow strictly]\n{instruction}")
        sections.append(f"[LANGUAGE ROUTING]\n{language_instruction}")
        sections.append(f"[ASSISTANT INSTRUCTION]\n{base}")
        return "\n\n".join(sections)

    def _language_instruction(self, language_code: str | None, modality: str) -> str:
        code = language_code or "unknown"
        if code == "ne":
            return (
                f"Input mode: {modality}. Detected user language: Nepali. "
                "You MUST respond ONLY in Nepali. Do not use any other language or script."
            )
        if code == "en":
            return (
                f"Input mode: {modality}. Detected user language: English. "
                "You MUST respond ONLY in English. Do not use any other language or script."
            )
        if code == "mixed":
            return (
                f"Input mode: {modality}. Detected user language: mixed Nepali-English. "
                "Respond in natural mixed Nepali-English (Romanized or Devanagari as the user did). "
                "Never use any third language."
            )
        # unknown: keep it bilingual but still constrained to the two allowed languages.
        return (
            f"Input mode: {modality}. Detected user language: unknown (assume Nepali/English context). "
            "Respond only in English or Nepali, matching the user's language. "
            "Do not use any other language or script."
        )

    def _retrieve_kb_context(self, query: str, knowledge_id: str | None = None) -> tuple[str, str | None, str | None]:
        # RAG-first: the banking assistant grounds EVERY turn (text + voice) in the
        # local Nabil Bank knowledge base. Retrieval is NOT gated on rag_enabled.
        # If a specific knowledge_id is selected, search only that collection;
        # otherwise search across all non-empty collections. If the KB is empty or
        # retrieval errors, we proceed gracefully with no context.
        if self.kb_service is None or not query.strip():
            return "", None, None

        collection_ids: list[str] | None = None
        rag_collection_id: str | None = None
        if knowledge_id and knowledge_id != "none":
            collection_ids = [knowledge_id]
            rag_collection_id = knowledge_id
        else:
            try:
                collections = self.kb_service.list_collections()
                # Only search collections that actually have indexed chunks.
                collection_ids = [
                    collection.id
                    for collection in collections
                    if getattr(collection, "chunk_count", 0) > 0
                ]
                if not collection_ids:
                    # Nothing indexed yet — nothing to ground on, proceed without context.
                    return "", None, None
                rag_collection_id = ",".join(collection_ids)
            except Exception as exc:
                logger.warning("Unable to list KB collections: %s", exc)
                return "", None, None

        try:
            context = self.kb_service.build_context(query, collection_ids)
            return context, "local_kb" if context else None, rag_collection_id if context else None
        except Exception as exc:
            logger.warning("Local KB retrieval failed: %s", exc)
            return "", None, None

    @staticmethod
    def _build_llm_input(
        text: str,
        language_code: str | None,
        kb_context: str,
        citations: list[dict[str, str]],
    ) -> str:
        parts: list[str] = []
        if kb_context:
            parts.append(
                "Before answering, check this local Nabil Bank knowledge base context. "
                "Use it when it is relevant. If the context is insufficient, say the knowledge base does not contain enough information.\n\n"
                f"--- KNOWLEDGE BASE CONTEXT ---\n{kb_context}\n--- END KNOWLEDGE BASE CONTEXT ---"
            )
        if citations:
            context_str = "\n".join(
                f"- {c['title']}: {c['snippet']} (Source: {c['url']})" for c in citations
            )
            parts.append(
                "Use these web search results only when they are relevant and cite the source URL in the answer.\n\n"
                f"--- WEB CONTEXT ---\n{context_str}\n--- END WEB CONTEXT ---"
            )
        parts.append(f"User question ({language_code or 'unknown'}): {text}")
        return "\n\n".join(parts)

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
        session_id: str | None = None,
        stt_provider_id: str | None = None,
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
                
        modality = "voice" if transcript_ms is not None else "text"
        knowledge_context, rag_path, rag_collection_id = self._retrieve_kb_context(text, knowledge_id)
        rag_used = bool(knowledge_context)
        llm_input = self._build_llm_input(text, input_language.language, knowledge_context, citations if internet_used else [])
        system_prompt = self._effective_system_prompt(input_language.language, modality)

        # 2. LLM Provider Routing and Fallbacks
        is_cloud_stt = (stt_provider_id == "openai") or (not stt_provider_id and getattr(self.settings, "stt_provider", "local") == "openai")
        
        is_cloud_voice = False
        if voice_id:
            vid = voice_id.lower()
            if "openai" in vid or vid in ("alloy", "echo", "fable", "onyx", "nova", "shimmer"):
                is_cloud_voice = True

        target_provider = llm_provider_id or self.settings.llm_provider or "local"
        if (is_cloud_stt or is_cloud_voice) and target_provider in ("local", "auto"):
            actual_provider = "openai"
        else:
            if target_provider == "auto":
                actual_provider = "local"
            else:
                actual_provider = target_provider


        if actual_provider == "local":
            provider_instance = self.llm_provider
        else:
            provider_instance = self._instantiate_provider(actual_provider)
        
        rag_fallback_used = False
        fallback_used = False
        fallback_reason = None
        llm_result = None

        try:
            llm_result = provider_instance.chat(llm_input, system_prompt=system_prompt)
        except Exception as e:
            if actual_provider in ("openai", "gemini") and self.settings.cloud_fallback_to_local:
                fallback_used = True
                fallback_reason = f"Cloud provider {actual_provider} failed: {e}. Falling back to local Ollama."
                actual_provider = "local"
                provider_instance = self.llm_provider
                llm_result = provider_instance.chat(llm_input, system_prompt=system_prompt)
            elif actual_provider == "local" and knowledge_id and knowledge_id != "none" and self.settings.rag_fallback_to_ollama:
                rag_fallback_used = True
                fallback_reason = f"Local RAG prompt failed: {e}. Retrying direct local chat."
                direct_input = self._build_llm_input(text, input_language.language, "", citations if internet_used else [])
                llm_result = provider_instance.chat(direct_input, system_prompt=system_prompt)
            else:
                raise

        response_language = self.language_router.detect(llm_result.text)

        # 2.1 Translations — each call is a full extra LLM round-trip, so skip them in
        # low-latency mode to keep chat fast. They return when low_latency_mode is off.
        transcript_translation = None
        response_translation = None
        if not getattr(self.settings, "low_latency_mode", False):
            input_lang_code = input_language.language
            if input_lang_code == "ne":
                transcript_translation = self._translate_via_llm(text, "Nepali", "English", provider_instance)
            elif input_lang_code == "en":
                transcript_translation = self._translate_via_llm(text, "English", "Nepali", provider_instance)
            elif input_lang_code == "mixed":
                transcript_translation = self._translate_via_llm(text, "Nepali/English mixed", "English", provider_instance)

            resp_lang_code = response_language.language
            if resp_lang_code == "ne":
                response_translation = self._translate_via_llm(llm_result.text, "Nepali", "English", provider_instance)
            elif resp_lang_code == "en":
                response_translation = self._translate_via_llm(llm_result.text, "English", "Nepali", provider_instance)
            elif resp_lang_code == "mixed":
                response_translation = self._translate_via_llm(llm_result.text, "Nepali/English mixed", "English", provider_instance)

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
            "rag_collection_id": rag_collection_id if rag_used else None,
            "rag_path": rag_path,
            "internet_used": internet_used,
            "transcript_translation": transcript_translation,
            "response_translation": response_translation,
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

        turn_id = str(uuid.uuid4())
        response_obj = ConversationResponse(
            id=turn_id,
            transcript=text,
            response=llm_result.text,
            transcript_translation=transcript_translation,
            response_translation=response_translation,
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
            rag_collection_id=rag_collection_id if rag_used else None,
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

        self._save_chat_turn(response_obj, session_id)
        return response_obj


    def _save_chat_turn(self, turn: ConversationResponse, session_id: str | None) -> None:
        from app.database import get_db_connection
        import json
        import time
        conn = get_db_connection()
        try:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            conn.execute(
                """
                INSERT INTO chat_turns (
                    id, session_id, timestamp, transcript, response, transcript_translation, response_translation, input_language, response_language,
                    audio_url, user_audio_url, tts_route, timings, rag_used, rag_collection_id,
                    rag_fallback_used, internet_used, citations, voice_id, requested_voice_id,
                    requested_voice_name, actual_voice_id, actual_voice_name, actual_engine,
                    actual_model_path, fallback_used, fallback_reason, llm_provider, rag_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    turn.id,
                    session_id,
                    timestamp,
                    turn.transcript,
                    turn.response,
                    turn.transcript_translation,
                    turn.response_translation,
                    turn.input_language,
                    turn.response_language,
                    turn.audio_url,
                    None,
                    json.dumps(turn.tts_route),
                    json.dumps(turn.timings.model_dump()),
                    1 if turn.rag_used else 0,
                    turn.rag_collection_id,
                    1 if turn.rag_fallback_used else 0,
                    1 if turn.internet_used else 0,
                    json.dumps(turn.citations or []),
                    turn.voice_id,
                    turn.requested_voice_id,
                    turn.requested_voice_name,
                    turn.actual_voice_id,
                    turn.actual_voice_name,
                    turn.actual_engine,
                    turn.actual_model_path,
                    1 if turn.fallback_used else 0,
                    turn.fallback_reason,
                    turn.llm_provider,
                    turn.rag_path
                )
            )
            conn.commit()
        except Exception as e:
            import sys
            print(f"Error saving chat turn: {e}", file=sys.stderr)
        finally:
            conn.close()


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

    def _translate_via_llm(self, text: str, source_lang: str, target_lang: str, provider_instance: Any) -> str:
        if not text or not text.strip():
            return ""
        prompt = (
            f"You are a professional, bilingual translator. Translate the following text from {source_lang} to {target_lang}. "
            f"Provide only the translation, with no explanation, introduction, or formatting. Keep the natural tone and style."
        )
        try:
            res = provider_instance.chat(text.strip(), system_prompt=prompt, options={"temperature": 0.1, "max_tokens": 250})
            return res.text.strip()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Translation failed: %s", e)
            return ""

