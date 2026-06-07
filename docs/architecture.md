# Architecture

SwarLocal is split into a browser frontend and a local FastAPI backend.

## Runtime Flow

1. Browser records microphone audio with `MediaRecorder`.
2. Frontend sends audio to `ws://localhost:8000/ws/voice` with the chosen `selected_brain` and `selected_voice_id` configuration.
3. `AudioPipeline` validates MIME type and size, converts browser audio to temporary 16 kHz mono WAV through ffmpeg, then deletes temporary turn audio after STT unless explicitly configured otherwise.
4. `FasterWhisperSTTProvider` transcribes the audio and returns text plus Whisper language metadata.
5. `LanguageRouter` selects Nepali, English, mixed, or unknown using Whisper confidence and script heuristics.
6. `ConversationService` resolves the active LLM provider (Local Ollama, OpenAI, or Google Gemini) and passes the query. If a cloud provider is active and fails, it optionally triggers a fallback to Local Ollama if `cloud_fallback_to_local` is enabled.
7. `LanguageRouter` splits the response into TTS chunks.
8. `PiperTTSProvider` generates or reuses cached WAV files under `.local/audio_cache`.
9. The frontend plays the returned `/audio/*.wav` URL and stores conversation history in browser local storage.

## Backend Modules

- `app/providers/stt.py`: Faster Whisper adapter.
- `app/providers/llm_base.py`: Protocol/interface defining LLM providers.
- `app/providers/llm_ollama.py`: Local Ollama adapter.
- `app/providers/llm_openai.py`: OpenAI API adapter using standard library `urllib`.
- `app/providers/llm_gemini.py`: Google Gemini API adapter using standard library `urllib`.
- `app/providers/tts.py`: Piper CLI adapter with audio caching and mixed-language concatenation.
- `app/providers/vad.py`: lightweight WAV energy checks.
- `app/services/language_router.py`: language detection and mixed-language TTS splitting.
- `app/services/audio_pipeline.py`: browser audio persistence and MIME handling.
- `app/services/conversation.py`: end-to-end turn orchestration, dynamic provider routing, fallback logic, and latency metrics.

## Local Data

- `.local/audio_work`: temporary decoded turn audio.
- `.local/audio_cache`: generated Piper WAV responses.
- `.local/settings.json`: runtime overrides including masked API keys and selected providers.
- Browser `localStorage`: conversation history, evaluation ratings, selected brain, and UI preferences.

No cloud service is required by default, and cloud LLM brains are strictly opt-in. Audio recording processing, STT, and TTS remain entirely on device.
