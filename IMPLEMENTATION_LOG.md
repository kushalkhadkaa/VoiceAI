# Implementation Log

## 2026-06-07

- Replaced the fake Piper “clone” path that copied built-in `.onnx` files with a real Chatterbox zero-shot clone path that builds `chatterbox_reference.wav` from consented recordings and routes synthesis through Chatterbox.
- Added `VoiceCloneService`, `ChatterboxTTSProvider`, Chatterbox routing in `TTSRouter`, and optional `make setup-voice-clone` dependencies.
- Changed Piper cloning to prepare a dataset and require `PIPER_TRAIN_COMMAND` before it can mark a Piper `.onnx` training job complete.
- Changed Voice Studio’s default clone engine to Chatterbox and kept Piper as an explicit fine-tuning/export path.
- Installed and smoke-tested Chatterbox locally; CPU synthesis generated a cloned WAV, while MPS was avoided by default after a Metal crash during testing.

## 2026-06-06

- Added a dedicated Logs menu section with browser app logs, backend audit logs, provider/runtime warnings, voice socket blockers, filters, export, and clear-local-app-logs.
- Added browser-side persistent event logging for refreshes, settings saves, conversation turns, recording start/stop, audio send failures, dataset recording saves/deletes, and UI errors.
- Fixed Voice Studio consent flow by normalizing `voice_id`/`id` after voice creation and routing pending-consent voices to the Consent step before recording.
- Changed pending Voice Studio gallery cards to show `Add consent` instead of sending users directly into recording, preventing “Recording is blocked: Voice consent is required first.”
- Fixed broken audio routing caused by strict custom voice selection when the published custom voice had no exported `.onnx` artifact.
- Added `single_tts_voice_model` so one Piper model can be used for Nepali and English chunks instead of switching between separate voices.
- Enabled live single-model TTS mode and verified English plus Nepali turns both synthesize through `ne_NP-chitwan-medium.onnx`.
- Marked published custom voices without model artifacts as not ready/disabled in the voice selector.
- Made HTTP text-turn fallback carry selected voice, knowledge, internet, and Local/OpenAI/Gemini brain routing.
- Fixed provider test persistence so a successful OpenAI/Gemini key test saves the key/model locally.
- Verified Faster Whisper STT with a local Nepali WAV upload.
- Reworked the frontend into a premium dark product UI with Voice, Voice Studio, Knowledge, Setup, Eval, Admin, and Settings navigation.
- Upgraded the Voice page into a live conversation layout with a center orb, animated waveform/rings, presets, grouped voice selector, badges, chat bubbles, live transcript, latency panel, and system usage.
- Added a Knowledge page for Open WebUI status, collection selection, default collection saving, and test questions.
- Added an Admin area with System Monitor, Voice Models, AI Providers, RAG / Knowledge, Audit Logs, Data Storage, Commercial Readiness, and Debug Tools tabs.
- Polished Voice Studio with workflow steps, filters, friendlier wording, sample progress, Clean all recordings / Create my voice actions, and Advanced Voice Controls with tooltips.
- Added `make ui-test` with static UI contract checks for navigation, orb UX, grouped voice selector, Voice Studio, Knowledge, Admin, disabled explanations, tooltips, responsive CSS, and reduced motion.
- Inspected the actual repo state under `/Users/kushalkhadka/Documents/VoiceAI`.
- Added strict conversation response fields: `requested_voice_id`, `requested_voice_name`, `actual_engine`, `actual_model_path`, and `fallback_used`.
- Made fallback-blocked TTS errors explicit when selected voice fallback is disabled.
- Exposed `force_selected_voice` and `fallback_allowed` through backend settings schema, frontend types, defaults, and Settings UI.
- Added compact CPU/RAM/disk system metrics to the voice conversation screen using the existing `/system/metrics` endpoint.
- Added tests for visible voice fallback metadata and strict cloned-voice fallback blocking.
- Added validation targets: `make model-test`, `make voice-test`, and `make rag-test`.
- Added `scripts/rag_smoke.py` to validate Open WebUI RAG configuration with exact failure reasons.
- Fixed E2E provider wiring by using the newer Ollama wrapper that accepts `system_prompt`.
- Changed RAG/Open WebUI API-key issues from voice-socket blockers to warnings so local text/audio turns remain available.
