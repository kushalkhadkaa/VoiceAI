# Validation Log

Updated: 2026-06-13 21:27:41 +0545

- PASS: `git push origin feat/voice-studio-ux` returned `Everything up-to-date` before this handoff-doc update.
- PASS: Backend health check at `http://127.0.0.1:8001/health` returned `{"ok":true,"app":"SwarLocal","version":"0.1.0"}`.
- PASS: Frontend dev server started at `http://127.0.0.1:5173/` with Vite.
- PASS: System metrics endpoint returned a healthy recommendation with roughly 8 GB RAM available and 27 GB disk free at the time checked.
- PASS: Provider status showed OpenAI available, local models available, and Gemini unavailable because the Gemini key is empty.
- NOTE: No full `make test` / `make lint` rerun was performed for this documentation-only handoff update.

- PASS: `make test` ran 55 backend tests successfully after real Chatterbox clone routing replaced fake Piper copy-cloning.
- PASS: `make lint` compiled backend/tests/scripts and ran frontend typecheck.
- PASS: `make e2e` completed local Ollama chat, Piper TTS, and full text-to-TTS after clone routing changes.
- PASS: `make ui-test` covered navigation, Voice page UX, Voice Studio, Knowledge, Admin, tooltips, responsive layout, and reduced motion.
- PASS: Chatterbox local CPU smoke generated `.local/audio_cache/62181e870f5ba487b589d61e2a1161ac.wav` through `ChatterboxTTSProvider`.
- PASS: `make doctor` reports critical runtime ready. Informational blockers remain for Open WebUI API key, OpenAI key, and Gemini key.

- PASS: `make test` ran 44 backend tests successfully.
- PASS: `make lint` compiled backend/tests/scripts and ran frontend typecheck.
- PASS: `npm --prefix frontend run typecheck`.
- PASS: `npm --prefix frontend run build`.
- PASS: `make doctor` reports critical runtime ready. Informational blockers remain for Open WebUI API key, OpenAI key, and Gemini key.
- PASS: `make model-test` reports local Ollama, `qwen2.5:7b`, `gemma3:4b`, Piper, STT, and DB readiness.
- PASS: `make voice-test` generated English/Nepali Piper audio and completed a full text-to-TTS turn.
- PASS: `make e2e` completed local Ollama chat, Piper TTS, and full text-to-TTS.
- PASS: Browser check at `http://127.0.0.1:5173/` shows `Ready for local voice turns` with CPU/RAM/Disk metrics visible.
- PASS: `make ui-test` covered navigation, Voice page UX, Voice Studio, Knowledge, Admin, tooltips, responsive layout, and reduced motion.
- PASS: Browser DOM check confirmed new navigation, presets, grouped voice selector, disabled explanations, no horizontal overflow, and conversation/latency panels.
- PASS: Local English text turn produced audio using `actual_model_path=models/piper/ne_NP-chitwan-medium.onnx`.
- PASS: Local Nepali text turn produced audio using `actual_model_path=models/piper/ne_NP-chitwan-medium.onnx`.
- PASS: `/stt/test` transcribed a saved Nepali WAV with Faster Whisper.
- PASS: Browser check confirms untrained custom voice option is disabled and app is ready for local voice turns.
- PASS: Browser check confirms Logs nav/page render with log details, export, clear app logs, provider warnings, and audit entries.
- PASS: Browser check confirms pending Voice Studio cards show `Add consent` and consent-completed cards show `Continue setup`.
- FAIL: `make rag-test` because Open WebUI API key is missing while RAG is enabled.
