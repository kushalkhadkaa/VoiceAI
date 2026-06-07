# Agent Handoff

Use `/Users/kushalkhadka/Documents/VoiceAI` as the real repo.

Local services observed:
- Ollama: `http://127.0.0.1:11434`
- Open WebUI: `http://127.0.0.1:8080`
- Frontend dev server target: `http://127.0.0.1:5173`
- Backend target: `http://127.0.0.1:8000`

Recent changes:
- Replaced fake Piper copy-cloning with real Chatterbox zero-shot clone references and Chatterbox TTS routing.
- Voice Studio now defaults new clone profiles to Chatterbox; Piper cloning requires `PIPER_TRAIN_COMMAND` to produce a real `.onnx`.
- Chatterbox is installed in the local `.venv`, model weights are cached under `.local/huggingface`, and CPU synthesis smoke-tested successfully.
- Added Logs page and browser-persistent app event logs.
- Fixed Voice Studio consent ID flow so new voices use `voice_id` correctly and pending voices route to consent before recording.
- Premium dark UI with new Knowledge and Admin pages.
- Voice screen now has presets, animated orb/waveform, grouped voice selector, chat bubbles, latency panel, and compact system metrics.
- Voice Studio has friendlier guided workflow framing and Advanced Voice Controls tooltips.
- Added `make ui-test` for static frontend UX contracts.
- Strict voice routing response fields and frontend display.
- Settings UI for `Force selected cloned voice only` and `Allow voice fallback`.
- Compact system metrics in the voice screen.
- Validation targets for model, voice, and RAG.
- Fixed E2E Ollama provider import.

Before continuing:
1. Run `make test` and `make e2e` after backend edits.
2. Do not treat `make rag-test` as green until Open WebUI API key is configured.
3. Preserve local voice/audio data under `.local/voices`; do not delete it unless the user asks.
4. Keep Chatterbox on CPU by default on this Mac; `CHATTERBOX_DEVICE=mps` caused a Metal crash during smoke testing.
