# SwarLocal Project State

Updated: 2026-06-06

SwarLocal is a local-first Nepali-English mixed-language voice agent with FastAPI backend, Vite/React frontend, Ollama reasoning, Faster Whisper STT, Piper TTS, Open WebUI RAG hooks, internet retrieval, Voice Studio, SQLite audit/persistence tables, and local model/provider diagnostics.

Current functional baseline:
- Local Ollama is reachable and `qwen2.5:7b` is installed.
- `gemma3:4b` is installed as a smaller Google-family local fallback model.
- Piper Nepali and English voices are present.
- Voice text-to-TTS smoke test passes end-to-end.
- Voice socket status now reports core local voice turns as ready even when optional RAG/cloud providers are not configured.
- Frontend now uses a premium dark UI with simple user-facing Voice/Voice Studio/Knowledge flows and a deeper Admin surface.
- Voice page includes presets, animated orb/waveform, grouped voice selector, chat bubbles, latency, and system mini metrics.
- Single-model TTS mode is enabled: English and Nepali chunks currently synthesize with `models/piper/ne_NP-chitwan-medium.onnx`.
- Custom voice profiles without trained/exported `.onnx` artifacts are shown as not ready instead of being selectable as working voices.
- Logs page is now available in the main menu and combines frontend app logs, backend audit logs, provider status, and voice socket warnings/errors.
- Voice Studio pending voices now require consent before recording; completed-consent voices can continue to recording.
- Voice Studio supports owners, consent, samples, quality scoring, cleanup endpoints, Chatterbox zero-shot clone references, publish/delete, audit logs, and conversation dropdown publishing.
- Chatterbox local voice cloning is wired as the default Voice Studio clone engine and was smoke-tested on CPU with a generated WAV.
- Strict voice routing metadata is returned on conversation turns.
- The app now exposes `Force selected cloned voice only` and `Allow voice fallback`.

Current blockers:
- Open WebUI is reachable, but no API key is configured, so RAG collection listing/query validation fails.
- OpenAI and Gemini are optional and not configured or tested because no API keys are saved.
- OpenAI key is currently not saved in backend runtime settings. A successful OpenAI Test now persists the key/model automatically.
- Piper model fine-tuning/export remains configurable through `PIPER_TRAIN_COMMAND`, not a complete bundled one-click training queue.
- Generated voice similarity scoring and richer background progress are still the next Voice Studio milestones.
