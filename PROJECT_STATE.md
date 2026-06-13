# SwarLocal Project State

Updated: 2026-06-13 21:27:41 +0545

Active repo: `/Users/kushalkhadka/VoiceAI`

Active branch: `feat/voice-studio-ux`

Latest pushed commit before this handoff-doc pass: `c877798322a91fa6b19f6d59d95888cad5660b8f`

SwarLocal is currently a local-first Nepali-English mixed-language voice agent for macOS with FastAPI backend, Vite/React frontend, Ollama reasoning, optional OpenAI/Gemini reasoning, Faster Whisper/OpenAI STT options, Piper/OpenAI/Chatterbox TTS paths, Open WebUI/RAG hooks, JS-aware site crawling, RAG evaluation tooling, Voice Studio, SQLite audit/persistence tables, and local model/provider diagnostics.

Current functional baseline:
- Backend was observed healthy on `http://127.0.0.1:8001`.
- Frontend dev server was started on `http://127.0.0.1:5173`.
- Local Ollama is reachable and multiple local models are installed, including `qwen3:1.7b`, `qwen2.5:7b`, `gemma3:4b`, `falcon:latest`, `llama3:latest`, and `mistral:latest`.
- Runtime settings currently show `llm_provider=gemini`, but Gemini has no key saved, so cloud Gemini availability is false.
- OpenAI API key is saved and masked by the API boundary; OpenAI provider reports available.
- Open WebUI API key is empty and `rag_enabled=false` in current runtime settings.
- Piper Nepali and English voices are present.
- Chatterbox local zero-shot voice cloning is wired and serialized so only one heavy synthesis job runs at a time.
- Text chat can return immediately while TTS runs non-blocking/on demand.
- RAG now includes JS-aware site crawl/extraction, document-scoped eval/chat, animated streaming eval UI, fix-in-KB flow, model picker, voice picker, and RAG auto-recovery.
- System reliability includes heartbeat/pulse, uptime reporting, provider-routing fixes, OpenAI retry logic, and heavy-job single-flight guards.
- Voice Studio includes redesigned UX, interactive recording, preflight checks, consent-first workflow, quality scoring, cleanup endpoints, Chatterbox clone references, publish/delete, audit logs, and conversation dropdown publishing.
- Frontend includes premium dark Voice, Voice Studio, Knowledge/RAG Eval, Admin, Settings, logs/telemetry surfaces, animated orb/waveform, grouped voice selector, chat bubbles, latency, and system metrics.

Recent pushed branch changes to inspect first:
- `c877798` reliability: system pulse/heartbeat, uptime, and RAG auto-recovery.
- `52368c7` stability: single-flight guard for heavy jobs.
- `0ca2e36` TTS: serialize Chatterbox synthesis.
- `11b7f0f` RAG eval: streaming animated evaluation, fix-in-KB, model and voice pickers.
- `48aa055` chat performance: non-blocking TTS.
- `2cbf0af`, `b5dbd80`, `4ae8f26`, `dddde0f`, `4d59463`, `400cb50`, `c75fe88`, and `3d53581` contain the rest of the current feature branch work.

Current blockers / caveats:
- The app does not expose actual OpenAI billing credit balance; it only confirms masked key and provider availability.
- Gemini is selected in runtime settings but unavailable until a Gemini API key is saved or the provider is switched.
- Open WebUI/RAG API key is empty; RAG is currently disabled in runtime settings.
- Piper model fine-tuning/export remains configurable through `PIPER_TRAIN_COMMAND`, not a bundled one-click training queue.
- Generated voice similarity scoring and richer background progress remain pending Voice Studio milestones.
- `.claude/` is local tool config and should remain untracked.
- `docs/RAG_EVAL_REPORT.md` is untracked and contains detailed bank knowledge-base eval answers; review privacy expectations before committing it.
