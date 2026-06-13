# Agent Handoff

Updated: 2026-06-13 21:27:41 +0545

Use `/Users/kushalkhadka/VoiceAI` as the active working repo. A separate checkout exists at `/Users/kushalkhadka/Documents/VoiceAI`, but the currently working/flawless app was observed running from `/Users/kushalkhadka/VoiceAI`.

Current git state:
- Branch: `feat/voice-studio-ux`
- Remote: `origin https://github.com/kushalkhadkaa/VoiceAI.git`
- Latest pushed commit: `c877798322a91fa6b19f6d59d95888cad5660b8f`
- `git push origin feat/voice-studio-ux` returned `Everything up-to-date` before this handoff update.
- Untracked and intentionally not pushed before this docs update: `.claude/` and `docs/RAG_EVAL_REPORT.md`.

Local services observed:
- Ollama: `http://127.0.0.1:11434`
- Open WebUI: `http://127.0.0.1:8080`
- Backend target: `http://127.0.0.1:8001`
- Frontend dev server target: `http://127.0.0.1:5173`
- Backend health: `{"ok":true,"app":"SwarLocal","version":"0.1.0"}`

Latest pushed changes to inspect first:
- `c877798` system pulse/heartbeat, uptime, and RAG auto-recovery.
- `52368c7` single-flight guard for heavy jobs to avoid concurrent CPU pile-ups.
- `0ca2e36` serialized Chatterbox synthesis to stop it jamming the API.
- `11b7f0f` streaming animated RAG evaluation, fix-in-KB, model and voice pickers.
- `48aa055` non-blocking TTS so text answers return instantly and speech can be generated on demand.
- Earlier branch commits include JS-aware RAG crawling, document-scoped evaluation/chat, explicit LLM provider routing, Voice Studio redesign, and Chatterbox audio-save fixes.

Provider/credit status seen through the app:
- OpenAI API key is configured and masked by the API boundary.
- OpenAI provider reports available.
- Gemini API key is empty and Gemini reports unavailable.
- Open WebUI API key is empty; current settings show `rag_enabled=false`.
- The local app does not expose actual OpenAI billing/credit balance, only key/provider availability.

Important implementation notes:
- Voice cloning is consent-first. Do not add workflows that clone another person's voice.
- Chatterbox is the default local zero-shot voice clone path; keep it CPU-default on this Mac unless the user explicitly wants to retry MPS.
- Piper fine-tuning/export still requires `PIPER_TRAIN_COMMAND`; do not fake a Piper clone by copying built-in `.onnx` files.
- Generated audio, recordings, runtime DBs, model caches, and user data belong under `.local/` or explicit export paths and must stay out of git.

Before continuing:
1. Check `git status --short --branch` and confirm you are in `/Users/kushalkhadka/VoiceAI`.
2. Read the latest branch commits with `git log --oneline main..HEAD`.
3. Treat `.claude/` as local tool config.
4. Review `docs/RAG_EVAL_REPORT.md` for privacy before committing it.
5. After backend edits, run at least `make test` and the relevant smoke target.
