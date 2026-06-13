# Next Actions

Updated: 2026-06-13 21:27:41 +0545

Start here for the next coding session:

1. Confirm the active repo is `/Users/kushalkhadka/VoiceAI` and branch is `feat/voice-studio-ux`.
2. Decide whether `docs/RAG_EVAL_REPORT.md` should be sanitized and committed, or kept as a local-only evaluation artifact.
3. Decide whether `.claude/` should stay local-only; default is do not commit it.
4. Switch runtime provider away from Gemini or add a Gemini key, because current settings select `llm_provider=gemini` while Gemini is unavailable.
5. Add or save an Open WebUI API key only if RAG should be live, then enable RAG and rerun the RAG smoke/eval flow.
6. Add a user-facing credit/billing status only if a provider API supports it safely; the current app only reports provider/key availability, not account balance.
7. Add generated preview scoring for Chatterbox/Piper output, not only uploaded recording quality scoring.
8. Add richer background progress for Chatterbox/Piper clone/test jobs.
9. Build a full Piper fine-tuning runner around `PIPER_TRAIN_COMMAND`, with progress and artifact validation for real `.onnx` exports.
10. Persist RAG eval runs, chat turns, and validation results more completely in SQLite if the UI should survive browser refreshes.
11. Add backup/export/import for voice packages.
12. Replace static UI checks with browser-driven Playwright tests when a frontend test runner is introduced.
