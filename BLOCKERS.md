# Blockers

Updated: 2026-06-13 21:27:41 +0545

- Runtime settings currently select `llm_provider=gemini`, but `gemini_api_key` is empty and Gemini reports unavailable.
  Fix: save a Gemini API key, or switch the provider to local/OpenAI in Settings.
- Open WebUI API key is empty and current runtime settings show `rag_enabled=false`.
  Fix: create an API key in Open WebUI, save it in SwarLocal Settings or `OPEN_WEBUI_API_KEY`, enable RAG, then rerun the RAG smoke/eval flow.
- The app can show OpenAI key/provider availability but not actual OpenAI account credit balance.
  Fix: add a safe billing/credit integration only if the selected provider exposes an appropriate endpoint and the user wants that feature.
- Piper fine-tuning/export is not yet a fully automated bundled workflow.
  Fix: set `PIPER_TRAIN_COMMAND` for a real Piper trainer, or continue using the default Chatterbox local clone engine for zero-shot cloning.
- `docs/RAG_EVAL_REPORT.md` is untracked and may contain detailed bank knowledge-base/person/contact answers.
  Fix: review and sanitize before committing, or keep it local-only.
- `.claude/` is untracked local tool configuration.
  Fix: keep it untracked unless the user explicitly wants that tool config shared.
