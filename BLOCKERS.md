# Blockers

Updated: 2026-06-06

- `make rag-test` fails because Open WebUI RAG is enabled but `open_webui_api_key` is empty.
  Fix: create an API key in Open WebUI, save it in SwarLocal Settings or `OPEN_WEBUI_API_KEY`, then rerun `make rag-test`.
- OpenAI cloud provider is not configured.
  Fix: add an OpenAI API key in Settings, then click Test OpenAI. A successful test now saves the key/model locally.
- Gemini cloud provider is not configured.
  Fix: add a Gemini API key only if Google cloud reasoning is desired, then click Test Gemini.
- Piper fine-tuning/export is not yet a fully automated bundled workflow.
  Fix: set `PIPER_TRAIN_COMMAND` for a real Piper trainer, or use the default Chatterbox local clone engine for zero-shot cloning.
- Existing Piper custom voices that were created before this fix may still contain copied base `.onnx` artifacts.
  Fix: create a new Chatterbox voice profile from consented recordings, or run/import a real Piper-compatible `.onnx` artifact.
