# Codex Project Notes

This repo is a local-first macOS voice AI assistant for Nepali and English.

- Keep backend changes modular under `backend/app/providers` and `backend/app/services`.
- Do not add cloud dependencies unless the user explicitly asks for them.
- Do not implement cloning of another person's voice. Dataset and training workflows must be consent-based and user-owned.
- Keep Whisper, Piper, and Ollama integrations lazy-loaded so tests can run before local models are installed.
- Prefer standard-library scripts for dataset preparation unless a dependency is already required by the runtime.
- Frontend work lives in `frontend/` and should remain a real React + Vite + TypeScript app.
- Generated audio, recordings, and user data belong under `.local/` or explicit dataset export directories and must stay out of git.
