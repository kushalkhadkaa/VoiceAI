# SwarLocal Audit

## Current Status

- The repo contains a FastAPI backend, React/Vite/TypeScript frontend, provider interfaces, dataset scripts, and setup docs.
- Unit-style tests exist for language routing and conversation orchestration.
- The scaffold can build without local Whisper/Piper/Ollama models because heavy providers are lazy.
- The app is not yet proven as a real end-to-end voice assistant on this machine until Ollama, ffmpeg, Piper, and matching Piper voices are installed and `make e2e` passes.

## Blockers

- Runtime dependencies are external and machine-specific: ffmpeg, Ollama, Piper, and `.onnx` plus `.onnx.json` voice files.
- Real microphone-to-STT validation requires browser permission and local decoding support.
- Faster Whisper model files may download on first use and are not bundled.

## High-Priority Fixes Completed In This Pass

- Added `scripts/check_environment.py` and `make doctor`.
- Made `/models/status` use the same doctor checks as the CLI.
- Added a Piper voice registry and `GET /voices`.
- Added audio upload validation, MIME checks, file size checks, duration checks, and temporary 16 kHz mono WAV conversion through ffmpeg.
- Switched Ollama provider to `/api/chat` with configurable prompt/options and clearer missing-Ollama errors.
- Added `scripts/e2e_voice_smoke.py` and `make e2e` for honest runtime validation.
- Added consent gating to the dataset recording UI.

## Medium-Priority Fixes

- Add more browser-level tests around microphone flows once local permissions are available.
- Add explicit voice download UI later, with user confirmation and source/license display.
- Add persisted dataset export packaging instead of browser multi-file downloads.
- Add richer romanized Nepali preference handling in Settings.

## What Is Already Good

- Local-first boundaries are clear.
- Heavy model imports are lazy.
- Provider modules are isolated and testable.
- Training docs correctly describe two language-specific same-speaker Piper voices instead of a fictional universal model.
- Frontend already has separate Voice, Setup, Dataset, Evaluation, and Settings surfaces.
