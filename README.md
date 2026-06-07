# SwarLocal

Local-first macOS voice AI assistant for Nepali and English.

The MVP records microphone audio in a React/Vite frontend, transcribes with Faster Whisper, chats with a local Ollama model, and speaks back with language-routed Piper voices.

## What It Builds

- Nepali and English voice turns through separate Piper voices.
- Mixed-language routing by script heuristics and Whisper metadata.
- FastAPI REST endpoints and WebSocket voice conversation.
- Setup checks for backend, Ollama, Piper, voices, ffmpeg, and microphone.
- A **My Voice Dataset** screen for consent-based recording and Piper dataset export.
- Training notes for two future same-speaker voices:
  - `myvoice_ne_NP_medium.onnx`
  - `myvoice_en_US_medium.onnx`

## macOS Setup

Prerequisites:

- Python 3.11 or newer
- Node.js 20 or newer
- ffmpeg
- Ollama
- Piper voice `.onnx` and `.onnx.json` files

Install system tools:

```bash
brew install ffmpeg
```

Install Ollama from the official macOS app or install script:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama and pull the default model:

```bash
ollama pull qwen3:1.7b
```

Install app dependencies:

```bash
cp .env.example .env
make setup
```

Check your local runtime:

```bash
make doctor
```

`make doctor` prints `READY` only when critical local pieces are present. If it prints `BLOCKED`, follow the listed fix commands. The command exits successfully so it can be used during setup without breaking your terminal workflow.

Create a voice folder and place Piper voices there:

```bash
mkdir -p models/piper
```

Recommended MVP voices:

- Nepali: `ne_NP-chitwan-medium` or `ne_NP-google-medium`
- English: `en_US-lessac-medium` or `en_US-ryan-medium`

Update `.env` or the Settings screen if your filenames differ:

```bash
PIPER_NEPALI_VOICE=./models/piper/ne_NP-chitwan-medium.onnx
PIPER_ENGLISH_VOICE=./models/piper/en_US-lessac-medium.onnx
```

Run the app:

```bash
make dev
```

Open:

```text
http://127.0.0.1:5173
```

## Validation

```bash
make test
make lint
npm --prefix frontend run build
make doctor
```

Real local runtime smoke test:

```bash
make e2e
```

`make e2e` does not require a microphone. It does require Ollama, the configured model, Piper, ffmpeg, and matching `.onnx` plus `.onnx.json` voice files. If those are missing, it fails with exact reasons.

Optional local zero-shot voice cloning uses Chatterbox:

```bash
make setup-voice-clone
```

The first Chatterbox synthesis downloads model weights into `.local/huggingface`. On macOS the app defaults Chatterbox to CPU because the MPS path can crash the backend on some Metal runtimes; set `CHATTERBOX_DEVICE=mps` only if you have verified it locally.

- `GET /health`
- `GET /settings`
- `POST /settings`
- `GET /models/status`
- `GET /doctor`
- `GET /voices`
- `POST /stt/test`
- `POST /chat/test`
- `POST /tts/test`
- `DELETE /local-data`
- `GET /ai-providers`
- `GET /ai-providers/status`
- `POST /ai-providers/test`
- `POST /ai-providers/test/local`
- `POST /ai-providers/test/openai`
- `POST /ai-providers/test/gemini`
- `POST /settings/ai-provider`
- `DELETE /settings/openai-key`
- `DELETE /settings/gemini-key`
- `WS /ws/voice`

## AI Provider Configuration

SwarLocal supports three LLM providers:
1. **Local Ollama** (Default): Offline, private, uses `qwen2.5:7b` (fallback: `gemma3:4b`).
2. **OpenAI**: Cloud-based, uses `gpt-4o-mini` (requires API key).
3. **Google Gemini**: Cloud-based, uses `gemini-1.5-flash` (requires API key).

Configure active providers, key masking, fallback capabilities, and connection testing on the Settings page or easily switch active providers on the Conversation page. Audio files, STT, and TTS remain entirely local; only prompt text is sent to cloud brains if selected.

## Offline Use

After Python packages, npm packages, Ollama models, Faster Whisper model files, and Piper voices are downloaded, normal conversation does not require internet access.

## Audio Handling

Normal push-to-talk audio is treated as temporary turn data. Browser audio is validated, converted through ffmpeg to 16 kHz mono WAV for STT, then deleted unless `KEEP_TURN_AUDIO=true` is explicitly configured. Dataset recordings are separate and require the consent checkbox in the Dataset screen.

## Voice Cloning Boundary

This repo supports consent-based cloning for the user's own voice. Chatterbox provides local zero-shot cloning from Voice Studio recordings. Piper remains the production `.onnx` path, but real Piper fine-tuning must be supplied through `PIPER_TRAIN_COMMAND`; the app no longer pretends a copied base Piper model is a trained clone. Do not clone another person or use recordings without clear permission.
