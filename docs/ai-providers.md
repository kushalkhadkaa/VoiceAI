# AI Provider Configuration

SwarLocal supports three LLM brains:
1. **Local Ollama** (Default)
2. **OpenAI** (Optional cloud)
3. **Google Gemini** (Optional cloud)

---

## 1. Local Ollama (Default)
* **Privacy**: 100% offline. No transcripts leave your Mac.
* **Requirements**: Ollama must be running on macOS.
* **Default model**: `qwen2.5:7b` (fallback: `gemma3:4b`).
* **Connection Test**: Verifies connection to Ollama API, checks if the selected model is pulled locally, and sends two short test queries (one in English, one in Nepali).

---

## 2. OpenAI (Cloud Brain)
* **Privacy**: Only the text prompt/context is sent to OpenAI chat completions. Transcription (STT) and voice synthesis (TTS) remain entirely on device.
* **Requirements**: OpenAI API Key and internet access.
* **Default model**: `gpt-4o-mini`.
* **Connection Test**: Verifies key presence, sends a short hello prompt to `https://api.openai.com/v1/chat/completions`, and returns latency metrics.

---

## 3. Google Gemini (Cloud Brain)
* **Privacy**: Only the text prompt/context is sent to Google's generative language API. Transcription (STT) and voice synthesis (TTS) remain entirely on device.
* **Requirements**: Google Gemini API Key and internet access.
* **Default model**: `gemini-1.5-flash`.
* **Connection Test**: Verifies key presence, sends a short test prompt to Google's generative API, and returns latency metrics.

---

## 4. Cloud Fallback to Local Ollama
In the Settings panel, you can toggle **"Fallback to Local Ollama if Cloud fails"** (`cloud_fallback_to_local`). 
* If enabled, any network timeout, invalid key, or cloud API error will automatically route the query to your local Ollama instance (`qwen2.5:7b` or `gemma3:4b`).
* A warning pill ("Fallback Active") will display on the conversation screen indicating that a fallback occurred.
