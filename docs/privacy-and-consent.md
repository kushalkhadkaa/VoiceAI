# Privacy And Consent

SwarLocal is local-first.

- Browser audio is sent only to the local FastAPI backend.
- STT runs through local Faster Whisper.
- Chat runs through local Ollama.
- TTS runs through local Piper.
- Generated audio and recordings stay under `.local/`.
- Conversation history is stored in the browser on the same machine.
- Push-to-talk turn audio is temporary by default and is deleted after STT processing.
- Dataset recordings are created only after the consent checkbox is selected.

Cloud services are not used unless the user explicitly adds and configures them later.

## Voice Data

Voice dataset tools are for the user's own recordings. Do not train on someone else's voice, public recordings, meetings, calls, or media clips unless there is explicit consent and the intended use is allowed.

The app does not implement instant voice cloning, cloud upload, or background recording.

## Deleting Data

Use the Settings screen's local data delete action or remove:

```bash
rm -rf .local
```

Browser history can also be cleared by deleting site data for the frontend origin.
