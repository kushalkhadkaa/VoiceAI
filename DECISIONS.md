# Decisions

- Piper remains the stable local production TTS engine.
- Mixed Nepali-English speech can route sentence/chunk level across same-speaker Nepali and English artifacts when a voice provides separate models.
- Fallback is allowed by default for casual use, but `Force selected cloned voice only` disables voice fallback for commercial QA.
- Open WebUI remains the local RAG/document manager.
- OpenAI and Gemini are optional reasoning providers only; voice audio stays local unless a future cloud voice provider is explicitly enabled.
- Internet retrieval remains off unless enabled by settings or requested from the conversation UI.

