# Evaluation

## Automated Checks

Run backend unit tests:

```bash
make test
```

Run import and type checks:

```bash
make lint
```

Evaluate dataset recordings:

```bash
python scripts/evaluate_recordings.py dataset/wav --output dataset/quality.csv
```

## Manual Voice Conversation Checks

Use the app's Evaluation view after real turns.

- Naturalness: 1 to 5
- Voice similarity: 1 to 5, only after custom voice training
- Nepali pronunciation: 1 to 5
- English pronunciation: 1 to 5

## Latency Metrics

Each turn reports:

- `audio_received_to_transcript_ms`
- `llm_first_token_ms`
- `tts_generation_ms`
- `total_turn_ms`

The first turn after startup may be slower because local models load into memory.
