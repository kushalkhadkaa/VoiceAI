# Voice Training

The MVP uses existing Piper voices:

- Nepali: `ne_NP-chitwan-medium` or `ne_NP-google-medium`
- English: `en_US-lessac-medium` or `en_US-ryan-medium`

There is no single standard Piper voice that simultaneously gives Nepali, English, and a cloned personal voice. The custom voice path is two same-speaker, language-specific Piper voices:

- `myvoice_ne_NP_medium.onnx`
- `myvoice_en_US_medium.onnx`

## Consent Boundary

Only train on recordings made or uploaded by the user. Do not clone another person's voice or use recordings without clear permission.

## Dataset Format

```text
dataset/
  wav/
    000001.wav
    000002.wav
  metadata.csv
```

`metadata.csv` uses Piper's single-speaker format:

```text
000001|Prompt text for this recording
000002|Another prompt
```

## Recording Quality

Good recordings are mono WAV, quiet background, no clipping, and usually 1.5 to 12 seconds long. Use:

```bash
python scripts/evaluate_recordings.py dataset/wav --output dataset/quality.csv
```

Rows marked `reject` should be re-recorded. Rows marked `review` can work, but they usually reduce final voice quality.

## Training Hardware

Piper inference can run locally on macOS. Fine-tuning is typically a Linux/NVIDIA GPU job. After training, copy the `.onnx` and `.onnx.json` files back into `models/piper/` and update `.env` or the Settings screen.
