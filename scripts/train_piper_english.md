# Piper English Fine-Tuning Notes

Target output:

```text
myvoice_en_US_medium.onnx
myvoice_en_US_medium.onnx.json
```

Use only recordings you created or explicitly own. Keep this dataset English-only, even if your natural conversation style is mixed.

1. Record English prompts in a quiet room with the same microphone position.
2. Export WAV files and `metadata.csv`, or run:

```bash
python scripts/prepare_voice_dataset.py raw_recordings/en prompts_en.txt --output dataset_en
python scripts/evaluate_recordings.py dataset_en/wav --output dataset_en/quality.csv
```

3. Remove clipped, noisy, too-short, or too-long recordings.
4. Fine-tune with Piper's training workflow on a Linux/NVIDIA GPU environment.
5. Copy the final `.onnx` and `.onnx.json` into `models/piper/`.
6. Set `PIPER_ENGLISH_VOICE=./models/piper/myvoice_en_US_medium.onnx`.

The app routes English responses to this voice and Nepali responses to the Nepali voice.
