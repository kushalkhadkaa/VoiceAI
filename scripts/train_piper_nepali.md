# Piper Nepali Fine-Tuning Notes

Target output:

```text
myvoice_ne_NP_medium.onnx
myvoice_ne_NP_medium.onnx.json
```

Use only recordings you created or explicitly own. Keep Nepali and English datasets separate so each Piper voice stays language-specific.

1. Record Nepali prompts in the app's **Dataset** view or another quiet recording workflow.
2. Export WAV files and `metadata.csv`, or run:

```bash
python scripts/prepare_voice_dataset.py raw_recordings/ne prompts_ne.txt --output dataset_ne
python scripts/evaluate_recordings.py dataset_ne/wav --output dataset_ne/quality.csv
```

3. Reject rows marked `reject`; re-record rows marked `review` when possible.
4. Train or fine-tune with Piper's training workflow on a Linux/NVIDIA GPU environment.
5. Copy the final `.onnx` and `.onnx.json` into `models/piper/`.
6. Set `PIPER_NEPALI_VOICE=./models/piper/myvoice_ne_NP_medium.onnx`.

Inference should run locally on macOS after the trained voice is copied back.
