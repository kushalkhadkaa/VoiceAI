# Multilingual (Nepali + English) TTS Fine-Tuning

Fine-tune **one** Piper voice that handles mixed Nepali-English (code-switched)
speech, starting from the bundled `models/piper/ne_NP-chitwan-medium.onnx`.
The result is `models/piper/ne_NP-nabil-multilingual-medium.onnx`.

All pipeline scripts live in `scripts/tts_finetune/` and write intermediate
data to `.local/tts_dataset/` (git-ignored).

## Prerequisites

| Requirement | How |
|---|---|
| Python venv | `D:\AI Voice Conversation\.venv\Scripts\python.exe` |
| ffmpeg | `python scripts/install_ffmpeg.py` (installs to `.venv\Scripts\` or `bin\`); scripts auto-detect PATH / venv / bin |
| yt-dlp | `pip install yt-dlp` (stage 1) |
| faster-whisper | already a backend dependency (stage 3) |
| demucs (optional) | `pip install demucs` — vocal isolation if sources have any music (stage 2) |
| Linux/WSL + GPU | Piper training (`piper_train`) does **not** run on Windows. Stage 4 detects Windows and prints the full WSL command sequence. |
| chitwan **.ckpt** | Fine-tuning resumes from a training checkpoint, not the `.onnx`. Download `ne/ne_NP/chitwan/medium/*.ckpt` from https://huggingface.co/datasets/rhasspy/piper-checkpoints |

## Picking good YouTube sources

> **Licensing warning:** only use content you have the rights to use — your own
> recordings, material with explicit permission from the creator, or suitably
> licensed content. Cloning someone's voice without consent may be illegal.

Good sources:
- **One clean speaker** per video (monologue podcasts are ideal)
- **No background music**, jingles, crowd noise, or phone-line audio
- **Natural Nepali-English code-switching** — Nepali tech podcasts, news
  interviews, lectures, finance/banking explainers
- Studio or quiet-room recording, consistent mic distance
- 10+ minutes per video

Edit `DEFAULT_SOURCES` in `scripts/tts_finetune/01_download_audio.py`, or pass
`--urls` / `--urls-file urls.txt` (one URL per line, `#` comments allowed).

## The 4 stages

Run everything: `python scripts/tts_finetune/run_pipeline.py`
(with `--skip-download`, `--skip-clean`, `--skip-dataset`, `--skip-train` to resume).

| Stage | Script | What it does | Output | Typical duration |
|---|---|---|---|---|
| 1 | `01_download_audio.py` | yt-dlp bestaudio → 22050 Hz mono WAV | `.local/tts_dataset/raw/` | minutes (network-bound) |
| 2 | `02_clean_audio.py` | optional demucs vocal isolation → ffmpeg `highpass=80, lowpass=8000, afftdn` → `silenceremove` + `loudnorm I=-19` | `.local/tts_dataset/clean/` | ~real-time without demucs; 5-10x slower with demucs on CPU |
| 3 | `03_segment_transcribe.py` | faster-whisper (`small`, `language=None` for auto / code-switch) VAD-segments into 3-12 s utterances; drops `avg_logprob < -1.0` and <2 s / >15 s clips; writes LJSpeech `wavs/` + `metadata.csv` (`filename|transcript`) | `.local/tts_dataset/dataset/` | ~0.3-1x audio length on CPU |
| 4 | `04_train_piper.py` | `piper_train.preprocess` (`--language ne --sample-rate 22050 --single-speaker`) → fine-tune with `--resume_from_checkpoint <chitwan.ckpt>` → `export_onnx` | `models/piper/ne_NP-nabil-multilingual-medium.onnx` | 1-6 h on a GPU for ~50 fine-tune epochs |

## Quality tips

- **Minimum 30 minutes** of clean segmented speech; **2+ hours is ideal**.
- Garbage in, garbage out: one noisy source can hurt the whole voice — listen
  to a few files in `clean/` before stage 3.
- Spot-check `metadata.csv`: fix obviously wrong transcripts (especially
  English words inside Nepali sentences) — transcript accuracy matters more
  than quantity.
- Keep one speaker. Mixing speakers produces a muddy, unstable voice.
- Fine-tune a modest number of extra epochs (the script defaults to base
  epoch + small delta). Overtraining on a small dataset causes robotic
  artifacts; export checkpoints every epoch and compare.

## Testing the voice in the app

1. Confirm `ne_NP-nabil-multilingual-medium.onnx` **and** its `.onnx.json`
   are in `models/piper/`.
2. Restart the backend.
3. In the app: **Settings → Voice & TTS → Nepali voice model path** → select
   the new model.
4. Speak or type a mixed sentence, e.g. *"मैले हिजो naya laptop किनेँ, battery
   life एकदम राम्रो छ।"* — both languages should come out in the same voice.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ffmpeg not found` | Run `python scripts/install_ffmpeg.py`; scripts check PATH, `.venv\Scripts`, `bin\` |
| `yt-dlp is not installed` | `pip install yt-dlp`; if downloads fail with 403/throttling, `pip install -U yt-dlp` |
| Stage 2 output sounds muffled | demucs over-isolated; re-run with `--no-demucs` for clean sources |
| Very few segments kept in stage 3 | Source too noisy (logprob filter) or long uninterrupted speech (>15 s); try cleaner sources or `--model medium` for better transcripts |
| Transcripts in wrong script (Latin vs Devanagari) | Whisper auto-detect flipped; acceptable for code-switching, but fix gross errors in `metadata.csv` by hand |
| `piper_train` won't install on Windows | Expected — use WSL; stage 4 prints the exact command sequence |
| `--resume_from_checkpoint` errors | You pointed at the `.onnx`; you need the `.ckpt` from the piper-checkpoints HF dataset |
| CUDA OOM during training | Lower `--batch-size` (16 → 8 → 4) |
| New voice sounds robotic / unstable | Too little data or overtrained — gather more audio, or export an earlier checkpoint |
| Voice not listed in app | Missing `.onnx.json` next to the `.onnx`, or backend not restarted |
