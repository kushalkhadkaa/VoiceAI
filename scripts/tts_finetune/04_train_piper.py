"""Stage 4: Fine-tune Piper from ne_NP-chitwan-medium on the prepared dataset.

Flow (all via `python -m piper_train`):
  1. piper_train.preprocess   --language ne --sample-rate 22050 --single-speaker
  2. piper_train              --resume_from_checkpoint <chitwan .ckpt>
  3. piper_train.export_onnx  -> models/piper/ne_NP-nabil-multilingual-medium.onnx

IMPORTANT: piper_train only runs on Linux (it depends on piper-phonemize and
an older PyTorch Lightning stack with no Windows wheels). On Windows this
script detects that and prints the exact WSL command sequence instead.

Note: fine-tuning resumes from a .ckpt training checkpoint, NOT the .onnx
file. Download the chitwan checkpoint from the Piper "checkpoints" collection
on Hugging Face (see docs/TTS_FINETUNE.md).
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATASET_DIR, MODELS_DIR, SAMPLE_RATE, TRAIN_DIR, banner, run

FINAL_NAME = "ne_NP-nabil-multilingual-medium"


def to_wsl_path(p: Path) -> str:
    s = str(p.resolve())
    drive, rest = s[0].lower(), s[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def print_wsl_instructions(dataset: Path, train_dir: Path, ckpt: Path | None, epochs: int) -> None:
    ds, td = to_wsl_path(dataset), to_wsl_path(train_dir)
    ck = to_wsl_path(ckpt) if ckpt else "/path/to/ne_NP-chitwan-medium.ckpt"
    onnx_out = to_wsl_path(MODELS_DIR / f"{FINAL_NAME}.onnx")
    print("""
[info] Piper training requires Linux. You are on Windows — run it inside WSL.
       Complete command sequence (copy/paste into a WSL Ubuntu shell):

  # one-time setup
  sudo apt update && sudo apt install -y python3.10-venv build-essential espeak-ng
  python3 -m venv ~/piper-train-venv && source ~/piper-train-venv/bin/activate
  pip install pip==23.3 wheel setuptools
  git clone https://github.com/rhasspy/piper ~/piper
  cd ~/piper/src/python && pip install -e . && pip install torchmetrics==0.11.4

  # 1. preprocess""")
    print(f"""  python -m piper_train.preprocess \\
    --language ne --input-dir {ds} --output-dir {td} \\
    --dataset-format ljspeech --single-speaker --sample-rate {SAMPLE_RATE}

  # 2. fine-tune from the chitwan checkpoint (.ckpt, not .onnx!)
  python -m piper_train \\
    --dataset-dir {td} --accelerator gpu --devices 1 --batch-size 16 \\
    --validation-split 0.05 --num-test-examples 2 \\
    --max_epochs {epochs} --resume_from_checkpoint {ck} \\
    --checkpoint-epochs 1 --precision 32 --quality medium

  # 3. export onnx
  python -m piper_train.export_onnx \\
    {td}/lightning_logs/version_0/checkpoints/*.ckpt \\
    {onnx_out}
  cp {td}/config.json {onnx_out}.json

After export, restart the app and point Settings -> Voice & TTS -> Nepali
voice model path at models/piper/{FINAL_NAME}.onnx (see docs/TTS_FINETUNE.md).
""".rstrip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate Piper preprocessing, fine-tuning and ONNX export "
                    "for the multilingual Nepali-English voice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR,
                        help="Dataset dir with wavs/ + metadata.csv")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR,
                        help="Working dir for preprocessing/checkpoints")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to ne_NP-chitwan-medium .ckpt to fine-tune from")
    parser.add_argument("--epochs", type=int, default=2200,
                        help="max_epochs (chitwan base is ~2164; +small delta to fine-tune)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--print-only", action="store_true",
                        help="Just print the training commands, do not run")
    args = parser.parse_args()

    banner("Stage 4: Piper fine-tuning")

    if not (args.dataset / "metadata.csv").exists():
        print(f"[error] {args.dataset / 'metadata.csv'} not found. Run 03_segment_transcribe.py first.")
        return 1
    n = len((args.dataset / "metadata.csv").read_text(encoding="utf-8").splitlines())
    print(f"[ok] dataset: {args.dataset} ({n} utterances)")

    if args.checkpoint and not args.checkpoint.exists():
        print(f"[warn] checkpoint not found: {args.checkpoint}")

    if platform.system() == "Windows" or args.print_only:
        if platform.system() == "Windows" and not args.print_only:
            print("[info] Windows detected.")
        print_wsl_instructions(args.dataset, args.train_dir, args.checkpoint, args.epochs)
        return 0

    # Linux/WSL path: run the flow directly
    try:
        import piper_train  # noqa: F401
    except ImportError:
        print("[error] piper_train is not installed. Install it (Linux only):")
        print("          git clone https://github.com/rhasspy/piper && "
              "cd piper/src/python && pip install -e .")
        print_wsl_instructions(args.dataset, args.train_dir, args.checkpoint, args.epochs)
        return 1

    if not args.checkpoint:
        print("[error] --checkpoint is required for fine-tuning (the .ckpt, not the .onnx).")
        print("        Download: https://huggingface.co/datasets/rhasspy/piper-checkpoints "
              "-> ne/ne_NP/chitwan/medium/")
        return 1

    args.train_dir.mkdir(parents=True, exist_ok=True)
    steps = [
        [sys.executable, "-m", "piper_train.preprocess",
         "--language", "ne", "--input-dir", args.dataset, "--output-dir", args.train_dir,
         "--dataset-format", "ljspeech", "--single-speaker", "--sample-rate", str(SAMPLE_RATE)],
        [sys.executable, "-m", "piper_train",
         "--dataset-dir", args.train_dir, "--accelerator", "gpu", "--devices", "1",
         "--batch-size", str(args.batch_size), "--validation-split", "0.05",
         "--num-test-examples", "2", "--max_epochs", str(args.epochs),
         "--resume_from_checkpoint", args.checkpoint,
         "--checkpoint-epochs", "1", "--precision", "32", "--quality", "medium"],
    ]
    for cmd in steps:
        if run(cmd).returncode != 0:
            print("[error] step failed, aborting.")
            return 1

    ckpts = sorted(args.train_dir.glob("lightning_logs/version_*/checkpoints/*.ckpt"))
    if not ckpts:
        print("[error] no checkpoint produced by training.")
        return 1
    onnx_out = MODELS_DIR / f"{FINAL_NAME}.onnx"
    if run([sys.executable, "-m", "piper_train.export_onnx", ckpts[-1], onnx_out]).returncode != 0:
        return 1
    cfg = args.train_dir / "config.json"
    if cfg.exists():
        (MODELS_DIR / f"{FINAL_NAME}.onnx.json").write_bytes(cfg.read_bytes())
    print(f"\n[summary] exported: {onnx_out}")
    print("          Select it in the app: Settings -> Voice & TTS -> Nepali voice model path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
