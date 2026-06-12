from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
import uuid
import wave
from pathlib import Path
from typing import Any


class VoiceCloneError(ValueError):
    pass


class VoiceCloneService:
    def __init__(self, voices_base_dir: Path) -> None:
        self.voices_base_dir = voices_base_dir

    def create_chatterbox_reference(self, voice_id: str, samples: list[Any], language: str) -> dict[str, Any]:
        wav_paths = self._existing_wav_paths(samples)
        if not wav_paths:
            raise VoiceCloneError("No recording files found on disk to build a clone reference.")

        voice_dir = self.voices_base_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        reference_path = voice_dir / "chatterbox_reference.wav"
        seconds = self._write_reference_wav(wav_paths, reference_path, max_seconds=60.0)
        if seconds < 2.0:
            raise VoiceCloneError("At least 2 seconds of clean reference audio are required for voice cloning.")

        manifest = {
            "engine": "chatterbox",
            "voice_id": voice_id,
            "language": language,
            "reference_path": str(reference_path),
            "source_sample_count": len(wav_paths),
            "reference_seconds": round(seconds, 3),
            "created_at": self._timestamp(),
            "note": "Zero-shot clone reference built from consented local Voice Studio recordings.",
        }
        manifest_path = voice_dir / "chatterbox_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "artifact_uri": f"chatterbox://{reference_path}",
            "config_path": str(manifest_path),
            "reference_path": str(reference_path),
            "reference_seconds": seconds,
            "sample_count": len(wav_paths),
        }

    def prepare_piper_dataset(self, voice_id: str, samples: list[Any], prompt_text_by_id: dict[str, str]) -> dict[str, Any]:
        wav_paths = self._existing_wav_paths(samples)
        if not wav_paths:
            raise VoiceCloneError("No recording files found on disk to prepare a Piper dataset.")

        voice_dir = self.voices_base_dir / voice_id
        dataset_dir = voice_dir / "piper_dataset"
        wav_dir = dataset_dir / "wav"
        wav_dir.mkdir(parents=True, exist_ok=True)
        metadata_rows: list[str] = []

        for sample in samples:
            wav_path = Path(sample["wav_path"])
            if not wav_path.exists():
                continue
            prompt_id = sample["prompt_id"]
            shutil.copy2(wav_path, wav_dir / f"{prompt_id}.wav")
            text = prompt_text_by_id.get(prompt_id, prompt_id)
            metadata_rows.append(f"{prompt_id}|{text}")

        if not metadata_rows:
            raise VoiceCloneError("No usable WAV rows were available for Piper training.")
        metadata_path = dataset_dir / "metadata.csv"
        metadata_path.write_text("\n".join(metadata_rows) + "\n", encoding="utf-8")
        manifest_path = dataset_dir / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "voice_id": voice_id,
                    "rows": len(metadata_rows),
                    "metadata_path": str(metadata_path),
                    "created_at": self._timestamp(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"dataset_dir": dataset_dir, "metadata_path": metadata_path, "rows": len(metadata_rows)}

    def run_piper_training_command(
        self,
        command_template: str,
        voice_id: str,
        language: str,
        dataset_dir: Path,
    ) -> dict[str, Any]:
        if not command_template.strip():
            raise VoiceCloneError(
                "Piper fine-tuning is not configured. Use Chatterbox for instant local cloning, "
                "or set PIPER_TRAIN_COMMAND to a command that writes .onnx and .onnx.json artifacts."
            )

        voice_dir = self.voices_base_dir / voice_id
        formatted = command_template.format(
            voice_id=voice_id,
            language=language,
            dataset_dir=str(dataset_dir),
            output_dir=str(voice_dir),
        )
        command = shlex.split(formatted)
        if not command:
            raise VoiceCloneError("PIPER_TRAIN_COMMAND is empty after formatting.")

        result = subprocess.run(command, cwd=Path.cwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="ignore") or result.stdout.decode("utf-8", errors="ignore")
            raise VoiceCloneError(f"Piper training command failed: {detail.strip()}")

        onnx_files = sorted(voice_dir.rglob("*.onnx"))
        ready = [path for path in onnx_files if Path(f"{path}.json").exists()]
        if not ready:
            raise VoiceCloneError("Piper training completed but did not produce a matching .onnx and .onnx.json artifact.")

        artifact = ready[0]
        return {
            "artifact_uri": str(artifact),
            "config_path": str(Path(f"{artifact}.json")),
            "artifact_count": len(ready),
        }

    @staticmethod
    def _existing_wav_paths(samples: list[Any]) -> list[Path]:
        paths: list[Path] = []
        for sample in samples:
            wav_path = Path(sample["wav_path"])
            if wav_path.exists():
                paths.append(wav_path)
        return paths

    @staticmethod
    def _write_reference_wav(wav_paths: list[Path], output_path: Path, max_seconds: float) -> float:
        params = None
        frames: list[bytes] = []
        written_frames = 0
        frame_rate = 0
        max_frames = None

        for wav_path in wav_paths:
            with wave.open(str(wav_path), "rb") as wav_file:
                current = wav_file.getparams()
                if params is None:
                    params = current
                    frame_rate = wav_file.getframerate()
                    max_frames = int(max_seconds * frame_rate)
                elif current[:3] != params[:3]:
                    raise VoiceCloneError("Reference recordings have incompatible WAV formats; please re-clean recordings.")

                remaining = max((max_frames or 0) - written_frames, 0)
                if remaining <= 0:
                    break
                chunk = wav_file.readframes(min(wav_file.getnframes(), remaining))
                frames.append(chunk)
                written_frames += len(chunk) // (wav_file.getsampwidth() * wav_file.getnchannels())

        if params is None or written_frames == 0:
            raise VoiceCloneError("Unable to write clone reference audio.")

        params = params._replace(nframes=written_frames)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as output:
            output.setparams(params)
            for frame_bytes in frames:
                output.writeframes(frame_bytes)

        return written_frames / float(frame_rate or 1)

    @staticmethod
    def new_job_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
