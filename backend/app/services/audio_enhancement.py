from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

class AudioEnhancementService:
    def __init__(self) -> None:
        pass

    def normalize_loudness(self, input_path: Path, output_path: Path, sample_rate: int = 22050) -> None:
        """
        Normalizes loudness to commercial target -23 LUFS using FFmpeg loudnorm.
        """
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for audio normalization.")
        
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            "s16",
            "-af",
            "loudnorm=I=-23:LRA=7:TP=-2",
            str(output_path),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode('utf-8', errors='ignore')}")

    def trim_silence(self, input_path: Path, output_path: Path) -> None:
        """
        Trims leading and trailing silence from audio using FFmpeg silenceremove.
        """
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for audio trimming.")
            
        # silenceremove syntax: start_periods:start_duration:start_threshold:end_periods:end_duration:end_threshold
        # Trims silence below -50dB
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.1:end_periods=-1:end_threshold=-50dB:end_silence=0.1",
            str(output_path),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg silenceremove failed: {result.stderr.decode('utf-8', errors='ignore')}")
