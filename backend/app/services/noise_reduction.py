from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

class NoiseReductionService:
    def __init__(self) -> None:
        pass

    def denoise_audio(self, input_path: Path, output_path: Path, noise_floor_db: float = -36.0) -> None:
        """
        Removes background stationary noise, hums, and hiss using highpass/lowpass filters and FFT denoiser.
        """
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for noise reduction.")
            
        # Combine highpass (rumble/hum), lowpass (high hiss), and afftdn (spectral noise reduction)
        filter_str = f"highpass=f=80,lowpass=f=12000,afftdn=nf={noise_floor_db}"
        
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            filter_str,
            str(output_path),
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg noise reduction failed: {result.stderr.decode('utf-8', errors='ignore')}")
