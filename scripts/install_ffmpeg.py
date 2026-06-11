import urllib.request
import zipfile
import io
import os
import shutil
from pathlib import Path

def main():
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    root_dir = Path(__file__).resolve().parents[1]
    dest_dir = root_dir / ".venv" / "Scripts"
    if not dest_dir.exists():
        dest_dir = root_dir / "bin"
        dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading FFmpeg from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            zip_data = response.read()
    except Exception as e:
        print(f"Failed to download FFmpeg: {e}")
        return 1
        
    print("Extracting FFmpeg binaries...")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
            # Find ffmpeg.exe, ffplay.exe, ffprobe.exe
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith(("ffmpeg.exe", "ffplay.exe", "ffprobe.exe")):
                    filename = os.path.basename(file_info.filename)
                    target_path = dest_dir / filename
                    print(f"Extracting {filename} to {target_path}")
                    with zip_ref.open(file_info) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
        print("FFmpeg installation completed successfully!")
        return 0
    except Exception as e:
        print(f"Failed to extract FFmpeg: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
