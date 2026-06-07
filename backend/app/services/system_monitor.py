from __future__ import annotations
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

class SystemMonitorService:
    def __init__(self) -> None:
        self.os_type = platform.system()
        self.os_version = platform.release()
        self.arch = platform.machine()
        self.cpu_cores = os.cpu_count() or 1
        
    def get_static_info(self) -> dict[str, Any]:
        """
        Gathers platform description, hardware specs, and binary versions.
        """
        # Retrieve binary versions safely via subprocess
        ffmpeg_ver = self._get_binary_version(["ffmpeg", "-version"])
        node_ver = self._get_binary_version(["node", "--version"])
        ollama_ver = self._get_binary_version(["ollama", "--version"])
        piper_ver = self._get_binary_version(["piper", "--version"])
        
        # Check Apple Silicon GPU/MPS availability
        mps_available = False
        if self.os_type == "Darwin":
            # Apple Silicon Macs support MPS
            mps_available = "arm" in self.arch.lower()

        # Disk space
        total_disk, used_disk, free_disk = shutil.disk_usage("/")
        
        return {
            "os": self.os_type,
            "os_version": self.os_version,
            "architecture": self.arch,
            "cpu_model": platform.processor() or "Unknown",
            "cpu_cores": self.cpu_cores,
            "ram_total_gb": round(self._get_total_ram_bytes() / (1024**3), 2),
            "disk_total_gb": round(total_disk / (1024**3), 2),
            "disk_free_gb": round(free_disk / (1024**3), 2),
            "gpu_mps_available": mps_available,
            "cuda_available": False,
            "python_version": platform.python_version(),
            "node_version": node_ver or "Not found",
            "ffmpeg_version": ffmpeg_ver or "Not found",
            "ollama_version": ollama_ver or "Not found",
            "piper_version": piper_ver or "Not found"
        }

    def get_realtime_metrics(self) -> dict[str, Any]:
        """
        Retrieves current CPU, RAM, and processes footprint.
        """
        ram_total = self._get_total_ram_bytes()
        ram_used = self._get_used_ram_bytes()
        ram_avail = max(0, ram_total - ram_used)
        
        cpu_usage = self._get_cpu_percentage()
        disk_total, _, disk_free = shutil.disk_usage("/")
        
        return {
            "cpu_percent": cpu_usage,
            "ram_used_gb": round(ram_used / (1024**3), 2),
            "ram_available_gb": round(ram_avail / (1024**3), 2),
            "disk_free_gb": round(disk_free / (1024**3), 2),
            "backend_process_memory_mb": round(self._get_backend_process_memory() / (1024**2), 2),
            "active_sessions": 1,
            "recommendation": self.get_performance_recommendation(cpu_usage, ram_avail)
        }

    def get_performance_recommendation(self, cpu_percent: float, ram_available_bytes: float) -> str:
        """
        Analyzes resource usage and outputs balancing recommendations.
        """
        ram_avail_gb = ram_available_bytes / (1024**3)
        if ram_avail_gb < 2.0:
            return "Use gemma3:4b for faster responses (RAM is low)"
        if cpu_percent > 80.0:
            return "Use push-to-talk in noisy room (CPU load is high)"
        return "System running optimally. Defaulting to qwen2.5:7b."

    def _get_binary_version(self, cmd: list[str]) -> str | None:
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if res.returncode == 0:
                line = res.stdout.strip().split("\n")[0]
                if cmd[0] == "ffmpeg":
                    # Extract version number
                    parts = line.split("version ")
                    return parts[1].split(" ")[0] if len(parts) > 1 else line
                return line
        except Exception:
            pass
        return None

    def _get_total_ram_bytes(self) -> int:
        if self.os_type == "Darwin":
            try:
                res = subprocess.run(["sysctl", "-n", "hw.memsize"], stdout=subprocess.PIPE, text=True, check=False)
                return int(res.stdout.strip())
            except Exception:
                pass
        elif self.os_type == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            return int(line.split()[1]) * 1024
            except Exception:
                pass
        return 16 * 1024 * 1024 * 1024  # Default mock 16GB

    def _get_used_ram_bytes(self) -> int:
        # Simple cross-platform approximation of free memory
        if self.os_type == "Darwin":
            try:
                # Parse vm_stat
                res = subprocess.run(["vm_stat"], stdout=subprocess.PIPE, text=True, check=False)
                pages_free = 0
                pages_speculative = 0
                page_size = 4096
                for line in res.stdout.split("\n"):
                    if "Pages free:" in line:
                        pages_free = int(line.split()[-1].replace(".", ""))
                    elif "Pages speculative:" in line:
                        pages_speculative = int(line.split()[-1].replace(".", ""))
                    elif "page size of" in line:
                        page_size = int(line.split()[2])
                free_bytes = (pages_free + pages_speculative) * page_size
                return self._get_total_ram_bytes() - free_bytes
            except Exception:
                pass
        elif self.os_type == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    mem_free = 0
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            return self._get_total_ram_bytes() - (int(line.split()[1]) * 1024)
                        elif line.startswith("MemFree:"):
                            mem_free = int(line.split()[1]) * 1024
                return self._get_total_ram_bytes() - mem_free
            except Exception:
                pass
        return 8 * 1024 * 1024 * 1024  # Default mock 8GB

    def _get_cpu_percentage(self) -> float:
        if self.os_type == "Darwin":
            try:
                # Run top command once to parse cpu idle rate
                res = subprocess.run(["top", "-l", "1", "-n", "0"], stdout=subprocess.PIPE, text=True, check=False)
                for line in res.stdout.split("\n"):
                    if "CPU usage" in line:
                        # "CPU usage: 12.34% user, 10.12% sys, 77.54% idle"
                        parts = line.split("idle")[0].split(",")[-1].strip().split(" ")
                        idle = float(parts[0].replace("%", ""))
                        return round(100.0 - idle, 1)
            except Exception:
                pass
        elif self.os_type == "Linux":
            try:
                # Read /proc/stat
                with open("/proc/stat", "r") as f:
                    first_line = f.readline()
                parts = first_line.split()[1:]
                vals = [float(x) for x in parts]
                idle = vals[3]
                total = sum(vals)
                return round(100.0 * (1.0 - idle / total), 1)
            except Exception:
                pass
        return 15.0  # Default mock 15%

    def _get_backend_process_memory(self) -> int:
        try:
            # sys.getsizeof() is just object level. For resident set size, call ps
            pid = os.getpid()
            if self.os_type == "Darwin" or self.os_type == "Linux":
                res = subprocess.run(["ps", "-p", str(pid), "-o", "rss="], stdout=subprocess.PIPE, text=True, check=False)
                # ps returns RSS in kilobytes
                return int(res.stdout.strip()) * 1024
        except Exception:
            pass
        return 120 * 1024 * 1024  # Default mock 120MB
