from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ProviderUnavailableError(RuntimeError):
    """Raised when a local provider dependency is missing or not configured."""


@dataclass(frozen=True, slots=True)
class STTResult:
    text: str
    language: str | None
    confidence: float | None
    duration_ms: float


class STTProvider(Protocol):
    def transcribe_file(self, audio_path: Path) -> STTResult:
        ...


class FasterWhisperSTTProvider:
    def __init__(self, model_size: str, device: str = "auto", compute_type: str = "auto") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise ProviderUnavailableError(
                    "faster-whisper is not installed. Run `make setup` first."
                ) from exc

            device = self.device if self.device != "auto" else None
            compute_type = self.compute_type if self.compute_type != "auto" else None

            # Auto-detect: try GPU first, fall back to CPU to avoid missing cuBLAS/cuDNN errors
            try:
                if device is None:
                    import ctypes, platform
                    cuda_ok = False
                    if platform.system() == "Windows":
                        for lib in ("cublas64_12.dll", "cublas64_11.dll", "cudart64_12.dll"):
                            try:
                                ctypes.CDLL(lib)
                                cuda_ok = True
                                break
                            except OSError:
                                pass
                    else:
                        try:
                            ctypes.CDLL("libcublas.so.12")
                            cuda_ok = True
                        except OSError:
                            try:
                                ctypes.CDLL("libcublas.so.11")
                                cuda_ok = True
                            except OSError:
                                pass
                    if not cuda_ok:
                        device = "cpu"
                        compute_type = compute_type or "int8"

                kwargs = {}
                if device is not None:
                    kwargs["device"] = device
                if compute_type is not None:
                    kwargs["compute_type"] = compute_type
                self._model = WhisperModel(self.model_size, **kwargs)
            except Exception as exc:
                # Hard fallback: CPU + int8 always works
                import logging
                logging.getLogger(__name__).warning(
                    "Whisper GPU init failed (%s), falling back to CPU/int8.", exc
                )
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe_file(self, audio_path: Path, language: str | None = None) -> STTResult:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        started = time.perf_counter()
        
        kwargs: dict = {"vad_filter": True, "beam_size": 1}
        if language in ("ne", "en"):
            kwargs["language"] = language
            
        segments, info = self.model.transcribe(str(audio_path), **kwargs)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        duration_ms = (time.perf_counter() - started) * 1000
        return STTResult(
            text=text,
            language=getattr(info, "language", None),
            confidence=getattr(info, "language_probability", None),
            duration_ms=duration_ms,
        )



class OpenAISTTProvider:
    def __init__(self, get_api_key, timeout_seconds: float = 30.0) -> None:
        self.get_api_key = get_api_key
        self.timeout_seconds = timeout_seconds

    def transcribe_file(self, audio_path: Path, language: str | None = None) -> STTResult:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        
        api_key = self.get_api_key()
        if not api_key:
            raise ValueError("OpenAI API key is missing. Please configure your OpenAI API key in settings.")
            
        import urllib.request
        import urllib.error
        import json
        import mimetypes
        import uuid
        import socket
        import ssl
        import http.client
        
        started = time.perf_counter()
        
        boundary = f"Boundary-{uuid.uuid4().hex}"
        mime_type, _ = mimetypes.guess_type(str(audio_path))
        if not mime_type:
            mime_type = "audio/wav"
            
        file_content = audio_path.read_bytes()
        file_name = audio_path.name
        
        body_parts = []
        body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n".encode("utf-8"))
        bilingual_prompt = "Namaste, tapai kasto chha? Mero naam... Please transcribe Nepali in Devanagari script (नेपाली) and English in English."
        body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n{bilingual_prompt}\r\n".encode("utf-8"))
        if language in ("ne", "en"):
            body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{language}\r\n".encode("utf-8"))
        body_parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\nContent-Type: {mime_type}\r\n\r\n".encode("utf-8")
        )
        body_parts.append(file_content)
        body_parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))


        
        body = b"".join(body_parts)
        
        url = "https://api.openai.com/v1/audio/transcriptions"
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST"
        )
        
        max_retries = 3
        last_exc = None
        resp_data = None
        
        for attempt in range(max_retries):
            try:
                timeout = self.timeout_seconds * (1.5 ** attempt)
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    resp_bytes = response.read()
                    resp_data = json.loads(resp_bytes.decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code in (408, 429) or exc.code >= 500:
                    import time as _time
                    _time.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    break
            except (urllib.error.URLError, TimeoutError, socket.timeout, ssl.SSLError, ConnectionError, http.client.HTTPException) as exc:
                last_exc = exc
                import time as _time
                _time.sleep(0.5 * (attempt + 1))
                continue
            except Exception as exc:
                last_exc = exc
                break
                
        if resp_data is None:
            if last_exc and isinstance(last_exc, urllib.error.HTTPError):
                try:
                    err_body = last_exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    err_body = str(last_exc)
                raise RuntimeError(f"OpenAI Transcription API error ({last_exc.code}): {err_body}") from last_exc
            raise RuntimeError(f"OpenAI Transcription connection error (after {max_retries} attempts): {last_exc}") from last_exc
            
        text = resp_data.get("text", "").strip()
        duration_ms = (time.perf_counter() - started) * 1000
        
        return STTResult(
            text=text,
            language=resp_data.get("language", None),
            confidence=1.0 if text else 0.0,
            duration_ms=duration_ms
        )


class STTRouter:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._local_provider = None
        self._openai_provider = None
        self._last_whisper_model_size = None
        self._last_whisper_device = None
        self._last_whisper_compute_type = None

    @property
    def local_provider(self) -> FasterWhisperSTTProvider:
        current_size = getattr(self.settings, "whisper_model_size", "small")
        current_device = getattr(self.settings, "whisper_device", "auto")
        current_comp = getattr(self.settings, "whisper_compute_type", "auto")

        if (self._local_provider is None or 
            self._last_whisper_model_size != current_size or
            self._last_whisper_device != current_device or
            self._last_whisper_compute_type != current_comp):
            self._local_provider = FasterWhisperSTTProvider(
                model_size=current_size,
                device=current_device,
                compute_type=current_comp,
            )
            self._last_whisper_model_size = current_size
            self._last_whisper_device = current_device
            self._last_whisper_compute_type = current_comp
        return self._local_provider

    @property
    def openai_provider(self) -> OpenAISTTProvider:
        if self._openai_provider is None:
            self._openai_provider = OpenAISTTProvider(
                get_api_key=lambda: getattr(self.settings, "openai_api_key", ""),
                timeout_seconds=getattr(self.settings, "cloud_timeout_seconds", 30.0),
            )
        else:
            self._openai_provider.timeout_seconds = getattr(self.settings, "cloud_timeout_seconds", 30.0)
        return self._openai_provider

    def transcribe_file(self, audio_path: Path, provider_name: str | None = None, language: str | None = None) -> STTResult:
        if not provider_name:
            provider_name = getattr(self.settings, "stt_provider", "local")
        if provider_name == "openai":
            return self.openai_provider.transcribe_file(audio_path, language=language)
        else:
            return self.local_provider.transcribe_file(audio_path, language=language)


