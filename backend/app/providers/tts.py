from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.services.heavy_jobs import HeavyJobBusy

from app.providers.stt import ProviderUnavailableError
from app.schemas import LanguageCode


class FallbackBlockedError(ValueError):
    """Raised when voice fallback is required but fallback is disabled/blocked by configuration."""
    pass


@dataclass(frozen=True, slots=True)
class TTSPart:
    text: str
    language: LanguageCode


@dataclass(frozen=True, slots=True)
class TTSResult:
    audio_path: Path
    generation_ms: float
    cached: bool
    requested_voice_id: str | None = None
    requested_voice_name: str | None = None
    actual_voice_id: str | None = None
    actual_voice_name: str | None = None
    actual_tts_engine: str = "piper"
    model_artifact_path: str | None = None
    language: str = "en"
    fallback_used: bool = False
    fallback_reason: str | None = None
    generated_audio_path: str | None = None


class TTSProvider(Protocol):
    def synthesize(
        self, parts: list[TTSPart], voice_id: str | None = None, fallback_allowed: bool = True
    ) -> TTSResult:
        ...


class PiperTTSProvider:
    def __init__(
        self,
        piper_binary: str,
        nepali_voice: Path,
        english_voice: Path,
        audio_cache_dir: Path,
        single_voice_model: bool = False,
    ) -> None:
        self.piper_binary = piper_binary
        self.nepali_voice = nepali_voice
        self.english_voice = english_voice
        self.single_voice_model = single_voice_model
        self.audio_cache_dir = audio_cache_dir
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
        import threading
        self._loaded_voices = {}
        self._lock = threading.RLock()

    def _get_voice_name(self, voice_id: str | None) -> str:
        if not voice_id or voice_id == "auto":
            return "Auto Select"
        BUILTIN_NAMES = {
            "ne_NP-chitwan-medium": "Chitwan Nepali (Medium)",
            "ne_NP-google-medium": "Google Nepali (Medium)",
            "en_US-lessac-medium": "Lessac English (Medium)",
            "en_US-ryan-medium": "Ryan English (Medium)",
            "ne": "Default Nepali",
            "en": "Default English",
        }
        if voice_id in BUILTIN_NAMES:
            return BUILTIN_NAMES[voice_id]
        
        # Try database lookup
        try:
            from app.database import get_db_connection
            conn = get_db_connection()
            row = conn.execute("SELECT name FROM voices WHERE id = ?;", (voice_id,)).fetchone()
            conn.close()
            if row:
                return row["name"]
        except Exception:
            pass
        return voice_id

    def synthesize(
        self, parts: list[TTSPart], voice_id: str | None = None, fallback_allowed: bool = True
    ) -> TTSResult:
        normalized = [part for part in parts if part.text.strip()]
        if not normalized:
            raise ValueError("Cannot synthesize empty text.")
        started = time.perf_counter()

        # Calculate voice traces for segments
        fallback_used = False
        fallback_reasons = []
        actual_voice_id = None
        actual_voice_name = None
        model_artifact_path = None
        languages = set()

        for part in normalized:
            languages.add(part.language)
            voice_path, act_id, act_name, fb_used, fb_reason = self._voice_for(part.language, voice_id)
            if fb_used:
                fallback_used = True
                if fb_reason:
                    fallback_reasons.append(fb_reason)
            if actual_voice_id is None:
                actual_voice_id = act_id
                actual_voice_name = act_name
                model_artifact_path = str(voice_path)

        if fallback_used and not fallback_allowed:
            raise FallbackBlockedError("; ".join(fallback_reasons) or "Voice fallback blocked by configuration.")

        requested_name = self._get_voice_name(voice_id)
        cache_key = self._cache_key(normalized, voice_id)
        output_path = self.audio_cache_dir / f"{cache_key}.wav"

        if output_path.exists():
            return TTSResult(
                audio_path=output_path,
                generation_ms=0,
                cached=True,
                requested_voice_id=voice_id,
                requested_voice_name=requested_name,
                actual_voice_id=actual_voice_id,
                actual_voice_name=actual_voice_name,
                actual_tts_engine="piper",
                model_artifact_path=model_artifact_path,
                language="-".join(sorted(list(languages))),
                fallback_used=fallback_used,
                fallback_reason="; ".join(fallback_reasons) if fallback_reasons else None,
                generated_audio_path=str(output_path),
            )
        self._ensure_ready()

        segment_paths: list[Path] = []
        for index, part in enumerate(normalized):
            voice, _, _, _, _ = self._voice_for(part.language, voice_id)
            segment_path = self.audio_cache_dir / f"{cache_key}.{index}.wav"
            self._run_piper(text=part.text, voice=voice, output_path=segment_path)
            
            # Post-process custom voice segments!
            is_custom = voice_id and voice_id not in {"auto", "ne_NP-chitwan-medium", "ne_NP-google-medium", "en_US-lessac-medium", "en_US-ryan-medium"} and not voice_id.startswith("openai-")
            if is_custom:
                apply_voice_cloning_effect(segment_path, voice_id, part.language)
                
            segment_paths.append(segment_path)

        if len(segment_paths) == 1:
            segment_paths[0].replace(output_path)
        else:
            self._concat_wavs(segment_paths, output_path)
            for segment_path in segment_paths:
                segment_path.unlink(missing_ok=True)

        return TTSResult(
            audio_path=output_path,
            generation_ms=(time.perf_counter() - started) * 1000,
            cached=False,
            requested_voice_id=voice_id,
            requested_voice_name=requested_name,
            actual_voice_id=actual_voice_id,
            actual_voice_name=actual_voice_name,
            actual_tts_engine="piper",
            model_artifact_path=model_artifact_path,
            language="-".join(sorted(list(languages))),
            fallback_used=fallback_used,
            fallback_reason="; ".join(fallback_reasons) if fallback_reasons else None,
            generated_audio_path=str(output_path),
        )

    def probe(self) -> list[tuple[str, bool, str]]:
        binary_ok = shutil.which(self.piper_binary) is not None
        return [
            ("piper", binary_ok, f"`{self.piper_binary}` found" if binary_ok else f"`{self.piper_binary}` not found"),
            (
                "nepali_voice",
                self.nepali_voice.exists(),
                str(self.nepali_voice) if self.nepali_voice.exists() else f"Missing {self.nepali_voice}",
            ),
            (
                "english_voice",
                self.english_voice.exists(),
                str(self.english_voice) if self.english_voice.exists() else f"Missing {self.english_voice}",
            ),
        ]

    def _ensure_ready(self) -> None:
        missing = [detail for _, ok, detail in self.probe() if not ok]
        if missing:
            raise ProviderUnavailableError("; ".join(missing))

    def _run_piper(self, text: str, voice: Path, output_path: Path) -> None:
        try:
            import wave
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise ProviderUnavailableError(
                "piper-tts is not installed. Run `make setup` first."
            ) from exc

        try:
            with self._lock:
                if voice not in self._loaded_voices:
                    self._loaded_voices[voice] = PiperVoice.load(voice)
                loaded_voice = self._loaded_voices[voice]
                with wave.open(str(output_path), "wb") as wav_file:
                    loaded_voice.synthesize_wav(text, wav_file)
        except Exception as exc:
            raise ProviderUnavailableError(f"Piper failed to synthesize audio with {voice}: {exc}") from exc

    def _voice_for(
        self, language: LanguageCode, voice_id: str | None = None
    ) -> tuple[Path, str, str, bool, str | None]:
        requested_name = self._get_voice_name(voice_id)
        
        # Default choices
        if self.single_voice_model:
            default_voice = self.nepali_voice
            default_id = self.nepali_voice.stem
        else:
            default_voice = self.nepali_voice if language == "ne" else self.english_voice
            default_id = "ne_NP-chitwan-medium" if language == "ne" else "en_US-lessac-medium"
        default_name = self._get_voice_name(default_id)

        if not voice_id or voice_id == "auto":
            return default_voice, default_id, default_name, False, None

        # Check built-in voice
        if voice_id in {"ne_NP-chitwan-medium", "ne_NP-google-medium", "en_US-lessac-medium", "en_US-ryan-medium"}:
            if self.single_voice_model:
                p = Path("models/piper") / f"{voice_id}.onnx"
                if p.exists():
                    return p, voice_id, self._get_voice_name(voice_id), False, None
            if language == "ne" and voice_id.startswith("ne_"):
                p = Path("models/piper") / f"{voice_id}.onnx"
                if p.exists():
                    return p, voice_id, self._get_voice_name(voice_id), False, None
            elif language == "en" and voice_id.startswith("en_"):
                p = Path("models/piper") / f"{voice_id}.onnx"
                if p.exists():
                    return p, voice_id, self._get_voice_name(voice_id), False, None
            
            # Built-in fallback due to language mismatch (e.g. Chitwan Nepali requested for English chunk)
            reason = f"Language mismatch: voice '{requested_name}' does not support {language.upper()}."
            return default_voice, default_id, default_name, True, reason

        # Check custom voice directory (recursive ONNX lookup)
        custom_dir = Path(".local/voices") / voice_id
        onnx_files = list(custom_dir.rglob("*.onnx")) if custom_dir.exists() else []

        if not onnx_files:
            reason = f"Voice '{requested_name}' artifact not found on disk."
            return default_voice, default_id, default_name, True, reason

        # Format: {voice_id}_ne_NP_medium.onnx or similar
        lang_suffix = "ne" if language == "ne" else "en"
        if self.single_voice_model and onnx_files:
            onnx_path = onnx_files[0]
            return onnx_path, voice_id, requested_name, False, None
        for onnx_path in onnx_files:
            if lang_suffix in onnx_path.name.lower():
                return onnx_path, voice_id, requested_name, False, None

        # Fallback to any ONNX in custom directory if no language match, otherwise use defaults
        for onnx_path in onnx_files:
            # If there's any ONNX, we use it as a fallback within the folder
            return onnx_path, voice_id, requested_name, True, f"Voice '{requested_name}' does not natively support {language.upper()}."

        reason = f"Voice '{requested_name}' does not support {language.upper()}."
        return default_voice, default_id, default_name, True, reason

    def _cache_key(self, parts: list[TTSPart], voice_id: str | None = None) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(str(self.nepali_voice).encode("utf-8"))
        h.update(str(self.english_voice).encode("utf-8"))
        h.update(str(self.single_voice_model).encode("utf-8"))
        if voice_id:
            h.update(voice_id.encode("utf-8"))
            # If it's a custom voice, include file paths and mtimes of ONNX files and pitch.txt
            is_custom = voice_id not in {"auto", "ne_NP-chitwan-medium", "ne_NP-google-medium", "en_US-lessac-medium", "en_US-ryan-medium"} and not voice_id.startswith("openai-")
            if is_custom:
                custom_dir = Path(".local/voices") / voice_id
                if custom_dir.exists():
                    for f in sorted(custom_dir.rglob("*")):
                        if f.is_file() and (f.suffix in {".onnx", ".json", ".txt"} or f.name == "pitch.txt"):
                            h.update(f.name.encode("utf-8"))
                            h.update(str(f.stat().st_mtime).encode("utf-8"))
        for part in parts:
            h.update(part.language.encode("utf-8"))
            h.update(part.text.strip().encode("utf-8"))
        return h.hexdigest()[:32]

    @staticmethod
    def _concat_wavs(paths: list[Path], output_path: Path) -> None:
        params = None
        frames: list[bytes] = []
        for path in paths:
            with wave.open(str(path), "rb") as wav:
                if params is None:
                    params = wav.getparams()
                elif wav.getparams()[:3] != params[:3]:
                    raise ProviderUnavailableError("Cannot concatenate outputs with different WAV formats.")
                frames.append(wav.readframes(wav.getnframes()))
        if params is None:
            raise ValueError("No WAV files to concatenate.")
        with wave.open(str(output_path), "wb") as output:
            output.setparams(params)
            for frame in frames:
                output.writeframes(frame)


class OpenAITTSProvider:
    def __init__(
        self,
        settings: Any,
        audio_cache_dir: Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.settings = settings
        self.audio_cache_dir = audio_cache_dir
        self.timeout_seconds = timeout_seconds
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def api_key(self) -> str:
        return getattr(self.settings, "openai_api_key", "")

    def synthesize(
        self, parts: list[TTSPart], voice_id: str | None = None, fallback_allowed: bool = True
    ) -> TTSResult:
        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Please configure your OpenAI API key in settings.")
            
        normalized = [part for part in parts if part.text.strip()]
        if not normalized:
            raise ValueError("Cannot synthesize empty text.")
            
        started = time.perf_counter()
        
        # Resolve voice name (e.g. openai-alloy -> alloy)
        voice_name = "alloy"
        if voice_id and voice_id.startswith("openai-"):
            voice_name = voice_id.split("-")[-1].lower()
            if voice_name not in {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}:
                voice_name = "alloy"

        cache_key = self._cache_key(normalized, voice_id)
        output_path = self.audio_cache_dir / f"{cache_key}.wav"

        languages = set(part.language for part in normalized)

        if output_path.exists():
            return TTSResult(
                audio_path=output_path,
                generation_ms=0,
                cached=True,
                requested_voice_id=voice_id,
                requested_voice_name=f"OpenAI {voice_name.capitalize()}",
                actual_voice_id=voice_id,
                actual_voice_name=f"OpenAI {voice_name.capitalize()}",
                actual_tts_engine="openai",
                model_artifact_path="https://api.openai.com/v1/audio/speech",
                language="-".join(sorted(list(languages))),
                fallback_used=False,
                fallback_reason=None,
                generated_audio_path=str(output_path),
            )

        segment_paths: list[Path] = []
        import urllib.request
        import urllib.error
        import socket
        import ssl
        import http.client
        
        for index, part in enumerate(normalized):
            segment_path = self.audio_cache_dir / f"{cache_key}.{index}.wav"
            
            payload = {
                "model": "tts-1",
                "input": part.text.strip(),
                "voice": voice_name,
                "response_format": "wav"
            }
            
            url = "https://api.openai.com/v1/audio/speech"
            max_retries = 3
            last_exc = None
            audio_bytes = None
            
            for attempt in range(max_retries):
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST"
                )
                try:
                    timeout = self.timeout_seconds * (1.5 ** attempt)
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        audio_bytes = response.read()
                    break  # Success!
                except urllib.error.HTTPError as exc:
                    last_exc = exc
                    # Retry on 408 Timeout, 429 Too Many Requests, or 5xx server errors
                    if exc.code in (408, 429) or exc.code >= 500:
                        if attempt < max_retries - 1:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                    break
                except (urllib.error.URLError, TimeoutError, socket.timeout, ssl.SSLError, ConnectionError, http.client.HTTPException) as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    break
                except Exception as exc:
                    last_exc = exc
                    break
            
            if audio_bytes is None:
                if last_exc and isinstance(last_exc, urllib.error.HTTPError):
                    err_body = last_exc.read().decode("utf-8", errors="ignore")
                    raise RuntimeError(f"OpenAI Speech API error ({last_exc.code}): {err_body}") from last_exc
                raise RuntimeError(f"OpenAI Speech connection error: {last_exc}") from last_exc
                
            try:
                segment_path.write_bytes(audio_bytes)
                segment_paths.append(segment_path)
            except Exception as exc:
                raise exc

        if len(segment_paths) == 1:
            segment_paths[0].replace(output_path)
        else:
            PiperTTSProvider._concat_wavs(segment_paths, output_path)
            for segment_path in segment_paths:
                segment_path.unlink(missing_ok=True)

        return TTSResult(
            audio_path=output_path,
            generation_ms=(time.perf_counter() - started) * 1000,
            cached=False,
            requested_voice_id=voice_id,
            requested_voice_name=f"OpenAI {voice_name.capitalize()}",
            actual_voice_id=voice_id,
            actual_voice_name=f"OpenAI {voice_name.capitalize()}",
            actual_tts_engine="openai",
            model_artifact_path="https://api.openai.com/v1/audio/speech",
            language="-".join(sorted(list(languages))),
            fallback_used=False,
            fallback_reason=None,
            generated_audio_path=str(output_path),
        )

    def _cache_key(self, parts: list[TTSPart], voice_id: str | None = None) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(b"openai")
        if voice_id:
            h.update(voice_id.encode("utf-8"))
        for part in parts:
            h.update(part.language.encode("utf-8"))
            h.update(part.text.strip().encode("utf-8"))
        return h.hexdigest()[:32]


class ElevenLabsTTSProvider:
    def __init__(
        self,
        settings: Any,
        audio_cache_dir: Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.settings = settings
        self.audio_cache_dir = audio_cache_dir
        self.timeout_seconds = timeout_seconds
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def api_key(self) -> str:
        return getattr(self.settings, "elevenlabs_api_key", "")

    def synthesize(
        self, parts: list[TTSPart], voice_id: str | None = None, fallback_allowed: bool = True
    ) -> TTSResult:
        if not self.api_key:
            raise ValueError("ElevenLabs API key is missing. Please configure it in settings.")
            
        normalized = [part for part in parts if part.text.strip()]
        if not normalized:
            raise ValueError("Cannot synthesize empty text.")
            
        started = time.perf_counter()
        
        # Load ElevenLabs voice ID
        el_voice_id = None
        if voice_id:
            voices_dir = Path(".local/voices")
            if hasattr(self.settings, "audio_work_dir") and self.settings.audio_work_dir:
                voices_dir = Path(self.settings.audio_work_dir).parent / "voices"
            el_id_path = voices_dir / voice_id / "elevenlabs_id.txt"
            if el_id_path.exists():
                el_voice_id = el_id_path.read_text(encoding="utf-8").strip()
            else:
                el_voice_id = voice_id  # fallback: use directly if it is the elevenlabs ID
                
        if not el_voice_id:
            raise ValueError(f"ElevenLabs voice ID not found for voice: {voice_id}")

        cache_key = self._cache_key(normalized, voice_id)
        output_path = self.audio_cache_dir / f"{cache_key}.wav"

        languages = set(part.language for part in normalized)

        if output_path.exists():
            return TTSResult(
                audio_path=output_path,
                generation_ms=0,
                cached=True,
                requested_voice_id=voice_id,
                requested_voice_name=f"ElevenLabs Voice",
                actual_voice_id=voice_id,
                actual_voice_name=f"ElevenLabs Voice",
                actual_tts_engine="elevenlabs",
                model_artifact_path=f"elevenlabs://{el_voice_id}",
                language="-".join(sorted(list(languages))),
                fallback_used=False,
                fallback_reason=None,
                generated_audio_path=str(output_path),
            )

        segment_paths: list[Path] = []
        import urllib.request
        import urllib.error
        import json
        import subprocess
        import shutil
        
        for index, part in enumerate(normalized):
            segment_path = self.audio_cache_dir / f"{cache_key}.{index}.wav"
            temp_mp3 = self.audio_cache_dir / f"{cache_key}.{index}.mp3"
            
            # Select model: multilingual v2 is excellent for mixed/Nepali/English
            payload = {
                "text": part.text.strip(),
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{el_voice_id}"
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "xi-api-key": self.api_key,
                },
                method="POST"
            )
            
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    audio_bytes = response.read()
                # Save MP3
                temp_mp3.write_bytes(audio_bytes)
                
                # Transcode MP3 to WAV using FFmpeg: mono, 22050Hz, s16 format
                if shutil.which("ffmpeg") is None:
                    raise RuntimeError("ffmpeg is required to decode ElevenLabs MP3 response.")
                cmd = [
                    "ffmpeg", "-y", "-i", str(temp_mp3),
                    "-ac", "1", "-ar", "22050", "-sample_fmt", "s16",
                    str(segment_path)
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                # Cleanup temp MP3
                if temp_mp3.exists():
                    temp_mp3.unlink()
                    
                segment_paths.append(segment_path)
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"ElevenLabs Speech API error ({exc.code}): {err_body}") from exc
            except Exception as exc:
                raise RuntimeError(f"ElevenLabs Speech connection error: {exc}") from exc

        if len(segment_paths) == 1:
            segment_paths[0].replace(output_path)
        else:
            PiperTTSProvider._concat_wavs(segment_paths, output_path)
            for segment_path in segment_paths:
                segment_path.unlink(missing_ok=True)

        return TTSResult(
            audio_path=output_path,
            generation_ms=(time.perf_counter() - started) * 1000,
            cached=False,
            requested_voice_id=voice_id,
            requested_voice_name=f"ElevenLabs Voice",
            actual_voice_id=voice_id,
            actual_voice_name=f"ElevenLabs Voice",
            actual_tts_engine="elevenlabs",
            model_artifact_path=f"elevenlabs://{el_voice_id}",
            language="-".join(sorted(list(languages))),
            fallback_used=False,
            fallback_reason=None,
            generated_audio_path=str(output_path),
        )

    def _cache_key(self, parts: list[TTSPart], voice_id: str | None = None) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(b"elevenlabs")
        if voice_id:
            h.update(voice_id.encode("utf-8"))
        for part in parts:
            h.update(part.language.encode("utf-8"))
            h.update(part.text.strip().encode("utf-8"))
        return h.hexdigest()[:32]


class ChatterboxTTSProvider:
    def __init__(
        self,
        settings: Any,
        audio_cache_dir: Path,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.settings = settings
        self.audio_cache_dir = audio_cache_dir
        self.timeout_seconds = timeout_seconds
        self.audio_cache_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[tuple[str, str], Any] = {}

    def synthesize(
        self, parts: list[TTSPart], voice_id: str | None = None, fallback_allowed: bool = True
    ) -> TTSResult:
        if not voice_id:
            raise ValueError("Chatterbox synthesis requires a custom voice id.")

        reference_path = self._reference_path(voice_id)
        if not reference_path.exists():
            raise ValueError(f"Chatterbox reference audio is missing for voice '{voice_id}'.")

        normalized = [part for part in parts if part.text.strip()]
        if not normalized:
            raise ValueError("Cannot synthesize empty text.")

        requested_name = self._get_voice_name(voice_id)
        started = time.perf_counter()
        languages = set(part.language for part in normalized)
        cache_key = self._cache_key(normalized, voice_id, reference_path)
        output_path = self.audio_cache_dir / f"{cache_key}.wav"
        language_note = self._language_note(languages)

        if output_path.exists():
            return TTSResult(
                audio_path=output_path,
                generation_ms=0,
                cached=True,
                requested_voice_id=voice_id,
                requested_voice_name=requested_name,
                actual_voice_id=voice_id,
                actual_voice_name=requested_name,
                actual_tts_engine="chatterbox",
                model_artifact_path=f"chatterbox://{reference_path}",
                language="-".join(sorted(list(languages))),
                fallback_used=bool(language_note),
                fallback_reason=language_note,
                generated_audio_path=str(output_path),
            )

        # One cloned-voice synthesis at a time, process-wide. If another is already
        # running, this raises HeavyJobBusy immediately (caller gets a clear "wait ~Ns"
        # message) instead of piling up and crashing the backend.
        from app.services.heavy_jobs import CLONED_VOICE
        segment_paths: list[Path] = []
        with CLONED_VOICE:
            for index, part in enumerate(normalized):
                segment_path = self.audio_cache_dir / f"{cache_key}.{index}.wav"
                self._generate_segment(part, reference_path, segment_path)
                segment_paths.append(segment_path)

        if len(segment_paths) == 1:
            segment_paths[0].replace(output_path)
        else:
            PiperTTSProvider._concat_wavs(segment_paths, output_path)
            for segment_path in segment_paths:
                segment_path.unlink(missing_ok=True)

        return TTSResult(
            audio_path=output_path,
            generation_ms=(time.perf_counter() - started) * 1000,
            cached=False,
            requested_voice_id=voice_id,
            requested_voice_name=requested_name,
            actual_voice_id=voice_id,
            actual_voice_name=requested_name,
            actual_tts_engine="chatterbox",
            model_artifact_path=f"chatterbox://{reference_path}",
            language="-".join(sorted(list(languages))),
            fallback_used=bool(language_note),
            fallback_reason=language_note,
            generated_audio_path=str(output_path),
        )

    def _generate_segment(self, part: TTSPart, reference_path: Path, output_path: Path) -> None:
        try:
            import torch
            import torchaudio as ta
        except ImportError as exc:
            raise ProviderUnavailableError(
                "Chatterbox local voice cloning is not installed. Run: "
                ".venv/bin/python -m pip install chatterbox-tts"
            ) from exc

        device = self._device(torch)
        exaggeration = getattr(self.settings, "chatterbox_exaggeration", 0.5)
        cfg_weight = getattr(self.settings, "chatterbox_cfg_weight", 0.5)
        temperature = getattr(self.settings, "chatterbox_temperature", 0.8)
        repetition_penalty = getattr(self.settings, "chatterbox_repetition_penalty", 1.2)

        try:
            model, language_id = self._model_for_language(part.language, device)
            kwargs: dict[str, Any] = {
                "audio_prompt_path": str(reference_path),
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            }
            if language_id is not None:
                kwargs["language_id"] = language_id
            wav = model.generate(part.text.strip(), **kwargs)
            self._save_wav(output_path, wav, model.sr)
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            if device != "cpu":
                self._models.pop(("tts", device), None)
                self._models.pop(("mtl", device), None)
                try:
                    model, language_id = self._model_for_language(part.language, "cpu")
                    kwargs = {
                        "audio_prompt_path": str(reference_path),
                        "exaggeration": exaggeration,
                        "cfg_weight": cfg_weight,
                        "temperature": temperature,
                        "repetition_penalty": repetition_penalty,
                    }
                    if language_id is not None:
                        kwargs["language_id"] = language_id
                    wav = model.generate(part.text.strip(), **kwargs)
                    self._save_wav(output_path, wav, model.sr)
                    return
                except Exception:
                    pass
            raise ProviderUnavailableError(f"Chatterbox failed to synthesize cloned voice audio: {exc}") from exc

    @staticmethod
    def _save_wav(output_path: Any, wav: Any, sr: int) -> None:
        """Write the generated waveform to disk.

        torchaudio 2.11 routes ``save`` through TorchCodec, which needs an
        ffmpeg build TorchCodec supports (<= 7). On systems with a newer ffmpeg
        that load fails, so we write via soundfile (libsndfile), which has no
        such dependency, and only fall back to torchaudio if soundfile is
        unavailable.
        """
        try:
            import numpy as np
            import soundfile as sf
            data = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
            if data.ndim == 2 and data.shape[0] < data.shape[1]:
                data = data.T  # (channels, frames) -> (frames, channels)
            sf.write(str(output_path), data, int(sr))
            return
        except Exception:
            import torchaudio as ta
            ta.save(str(output_path), wav, int(sr))

    def _model_for_language(self, language: LanguageCode, device: str) -> tuple[Any, str | None]:
        if language == "en":
            key = ("tts", device)
            if key not in self._models:
                try:
                    from chatterbox.tts import ChatterboxTTS
                except ImportError as exc:
                    raise ProviderUnavailableError(
                        "Chatterbox local voice cloning is not installed. Run: "
                        ".venv/bin/python -m pip install chatterbox-tts"
                    ) from exc
                self._models[key] = ChatterboxTTS.from_pretrained(device=device)
            return self._models[key], None

        key = ("mtl", device)
        if key not in self._models:
            try:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            except ImportError as exc:
                raise ProviderUnavailableError(
                    "Chatterbox multilingual voice cloning is not installed. Run: "
                    ".venv/bin/python -m pip install chatterbox-tts"
                ) from exc
            self._models[key] = ChatterboxMultilingualTTS.from_pretrained(device=device)
        return self._models[key], self._language_id(language)

    @staticmethod
    def _device(torch_module: Any) -> str:
        requested = str(getattr(torch_module, "device", "") or "").strip()
        configured = ""
        try:
            import os
            configured = os.getenv("CHATTERBOX_DEVICE", "").strip().lower()
        except Exception:
            configured = ""
        if configured and configured != "auto":
            return configured
        try:
            if torch_module.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        # Chatterbox can expose an MPS path, but on some macOS/Metal versions
        # generation aborts the whole Python process. Use CPU by default on Mac;
        # advanced users can opt in with CHATTERBOX_DEVICE=mps.
        return "cpu" if not requested else "cpu"

    @staticmethod
    def _language_id(language: LanguageCode) -> str:
        if language == "en":
            return "en"
        if language == "ne":
            return "hi"
        if language == "mixed":
            return "hi"
        return "en"

    @staticmethod
    def _language_note(languages: set[LanguageCode]) -> str | None:
        if "ne" in languages or "mixed" in languages:
            return "Chatterbox has no native Nepali checkpoint; using its Hindi multilingual route for Nepali text."
        return None

    def _reference_path(self, voice_id: str) -> Path:
        voices_dir = Path(".local/voices")
        if hasattr(self.settings, "audio_work_dir") and self.settings.audio_work_dir:
            voices_dir = Path(self.settings.audio_work_dir).parent / "voices"
        return voices_dir / voice_id / "chatterbox_reference.wav"

    def _get_voice_name(self, voice_id: str) -> str:
        try:
            from app.database import get_db_connection
            conn = get_db_connection()
            row = conn.execute("SELECT name FROM voices WHERE id = ?;", (voice_id,)).fetchone()
            conn.close()
            if row:
                return row["name"]
        except Exception:
            pass
        return voice_id

    def _cache_key(self, parts: list[TTSPart], voice_id: str, reference_path: Path) -> str:
        h = hashlib.sha256()
        h.update(b"chatterbox")
        h.update(voice_id.encode("utf-8"))
        h.update(str(reference_path).encode("utf-8"))
        h.update(str(reference_path.stat().st_mtime).encode("utf-8"))
        h.update(str(getattr(self.settings, "chatterbox_exaggeration", 0.5)).encode("utf-8"))
        h.update(str(getattr(self.settings, "chatterbox_cfg_weight", 0.5)).encode("utf-8"))
        h.update(str(getattr(self.settings, "chatterbox_temperature", 0.8)).encode("utf-8"))
        h.update(str(getattr(self.settings, "chatterbox_repetition_penalty", 1.2)).encode("utf-8"))
        manifest_path = reference_path.with_name("chatterbox_manifest.json")
        if manifest_path.exists():
            h.update(manifest_path.read_bytes())
        for part in parts:
            h.update(part.language.encode("utf-8"))
            h.update(part.text.strip().encode("utf-8"))
        return h.hexdigest()[:32]


class TTSRouter:
    def __init__(
        self,
        piper_provider: PiperTTSProvider,
        openai_provider: OpenAITTSProvider,
        elevenlabs_provider: ElevenLabsTTSProvider | None = None,
        chatterbox_provider: ChatterboxTTSProvider | None = None,
    ) -> None:
        self.piper_provider = piper_provider
        self.openai_provider = openai_provider
        self.elevenlabs_provider = elevenlabs_provider
        self.chatterbox_provider = chatterbox_provider

    def synthesize(
        self, parts: list[TTSPart], voice_id: str | None = None, fallback_allowed: bool = True
    ) -> TTSResult:
        if voice_id and voice_id.startswith("openai-"):
            return self.openai_provider.synthesize(parts, voice_id=voice_id, fallback_allowed=fallback_allowed)
            
        engine = self._custom_voice_engine(voice_id)

        if engine == "chatterbox":
            try:
                if self.chatterbox_provider is None:
                    raise ValueError("Chatterbox provider is not configured.")
                return self.chatterbox_provider.synthesize(parts, voice_id=voice_id, fallback_allowed=fallback_allowed)
            except HeavyJobBusy:
                # Don't silently swap to another voice — let the caller tell the
                # user "a cloned voice is already running, please wait".
                raise
            except Exception as exc:
                if not fallback_allowed:
                    raise
                res = self.piper_provider.synthesize(parts, voice_id=voice_id, fallback_allowed=fallback_allowed)
                return TTSResult(
                    audio_path=res.audio_path,
                    generation_ms=res.generation_ms,
                    cached=res.cached,
                    requested_voice_id=voice_id,
                    requested_voice_name=res.requested_voice_name,
                    actual_voice_id=res.actual_voice_id,
                    actual_voice_name=res.actual_voice_name,
                    actual_tts_engine=res.actual_tts_engine,
                    model_artifact_path=res.model_artifact_path,
                    language=res.language,
                    fallback_used=True,
                    fallback_reason=f"Chatterbox synthesis failed ({exc}). Falling back to Piper.",
                    generated_audio_path=res.generated_audio_path,
                )

        # Check if the voice ID corresponds to elevenlabs
        is_elevenlabs = engine == "elevenlabs"
        if voice_id:
            if voice_id.startswith("elevenlabs-"):
                is_elevenlabs = True
            else:
                # check file or database
                voices_dir = Path(".local/voices")
                if self.elevenlabs_provider and hasattr(self.elevenlabs_provider.settings, "audio_work_dir") and self.elevenlabs_provider.settings.audio_work_dir:
                    voices_dir = Path(self.elevenlabs_provider.settings.audio_work_dir).parent / "voices"
                if (voices_dir / voice_id / "elevenlabs_id.txt").exists():
                    is_elevenlabs = True
                else:
                    try:
                        from app.database import get_db_connection
                        conn = get_db_connection()
                        row = conn.execute("SELECT engine FROM voices WHERE id = ?;", (voice_id,)).fetchone()
                        conn.close()
                        if row and row["engine"] == "elevenlabs":
                            is_elevenlabs = True
                    except Exception as e:
                        pass

        if is_elevenlabs:
            try:
                return self.elevenlabs_provider.synthesize(parts, voice_id=voice_id, fallback_allowed=fallback_allowed)
            except Exception as exc:
                if not fallback_allowed:
                    raise
                # Fallback to piper!
                res = self.piper_provider.synthesize(parts, voice_id=voice_id, fallback_allowed=fallback_allowed)
                return TTSResult(
                    audio_path=res.audio_path,
                    generation_ms=res.generation_ms,
                    cached=res.cached,
                    requested_voice_id=voice_id,
                    requested_voice_name=res.requested_voice_name,
                    actual_voice_id=res.actual_voice_id,
                    actual_voice_name=res.actual_voice_name,
                    actual_tts_engine=res.actual_tts_engine,
                    model_artifact_path=res.model_artifact_path,
                    language=res.language,
                    fallback_used=True,
                    fallback_reason=f"ElevenLabs synthesis failed ({exc}). Falling back to Piper.",
                    generated_audio_path=res.generated_audio_path,
                )

        return self.piper_provider.synthesize(parts, voice_id=voice_id, fallback_allowed=fallback_allowed)

    def probe(self) -> list[tuple[str, bool, str]]:
        return self.piper_provider.probe()

    def _custom_voice_engine(self, voice_id: str | None) -> str | None:
        if not voice_id or voice_id == "auto" or voice_id.startswith("openai-"):
            return None
        voices_dir = Path(".local/voices")
        for provider in (self.elevenlabs_provider, self.chatterbox_provider):
            if provider and hasattr(provider, "settings") and getattr(provider.settings, "audio_work_dir", None):
                voices_dir = Path(provider.settings.audio_work_dir).parent / "voices"
                break
        if (voices_dir / voice_id / "chatterbox_reference.wav").exists():
            return "chatterbox"
        if (voices_dir / voice_id / "elevenlabs_id.txt").exists():
            return "elevenlabs"
        try:
            from app.database import get_db_connection
            conn = get_db_connection()
            row = conn.execute("SELECT engine FROM voices WHERE id = ?;", (voice_id,)).fetchone()
            conn.close()
            if row:
                return row["engine"]
        except Exception:
            pass
        return None


def estimate_pitch(wav_path: Path) -> float | None:
    try:
        import wave
        import struct
        import math
        
        if not wav_path.exists():
            return None
            
        with wave.open(str(wav_path), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            
        if len(frames) == 0:
            return None
            
        if sample_width == 1:
            samples = [value - 128 for value in frames]
        elif sample_width == 2:
            samples = list(struct.unpack(f"<{len(frames) // 2}h", frames))
        else:
            return None
            
        num_samples = len(samples)
        if num_samples < 1000:
            return None
            
        start = max(0, num_samples // 2 - 20000)
        end = min(num_samples, num_samples // 2 + 20000)
        chunk = samples[start:end]
        
        min_lag = int(frame_rate / 300)
        max_lag = int(frame_rate / 80)
        
        best_lag = 0
        best_corr = -1.0
        
        for lag in range(min_lag, max_lag):
            corr = 0.0
            norm1 = 0.0
            norm2 = 0.0
            step = 4
            for i in range(0, len(chunk) - lag - step, step):
                corr += chunk[i] * chunk[i + lag]
                norm1 += chunk[i] * chunk[i]
                norm2 += chunk[i + lag] * chunk[i + lag]
            if norm1 > 0.0 and norm2 > 0.0:
                corr = corr / math.sqrt(norm1 * norm2)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
                
        if best_lag > 0:
            return frame_rate / best_lag
    except Exception:
        pass
    return None


def apply_voice_cloning_effect(wav_path: Path, voice_id: str, language: str) -> None:
    try:
        import json
        pitch_file = Path(".local/voices") / voice_id / "pitch.txt"
        if pitch_file.exists():
            try:
                avg_user_pitch = float(pitch_file.read_text(encoding="utf-8").strip())
            except Exception:
                avg_user_pitch = 160.0
        else:
            # Fallback scan of normalized WAV files
            ref_dir = Path(".local/voices") / voice_id / "normalized"
            wav_files = list(ref_dir.glob("*.wav")) if ref_dir.exists() else []
            pitches = []
            for wf in wav_files:
                p = estimate_pitch(wf)
                if p is not None:
                    pitches.append(p)
            avg_user_pitch = sum(pitches) / len(pitches) if pitches else 160.0
            
        base_pitch = 140.0 if language == "ne" else 180.0
        ratio = avg_user_pitch / base_pitch
        
        # Limit ratio range to avoid extreme distortion
        ratio = max(0.65, min(1.65, ratio))
        
        import shutil
        import subprocess
        if shutil.which("ffmpeg") is not None:
            temp_out = wav_path.with_name(f"vc_{wav_path.name}")
            
            import wave
            with wave.open(str(wav_path), "rb") as w:
                sample_rate = w.getframerate()
                
            filter_str = f"asetrate={int(sample_rate * ratio)},atempo={1.0/ratio:.4f}"
            
            # Equalizer tweaks to match speaker timbre
            if avg_user_pitch < 150.0:
                # Male boost
                filter_str += ",equalizer=f=350:width_type=h:w=150:g=2.0"
            else:
                # Female boost
                filter_str += ",equalizer=f=3000:width_type=h:w=1000:g=1.5"
            
            # Prevent clipping using a peak limiter
            filter_str += ",alimiter=limit=0.95"
                
            cmd = [
                "ffmpeg", "-y", "-i", str(wav_path),
                "-af", filter_str,
                str(temp_out)
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=15)
            if temp_out.exists():
                temp_out.replace(wav_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
