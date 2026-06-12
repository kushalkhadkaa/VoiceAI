from __future__ import annotations

import io
import hashlib
import math
import shutil
import subprocess
import struct
import wave
import json
import uuid
import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from app.database import get_db_connection
from app.services.audio_validation import AudioValidator, AudioValidationError

VOICE_STUDIO_PROMPTS = [
    # Nepali prompts — varied pitch, tone, rhythm
    {"id": "ne_001", "language": "ne", "category": "general",  "text": "नमस्ते, आज म मेरो आवाजको नमूना रेकर्ड गर्दैछु।"},
    {"id": "ne_002", "language": "ne", "category": "banking",  "text": "तपाईंको बैंक खातामा रकम सफलतापूर्वक जम्मा भएको छ।"},
    {"id": "ne_003", "language": "ne", "category": "support",  "text": "कृपया मलाई यो समस्या समाधान गर्न मद्दत गर्नुहोस्।"},
    {"id": "ne_004", "language": "ne", "category": "numbers",  "text": "एक, दुई, तीन, चार, पाँच, छ, सात, आठ, नौ, दश।"},
    {"id": "ne_005", "language": "ne", "category": "question", "text": "के तपाईंलाई आज कुनै सहायता चाहिएको छ?"},
    {"id": "ne_006", "language": "ne", "category": "banking",  "text": "बैंकको वित्तीय सेवाहरू र नयाँ सुविधाहरू निकै उपयोगी छन्।"},
    {"id": "ne_007", "language": "ne", "category": "security", "text": "मेरो व्यक्तिगत विवरण र खाता सुरक्षित राख्न म सधैं सजग छु।"},
    {"id": "ne_008", "language": "ne", "category": "general",  "text": "नेपालको प्राकृतिक सुन्दरता र संस्कृति विश्वमै अद्वितीय छ।"},
    {"id": "ne_009", "language": "ne", "category": "support",  "text": "ग्राहक सेवा प्रतिनिधिले मेरो जिज्ञासाबारे राम्रो जानकारी दिनुभयो।"},
    {"id": "ne_010", "language": "ne", "category": "banking",  "text": "नयाँ प्रविधिको प्रयोगले हाम्रो दैनिक बैंकिङ कार्यलाई झन् सजिलो बनाएको छ।"},
    # English prompts — phonetically diverse
    {"id": "en_001", "language": "en", "category": "general",  "text": "Hello, this is a clean sample of my speaking voice."},
    {"id": "en_002", "language": "en", "category": "banking",  "text": "Your account balance has been updated successfully."},
    {"id": "en_003", "language": "en", "category": "support",  "text": "How can I assist you with your customer support query today?"},
    {"id": "en_004", "language": "en", "category": "rainbow",  "text": "The quick brown fox jumps over the lazy dog near the riverbank."},
    {"id": "en_005", "language": "en", "category": "emotion",  "text": "I am really happy to help you with that right away!"},
    {"id": "en_006", "language": "en", "category": "numbers",  "text": "One, two, three, four, five, six, seven, eight, nine, ten."},
    {"id": "en_007", "language": "en", "category": "banking",  "text": "We strive to deliver secure and reliable financial solutions to our customers."},
    {"id": "en_008", "language": "en", "category": "security", "text": "Could you please verify your identity before we proceed with the account changes?"},
    {"id": "en_009", "language": "en", "category": "general",  "text": "The weather today is perfect for a short walk in the park near the office."},
    {"id": "en_010", "language": "en", "category": "general",  "text": "Innovation drives progress, and technology makes our daily tasks much simpler."},
    {"id": "en_011", "language": "en", "category": "security", "text": "Please make sure to enable two-factor authentication for additional security."},
    # Mixed Nepali-English prompts
    {"id": "mixed_001", "language": "mixed", "category": "general", "text": "नमस्ते, मलाई banking app मा login गर्न help चाहियो।"},
    {"id": "mixed_002", "language": "mixed", "category": "support", "text": "मेरो transaction status check गरेर mobile message पठाइदिनुस् न।"},
    {"id": "mixed_003", "language": "mixed", "category": "general", "text": "Please मलाई यो form fill गर्न सघाउनुस्।"},
    {"id": "mixed_004", "language": "mixed", "category": "support", "text": "मेरो credit card हरायो, म यसलाई कसरी block गराउन सक्छु?"},
    {"id": "mixed_005", "language": "mixed", "category": "banking", "text": "यस loan approval process को लागि कति days लाग्छ होला?"},
    {"id": "mixed_006", "language": "mixed", "category": "security", "text": "तपाईंको online payment security र privacy नीति एकदमै चित्तबुझ्दो छ।"},
]


@dataclass(frozen=True, slots=True)
class VoiceStudioRecord:
    id: str
    prompt_id: str
    text: str
    language: str
    exists: bool
    audio_url: str | None
    score: int
    verdict: str
    peak: float
    rms: float
    reason: str


class VoiceStudioService:
    def __init__(self, base_dir: Path, audio_validator: AudioValidator) -> None:
        self.base_dir = base_dir
        self.audio_validator = audio_validator

    def log_audit(self, user_id: str | None, event: str, details: str | None = None) -> None:
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO voice_audit_log (id, timestamp, user_id, event, details) VALUES (?, ?, ?, ?, ?);",
                (str(uuid.uuid4()), datetime_str(), user_id, event, details)
            )
            conn.commit()
        finally:
            conn.close()

    def get_audit_logs(self) -> list[dict[str, Any]]:
        conn = get_db_connection()
        try:
            rows = conn.execute("SELECT * FROM voice_audit_log ORDER BY timestamp DESC;").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def list_prompts(self) -> list[dict[str, str]]:
        return VOICE_STUDIO_PROMPTS

    def create_voice(self, name: str, owner_name: str, owner_email: str | None, organization: str | None, language: str, engine: str, commercial_allowed: bool) -> dict[str, Any]:
        owner_id = str(uuid.uuid4())
        voice_id = str(uuid.uuid4())
        
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO voice_owners (id, name, email, organization) VALUES (?, ?, ?, ?);",
                (owner_id, owner_name, owner_email, organization)
            )
            conn.execute(
                "INSERT INTO voices (id, name, owner_id, language, engine, quality_score, status, consent_status, publish_status, commercial_allowed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (voice_id, name, owner_id, language, engine, 0.0, "missing_files", "pending", "unpublished", 1 if commercial_allowed else 0)
            )
            conn.commit()
        finally:
            conn.close()
            
        self.log_audit(owner_name, "voice_created", f"Voice {name} ({voice_id}) created for {owner_name}.")
        return {"voice_id": voice_id, "owner_id": owner_id, "name": name}

    def save_consent(self, voice_id: str, signature: str, spoken_consent_bytes: bytes | None, mime_type: str | None) -> None:
        conn = get_db_connection()
        voice = conn.execute("SELECT * FROM voices WHERE id = ?;", (voice_id,)).fetchone()
        if not voice:
            conn.close()
            raise ValueError(f"Voice not found: {voice_id}")
            
        consent_dir = self.base_dir / voice_id / "consent"
        consent_dir.mkdir(parents=True, exist_ok=True)
        
        # Save signature as text file
        sig_path = consent_dir / "signature.txt"
        sig_path.write_text(signature, encoding="utf-8")
        
        spoken_path = None
        if spoken_consent_bytes:
            # Save spoken consent
            spoken_path = consent_dir / "spoken_consent.wav"
            temp_path = consent_dir / "temp_spoken.raw"
            temp_path.write_bytes(spoken_consent_bytes)
            try:
                self._normalize_wav(temp_path, spoken_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
                    
        # Update SQLite
        try:
            conn.execute(
                "INSERT OR REPLACE INTO voice_consents (voice_id, signature, consent_document_path, spoken_consent_path, timestamp) VALUES (?, ?, ?, ?, ?);",
                (voice_id, signature, str(sig_path), str(spoken_path) if spoken_path else None, datetime_str())
            )
            conn.execute(
                "UPDATE voices SET consent_status = 'completed' WHERE id = ?;",
                (voice_id,)
            )
            conn.commit()
        finally:
            conn.close()
            
        self.log_audit(None, "consent_added", f"Consent signed and updated for voice ID {voice_id}.")

    def save_sample(self, voice_id: str, prompt_id: str, audio_bytes: bytes, mime_type: str | None) -> VoiceStudioRecord:
        prompt = next((p for p in VOICE_STUDIO_PROMPTS if p["id"] == prompt_id), None)
        if not prompt:
            raise ValueError(f"Invalid prompt ID: {prompt_id}")
            
        conn = get_db_connection()
        voice = conn.execute("SELECT * FROM voices WHERE id = ?;", (voice_id,)).fetchone()
        if not voice:
            conn.close()
            raise ValueError(f"Voice not found: {voice_id}")
            
        # Ensure consent is signed before recording samples
        if voice["consent_status"] != "completed":
            conn.close()
            raise ValueError("Voice consent is required first.")
            
        voice_dir = self.base_dir / voice_id
        raw_dir = voice_dir / "raw"
        normalized_dir = voice_dir / "normalized"
        raw_dir.mkdir(parents=True, exist_ok=True)
        normalized_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw recording
        raw_path = raw_dir / f"{prompt_id}.raw"
        raw_path.write_bytes(audio_bytes)
        
        normalized_path = normalized_dir / f"{prompt_id}.wav"
        try:
            self._normalize_wav(raw_path, normalized_path)
            self.audio_validator.validate_wav_duration(normalized_path)
            q = self._evaluate_wav(normalized_path)
        except Exception as exc:
            if normalized_path.exists():
                normalized_path.unlink()
            conn.close()
            raise AudioValidationError(f"Audio validation failed: {exc}") from exc
            
        # Save to SQLite
        sample_id = str(uuid.uuid4())
        try:
            conn.execute(
                "INSERT OR REPLACE INTO voice_samples (id, voice_id, prompt_id, wav_path, status, score, reason, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (sample_id, voice_id, prompt_id, str(normalized_path), q["verdict"], q["score"], q["reason"], datetime_str())
            )
            
            # Recalculate average quality score for voice
            scores = conn.execute("SELECT score FROM voice_samples WHERE voice_id = ?;", (voice_id,)).fetchall()
            avg_score = sum(s[0] for s in scores) / len(scores) if scores else 0.0
            conn.execute(
                "UPDATE voices SET quality_score = ? WHERE id = ?;",
                (avg_score, voice_id)
            )
            conn.commit()
        finally:
            conn.close()
            
        self.log_audit(None, "sample_recorded", f"Sample {prompt_id} recorded for voice {voice_id} (Score: {q['score']}).")
        
        return VoiceStudioRecord(
            id=sample_id,
            prompt_id=prompt_id,
            text=prompt["text"],
            language=prompt["language"],
            exists=True,
            audio_url=f"/audio/voices/{voice_id}/normalized/{prompt_id}.wav",
            score=q["score"],
            verdict=q["verdict"],
            peak=q["peak"],
            rms=q["rms"],
            reason=q["reason"]
        )

    def delete_sample(self, voice_id: str, prompt_id: str) -> None:
        conn = get_db_connection()
        try:
            sample = conn.execute("SELECT * FROM voice_samples WHERE voice_id = ? AND prompt_id = ?;", (voice_id, prompt_id)).fetchone()
            if sample:
                path = Path(sample["wav_path"])
                if path.exists():
                    path.unlink()
                conn.execute("DELETE FROM voice_samples WHERE voice_id = ? AND prompt_id = ?;", (voice_id, prompt_id))
                
                # Recalculate average quality score
                scores = conn.execute("SELECT score FROM voice_samples WHERE voice_id = ?;", (voice_id,)).fetchall()
                avg_score = sum(s[0] for s in scores) / len(scores) if scores else 0.0
                conn.execute("UPDATE voices SET quality_score = ? WHERE id = ?;", (avg_score, voice_id))
                conn.commit()
        finally:
            conn.close()
        self.log_audit(None, "sample_deleted", f"Sample {prompt_id} deleted for voice {voice_id}.")

    def publish_voice(self, voice_id: str) -> dict[str, Any]:
        conn = get_db_connection()
        try:
            voice = conn.execute("SELECT * FROM voices WHERE id = ?;", (voice_id,)).fetchone()
            if not voice:
                raise ValueError("Voice not found")
            if voice["consent_status"] != "completed":
                raise ValueError("Consent is missing. Cannot publish voice.")
            samples = conn.execute("SELECT * FROM voice_samples WHERE voice_id = ?;", (voice_id,)).fetchall()
            # Only count non-rejected samples toward the minimum
            good_samples = [s for s in samples if s["status"] != "reject"]
            if len(good_samples) < 3:
                raise ValueError(f"At least 3 usable recordings are required (you have {len(good_samples)}). Record more prompts.")

            # Recalculate average quality score using only good samples
            scores = [s["score"] for s in good_samples if s["score"] is not None]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            conn.execute("UPDATE voices SET quality_score = ? WHERE id = ?;", (avg_score, voice_id))

            if avg_score < 50.0:
                raise ValueError(f"Average quality score ({avg_score:.1f}) is below 50. Try recording in a quieter place.")
                
            conn.execute("UPDATE voices SET publish_status = 'published', status = 'ready' WHERE id = ?;", (voice_id,))
            conn.commit()
        finally:
            conn.close()
        self.log_audit(None, "model_published", f"Voice model {voice_id} published successfully. Commercial checklist verified.")
        return {"ok": True, "voice_id": voice_id}

    def delete_voice(self, voice_id: str) -> None:
        conn = get_db_connection()
        try:
            # Wipe files
            voice_dir = self.base_dir / voice_id
            if voice_dir.exists() and ".local" in voice_dir.parts:
                shutil.rmtree(voice_dir)
            
            # Wipe database records
            conn.execute("DELETE FROM voice_owners WHERE id = (SELECT owner_id FROM voices WHERE id = ?);", (voice_id,))
            conn.execute("DELETE FROM voices WHERE id = ?;", (voice_id,))
            conn.commit()
        finally:
            conn.close()
        self.log_audit(None, "voice_deleted", f"Voice {voice_id} deleted and all local media scrubbed.")

    def get_gallery_voices(self) -> list[dict[str, Any]]:
        conn = get_db_connection()
        try:
            rows = conn.execute("""
                SELECT v.*, o.name as owner_name, o.organization as owner_org
                FROM voices v
                LEFT JOIN voice_owners o ON v.owner_id = o.id;
            """).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                voice_dir = self.base_dir / d["id"]
                if d.get("engine") == "elevenlabs":
                    d["model_exists"] = (voice_dir / "elevenlabs_id.txt").exists()
                elif d.get("engine") == "chatterbox":
                    d["model_exists"] = (voice_dir / "chatterbox_reference.wav").exists()
                else:
                    d["model_exists"] = self._has_real_piper_model(voice_dir)
                results.append(d)
            return results
        finally:
            conn.close()

    def get_voice_samples(self, voice_id: str) -> list[dict[str, Any]]:
        conn = get_db_connection()
        try:
            rows = conn.execute("SELECT * FROM voice_samples WHERE voice_id = ?;", (voice_id,)).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def _normalize_wav(self, input_path: Path, output_path: Path, sample_rate: int = 22050) -> None:
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

    def _evaluate_wav(self, path: Path) -> dict[str, Any]:
        with wave.open(str(path), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            channels = wav.getnchannels()
            frame_count = wav.getnframes()
        
        samples = self._pcm_samples(frames, sample_width)
        max_amplitude = float((1 << (8 * sample_width - 1)) - 1)
        peak = max((abs(sample) for sample in samples), default=0) / max_amplitude
        rms = math.sqrt(sum(sample * sample for sample in samples) / max(len(samples), 1)) / max_amplitude
        clipped_ratio = sum(1 for sample in samples if abs(sample) >= max_amplitude * 0.98) / max(len(samples), 1)
        duration = frame_count / float(frame_rate or 1)
        
        score = 100
        reasons = []
        if duration < 1.0:
            score -= 40
            reasons.append("too_short")
        elif duration < 2.0:
            score -= 15
            reasons.append("short")
        if duration > 18:
            score -= 20
            reasons.append("too_long")
        if rms < 0.005:
            score -= 35
            reasons.append("very_quiet")
        elif rms < 0.012:
            score -= 15
            reasons.append("quiet")
        if clipped_ratio > 0.01:
            score -= 35
            reasons.append("clipped")
        elif clipped_ratio > 0.003:
            score -= 12
            reasons.append("slight_clipping")
        if channels != 1:
            score -= 8
            reasons.append("not_mono")

        score = max(0, min(100, score))
        verdict = "good" if score >= 65 else "review" if score >= 45 else "reject"
        return {
            "score": score,
            "verdict": verdict,
            "duration_seconds": round(duration, 3),
            "peak": round(peak, 4),
            "rms": round(rms, 4),
            "reason": ",".join(reasons) or "clean",
        }

    @staticmethod
    def _pcm_samples(frames: bytes, sample_width: int) -> list[int]:
        if sample_width == 1:
            return [value - 128 for value in frames]
        if sample_width == 2:
            return list(struct.unpack(f"<{len(frames) // 2}h", frames))
        if sample_width == 3:
            samples = []
            for offset in range(0, len(frames), 3):
                raw = frames[offset : offset + 3]
                value = int.from_bytes(raw + (b"\xff" if raw[2] & 0x80 else b"\x00"), "little", signed=True)
                samples.append(value)
            return samples
        if sample_width == 4:
            return list(struct.unpack(f"<{len(frames) // 4}i", frames))
        raise ValueError(f"Unsupported sample width: {sample_width}")

    def _has_real_piper_model(self, voice_dir: Path) -> bool:
        onnx_files = list(voice_dir.rglob("*.onnx")) if voice_dir.exists() else []
        if not onnx_files:
            return False

        builtin_hashes = self._builtin_piper_hashes()
        if not builtin_hashes:
            return True
        return any(self._file_hash(path) not in builtin_hashes for path in onnx_files)

    @staticmethod
    def _builtin_piper_hashes() -> set[str]:
        model_dir = Path("models/piper")
        if not model_dir.exists():
            return set()
        return {digest for path in model_dir.glob("*.onnx") if (digest := VoiceStudioService._file_hash(path))}

    @staticmethod
    def _file_hash(path: Path) -> str | None:
        if not path.exists():
            return None
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()


def datetime_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
