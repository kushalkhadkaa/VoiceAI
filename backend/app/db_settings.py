"""
SQLite-backed persistent settings store.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Single consolidated MySQL database for the banking app. App settings are stored
# in SQLite (the local app store); if a MySQL mirror is ever re-enabled it must
# target kyc_db, never the legacy voiceai database.
MYSQL_DB = "kyc_db"

def _ensure_settings_table() -> None:
    """Ensure the app_settings table exists in the SQLite database."""
    try:
        from app.database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT,
                value_type TEXT DEFAULT 'string',
                category TEXT DEFAULT 'general',
                is_secret INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as exc:
        logger.warning(f"Failed to ensure SQLite app_settings table: {exc}")

def load_settings() -> dict[str, Any]:
    """Load all settings from SQLite database."""
    _ensure_settings_table()
    try:
        from app.database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT setting_key, setting_value, value_type FROM app_settings;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        result = {}
        for row in rows:
            key = row["setting_key"]
            value = row["setting_value"]
            vtype = row["value_type"] if "value_type" in row.keys() else "string"
            
            if value is None:
                result[key] = None
            elif vtype == "bool":
                result[key] = value.lower() in ("1", "true", "yes")
            elif vtype == "int":
                result[key] = int(value)
            elif vtype == "float":
                result[key] = float(value)
            elif vtype == "json":
                try:
                    result[key] = json.loads(value)
                except Exception:
                    result[key] = value
            else:
                result[key] = value
        return result
    except Exception as exc:
        logger.warning(f"Failed to load settings from SQLite: {exc}")
        return {}

def save_settings(payload: dict[str, Any]) -> bool:
    """Upsert settings into SQLite."""
    _ensure_settings_table()
    try:
        from app.database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        for key, value in payload.items():
            # Skip updating if key is masked or contains placeholder characters to prevent overwriting
            if key in ("openai_api_key", "gemini_api_key", "open_webui_api_key", "elevenlabs_api_key"):
                if value is not None and ("..." in str(value) or "…" in str(value) or "****" in str(value)):
                    continue
            
            if isinstance(value, bool):
                vtype, stored = "bool", "1" if value else "0"
            elif isinstance(value, int):
                vtype, stored = "int", str(value)
            elif isinstance(value, float):
                vtype, stored = "float", str(value)
            elif isinstance(value, (list, dict)):
                vtype, stored = "json", json.dumps(value)
            else:
                vtype, stored = "string", str(value) if value is not None else None

            is_secret = 1 if "api_key" in key else 0
            category = _category_for(key)
            cursor.execute(
                """INSERT INTO app_settings (setting_key, setting_value, value_type, category, is_secret)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(setting_key) DO UPDATE SET
                     setting_value = excluded.setting_value,
                     value_type = excluded.value_type,
                     category = excluded.category,
                     is_secret = excluded.is_secret,
                     updated_at = CURRENT_TIMESTAMP;""",
                (key, stored, vtype, category, is_secret),
            )
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as exc:
        logger.warning(f"Failed to save settings to SQLite: {exc}")
        return False

def log_audit(event_type: str, entity_type: str, entity_id: str, description: str) -> None:
    """Write an audit event to SQLite (best-effort, never raises)."""
    try:
        from app.database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS voice_audit_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                event TEXT NOT NULL,
                details TEXT
            );"""
        )
        import time
        import uuid
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn.execute(
            "INSERT INTO voice_audit_log (id, timestamp, event, details) VALUES (?, ?, ?, ?);",
            (str(uuid.uuid4()), timestamp, f"{event_type}:{entity_type}:{entity_id}", description)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception:
        pass

def _category_for(key: str) -> str:
    if "openai" in key:
        return "openai"
    if "gemini" in key:
        return "gemini"
    if "ollama" in key or "local_model" in key:
        return "ollama"
    if "open_webui" in key:
        return "open_webui"
    if "piper" in key:
        return "piper"
    if "whisper" in key or "stt" in key:
        return "stt"
    if "voice" in key or "tts" in key:
        return "voice"
    if "rag" in key or "internet" in key:
        return "rag"
    return "general"

def is_mysql_available() -> bool:
    return False
