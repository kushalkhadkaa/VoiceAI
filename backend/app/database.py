from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_FILE = Path(".local/swarlocal.db")

def get_db_connection() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db() -> None:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. voice_owners
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_owners (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        organization TEXT
    );
    """)
    
    # 2. voices
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voices (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        owner_id TEXT REFERENCES voice_owners(id) ON DELETE CASCADE,
        language TEXT NOT NULL,
        engine TEXT NOT NULL,
        quality_score REAL DEFAULT 0.0,
        status TEXT DEFAULT 'missing_files',
        consent_status TEXT DEFAULT 'pending',
        publish_status TEXT DEFAULT 'unpublished',
        commercial_allowed INTEGER DEFAULT 0
    );
    """)
    
    # 3. voice_consents
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_consents (
        voice_id TEXT PRIMARY KEY REFERENCES voices(id) ON DELETE CASCADE,
        signature TEXT NOT NULL,
        consent_document_path TEXT,
        spoken_consent_path TEXT,
        timestamp TEXT NOT NULL
    );
    """)
    
    # 4. voice_samples
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_samples (
        id TEXT PRIMARY KEY,
        voice_id TEXT REFERENCES voices(id) ON DELETE CASCADE,
        prompt_id TEXT NOT NULL,
        wav_path TEXT NOT NULL,
        status TEXT NOT NULL,
        score INTEGER,
        reason TEXT,
        timestamp TEXT NOT NULL
    );
    """)
    
    # 5. voice_training_jobs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_training_jobs (
        id TEXT PRIMARY KEY,
        voice_id TEXT REFERENCES voices(id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        progress REAL DEFAULT 0.0,
        timestamp TEXT NOT NULL
    );
    """)
    
    # 6. voice_model_artifacts
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_model_artifacts (
        voice_id TEXT PRIMARY KEY REFERENCES voices(id) ON DELETE CASCADE,
        language TEXT NOT NULL,
        onnx_path TEXT NOT NULL,
        config_path TEXT NOT NULL
    );
    """)
    
    # 7. voice_permissions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_permissions (
        voice_id TEXT REFERENCES voices(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        permission TEXT NOT NULL,
        PRIMARY KEY (voice_id, role, permission)
    );
    """)
    
    # 8. voice_usage_events
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_usage_events (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        event_type TEXT NOT NULL,
        details TEXT
    );
    """)
    
    # 9. voice_audit_log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_audit_log (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        user_id TEXT,
        event TEXT NOT NULL,
        details TEXT
    );
    """)
    
    conn.commit()
    conn.close()
