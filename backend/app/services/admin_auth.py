"""Admin authentication: MySQL-backed users with PBKDF2 hashing and in-memory sessions."""
from __future__ import annotations

import hashlib
import logging
import secrets
import time

logger = logging.getLogger(__name__)

import os

_PBKDF2_ITERATIONS = 100_000
_SESSION_TTL_SECONDS = 12 * 3600
_DEFAULT_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
_DEFAULT_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# Single consolidated MySQL database for the banking app (admin_users live here).
MYSQL_DB = "kyc_db"

# token -> {"username": str, "role": str, "expires": float}
_sessions: dict[str, dict] = {}
_table_ready = False


def _connect():
    import mysql.connector

    return mysql.connector.connect(
        host="localhost",
        port=3306,
        user="root",
        password="",
        database=MYSQL_DB,
        connection_timeout=5,
    )


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS
    ).hex()


def _ensure_table(conn) -> None:
    global _table_ready
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS admin_users (
          id INT AUTO_INCREMENT PRIMARY KEY,
          username VARCHAR(64) NOT NULL UNIQUE,
          password_hash VARCHAR(128) NOT NULL,
          salt VARCHAR(32) NOT NULL,
          role VARCHAR(32) DEFAULT 'admin',
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          last_login_at TIMESTAMP NULL
        )"""
    )
    cursor.execute("SELECT COUNT(*) FROM admin_users")
    (count,) = cursor.fetchone()
    if count == 0:
        salt = secrets.token_hex(16)
        cursor.execute(
            "INSERT INTO admin_users (username, password_hash, salt, role) VALUES (%s, %s, %s, %s)",
            (_DEFAULT_USERNAME, _hash_password(_DEFAULT_PASSWORD, salt), salt, "admin"),
        )
        conn.commit()
    cursor.close()
    _table_ready = True


def _issue_token(username: str, role: str) -> dict:
    token = secrets.token_urlsafe(32)
    _sessions[token] = {
        "username": username,
        "role": role,
        "expires": time.time() + _SESSION_TTL_SECONDS,
    }
    return {"token": token, "username": username, "role": role}


def login(username: str, password: str) -> dict | None:
    username = (username or "").strip()
    password = password or ""
    try:
        conn = _connect()
        try:
            _ensure_table(conn)
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT username, password_hash, salt, role FROM admin_users WHERE username = %s",
                (username,),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.close()
                return None
            computed = _hash_password(password, row["salt"])
            if not secrets.compare_digest(computed, row["password_hash"]):
                cursor.close()
                return None
            cursor.execute(
                "UPDATE admin_users SET last_login_at = CURRENT_TIMESTAMP WHERE username = %s",
                (username,),
            )
            conn.commit()
            cursor.close()
            return _issue_token(row["username"], row["role"] or "admin")
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Admin auth MySQL unavailable, using built-in fallback: %s", exc)
        if username == _DEFAULT_USERNAME and secrets.compare_digest(password, _DEFAULT_PASSWORD):
            return _issue_token(_DEFAULT_USERNAME, "admin")
        return None


def verify(token: str) -> dict | None:
    session = _sessions.get(token or "")
    if session is None:
        return None
    if session["expires"] < time.time():
        _sessions.pop(token, None)
        return None
    return {"username": session["username"], "role": session["role"]}


def logout(token: str) -> None:
    _sessions.pop(token or "", None)
