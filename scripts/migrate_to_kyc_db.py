"""Non-destructive DB consolidation: voiceai -> kyc_db.

Creates kyc_db if missing, recreates admin_users (and app_settings if present in
voiceai) in kyc_db with the same schema, and copies all rows with INSERT IGNORE.
The source voiceai database is left untouched (copy, not move).

Run with the venv python:
    D:\\AI Voice Conversation\\.venv\\Scripts\\python.exe scripts\\migrate_to_kyc_db.py
"""
from __future__ import annotations

import sys

SOURCE_DB = "voiceai"
TARGET_DB = "kyc_db"

MYSQL = dict(host="localhost", port=3306, user="root", password="", connection_timeout=10)


def _connect(database: str | None = None):
    import mysql.connector

    kwargs = dict(MYSQL)
    if database:
        kwargs["database"] = database
    return mysql.connector.connect(**kwargs)


def _table_exists(cursor, database: str, table: str) -> bool:
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=%s AND table_name=%s",
        (database, table),
    )
    (count,) = cursor.fetchone()
    return count > 0


def _row_count(cursor, database: str, table: str) -> int:
    cursor.execute(f"SELECT COUNT(*) FROM `{database}`.`{table}`")
    (count,) = cursor.fetchone()
    return int(count)


def _copy_table(cursor, table: str) -> dict:
    """Recreate `table` in TARGET_DB from SOURCE_DB schema and copy rows (INSERT IGNORE)."""
    if not _table_exists(cursor, SOURCE_DB, table):
        return {"table": table, "skipped": True, "reason": f"{SOURCE_DB}.{table} does not exist"}

    # Create target table with identical schema (no data) if missing.
    cursor.execute(
        f"CREATE TABLE IF NOT EXISTS `{TARGET_DB}`.`{table}` LIKE `{SOURCE_DB}`.`{table}`"
    )
    # Copy rows, ignoring duplicates (unique keys preserved).
    cursor.execute(
        f"INSERT IGNORE INTO `{TARGET_DB}`.`{table}` SELECT * FROM `{SOURCE_DB}`.`{table}`"
    )
    source_rows = _row_count(cursor, SOURCE_DB, table)
    target_rows = _row_count(cursor, TARGET_DB, table)
    return {
        "table": table,
        "skipped": False,
        "source_rows": source_rows,
        "target_rows": target_rows,
    }


def main() -> int:
    results: list[dict] = []

    # 1) Ensure kyc_db exists. Connect without a default DB.
    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{TARGET_DB}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.commit()
        cursor.close()
    finally:
        conn.close()

    # 2) Copy admin_users and app_settings from voiceai -> kyc_db.
    conn = _connect(TARGET_DB)
    source_present = True
    try:
        cursor = conn.cursor()
        # Verify source database exists at all.
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.schemata WHERE schema_name=%s",
            (SOURCE_DB,),
        )
        (src_exists,) = cursor.fetchone()
        if not src_exists:
            source_present = False
            # No source -> ensure admin_users table at least exists in target.
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS `kyc_db`.`admin_users` (
                  id INT AUTO_INCREMENT PRIMARY KEY,
                  username VARCHAR(64) NOT NULL UNIQUE,
                  password_hash VARCHAR(128) NOT NULL,
                  salt VARCHAR(32) NOT NULL,
                  role VARCHAR(32) DEFAULT 'admin',
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  last_login_at TIMESTAMP NULL
                )"""
            )
            conn.commit()
            results.append({"table": "admin_users", "skipped": True,
                            "reason": f"source DB {SOURCE_DB} absent; created empty admin_users in {TARGET_DB}"})
        else:
            for table in ("admin_users", "app_settings"):
                res = _copy_table(cursor, table)
                conn.commit()
                results.append(res)

        # Final confirmation counts in target.
        target_counts = {}
        for table in ("admin_users", "app_settings"):
            if _table_exists(cursor, TARGET_DB, table):
                target_counts[table] = _row_count(cursor, TARGET_DB, table)
        cursor.close()
    finally:
        conn.close()

    print("=== DB consolidation voiceai -> kyc_db ===")
    print(f"source_db_present={source_present}")
    for r in results:
        print(r)
    print(f"target_counts={target_counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
