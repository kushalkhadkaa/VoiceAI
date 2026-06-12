import sqlite3
import sys

sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect(".local/swarlocal.db")
conn.row_factory = sqlite3.Row

print("=== CHECKING API KEYS CONFIG ===")
for key in ["openai_api_key", "gemini_api_key", "open_webui_api_key"]:
    row = conn.execute("SELECT setting_value FROM app_settings WHERE setting_key = ?", (key,)).fetchone()
    val = row["setting_value"] if row else None
    is_set = bool(val and val.strip())
    masked = f"{val[:6]}...{val[-4:]}" if is_set and len(val) > 10 else ("YES" if is_set else "NO")
    print(f"Key: {key} | Is Configured: {is_set} | Value Mask: {masked}")

conn.close()
