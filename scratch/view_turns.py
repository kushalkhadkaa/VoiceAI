import sqlite3
import sys

sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect(".local/swarlocal.db")
conn.row_factory = sqlite3.Row

print("--- READY CHATTERBOX VOICES ---")
voices = conn.execute("SELECT * FROM voices WHERE engine='chatterbox'").fetchall()
for v in voices:
    print(dict(v))

print("\n--- CHATTERBOX VOICE ARTIFACTS ---")
artifacts = conn.execute("SELECT * FROM voice_model_artifacts WHERE voice_id IN (SELECT id FROM voices WHERE engine='chatterbox')").fetchall()
for a in artifacts:
    print(dict(a))

conn.close()
