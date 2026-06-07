# Commercial Readiness

Current status: progressing, not complete.

Ready foundations:
- Local-first model path with Ollama, Faster Whisper, and Piper.
- Consent-gated Voice Studio recordings.
- SQLite tables for owners, voices, consent, samples, training jobs, artifacts, permissions, usage events, and audit logs.
- Generated audio sidecar metadata for voice/model provenance when turn audio retention is enabled.
- Safe voice deletion path for local voice data.
- Strict voice routing fields and no-silent-fallback option.

Not yet complete:
- Automated production fine-tuning pipeline.
- Generated voice similarity score and retraining recommendations.
- Chat/performance/evaluation persistence in SQLite.
- Voice package export/import and backup/restore workflow.
- Role/permission UI beyond database-ready structure.

