# Next Actions

1. Create an Open WebUI API key, save it in SwarLocal settings, then rerun `make rag-test`.
2. Add a full Piper fine-tuning runner around `PIPER_TRAIN_COMMAND`, with richer progress and artifact validation for `.onnx` exports.
3. Add preview generation and voice similarity scoring for Chatterbox/Piper generated samples, not only uploaded recordings.
4. Persist chat turns and validation results in SQLite instead of only browser local storage.
5. Add richer turn-detection controls for Push to Talk, Auto Detect, Hybrid, room presets, silence timeout, noise gate, and calibration.
6. Add backup/export/import for voice packages.
7. Replace static UI checks with browser-driven Playwright tests if/when a frontend test runner is added.
