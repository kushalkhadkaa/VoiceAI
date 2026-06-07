PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
VENV_PYTHON ?= /opt/homebrew/bin/python3.11
VENV_BIN := $(CURDIR)/.venv/bin
BACKEND_DIR := backend
FRONTEND_DIR := frontend

.PHONY: setup setup-voice-clone dev backend frontend test lint doctor e2e model-test voice-test rag-test ui-test download-models download-piper-voices eval clean

setup:
	$(VENV_PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r $(BACKEND_DIR)/requirements.txt
	cd $(FRONTEND_DIR) && npm install

setup-voice-clone:
	. .venv/bin/activate && pip install -r $(BACKEND_DIR)/requirements-voice-clone.txt

dev:
	$(MAKE) -j2 backend frontend

backend:
	PATH="$(VENV_BIN):$$PATH" $(PYTHON) -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --app-dir $(BACKEND_DIR)

frontend:
	cd $(FRONTEND_DIR) && npm run dev -- --host 127.0.0.1 --port 5173

test:
	PATH="$(VENV_BIN):$$PATH" PYTHONPATH=$(BACKEND_DIR) $(PYTHON) -m unittest discover -s $(BACKEND_DIR)/tests -p "test_*.py"

lint:
	PATH="$(VENV_BIN):$$PATH" PYTHONPATH=$(BACKEND_DIR) $(PYTHON) -m compileall $(BACKEND_DIR)/app $(BACKEND_DIR)/tests scripts
	cd $(FRONTEND_DIR) && npm run typecheck

doctor:
	PATH="$(VENV_BIN):$$PATH" PYTHONPATH=$(BACKEND_DIR) $(PYTHON) scripts/check_environment.py

e2e:
	PATH="$(VENV_BIN):$$PATH" PYTHONPATH=$(BACKEND_DIR) $(PYTHON) scripts/e2e_voice_smoke.py

model-test:
	PATH="$(VENV_BIN):$$PATH" PYTHONPATH=$(BACKEND_DIR) $(PYTHON) scripts/check_environment.py --json

voice-test:
	PATH="$(VENV_BIN):$$PATH" PYTHONPATH=$(BACKEND_DIR) $(PYTHON) scripts/e2e_voice_smoke.py

rag-test:
	PATH="$(VENV_BIN):$$PATH" PYTHONPATH=$(BACKEND_DIR) $(PYTHON) scripts/rag_smoke.py

ui-test:
	$(PYTHON) scripts/ui_static_checks.py

download-models:
	mkdir -p models/piper
	ollama pull qwen2.5:7b
	ollama pull gemma3:4b
	@echo "Run: make download-piper-voices"

download-piper-voices:
	PATH="$(VENV_BIN):$$PATH" PYTHONPATH=$(BACKEND_DIR) $(PYTHON) scripts/download_piper_voices.py

eval:
	PATH="$(VENV_BIN):$$PATH" PYTHONPATH=$(BACKEND_DIR) $(PYTHON) scripts/evaluate_recordings.py raw_recordings --output .local/evaluation.csv

clean:
	rm -rf frontend/dist .pytest_cache coverage
