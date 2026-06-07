#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "frontend" / "src" / "App.tsx"
CSS = ROOT / "frontend" / "src" / "styles.css"


CHECKS = {
    "main_navigation": ["Voice Studio", "Knowledge", "Setup", "Eval", "Admin", "Logs", "Settings"],
    "voice_orb_states": ["data-orb-state", "Listening for your voice", "Searching knowledge", "Speaking now"],
    "voice_dropdown_groups": ["<optgroup label=\"Recommended\">", "My cloned voices", "Built-in voices", "Experimental voices"],
    "voice_presets": ["Fast response", "Best quality", "Noisy room", "My cloned voice only", "Local only"],
    "voice_studio_wizard": ["Create Voice", "Record and Clean", "Create my voice", "Advanced Voice Controls"],
    "knowledge_page": ["function KnowledgeView", "Open WebUI API key needed", "Test knowledge"],
    "admin_page": ["function AdminView", "System Monitor", "Commercial Readiness", "Debug Tools"],
    "logs_page": ["function LogsView", "Log details", "Export logs", "Clear app logs"],
    "disabled_explanations": ["No generated answer yet", "Nothing is speaking right now", "Add Open WebUI API key first"],
    "tooltips": ["VAD sensitivity controls", "Denoise strength controls", "Engine selection chooses"],
}

CSS_CHECKS = {
    "premium_dark": ["color-scheme: dark", "--surface: #101414"],
    "responsive": ["@media (max-width: 760px)", "grid-template-columns: 1fr"],
    "reduced_motion": ["prefers-reduced-motion: reduce"],
    "orb_animation": ["@keyframes waveform", "@keyframes voice-ring"],
}


def main() -> int:
    app = APP.read_text(encoding="utf-8")
    css = CSS.read_text(encoding="utf-8")
    failures: list[str] = []

    for group, needles in CHECKS.items():
        for needle in needles:
            if needle not in app:
                failures.append(f"{group}: missing {needle!r}")

    for group, needles in CSS_CHECKS.items():
        for needle in needles:
            if needle not in css:
                failures.append(f"{group}: missing {needle!r}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 2

    print("[PASS] UI static checks covered navigation, voice UX, Voice Studio, Knowledge, Admin, tooltips, responsive layout, and reduced motion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
