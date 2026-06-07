#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import Settings
from app.services.environment import format_human_report, run_environment_checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SwarLocal macOS runtime readiness.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of human output.")
    args = parser.parse_args()
    report = run_environment_checks(Settings.from_env())
    print(report.to_json() if args.json else format_human_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
