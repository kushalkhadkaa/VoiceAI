"""Process-wide guards for heavy, resource-hungry operations.

Some operations (local cloned-voice synthesis via Chatterbox, voice building)
peg every CPU core and a lot of RAM for minutes at a time. Running two at once
on a modest machine starves the API and can OOM-kill the backend — which the
user sees as "Knowledge service unavailable".

Each HeavyJob is a non-blocking, process-wide single-flight gate: if one is
already running, the next caller is rejected *immediately* with a clear,
friendly message telling them what is running and roughly how long to wait —
rather than silently queuing (which would look like a hang) or piling up.
"""
from __future__ import annotations

import threading
import time


def _human(seconds: int) -> str:
    seconds = max(1, int(seconds))
    if seconds < 90:
        return f"about {seconds} second{'s' if seconds != 1 else ''}"
    minutes = round(seconds / 60)
    return f"about {minutes} minute{'s' if minutes != 1 else ''}"


class HeavyJobBusy(Exception):
    """Raised when a heavy job is asked to start while another is in progress."""

    def __init__(self, label: str, started_at: float | None, typical_seconds: int) -> None:
        elapsed = int(time.time() - started_at) if started_at else 0
        remaining = max(2, typical_seconds - elapsed)
        self.label = label
        self.elapsed_seconds = elapsed
        self.retry_after = remaining
        self.message = (
            f"{label} is already in progress on this computer (started {elapsed}s ago). "
            f"To keep the app fast and avoid a crash, only one runs at a time — "
            f"please wait {_human(remaining)} and try again."
        )
        super().__init__(self.message)


class HeavyJob:
    def __init__(self, label: str, typical_seconds: int) -> None:
        self.label = label
        self.typical_seconds = typical_seconds
        self._lock = threading.Lock()
        self._started_at: float | None = None

    @property
    def running(self) -> bool:
        return self._started_at is not None

    def status(self) -> dict | None:
        if self._started_at is None:
            return None
        elapsed = int(time.time() - self._started_at)
        return {
            "label": self.label,
            "elapsed_seconds": elapsed,
            "typical_seconds": self.typical_seconds,
            "remaining_seconds": max(0, self.typical_seconds - elapsed),
        }

    def __enter__(self) -> "HeavyJob":
        if not self._lock.acquire(blocking=False):
            raise HeavyJobBusy(self.label, self._started_at, self.typical_seconds)
        self._started_at = time.time()
        return self

    def __exit__(self, *exc: object) -> None:
        self._started_at = None
        try:
            self._lock.release()
        except RuntimeError:
            pass


# Named, process-wide heavy jobs. Typical durations are rough CPU estimates used
# only to tell the user how long to wait.
CLONED_VOICE = HeavyJob("A cloned-voice clip", typical_seconds=45)
VOICE_BUILD = HeavyJob("A custom voice build", typical_seconds=120)

_ALL = [CLONED_VOICE, VOICE_BUILD]


def running_jobs() -> list[dict]:
    """All heavy jobs currently in progress (for a status endpoint)."""
    return [s for job in _ALL if (s := job.status())]
