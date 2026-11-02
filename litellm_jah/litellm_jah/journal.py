"""The JAH log files: ``steps.jsonl`` and ``decisions.jsonl`` in the exact schema the standalone proxy's
``jah report`` and ``jah fit`` read (``jevagentharness.report`` / ``jevagentharness.fit``).

Both files are created (empty) when the journal is opened, so ``jah report <log_dir>`` works on a plugin log
directory from the first minute, before any row is written.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any


def head(session: str, model_name: str, step: int) -> dict[str, Any]:
    """The first fields of every JAH log row."""
    return {"ts": round(time.time(), 3), "session": session, "model_name": model_name, "step": step}


class Journal:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        for name in ("steps.jsonl", "decisions.jsonl"):
            (self.log_dir / name).touch(exist_ok=True)
        self._lock = asyncio.Lock()

    async def _append(self, name: str, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"

        def write() -> None:
            with open(self.log_dir / name, "a") as f:
                f.write(line)

        # the lock serializes the writes; to_thread keeps the blocking I/O off the event loop
        async with self._lock:
            await asyncio.to_thread(write)

    async def decision(self, row: dict[str, Any]) -> None:
        await self._append("decisions.jsonl", row)

    async def step(self, row: dict[str, Any]) -> None:
        await self._append("steps.jsonl", row)
