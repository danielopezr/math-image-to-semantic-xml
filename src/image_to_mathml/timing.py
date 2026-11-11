"""Phase timing helpers for pipeline reporting."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhaseTiming:
    phase: str
    tool: str
    duration_ms: float
    attempt: int | None = None
    note: str = ""
    group: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "tool": self.tool,
            "duration_ms": self.duration_ms,
            "attempt": self.attempt,
            "note": self.note,
            "group": self.group,
        }


@dataclass
class TimingRecorder:
    entries: list[PhaseTiming] = field(default_factory=list)
    _group: str = field(default="", init=False, repr=False)

    @contextmanager
    def group(self, name: str):
        previous = self._group
        self._group = name
        try:
            yield
        finally:
            self._group = previous

    @contextmanager
    def measure(
        self,
        phase: str,
        tool: str,
        *,
        attempt: int | None = None,
        note: str = "",
        group: str | None = None,
    ):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.entries.append(
                PhaseTiming(
                    phase=phase,
                    tool=tool,
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                    attempt=attempt,
                    note=note,
                    group=self._group if group is None else group,
                )
            )

    def as_list(self) -> list[PhaseTiming]:
        return list(self.entries)
