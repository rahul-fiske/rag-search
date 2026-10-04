"""Cost measurement for conversion steps (stdlib only): wall time, CPU time, peak memory.

``Meter`` wraps a step in one process.  CPU time is the process's own plus, for a step that runs
a child process (the docling subprocess), the children's.  The peak is the process's high-water
mark of resident memory, so it is a property of the process, not of one document.

Energy is not measured: on macOS that needs root (``powermetrics``).
"""

from __future__ import annotations

import sys
import time
from typing import Any

try:                                   # not on Windows
    import resource
except ImportError:                    # pragma: no cover
    resource = None                    # type: ignore[assignment]


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (0 when the platform cannot say)."""
    if resource is None:
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024      # bytes vs KB


def children_cpu_s() -> float:
    if resource is None:
        return 0.0
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


class Meter:
    """``with Meter() as m: ...`` then ``m.wall_s``, ``m.cpu_s``, ``m.result()``."""

    def __init__(self) -> None:
        self.wall_s = 0.0
        self.cpu_s = 0.0

    def start(self) -> "Meter":
        self._w = time.perf_counter()
        self._c = time.process_time() + children_cpu_s()
        return self

    __enter__ = start

    def cpu_now(self) -> float:
        """CPU seconds since ``start`` (the meter keeps running)."""
        return max(0.0, time.process_time() + children_cpu_s() - self._c)

    def __exit__(self, *exc: Any) -> None:
        self.wall_s = time.perf_counter() - self._w
        self.cpu_s = max(0.0, time.process_time() + children_cpu_s() - self._c)

    def result(self) -> dict[str, float]:
        return {"wall_s": round(self.wall_s, 2), "cpu_s": round(self.cpu_s, 2),
                "peak_mb": round(peak_rss_mb())}
