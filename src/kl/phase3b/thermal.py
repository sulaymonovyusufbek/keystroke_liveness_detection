"""GPU thermal guard for Phase 3b training (operational safety only: it never touches models, data, RNGs or scores).

Between epochs :meth:`ThermalGuard.check` reads the GPU temperature. At >= ``PAUSE_AT`` it sleeps until the GPU is at
<= ``RESUME_AT`` and then lets training continue. If the temperature is still >= ``ABORT_AT`` after ``ABORT_AFTER_S`` of
pausing, it writes ``thermal_stop.json`` and exits the process with ``THERMAL_EXIT``. The model is then incomplete, so
it is retrained from scratch with the same seed on the next launch. Every pause is logged (timestamp and temperatures)
in the step log and in ``runtime_log.json["thermal_pauses"]``.

The thresholds live here, NOT in config_phase3b.yaml, because that file is covered by the pre-registration hash.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from .runner import update_runtime

PAUSE_AT = 85        # degC: pause at or above
RESUME_AT = 70       # degC: resume at or below
ABORT_AT = 90        # degC: still at or above this after ABORT_AFTER_S of pausing -> stop cleanly
ABORT_AFTER_S = 30 * 60
POLL_S = 30
THERMAL_EXIT = 76    # exit code of a training script stopped by the guard (the launcher recognises it)


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def read_gpu_temp() -> int | None:
    """Temperature of GPU 0 in degC from ``nvidia-smi``, or ``None`` if it cannot be read."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - no GPU / no driver / unparsable output
        return None


class ThermalGuard:
    """Callable between epochs; returns the seconds spent paused (0.0 if no pause)."""

    def __init__(self, out: Path, log: logging.Logger, read_temp: Callable[[], int | None] = read_gpu_temp,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        self.out, self.log, self.read_temp, self.sleep, self.clock = out, log, read_temp, sleep, clock
        self._warned = False

    def check(self, run: str, epoch: int) -> float:
        t = self.read_temp()
        if t is None:
            if not self._warned:
                self.log.warning("thermal guard: GPU temperature unreadable (nvidia-smi); guard inactive until it can be read")
                self._warned = True
            return 0.0
        if t < PAUSE_AT:
            return 0.0
        start, started_at, peak = self.clock(), _now(), t
        self.log.warning("THERMAL PAUSE %s: %s after epoch %d: GPU %d C >= %d C; sleeping until <= %d C", started_at, run, epoch, t, PAUSE_AT, RESUME_AT)
        print(f"[thermal {started_at}] pause: {run} epoch {epoch}, GPU {t} C", flush=True)
        while True:
            self.sleep(POLL_S)
            t2 = self.read_temp()
            waited = self.clock() - start
            if t2 is None:
                continue                                  # keep waiting; a transient read failure is not "cool"
            peak = max(peak, t2)
            if t2 <= RESUME_AT:
                break
            if waited >= ABORT_AFTER_S and t2 >= ABORT_AT:
                self._record(run, epoch, started_at, t, t2, peak, waited, "stopped")
                (self.out / "thermal_stop.json").write_text(json.dumps(
                    {"stopped_at": _now(), "run": run, "after_epoch": epoch, "pause_started": started_at, "temp_at_pause": t,
                     "temp_at_stop": t2, "peak": peak, "paused_seconds": round(waited, 1)}, indent=2))
                self.log.error("THERMAL STOP %s: %s: GPU still %d C >= %d C after %.0f min of pausing; stopping cleanly "
                               "(run incomplete, retrained from scratch with the same seed on relaunch)", _now(), run, t2, ABORT_AT, waited / 60)
                sys.exit(THERMAL_EXIT)
        waited = self.clock() - start
        self._record(run, epoch, started_at, t, t2, peak, waited, "resumed")
        self.log.warning("THERMAL RESUME %s: %s: GPU %d C <= %d C after %.0f s pause (peak %d C)", _now(), run, t2, RESUME_AT, waited, peak)
        print(f"[thermal {_now()}] resume: {run}, GPU {t2} C after {waited:.0f} s", flush=True)
        return waited

    def _record(self, run: str, epoch: int, started_at: str, t0: int, t1: int, peak: int, waited: float, outcome: str) -> None:
        update_runtime(self.out, "thermal_pauses", started_at, {"run": run, "after_epoch": epoch, "temp_at_pause": t0, "temp_at_end": t1,
                                                                "peak": peak, "paused_seconds": round(waited, 1), "ended": _now(), "outcome": outcome})
