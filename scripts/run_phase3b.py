#!/usr/bin/env python
"""Phase 3b launcher: parts (a) E4 = 31_train_d4, (b) E5 = 32_train_d5 --experiment e5, (c) E6 = 32_train_d5
--experiment e6, STRICTLY one after the other, with 33_report_phase3b after each part.

* A lock file (<out>/phase3b/.launcher.lock) stops a second launcher; every training script also holds
  <out>/phase3b/.training.lock, so only one training runs on the GPU at a time, even one started by hand.
* Parts are waited for with ``Popen.wait()``, which does NOT return while a child is paused with SIGSTOP (the bash loop
  used on 2026-09-25 did return, exit 147, and started the next part next to the paused one). Pausing the training
  PID printed below therefore pauses the whole pipeline; ``kill -CONT`` resumes it.
* Models that are already COMPLETE (weights + valid score file + test-ledger entry + runtime entry) are skipped by the
  training scripts, so a re-launch continues where the last one stopped. A model interrupted mid-training is retrained
  from scratch with the same seed (deterministic).
* If a part exits non-zero the launcher regenerates the report and stops (later parts are not started).
* SIGTERM / SIGINT to the launcher are forwarded to the running part (CONT, then TERM) before the launcher exits.
* Old logs of steps 31-33 are copied to <out>/phase3b/log_archive/<timestamp>/ before they are overwritten.
* Thermal guard (kl.phase3b.thermal): a training script stopped because the GPU stayed >= 90 C exits with
  THERMAL_EXIT; the launcher then prepends the stop state to docs/PHASE3B_RESUME.md and stops (waits for the user).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kl.phase3b.runner import LockBusyError, acquire_lock, gpu_lock  # noqa: E402
from kl.phase3b.thermal import THERMAL_EXIT  # noqa: E402

PARTS = {"e4": ["scripts/31_train_d4.py"], "e5": ["scripts/32_train_d5.py", "--experiment", "e5"], "e6": ["scripts/32_train_d5.py", "--experiment", "e6"]}
LOGS = ["31_train_d4.log", "32_train_d5_e5.log", "32_train_d5_e6.log", "33_report_phase3b.log"]
_child: subprocess.Popen | None = None


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def say(msg: str) -> None:
    print(f"[launcher {now()}] {msg}", flush=True)


def _forward(signum, _frame) -> None:
    if _child is not None and _child.poll() is None:
        say(f"signal {signum}: stopping part PID {_child.pid} (CONT, then TERM)")
        _child.send_signal(signal.SIGCONT)
        _child.terminate()
        _child.wait()
    say("launcher stopped")
    sys.exit(128 + signum)


def run(cmd: list[str], status_path: Path, part: str) -> int:
    global _child
    _child = subprocess.Popen([sys.executable, *cmd], cwd=ROOT)
    say(f"part {part}: started PID {_child.pid}: {' '.join(cmd)}  (pause with: kill -STOP {_child.pid}; resume: kill -CONT {_child.pid})")
    status_path.write_text(json.dumps({"launcher_pid": os.getpid(), "part": part, "child_pid": _child.pid, "cmd": cmd, "started": now()}, indent=2))
    rc = _child.wait()          # blocks through SIGSTOP pauses; returns only when the child has EXITED
    _child = None
    return rc


def write_thermal_resume(out: Path, part: str) -> None:
    """Prepend the thermal-stop state (what stopped, temperatures, COMPLETE runs so far) to docs/PHASE3B_RESUME.md."""
    stop = json.loads((out / "thermal_stop.json").read_text()) if (out / "thermal_stop.json").exists() else {}
    rt = json.loads((out / "runtime_log.json").read_text()) if (out / "runtime_log.json").exists() else {}
    done = sorted(rt.get("runs", {}))
    lines = [f"# THERMAL STOP {now()} (automatic; waiting for the user)", "",
             f"Part {part} was stopped by the thermal guard: GPU still >= 90 C after 30 min of pausing.", "",
             f"- interrupted run: `{stop.get('run')}` after epoch {stop.get('after_epoch')} (nothing saved; retrained from scratch, same seed)",
             f"- pause started {stop.get('pause_started')} at {stop.get('temp_at_pause')} C; stopped {stop.get('stopped_at')} at "
             f"{stop.get('temp_at_stop')} C (peak {stop.get('peak')} C)",
             f"- COMPLETE runs so far ({len(done)}): " + ", ".join(f"`{k}`" for k in done),
             "- later parts were NOT started. Resume: same command as below (COMPLETE runs are skipped).",
             "- all pauses: `runtime_log.json` -> `thermal_pauses`", "", "---", ""]
    p = ROOT / "docs" / "PHASE3B_RESUME.md"
    p.write_text("\n".join(lines) + (p.read_text() if p.exists() else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), default=list(PARTS))
    args = ap.parse_args()
    mode = "quick" if args.quick else "full"
    out = ROOT / "outputs" / mode / "phase3b"
    try:
        launcher_lock = acquire_lock(out / ".launcher.lock", "launcher")  # noqa: F841 - held until exit
        probe = gpu_lock(out)                                             # nothing may be training right now
        probe.close()
    except LockBusyError as e:
        say(str(e))
        sys.exit(75)
    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    logs = ROOT / "outputs" / mode / "logs"
    old = [logs / n for n in LOGS if (logs / n).exists()]
    if old:
        dest = out / "log_archive" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest.mkdir(parents=True, exist_ok=True)
        for f in old:
            shutil.copy2(f, dest / f.name)
        say(f"archived {len(old)} earlier logs to {dest.relative_to(ROOT)}")

    extra = ["--quick"] if args.quick else []
    status = out / "launcher_status.json"
    say(f"mode {mode}; parts {args.parts} (strictly sequential)")
    for part in args.parts:
        rc = run(PARTS[part] + extra, status, part)
        say(f"part {part}: exit code {rc}")
        rrc = run(["scripts/33_report_phase3b.py"] + extra, status, f"report after {part}")
        say(f"report after {part}: exit code {rrc}")
        if rc == THERMAL_EXIT:
            write_thermal_resume(out, part)
            say(f"part {part}: THERMAL STOP; state written to docs/PHASE3B_RESUME.md; later parts NOT started")
            sys.exit(rc)
        if rc != 0:
            say(f"part {part} failed (exit {rc}): later parts NOT started")
            sys.exit(rc)
    status.write_text(json.dumps({"finished": now(), "parts": args.parts}, indent=2))
    say("ALL DONE")


if __name__ == "__main__":
    main()
