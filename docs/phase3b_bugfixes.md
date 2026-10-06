# Phase 3b bug-fix and incident log

Rules (docs/PHASE3B_PREREGISTRATION.md): after results are seen, only genuine bugs are fixed, never tuning. Each
entry gives the date, what changed, why it is a bug and not tuning, and whether any result had been seen.
RESULTS_PHASE3B.md reproduces this file.

## Incidents (not Phase 3b bugs)

- **2026-09-25, before any Phase 3b model was trained: frozen `src/kl/features.py` was found modified.** Its first
  line started with two stray characters (`cl"""Session -> …`), which made the file unparseable, and it failed the
  Phase 1 hash. The file's mtime was 2026-09-25 13:36:26, before this Phase 3b session started; no tool in this
  session wrote to it. Removing exactly those two bytes gives back the frozen SHA-256 `5be4f320…a63ec` from
  `docs/phase1_frozen_hashes.json`, so the file was restored to its frozen bytes. The modified copy was kept
  (`features.py.with_stray_cl` in the session scratchpad). All Phase 1/2/3a guards passed afterwards. No result
  depended on the modified file: it could not even be imported.

- **2026-09-25 18:17–18:23, full run: three Phase 3b processes were alive at once, and the run was stopped.** The
  user paused the D4 training (`kill -STOP`, PID 79790) to free the GPU. The launcher was a plain bash loop, and a
  non-interactive bash treats a SIGSTOP-paused foreground child as *finished*: it logged `exit=147` (= 128 + SIGSTOP),
  ran the report step and started part (b) E5 (PID 97986). When that was paused too (exit 147 at 18:20:02) it started
  part (c) E6 (PID 103032). Only one process was ever *computing* at a time, because each earlier one was paused, but
  all three held GPU memory. At the user's request everything was stopped at 18:23 (loop frozen first, then CONT+TERM
  for each process). Damage check: no score file, model file, test-ledger entry or training curve was written after
  18:06:09. The only files written during the overlap were derived report files (regenerated on every report) and
  the `device` entries of `runtime_log.json`. All 8 complete D4 models were checked (score file loads, expected
  arrays, finite, sizes, ledger entry; reloaded weights reproduce the stored val scores, CPU vs GPU max |diff|
  ≤ 2.1e-4), so none is invalid. The three interrupted models (D4-nokey seed 44, D5←A1 seed 42, D5←all-but-A1 seed 42)
  had saved nothing and not touched the test split. They are retrained from scratch with the same seed. See
  `docs/PHASE3B_RESUME.md`.

- **2026-09-26 10:06, frozen `src/kl/features.py` corrupted outside any Phase 3b process.** Before the resume, the
  file started with the 6 stray bytes `claude` (mtime 10:06:03, a minute before the resume session's first command;
  most likely typed into the open editor, which autosaves). Every freeze guard would have raised. With the user's
  approval, exactly those 6 bytes were removed. The file's SHA-256 is again `5be4f320…a63ec`, identical to
  `docs/phase1_frozen_hashes.json`. `assert_frozen_p123` passes for full and quick, and all tests pass. No Phase 3b
  step ran while the file was corrupted.

## Bug fixes

- **2026-09-25, launcher (after the incident above; D4 results for 8 models had been logged, no E4 table existed yet).**
  *Bug:* the bash-loop launcher started the next part when the current one was paused with SIGSTOP, so parts could
  overlap on the GPU. *Fix:* new `scripts/run_phase3b.py`. It runs parts strictly one after the other and waits with
  `Popen.wait()`, which does not return while a child is paused. A launcher lock (`.launcher.lock`) stops a second
  launcher. A training lock (`.training.lock`, taken by `31_train_d4.py` and `32_train_d5.py` for their whole
  lifetime and kept while paused) stops a second training process, however it is started. If a part fails, later
  parts are not started. SIGTERM is forwarded to the running part. Old logs are archived before being overwritten.
  The training scripts now skip only runs that are **complete** (weights + loadable score file with every expected
  finite array + test-ledger entry + runtime entry, `runner.run_state`), rather than any run with a score file. A run
  that used the test split but has no complete result refuses to retrain without `--allow-rerun`. *Why this is not
  tuning:* nothing about models, data, seeds, scores, thresholds or evaluation changed. Only the order of execution
  and the bookkeeping changed, and every model is still trained exactly as pre-registered. Tests: lock exclusivity,
  lock held while paused, `Popen.wait` across SIGSTOP, completion states.

- **2026-09-26, thermal guard (operational safety, requested by the user; not a protocol change).** New
  `src/kl/phase3b/thermal.py`. After every epoch, `nets.fit` calls an optional `between_epochs` hook. The training
  scripts pass `ThermalGuard.check`, which reads `nvidia-smi --query-gpu=temperature.gpu`. At ≥ 85 °C it sleeps until
  ≤ 70 °C and logs every pause (timestamp, temperatures, peak, duration) in the step log and in
  `runtime_log.json["thermal_pauses"]`. If the GPU is still ≥ 90 °C after 30 min of pausing, it writes
  `thermal_stop.json` and exits with code 76. The launcher then prepends the stop state to `PHASE3B_RESUME.md` and
  stops; the interrupted run is retrained from scratch with the same seed on relaunch. The thresholds are constants
  in code, not in `config_phase3b.yaml`, so the pre-registration hash is unchanged. The hook only sleeps and never
  touches the model, optimiser, data or RNGs. A test checks that the weights are bit-identical with and without it.
  `train_seconds` / `seconds_per_epoch` now exclude pause time, and the pause time is logged separately
  (`thermal_paused_seconds`).

## Incident 2026-10-06 — stray text in `src/kl/sessions.py` (found before Phase 4)

- **Found:** 2026-10-06 13:54 CEST, by the start-of-session checks (`pytest` failed at collection with
  `NameError: name 'cl' is not defined`; `sha256sum -c docs/MANIFEST_SHA256.txt` flagged `src/kl/sessions.py`).
  The Phase 1–3c freeze guards passed, because `sessions.py` is not in those freeze records.
- **File / line / stray text:** `src/kl/sessions.py`, line 40 (an empty line between the module setup and
  `def sort_keystrokes`), now contained the two characters `cl`. File mtime 2026-10-06 13:53:27. A diff against the
  2026-09-26 backup showed no other difference.
- **Cause:** accidental typing in the PyCharm editor (the same pattern as the two `features.py` incidents above;
  most likely the start of "claude" typed into the editor instead of the terminal). The file was writable because
  it was not on the read-only list.
- **Fix (2026-10-06 14:00:23 CEST, approved by the user):** removed exactly the two characters `cl` on line 40 and
  kept the empty line. Nothing else changed.
- **Hash after the fix:** `ff0d10b2a6e67dc57b55393994a44289b7b15ccfaa0e860e1891d978dd9d053e`, identical to
  `docs/MANIFEST_SHA256.txt` and to the copy in `backups/keystroke_liveness_backup_20260926.tar.gz`.
- **Checks after the fix:** `pytest`: 122 passed; all freeze guards (Phases 1, 2, 3a, 3b pre-registration,
  3b outputs, 3c) pass in full and quick mode; the manifest check now flags only `docs/PROJECT_STATE.md` (expected,
  see its section j).
- **No result was affected:** while it held the stray text, the module could not even be imported. No step ran
  between 13:53:27 and the fix, and no output file changed (all output hashes in the freeze records match).
- **Prevention:** every `src/` file listed in `docs/MANIFEST_SHA256.txt` is now read-only. The 7 that were still
  writable were added to `docs/environment/readonly_files.txt`: `src/kl/__init__.py`, `config.py`, `evaluate.py`,
  `io_aalto.py`, `report.py`, `sessions.py`, `split.py`.
