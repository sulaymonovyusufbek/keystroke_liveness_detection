"""Phase 3b: sequence builder, padding invariance, mixture NLL, determinism, bot regeneration, guards."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import multivariate_normal, norm

from kl.attacks.a1_gaussian import A1Gaussian
from kl.attacks.a2_digraph import A2Digraph
from kl.attacks.a4_imitator import A4Imitator
from kl.attacks.a5_replay import A5bTransplant
from kl.attacks.pool import SessionBank, fit_pool_stats
from kl.config import ROOT
from kl.features import FEATURE_NAMES, FeatureParams, worked_example
from kl.phase2.generate import generate_bot_features
from kl.phase2.guard import FrozenViolation
from kl.phase3b import nets
from kl.phase3b.evaluation import Scored, TestAccess, TestReuseError, TestSplit, ThresholdDataError, seed_cell, thresholds_from_val_humans
from kl.phase3b.guard import read_prereg_record, verify_prereg
from kl.phase3b.regen import regenerate_bots, verify_against_phase2
from kl.phase3b.sequences import UNK, Encoder, NegativeHoldError, SeqSet, seqset_from_frame, session_arrays
from synth import SENTENCES, synth_pool

PARAMS = dict(min_samples=30, min_session_pairs=10, min_rate_transitions=3, pause_threshold_ms=500, max_pp_ms=5000, max_hold_ms=2000)


# ---------------------------------------------------------------- helpers
def _train_set(n_sessions: int = 60, seed: int = 0) -> SeqSet:
    df = synth_pool(n_users=max(n_sessions // 15, 2), per_user=15, seed=seed)
    return seqset_from_frame(df, "train_all", "train", "human", None)


def _encoded(n: int = 40, seed: int = 0, max_len: int | None = None):
    s = _train_set(max(n, 30), seed)
    enc = Encoder.fit(s, min_count=5, percentile=99)
    if max_len is not None:
        enc.max_len = max_len
    return enc, enc.encode(s.take(np.arange(min(n, s.n_sessions))))


def _two_sets():
    """Encoder + a 'human' set of 30 and a 'bot' set of 45 sessions encoded with the SAME encoder."""
    s = _train_set(60)
    enc = Encoder.fit(s, min_count=5, percentile=99)
    return enc, enc.encode(s.take(np.arange(30))), enc.encode(s.take(np.arange(15, 60)))


def _pad_more(batch: dict, extra: int) -> dict:
    out = dict(batch)
    for k in ("keys", "h", "p", "pmask"):
        a = batch[k]
        out[k] = np.concatenate([a, np.zeros((a.shape[0], extra), dtype=a.dtype)], axis=1)
    return out


# ---------------------------------------------------------------- 1. sequence builder
def test_sequence_builder_worked_example():
    keycode, hold, pp = session_arrays([0, 200, 450], [100, 280, 560], [65, 66, 67])
    assert hold.tolist() == [100, 80, 110]
    assert pp[:2].tolist() == [200, 250] and np.isnan(pp[2])                # last p undefined -> masked
    assert keycode.tolist() == [65, 66, 67]


def test_sequence_builder_sorts_by_press_with_keystroke_id_tie_break():
    keycode, hold, pp = session_arrays([450, 0, 200, 200], [560, 100, 280, 290], [67, 65, 66, 16], kid=[4, 1, 3, 2])
    assert keycode.tolist() == [65, 16, 66, 67]                             # equal PRESS: KEYSTROKE_ID decides
    assert hold.tolist() == [100, 90, 80, 110] and pp[:3].tolist() == [200, 0, 250]


def test_frame_builder_and_encoder_mask_the_last_p():
    df = worked_example().assign(session_id="1_1")
    s = seqset_from_frame(df, "train_all", "train", "human", None)
    assert s.hold.tolist() == [100, 80, 110] and s.pp[:2].tolist() == [200, 250] and np.isnan(s.pp[2])
    enc = Encoder.fit(s, min_count=1, percentile=99)
    e = enc.encode(s)
    assert e.lens.tolist() == [3] and e.pmask[0].tolist() == [True, True, False]
    assert e.p[0, 2] == 0.0                                                 # masked p stored as 0
    assert np.allclose(e.h[0] * enc.sd_h + enc.mu_h, np.log1p([100, 80, 110]), atol=1e-6)
    assert np.allclose(e.p[0, :2] * enc.sd_p + enc.mu_p, np.log1p([200, 250]), atol=1e-6)


def test_encoder_vocabulary_unk_truncation_and_negative_holds():
    s = _train_set(60)
    enc = Encoder.fit(s, min_count=20, percentile=99)
    codes, counts = np.unique(s.keycode, return_counts=True)
    assert set(enc.keycodes.tolist()) == set(codes[counts >= 20].tolist())
    assert (enc.key_ids(np.array([999999])) == UNK).all()
    enc.max_len = 10
    e = enc.encode(s)
    assert e.n_truncated == int((s.lengths > 10).sum()) and (e.lens <= 10).all()
    last = e.lens - 1                                                        # the last KEPT key is the last key
    assert not e.pmask[np.arange(len(e)), last].any() and e.pmask[np.arange(len(e)), last - 1].all()
    with pytest.raises(ValueError):
        Encoder.fit(SeqSet(**{**s.__dict__, "split": "val"}), 20, 99)       # the encoder sees TRAIN humans only
    bad = SeqSet(**{**s.__dict__, "hold": np.r_[-5.0, s.hold[1:]]})
    with pytest.raises(NegativeHoldError):
        enc.encode(bad)


# ---------------------------------------------------------------- 2. padding invariance
@pytest.mark.parametrize("use_keys", [True, False])
def test_d4_nll_identical_with_extra_padding(use_keys):
    torch.manual_seed(0)
    enc, e = _encoded(12)
    model = nets.D4Model(enc.vocab_size, use_keys=use_keys).double().eval()
    b = e.batch(np.arange(len(e)))
    to = lambda x: {k: (torch.from_numpy(v).double() if v.dtype.kind == "f" else torch.from_numpy(v)) for k, v in x.items()}  # noqa: E731
    with torch.no_grad():
        a1, v1 = model.step_nll(to(b))
        a2, v2 = model.step_nll(to(_pad_more(b, 17)))
        s1, s2 = model.session_scores(to(b)), model.session_scores(to(_pad_more(b, 17)))
    w = b["h"].shape[1]
    assert torch.equal(v1, v2[:, :w]) and not v2[:, w:].any()
    assert torch.allclose(a1, a2[:, :w], atol=1e-12, rtol=0) and torch.allclose(s1, s2, atol=1e-12, rtol=0)
    # a session scored alone equals its score inside a batch padded to a longer neighbour
    with torch.no_grad():
        single = model.session_scores(to(e.batch(np.array([int(np.argmin(e.lens))]))))
    assert torch.allclose(single, s1[int(np.argmin(e.lens))].reshape(1), atol=1e-12, rtol=0)


def test_d5_logits_identical_with_extra_padding():
    torch.manual_seed(0)
    enc, e = _encoded(12)
    model = nets.D5Model(enc.vocab_size).double().eval()
    b = e.batch(np.arange(len(e)))
    to = lambda x: {k: (torch.from_numpy(v).double() if v.dtype.kind == "f" else torch.from_numpy(v)) for k, v in x.items()}  # noqa: E731
    with torch.no_grad():
        l1, l2 = model.logits(to(b)), model.logits(to(_pad_more(b, 23)))
        single = model.logits(to(e.batch(np.array([int(np.argmin(e.lens))]))))
    assert torch.allclose(l1, l2, atol=1e-12, rtol=0)
    assert torch.allclose(single, l1[int(np.argmin(e.lens))].reshape(1), atol=1e-12, rtol=0)


# ---------------------------------------------------------------- 3. mixture NLL
def test_mixture_nll_k1_equals_analytic_bivariate_gaussian():
    rng = np.random.default_rng(0)
    n = 50
    h, p = rng.normal(size=n), rng.normal(size=n)
    mu_h, mu_p = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
    s_h, s_p = rng.uniform(0.2, 2, size=(n, 1)), rng.uniform(0.2, 2, size=(n, 1))
    T = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    ours = nets.mixture_nll(T(h), T(p), torch.ones(n, dtype=torch.bool), T(rng.normal(size=(n, 1))), T(mu_h), T(mu_p), T(s_h), T(s_p)).numpy()
    ref = np.array([-multivariate_normal(mean=[mu_h[i, 0], mu_p[i, 0]], cov=np.diag([s_h[i, 0] ** 2, s_p[i, 0] ** 2])).logpdf([h[i], p[i]]) for i in range(n)])
    assert np.max(np.abs(ours - ref)) < 1e-6
    marg = nets.mixture_nll(T(h), T(p), torch.zeros(n, dtype=torch.bool), T(np.zeros((n, 1))), T(mu_h), T(mu_p), T(s_h), T(s_p)).numpy()
    assert np.max(np.abs(marg + norm.logpdf(h, mu_h[:, 0], s_h[:, 0]))) < 1e-6   # masked p: marginal over h


def test_mixture_nll_is_a_proper_mixture():
    T = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    logits, mu_h, mu_p, s = T([[0.3, -0.2]]), T([[0.0, 1.0]]), T([[0.5, -1.0]]), T([[0.7, 1.3]])
    ours = float(nets.mixture_nll(T([0.2]), T([0.1]), torch.tensor([True]), logits, mu_h, mu_p, s, s))
    w = np.exp([0.3, -0.2]) / np.exp([0.3, -0.2]).sum()
    dens = sum(w[k] * norm.pdf(0.2, [0.0, 1.0][k], [0.7, 1.3][k]) * norm.pdf(0.1, [0.5, -1.0][k], [0.7, 1.3][k]) for k in range(2))
    assert ours == pytest.approx(-np.log(dens), abs=1e-10)


# ---------------------------------------------------------------- 4. determinism
def _one_epoch_d4(seed: int) -> dict:
    enc, e = _encoded(40)
    nets.set_determinism(seed)
    model = nets.D4Model(enc.vocab_size)
    gen = torch.Generator().manual_seed(seed)
    nets.fit(model, nets.d4_epoch_batches(model, e, 8, gen, torch.device("cpu")), lambda: nets.d4_mean_nll(model, e, 64, torch.device("cpu")),
             1e-3, 1, 3, 1.0)
    return {k: v.clone() for k, v in model.state_dict().items()}


def _one_epoch_d5(seed: int) -> dict:
    enc, e, b = _two_sets()
    nets.set_determinism(seed)
    model = nets.D5Model(enc.vocab_size)
    gen = torch.Generator().manual_seed(seed)
    nets.fit(model, nets.d5_epoch_batches(model, e, b, 8, gen, torch.device("cpu")), lambda: nets.d5_val_loss(model, e, b, 64, torch.device("cpu")),
             1e-3, 1, 3, 1.0)
    return {k: v.clone() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("runner", [_one_epoch_d4, _one_epoch_d5])
def test_same_seed_gives_identical_weights_after_one_epoch(runner):
    a, b, c = runner(42), runner(42), runner(43)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)


def test_early_stopping_restores_the_best_epoch():
    enc, e = _encoded(20)
    nets.set_determinism(0)
    model = nets.D4Model(enc.vocab_size)
    vals = iter([1.0, 0.5, 0.7, 0.8, 0.9, 0.1])
    snapshots = []

    def val_loss():
        snapshots.append({k: v.clone() for k, v in model.state_dict().items()})
        return next(vals)
    res = nets.fit(model, nets.d4_epoch_batches(model, e, 8, torch.Generator().manual_seed(0), torch.device("cpu")), val_loss, 1e-3, 30, 3, 1.0)
    assert res.best_epoch == 2 and res.epochs_run == 5 and res.stopped_early               # 3 epochs without improvement
    assert all(torch.equal(model.state_dict()[k], snapshots[1][k]) for k in snapshots[1])


def test_d5_epochs_are_class_balanced():
    enc, h, b = _two_sets()
    model = nets.D5Model(enc.vocab_size)
    seen = []
    orig = nets.F.binary_cross_entropy_with_logits
    try:
        nets.F.binary_cross_entropy_with_logits = lambda logits, y: (seen.append(y.numpy().copy()), orig(logits, y))[1]
        for _ in nets.d5_epoch_batches(model, h, b, 8, torch.Generator().manual_seed(0), torch.device("cpu"))(1):
            pass
    finally:
        nets.F.binary_cross_entropy_with_logits = orig
    y = np.concatenate(seen)
    assert len(y) == 2 * len(h) and y.sum() == len(h)                       # N_h humans + N_h bots per epoch


# ---------------------------------------------------------------- 5. regenerated bots == Phase 2 features
@pytest.fixture(scope="module")
def pool():
    df = synth_pool(n_users=60, per_user=20, seed=0)
    bank = SessionBank.from_frame(df)
    return bank, fit_pool_stats(bank, range(bank.n_sessions), PARAMS)


@pytest.mark.parametrize("name", ["A1", "A2", "A4", "A5b"])
def test_regenerated_bots_reproduce_phase2_features(pool, name):
    bank, stats = pool
    att = {"A1": A1Gaussian(), "A2": A2Digraph(stats), "A4": A4Imitator(stats), "A5b": A5bTransplant(bank, list(range(bank.n_sessions)))}[name]
    matched = pd.DataFrame({"session_id": [f"9_{i}" for i in range(40)], "user_id": 9, "sentence": [SENTENCES[i % len(SENTENCES)] for i in range(40)]})
    stored = generate_bot_features(att, matched, "test", 42, FeatureParams(), n_jobs=1, chunk_size=7)        # the Phase 2 code path
    feats, seqs = regenerate_bots(att, matched, "test", 42, FeatureParams(), n_jobs=1, chunk_size=13)
    rep = verify_against_phase2(feats, stored)
    assert rep["ok"] and rep["max_abs_diff"] == 0.0
    assert seqs.n_sessions == len(stored) and np.array_equal(seqs.lengths, stored["n_keys"].to_numpy())
    tampered = stored.copy()
    tampered.loc[3, "hold_mean"] += 1e-6
    assert not verify_against_phase2(feats, tampered)["ok"]
    assert not verify_against_phase2(feats, stored.iloc[::-1].reset_index(drop=True))["ok"]            # order matters


def test_regenerated_bots_match_the_stored_quick_run_prefix():
    """On the real quick-run outputs (if present): the first bots of every split / attacker reproduce Phase 2 exactly."""
    import joblib
    from kl.config import load_settings
    from kl.phase2.generate import build_attackers
    st = load_settings(quick=True)
    p2 = st.out_dir / "phase2"
    if not (p2 / "pool.joblib").exists():
        pytest.skip("quick-run Phase 2 outputs not available")
    pool_ = joblib.load(p2 / "pool.joblib")
    names = ["A1", "A2", "A3", "A4", "A5a", "A5b"]
    atts = build_attackers(pool_["stats"], pool_["bank"], pool_["fit_idx"], st.cfg["a1"], names)
    sent = pd.read_parquet(st.interim_dir / "session_index.parquet", columns=["session_id", "SENTENCE"]).set_index("session_id")["SENTENCE"]
    for split in ("train", "test"):
        feats = pd.read_parquet(st.interim_dir / f"features_{split}.parquet", columns=["session_id", "user_id"]).iloc[:60]
        matched = feats.assign(sentence=sent.loc[feats["session_id"]].to_numpy())
        for a in names:
            regen, _ = regenerate_bots(atts[a], matched, split, st.seed, FeatureParams.from_config(st.cfg["features"]))
            stored = pd.read_parquet(p2 / f"bots_{a}_{split}.parquet").iloc[:60].reset_index(drop=True)
            assert verify_against_phase2(regen, stored)["ok"], (a, split)


# ---------------------------------------------------------------- 6. guards
def test_thresholds_come_from_validation_humans_only():
    s = np.linspace(0, 1, 101)
    taus = thresholds_from_val_humans(Scored(s, "val", "human"))
    assert taus["tau5"] == pytest.approx(0.05) and taus["tau10"] == pytest.approx(0.10)
    for split, kind in (("test", "human"), ("train", "human"), ("val", "bot"), ("test", "bot")):
        with pytest.raises(ThresholdDataError):
            thresholds_from_val_humans(Scored(s, split, kind))


def test_test_split_is_handed_out_once_per_detector(tmp_path):
    enc, e = _encoded(10)
    te = e.__class__(**{**e.__dict__, "split": "test"})
    ledger = tmp_path / "test_access.json"
    ts = TestSplit(te, {"A1": te}, TestAccess(ledger))
    ts.for_detector("D4/seed42")
    with pytest.raises(TestReuseError):
        ts.for_detector("D4/seed42")                                         # same process
    ts.for_detector("D4/seed43")                                             # another (detector, seed) is fine
    again = TestSplit(te, {"A1": te}, TestAccess(ledger))
    with pytest.raises(TestReuseError):
        again.for_detector("D4/seed42")                                      # a later run is refused ...
    TestSplit(te, {"A1": te}, TestAccess(ledger, allow_rerun=True)).for_detector("D4/seed42")   # ... unless --allow-rerun
    assert len(json.loads(ledger.read_text())["D4/seed42"]) == 2
    with pytest.raises(ValueError):
        TestSplit(e, {"A1": te}, TestAccess(tmp_path / "x.json"))            # only test-split sets


def test_seed_cell_aggregates_over_seeds_and_bootstraps_seed_42():
    from kl.phase2.evaluation import user_weights
    rng = np.random.default_rng(0)
    hu, bu = rng.integers(0, 30, 300), rng.integers(0, 30, 300)
    per_seed = {s: {"h": rng.normal(1, 1, 300), "b": rng.normal(0, 1, 300), "taus": {"tau5": -0.6, "tau10": -0.3}} for s in (42, 43, 44)}
    row = seed_cell("D5", "A1", "A2", per_seed, hu, bu, user_weights(30, 200, 42), 0.95)
    aucs = [row[f"auc_seed{s}"] for s in (42, 43, 44)]
    assert row["auc_mean"] == pytest.approx(np.mean(aucs)) and row["auc_sd"] == pytest.approx(np.std(aucs, ddof=1))
    assert row["auc_ci_lo"] <= row["auc_seed42"] <= row["auc_ci_hi"]
    assert row["apcer_fpr_tau5_seed43"] == pytest.approx(np.mean(per_seed[43]["b"] >= -0.6))
    flipped = seed_cell("D5", "A1", "A2", {s: {**d, "h": -d["h"], "b": -d["b"]} for s, d in per_seed.items()}, hu, bu, user_weights(30, 50, 42), 0.95)
    assert flipped["auc_mean"] < 0.5                                         # AUC < 0.5 is reported as is


def test_prereg_guard_detects_changes(tmp_path):
    rec = read_prereg_record()
    assert set(rec) >= {"prereg_sha256", "config_sha256", "recorded_at"}
    verify_prereg()                                                          # the real record holds
    for f in (rec["prereg_file"], rec["config_file"]):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_bytes((ROOT / f).read_bytes())
    rec_path = tmp_path / "rec.txt"
    rec_path.write_text((ROOT / "docs" / "phase3b_prereg_hash.txt").read_text())
    verify_prereg(rec_path, tmp_path)
    (tmp_path / rec["config_file"]).write_text((ROOT / rec["config_file"]).read_text() + "\n# tuned\n")
    with pytest.raises(FrozenViolation):
        verify_prereg(rec_path, tmp_path)


def test_phase3a_record_covers_its_sources_and_still_matches():
    from kl.phase2.guard import verify_files
    rec = json.loads((ROOT / "docs" / "phase3a_frozen_hashes.json").read_text())
    assert {"src/kl/phase3a/analyses.py", "scripts/20_run_diagnostics.py", "config_phase3a.yaml"} <= set(rec["source_files"])
    verify_files(rec["source_files"])


# ---------------------------------------------------------------- launcher: lock, completion state, pause handling
def test_training_lock_is_exclusive_and_released(tmp_path):
    from kl.phase3b.runner import LockBusyError, acquire_lock, gpu_lock
    first = gpu_lock(tmp_path)
    with pytest.raises(LockBusyError):
        acquire_lock(tmp_path / ".training.lock", "training")
    first.close()
    gpu_lock(tmp_path).close()                                               # free again once the holder is gone


def test_lock_held_by_a_paused_process_blocks_a_second_start(tmp_path):
    import signal
    import subprocess
    import sys
    import time
    from kl.phase3b.runner import LockBusyError, gpu_lock
    code = f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); from kl.phase3b.runner import gpu_lock; l = gpu_lock(__import__('pathlib').Path({str(tmp_path)!r})); print('locked', flush=True); import time; time.sleep(60)"
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        child.send_signal(signal.SIGSTOP)
        time.sleep(0.2)
        with pytest.raises(LockBusyError):
            gpu_lock(tmp_path)
    finally:
        child.send_signal(signal.SIGCONT)
        child.terminate()
        child.wait()
    gpu_lock(tmp_path).close()


def test_popen_wait_does_not_return_while_the_child_is_paused():
    """The launcher bug of 2026-09-25: bash treated a SIGSTOP-paused part as finished. Popen.wait() must not."""
    import signal
    import subprocess
    import sys
    import threading
    import time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)"])
    child.send_signal(signal.SIGSTOP)
    done = threading.Event()
    threading.Thread(target=lambda: (child.wait(), done.set()), daemon=True).start()
    time.sleep(2.0)                                                          # the child would have exited by now if running
    assert not done.is_set()
    child.send_signal(signal.SIGCONT)
    assert done.wait(10) and child.returncode == 0


def test_run_state_complete_interrupted_not_done(tmp_path):
    from kl.config import write_json
    from kl.phase3b.runner import run_state, save_scores, score_path
    (tmp_path / "scores").mkdir()
    (tmp_path / "models").mkdir()
    model = tmp_path / "models" / "D4_seed42.pt"
    key = "D4/seed42"
    assert run_state(tmp_path, key, model, ["test_m"]) == "not-done"
    (tmp_path / "test_access.json").write_text(json.dumps({key: ["t"]}))
    assert run_state(tmp_path, key, model, ["test_m"]) == "interrupted-after-test-access"
    save_scores(score_path(tmp_path, key), test_m=np.ones(3))
    model.write_bytes(b"w")
    write_json(tmp_path / "runtime_log.json", {"runs": {key: {}}})
    assert run_state(tmp_path, key, model, ["test_m"]) == "complete"
    assert run_state(tmp_path, key, model, ["test_m", "bot_m_A1"]) == "interrupted-after-test-access"   # missing array
    save_scores(score_path(tmp_path, key), test_m=np.array([1.0, np.nan]))
    assert run_state(tmp_path, key, model, ["test_m"]) == "interrupted-after-test-access"               # non-finite


# ---------------------------------------------------------------------------------------------- thermal guard
def _fake_thermal(temps):
    from kl.phase3b import thermal
    it, clock = iter(temps), [0.0]
    return thermal, (lambda: next(it)), (lambda s: clock.__setitem__(0, clock[0] + s)), (lambda: clock[0])


def test_thermal_guard_no_pause_below_threshold(tmp_path):
    import logging
    thermal, read, sleep, clock = _fake_thermal([84])
    g = thermal.ThermalGuard(tmp_path, logging.getLogger("t"), read, sleep, clock)
    assert g.check("run", 1) == 0.0
    assert not (tmp_path / "runtime_log.json").exists()


def test_thermal_guard_pauses_until_70_and_logs(tmp_path):
    import logging
    thermal, read, sleep, clock = _fake_thermal([86, 80, 75, 70])
    g = thermal.ThermalGuard(tmp_path, logging.getLogger("t"), read, sleep, clock)
    assert g.check("D4/seed42", 3) == 3 * thermal.POLL_S
    rec = list(json.loads((tmp_path / "runtime_log.json").read_text())["thermal_pauses"].values())
    assert rec[0]["outcome"] == "resumed" and rec[0]["temp_at_pause"] == 86 and rec[0]["temp_at_end"] == 70 and rec[0]["after_epoch"] == 3


def test_thermal_guard_stops_after_30_min_at_90(tmp_path):
    import logging
    n = thermal_n = 1800 // 30
    thermal, read, sleep, clock = _fake_thermal([91] * (n + 2))
    g = thermal.ThermalGuard(tmp_path, logging.getLogger("t"), read, sleep, clock)
    with pytest.raises(SystemExit) as e:
        g.check("D5<-A1/seed42", 6)
    assert e.value.code == thermal.THERMAL_EXIT
    assert json.loads((tmp_path / "thermal_stop.json").read_text())["paused_seconds"] >= thermal.ABORT_AFTER_S
    del thermal_n


def test_fit_is_unchanged_by_the_between_epochs_hook():
    torch.manual_seed(0)
    m1 = torch.nn.Linear(3, 1)
    m2 = torch.nn.Linear(3, 1)
    m2.load_state_dict(m1.state_dict())
    x = torch.randn(16, 3)

    def batches(m):
        return lambda epoch: iter([(m(x).pow(2).mean(), 16)])
    r1 = nets.fit(m1, batches(m1), lambda: float(m1(x).pow(2).mean()), 0.01, 5, 2, 1.0)
    r2 = nets.fit(m2, batches(m2), lambda: float(m2(x).pow(2).mean()), 0.01, 5, 2, 1.0, between_epochs=lambda e: 7.0)
    assert all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    assert r2.paused_seconds == 35.0 and r1.best_epoch == r2.best_epoch
