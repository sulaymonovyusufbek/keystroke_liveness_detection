"""Attacker statistics fitted on attacker_pool sessions ONLY (Phase 2).

:class:`SessionBank` stores keystroke sessions in flat, memmap-friendly numpy arrays; :func:`fit_pool_stats`
extracts every population-level statistic used by attackers A2-A5 from a chosen subset of its sessions.
Nothing here ever sees train / val / test data, and no detector is involved.

Conventions
-----------
* Rows of a session are sorted by PRESS_TIME (KEYSTROKE_ID as tie-break only).
* A *character row* is a row whose LETTER is a single character consistent with its KEYCODE
  (``char_to_key(letter)[0] == keycode``); LETTER is unreliable in the corpus, so anything inconsistent is treated
  as a non-character row. Shift (16) rows are ignored when looking at neighbouring keys; Backspace (8) rows are
  correction rows; everything else (Delete, arrows, ...) interrupts adjacency.
* ``pp`` = press-to-press = PRESS_next - PRESS_this, ``flight`` = PRESS_next - RELEASE_this, both between
  consecutive non-Shift rows. A *pause* is ``flight > pause_threshold_ms`` (same definition as the features).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from .a1_gaussian import char_to_key

SHIFT_KEYCODE, BACKSPACE_KEYCODE, DELETE_KEYCODE = 16, 8, 46
SPACE_CP = 32
MAX_DELAY = 3  # correction delay d in {0, 1, 2, 3} extra characters


def cp(ch: str) -> int:
    """Code point of the lower-cased character (0 for an empty string)."""
    lo = ch.lower()
    return ord(lo[0]) if lo else 0


# --------------------------------------------------------------------------------------------- session bank
class SessionBank:
    """Flat storage of many keystroke sessions (sorted by PRESS_TIME, KEYSTROKE_ID within a session)."""

    def __init__(
        self, session_ids: np.ndarray, user_ids: np.ndarray, offsets: np.ndarray, kid: np.ndarray, press: np.ndarray,
        release: np.ndarray, keycode: np.ndarray, letter_id: np.ndarray, letter_table: list[str],
        sentence_id: np.ndarray, input_id: np.ndarray, text_table: list[str],
    ) -> None:
        self.session_ids, self.user_ids, self.offsets = session_ids, user_ids, offsets
        self.kid, self.press, self.release, self.keycode, self.letter_id = kid, press, release, keycode, letter_id
        self.letter_table, self.sentence_id, self.input_id, self.text_table = letter_table, sentence_id, input_id, text_table
        self.char_code = self._char_codes()
        self.session_of_row = np.repeat(np.arange(len(session_ids)), np.diff(offsets))

    # -- construction -------------------------------------------------------------------------------
    @classmethod
    def from_frames(cls, rows: pd.DataFrame, meta: pd.DataFrame) -> "SessionBank":
        """Build from keystroke ``rows`` (session_id, KEYSTROKE_ID, PRESS_TIME, RELEASE_TIME, LETTER, KEYCODE)
        and one ``meta`` row per session (session_id, PARTICIPANT_ID, SENTENCE, USER_INPUT)."""
        codes, uniques = pd.factorize(rows["session_id"].to_numpy(dtype=object), sort=True)
        kid = np.nan_to_num(rows["KEYSTROKE_ID"].to_numpy(dtype="float64", na_value=np.nan), nan=0.0)
        press = rows["PRESS_TIME"].to_numpy(dtype="float64")
        order = np.lexsort((kid, press, codes))
        codes = codes[order]
        bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
        letters = rows["LETTER"].to_numpy(dtype=object)[order]
        lid, ltable = pd.factorize(letters)
        meta = meta.drop_duplicates("session_id").set_index("session_id").loc[list(uniques)]
        texts = pd.concat([meta["SENTENCE"], meta["USER_INPUT"]]).astype(object)
        tid, ttable = pd.factorize(texts.to_numpy(dtype=object))
        n = len(uniques)
        return cls(
            session_ids=np.array(uniques, dtype=object), user_ids=meta["PARTICIPANT_ID"].to_numpy(dtype="int64"),
            offsets=bounds.astype("int64"), kid=kid[order].astype("int64"), press=press[order],
            release=rows["RELEASE_TIME"].to_numpy(dtype="float64")[order],
            keycode=np.nan_to_num(rows["KEYCODE"].to_numpy(dtype="float64", na_value=np.nan), nan=-1.0)[order].astype("int32"),
            letter_id=lid.astype("int32"), letter_table=[str(x) for x in ltable],
            sentence_id=tid[:n].astype("int32"), input_id=tid[n:].astype("int32"), text_table=[str(x) for x in ttable],
        )

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "SessionBank":
        """Build from a single frame with ``session_id`` and all Aalto columns (convenience for tests)."""
        return cls.from_frames(df, df)

    # -- access ---------------------------------------------------------------------------------------
    @property
    def n_sessions(self) -> int:
        """Number of sessions."""
        return len(self.session_ids)

    def n_rows(self, i: int) -> int:
        """Keystroke rows of session ``i``."""
        return int(self.offsets[i + 1] - self.offsets[i])

    def sentence(self, i: int) -> str:
        """Prompt sentence of session ``i``."""
        return self.text_table[self.sentence_id[i]]

    def user_input(self, i: int) -> str:
        """Typed text of session ``i``."""
        return self.text_table[self.input_id[i]]

    def frame(self, i: int) -> pd.DataFrame:
        """Session ``i`` as an Aalto-schema frame with the ORIGINAL timings (placeholder ids -1)."""
        sl = slice(int(self.offsets[i]), int(self.offsets[i + 1]))
        n = sl.stop - sl.start
        return pd.DataFrame({
            "PARTICIPANT_ID": np.full(n, -1, dtype="int64"), "TEST_SECTION_ID": np.full(n, -1, dtype="int64"),
            "SENTENCE": [self.sentence(i)] * n, "USER_INPUT": [self.user_input(i)] * n,
            "KEYSTROKE_ID": self.kid[sl].copy(), "PRESS_TIME": self.press[sl].copy(), "RELEASE_TIME": self.release[sl].copy(),
            "LETTER": [self.letter_table[k] for k in self.letter_id[sl]], "KEYCODE": self.keycode[sl].astype("int64"),
        })

    def hold_flight(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """``(hold[0..m-1], flight[0..m-2])`` of session ``i`` over ALL its rows in press order."""
        sl = slice(int(self.offsets[i]), int(self.offsets[i + 1]))
        p, r = self.press[sl], self.release[sl]
        return r - p, p[1:] - r[:-1]

    def _char_codes(self) -> np.ndarray:
        expected = np.full(len(self.letter_table), -1, dtype="int32")
        cps = np.zeros(len(self.letter_table), dtype="int32")
        for k, letter in enumerate(self.letter_table):
            if len(letter) == 1:
                code = char_to_key(letter)[0]
                expected[k] = -1 if code == 229 else code
                cps[k] = cp(letter)
        ok = (expected[self.letter_id] == self.keycode) & ~np.isin(self.keycode, (SHIFT_KEYCODE, BACKSPACE_KEYCODE, DELETE_KEYCODE))
        return np.where(ok, cps[self.letter_id], 0).astype("int32")


# --------------------------------------------------------------------------------------------- fitted stats
def _arr(x: Any, fallback: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype="float64")
    return a if len(a) else np.asarray(fallback, dtype="float64")


@dataclass
class PoolStats:
    """All population-level statistics used by attackers A2-A5 (fitted on attacker_pool FIT sessions only)."""

    params: dict[str, Any]
    n_fit_sessions: int
    dg_pp: dict[int, np.ndarray]          # digraph key -> pp samples (n >= min_samples)
    sec_pp: dict[int, np.ndarray]         # second char -> pp samples
    glob_pp: np.ndarray
    key_hold: dict[int, np.ndarray]       # char -> hold samples
    glob_hold: np.ndarray
    mu_dg: dict[int, float]               # mean log pp (non-pause transitions)
    mu_sec: dict[int, float]
    mu_glob: float
    speed: np.ndarray                     # session offsets s_j
    speed_mean: float
    phi: float
    sigma: float
    hold_slope: float
    hold_intercept: float
    pause_flights: np.ndarray
    bs_pp: dict[str, np.ndarray]          # "c2b", "b2b", "b2c"
    bs_hold: np.ndarray
    delay_p: np.ndarray                   # P(d), d = 0..3
    frac_long_runs: float
    rates: np.ndarray                     # rows (c, u, p_boundary, p_within)
    pso: np.ndarray                       # per-session share of Shift holds released after the next key press
    shift_lead: np.ndarray
    shift_lag: np.ndarray
    shift_tail: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)
    _cum_delay: np.ndarray = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._cum_delay = np.cumsum(self.delay_p)

    # -- helpers ------------------------------------------------------------------------------------
    @property
    def max_pp(self) -> int:
        """Upper clip of a press-to-press interval (ms)."""
        return int(self.params["max_pp_ms"])

    @property
    def max_hold(self) -> int:
        """Upper clip of a hold (ms)."""
        return int(self.params["max_hold_ms"])

    @property
    def mean_bs_per_event(self) -> float:
        """Expected Backspace presses per correction event under the fitted delay distribution."""
        return float(np.dot(self.delay_p, np.arange(1, MAX_DELAY + 2)))

    @staticmethod
    def _pick(arr: np.ndarray, rng: np.random.Generator) -> float:
        return float(arr[rng.integers(len(arr))])

    # -- A2 ---------------------------------------------------------------------------------------------
    def pp_samples(self, prev: str, cur: str) -> np.ndarray:
        """pp samples of the digraph ``(prev, cur)`` with backoff: digraph -> second character -> global."""
        a = self.dg_pp.get((cp(prev) << 21) | cp(cur))
        if a is None:
            a = self.sec_pp.get(cp(cur))
        return self.glob_pp if a is None else a

    def sample_pp(self, prev: str, cur: str, rng: np.random.Generator) -> int:
        """One pp draw (ms) for the digraph."""
        return int(self._pick(self.pp_samples(prev, cur), rng))

    def hold_samples(self, ch: str) -> np.ndarray:
        """Hold samples of the key (backoff: global)."""
        a = self.key_hold.get(cp(ch))
        return self.glob_hold if a is None else a

    def sample_hold(self, ch: str, rng: np.random.Generator) -> float:
        """One hold draw (ms) for the key."""
        return self._pick(self.hold_samples(ch), rng)

    # -- A3 ---------------------------------------------------------------------------------------------
    def mu(self, prev: str, cur: str) -> float:
        """Mean log pp of the digraph with the same backoff chain."""
        v = self.mu_dg.get((cp(prev) << 21) | cp(cur))
        if v is None:
            v = self.mu_sec.get(cp(cur))
        return self.mu_glob if v is None else v

    def sample_speed(self, rng: np.random.Generator) -> float:
        """Session speed offset s drawn from the pool's per-session offsets."""
        return self._pick(self.speed, rng)

    def hold_scale(self, s: float) -> float:
        """Multiplicative hold scaling ``exp(slope * (s - mean(s)))`` from the pool's hold-speed regression."""
        return float(np.exp(self.hold_slope * (s - self.speed_mean)))

    def scaled_hold(self, ch: str, s: float, rng: np.random.Generator) -> int:
        """Per-key empirical hold scaled by the session speed, clipped to [1, max_hold] ms."""
        return int(min(max(round(self.sample_hold(ch, rng) * self.hold_scale(s)), 1), self.max_hold))

    def clip_pp(self, pp: float) -> int:
        """Round and clip a press-to-press interval to [1, max_pp] ms."""
        return int(min(max(round(pp), 1), self.max_pp))

    # -- A4 ---------------------------------------------------------------------------------------------
    def sample_rates(self, rng: np.random.Generator) -> tuple[float, float, float, float]:
        """Joint per-session draw ``(c, u, p_boundary, p_within)`` from one pool session."""
        r = self.rates[rng.integers(len(self.rates))]
        return float(r[0]), float(r[1]), float(r[2]), float(r[3])

    def sample_pso(self, rng: np.random.Generator) -> float:
        """Per-session probability that a Shift hold outlasts the next key press."""
        return self._pick(self.pso, rng)

    def sample_delay(self, rng: np.random.Generator) -> int:
        """Correction delay d (extra characters typed before the Backspaces), 0..3."""
        return int(min(np.searchsorted(self._cum_delay, rng.random(), side="right"), MAX_DELAY))

    def sample_pause_flight(self, rng: np.random.Generator) -> int:
        """Flight (ms, > pause threshold) of an inserted pause."""
        return int(self._pick(self.pause_flights, rng))

    def sample_bs_pp(self, kind: str, rng: np.random.Generator) -> int:
        """pp for a transition involving Backspace (``c2b`` char->bs, ``b2b``, ``b2c``)."""
        return max(1, int(self._pick(self.bs_pp[kind], rng)))

    def sample_bs_hold(self, rng: np.random.Generator) -> int:
        """Backspace hold (ms), clipped to [1, max_hold]."""
        return int(min(max(round(self._pick(self.bs_hold, rng)), 1), self.max_hold))

    def sample_shift_lead(self, rng: np.random.Generator) -> int:
        """Shift press -> key press lead (ms)."""
        return max(1, int(self._pick(self.shift_lead, rng)))

    def sample_shift_lag(self, rng: np.random.Generator) -> int:
        """Shift release minus key release (ms, may be negative) for non-overlapping Shift holds."""
        return int(self._pick(self.shift_lag, rng))

    def sample_shift_tail(self, rng: np.random.Generator) -> int:
        """Shift release minus the press of the next key (ms, > 0) for overlapping Shift holds."""
        return max(1, int(self._pick(self.shift_tail, rng)))

    # -- reproducibility ------------------------------------------------------------------------------
    def digraph_table(self) -> pd.DataFrame:
        """Digraphs with >= min_samples pp samples: counts, mean pp, mean log pp (non-pause)."""
        rows = []
        for k, a in self.dg_pp.items():
            rows.append({"first": chr(k >> 21), "second": chr(k & ((1 << 21) - 1)), "n_pp": len(a), "mean_pp": float(a.mean()),
                         "mean_log_pp_nonpause": self.mu_dg.get(k, float("nan"))})
        return pd.DataFrame(rows).sort_values("n_pp", ascending=False).reset_index(drop=True)

    def describe(self) -> dict[str, Any]:
        """Scalar parameters and quantile summaries of every fitted distribution (for ATTACKS.md)."""
        def q(a: np.ndarray) -> dict[str, float]:
            a = np.asarray(a, dtype="float64")
            if not len(a):
                return {"n": 0}
            qs = np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95])
            return {"n": int(len(a)), "mean": float(a.mean()), "std": float(a.std()), "q05": float(qs[0]), "q25": float(qs[1]),
                    "median": float(qs[2]), "q75": float(qs[3]), "q95": float(qs[4])}
        return {
            "params": self.params, "n_fit_sessions": self.n_fit_sessions,
            "a2": {"n_digraphs_with_own_distribution": len(self.dg_pp), "n_second_char_distributions": len(self.sec_pp),
                   "n_key_hold_distributions": len(self.key_hold), "global_pp": q(self.glob_pp), "global_hold": q(self.glob_hold)},
            "a3": {"mu_global_log_pp": self.mu_glob, "n_mu_digraphs": len(self.mu_dg), "phi": self.phi, "sigma": self.sigma,
                   "speed_offset_s": q(self.speed), "speed_mean": self.speed_mean,
                   "hold_slope": self.hold_slope, "hold_intercept": self.hold_intercept},
            "a4": {"delay_p": [float(x) for x in self.delay_p], "frac_backspace_runs_longer_than_4": self.frac_long_runs,
                   "mean_backspaces_per_event": self.mean_bs_per_event,
                   "rate_c_backspace_rows_per_char": q(self.rates[:, 0]), "rate_u_lev_norm": q(self.rates[:, 1]),
                   "rate_p_pause_word_boundary": q(self.rates[:, 2]), "rate_p_pause_within_word": q(self.rates[:, 3]),
                   "pause_flight_ms": q(self.pause_flights), "p_shift_overlap_per_session": q(self.pso),
                   "shift_lead_ms": q(self.shift_lead), "shift_lag_ms": q(self.shift_lag), "shift_tail_ms": q(self.shift_tail),
                   "backspace_hold_ms": q(self.bs_hold), **{f"pp_{k}_ms": q(v) for k, v in self.bs_pp.items()}},
            "extras": self.extras,
        }


# --------------------------------------------------------------------------------------------- fitting
def _group_arrays(keys: np.ndarray, values: np.ndarray, min_n: int, dtype: str = "int32") -> dict[int, np.ndarray]:
    order = np.argsort(keys, kind="stable")
    ks, vs = keys[order], values[order]
    uniq, start, cnt = np.unique(ks, return_index=True, return_counts=True)
    return {int(k): vs[s:s + c].astype(dtype) for k, s, c in zip(uniq, start, cnt) if c >= min_n}


def _group_means(keys: np.ndarray, values: np.ndarray, min_n: int) -> dict[int, float]:
    s = pd.Series(values).groupby(keys)
    mean, cnt = s.mean(), s.count()
    return {int(k): float(mean[k]) for k in mean.index[cnt.to_numpy() >= min_n]}


def _lookup(keys: np.ndarray, table: Mapping[int, float]) -> np.ndarray:
    if not table:
        return np.full(len(keys), np.nan)
    return pd.Series(dict(table)).reindex(keys).to_numpy(dtype="float64")


def fit_pool_stats(bank: SessionBank, idx: Sequence[int], params: Mapping[str, Any]) -> PoolStats:
    """Fit every attacker statistic on the sessions ``idx`` of ``bank`` (attacker_pool FIT sessions).

    ``params``: ``min_samples`` (30), ``min_session_pairs`` (10), ``min_rate_transitions`` (3),
    ``pause_threshold_ms`` (500), ``max_pp_ms`` (5000), ``max_hold_ms`` (2000).
    """
    min_n, min_pairs, min_rate = int(params["min_samples"]), int(params["min_session_pairs"]), int(params["min_rate_transitions"])
    pause_ms = float(params["pause_threshold_ms"])
    idx = np.asarray(sorted(idx), dtype="int64")
    n_all = bank.n_sessions
    sel = np.zeros(n_all, dtype=bool)
    sel[idx] = True
    sess_of_row = bank.session_of_row
    in_sel = sel[sess_of_row]
    keycode, press, release, cc = bank.keycode, bank.press, bank.release, bank.char_code

    # ---- consecutive non-Shift row pairs -------------------------------------------------------------
    R = np.flatnonzero(in_sel & (keycode != SHIFT_KEYCODE))
    a, b = R[:-1], R[1:]
    same = sess_of_row[a] == sess_of_row[b]
    a, b = a[same], b[same]
    sess = sess_of_row[a]
    pa, pb = cc[a], cc[b]
    a_bs, b_bs = keycode[a] == BACKSPACE_KEYCODE, keycode[b] == BACKSPACE_KEYCODE
    pp = press[b] - press[a]
    flight = press[b] - release[a]
    is_cc = (pa > 0) & (pb > 0)
    nonpause = flight <= pause_ms
    adjacent = np.r_[b[:-1] == a[1:], False]  # pair k+1 starts where pair k ends

    # ---- A2: empirical pp per digraph / second char / global; hold per key ---------------------------
    dg_key = (pa.astype("int64") << 21) | pb.astype("int64")
    pp_i = np.clip(pp, 0, params["max_pp_ms"])
    dg_pp = _group_arrays(dg_key[is_cc], pp_i[is_cc], min_n)
    sec_pp = _group_arrays(pb[is_cc].astype("int64"), pp_i[is_cc], min_n)
    glob_pp = pp_i[is_cc].astype("int32")
    crow = in_sel & (cc > 0)
    hold_all = release - press
    key_hold = _group_arrays(cc[crow].astype("int64"), np.clip(hold_all[crow], 0, params["max_hold_ms"]), min_n)
    glob_hold = np.clip(hold_all[crow], 0, params["max_hold_ms"]).astype("int32")

    # ---- A3: log-pp model on non-pause char->char transitions -----------------------------------------
    m = is_cc & nonpause
    logpp = np.log(np.maximum(pp, 1.0))
    mu_dg = _group_means(dg_key[m], logpp[m], min_n)
    mu_sec = _group_means(pb[m].astype("int64"), logpp[m], min_n)
    mu_glob = float(logpp[m].mean())
    mu_pair = _lookup(dg_key, mu_dg)
    mu_pair = np.where(np.isnan(mu_pair), _lookup(pb.astype("int64"), mu_sec), mu_pair)
    mu_pair = np.where(np.isnan(mu_pair), mu_glob, mu_pair)
    resid0 = np.where(m, logpp - mu_pair, np.nan)
    g = pd.Series(resid0[m]).groupby(sess[m])
    s_mean, s_cnt = g.mean(), g.count()
    s_series = s_mean[s_cnt >= min_pairs]
    s_full = np.full(n_all, np.nan)
    s_full[s_series.index.to_numpy()] = s_series.to_numpy()
    s_pair = s_full[sess]
    valid = m & ~np.isnan(s_pair)
    e = np.where(valid, resid0 - s_pair, np.nan)
    both = adjacent & valid & np.r_[valid[1:], False]
    e_prev, e_next = e[both], np.r_[e[1:], np.nan][both]
    phi = float(np.dot(e_prev, e_next) / np.dot(e_prev, e_prev)) if len(e_prev) else 0.0
    sigma = float(np.std(e_next - phi * e_prev)) if len(e_prev) else 0.0
    # hold-speed regression: per-session mean log hold on the session offset s
    lh_sum = np.bincount(sess_of_row[crow], weights=np.log(np.maximum(hold_all[crow], 1.0)), minlength=n_all)
    lh_cnt = np.bincount(sess_of_row[crow], minlength=n_all)
    have = ~np.isnan(s_full) & (lh_cnt > 0)
    x, y = s_full[have], lh_sum[have] / lh_cnt[have]
    hold_slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x)) if have.sum() > 2 and np.var(x) > 0 else 0.0
    hold_intercept = float(y.mean() - hold_slope * x.mean()) if have.sum() else 0.0

    # ---- A4: pauses, Backspace transitions, correction runs --------------------------------------------
    pause = is_cc & ~nonpause
    boundary = is_cc & (pa == SPACE_CP)
    within = is_cc & ~boundary

    def rate(mask_all: np.ndarray, mask_hit: np.ndarray) -> np.ndarray:
        n = np.bincount(sess[mask_all], minlength=n_all).astype("float64")
        k = np.bincount(sess[mask_all & mask_hit], minlength=n_all).astype("float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            r = k / n
        r[n < min_rate] = np.nan
        return r
    p_b, p_w = rate(boundary, pause), rate(within, pause)
    pause_flights = flight[pause & (flight <= params["max_pp_ms"])]
    c2b = ~a_bs & (pa > 0) & b_bs
    b2b = a_bs & b_bs
    b2c = a_bs & (pb > 0)
    bs_pp = {"c2b": _arr(pp[c2b], [250.0]), "b2b": _arr(pp[b2b], [150.0]), "b2c": _arr(pp[b2c], [250.0])}
    brow = in_sel & (keycode == BACKSPACE_KEYCODE)
    bs_hold = _arr(hold_all[brow], [90.0])
    isb = keycode[R] == BACKSPACE_KEYCODE
    sR = sess_of_row[R]
    prev_bs_same = np.r_[False, isb[:-1] & (sR[1:] == sR[:-1])]
    run_start = isb & ~prev_bs_same
    run_id = np.cumsum(run_start)
    lengths = np.bincount(run_id[isb]) if isb.any() else np.zeros(1, dtype=int)
    lengths = lengths[1:]
    short = lengths[(lengths >= 1) & (lengths <= MAX_DELAY + 1)]
    delay_counts = np.bincount(short - 1, minlength=MAX_DELAY + 1).astype("float64") if len(short) else np.array([1.0, 0, 0, 0])
    delay_p = delay_counts / delay_counts.sum()
    frac_long = float((lengths > MAX_DELAY + 1).mean()) if len(lengths) else 0.0
    bs_rows = np.bincount(sess_of_row[brow], minlength=n_all).astype("float64")
    n_chars = np.bincount(sess_of_row[crow], minlength=n_all).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        c_rate = bs_rows / n_chars
    u_rate = np.full(n_all, np.nan)
    for i in idx:
        s_, u_ = bank.sentence(i), bank.user_input(i)
        d = max(len(s_), len(u_))
        u_rate[i] = Levenshtein.distance(s_, u_) / d if d else 0.0

    # ---- Shift holds ------------------------------------------------------------------------------------
    lead, lag, tail = [], [], []
    n_sh = np.zeros(n_all)
    n_ov = np.zeros(n_all)
    for j in np.flatnonzero(in_sel & (keycode == SHIFT_KEYCODE)):
        i = int(sess_of_row[j])
        end = int(bank.offsets[i + 1])
        r = j + 1
        while r < end and (keycode[r] == SHIFT_KEYCODE or press[r] <= press[j]):
            r += 1
        if r >= end or cc[r] <= 0 or press[r] - press[j] > 2000:
            continue
        k1 = r
        k2 = k1 + 1
        while k2 < end and keycode[k2] == SHIFT_KEYCODE:
            k2 += 1
        n_sh[i] += 1
        lead.append(press[k1] - press[j])
        if k2 < end and press[k2] < release[j]:
            n_ov[i] += 1
            tail.append(release[j] - press[k2])
        else:
            lag.append(release[j] - release[k1])
    with np.errstate(invalid="ignore", divide="ignore"):
        pso_all = n_ov / n_sh
    pso_all[n_sh == 0] = np.nan

    # ---- per-session tables ---------------------------------------------------------------------------
    ok = np.zeros(n_all, dtype=bool)
    ok[idx] = True
    rate_ok = ok & (n_chars >= min_pairs) & ~np.isnan(p_b) & ~np.isnan(p_w) & ~np.isnan(c_rate) & ~np.isnan(u_rate)
    rates = np.column_stack([c_rate[rate_ok], u_rate[rate_ok], p_b[rate_ok], p_w[rate_ok]])
    pso = pso_all[ok & ~np.isnan(pso_all)]
    speed = s_series.to_numpy(dtype="float64")
    if not len(rates):
        rates = np.zeros((1, 4))
    stats = PoolStats(
        params=dict(params), n_fit_sessions=int(len(idx)), dg_pp=dg_pp, sec_pp=sec_pp, glob_pp=glob_pp,
        key_hold=key_hold, glob_hold=glob_hold, mu_dg=mu_dg, mu_sec=mu_sec, mu_glob=mu_glob,
        speed=speed if len(speed) else np.zeros(1), speed_mean=float(speed.mean()) if len(speed) else 0.0,
        phi=phi, sigma=sigma, hold_slope=hold_slope, hold_intercept=hold_intercept,
        pause_flights=_arr(pause_flights, [800.0]), bs_pp=bs_pp, bs_hold=bs_hold, delay_p=delay_p, frac_long_runs=frac_long,
        rates=rates, pso=pso if len(pso) else np.zeros(1),
        shift_lead=np.clip(_arr(lead, [60.0]), 1, 2000), shift_lag=_arr(lag, [30.0]), shift_tail=_arr(tail, [50.0]),
        extras={"n_char_pairs": int(is_cc.sum()), "n_nonpause_pairs": int(m.sum()), "n_pause_pairs": int(pause.sum()),
                "n_sessions_with_speed": int(len(speed)), "n_sessions_with_rates": int(len(rates)),
                "n_backspace_runs": int(len(lengths)), "n_shift_holds": int(n_sh.sum()), "n_shift_overlap_holds": int(n_ov.sum()),
                "n_adjacent_residual_pairs": int(len(e_prev))},
    )
    return stats
