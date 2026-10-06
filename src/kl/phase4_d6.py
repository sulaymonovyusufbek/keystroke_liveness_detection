"""D6: key-pair timing lookup model (docs/PHASE4_SPEC.md, A4; pre-registered, humans only, added after the main
evaluation).

Steps are exactly D4's: the first ``max_len`` keys of a session (frozen Phase 3b encoder), key ids from the same
vocabulary (0 = PAD, 1 = UNK), ``h_i = log1p(hold_i)`` at every kept key and ``p_i = log1p(pp_i)`` at every kept key
but the last (where D4 masks ``p``). Values are the RAW log times (no standardisation).

Contexts (a context is used only if it has at least ``N_MIN`` training observations, otherwise the next level):
    hold  h_i : key k_i                   -> global
    press p_i : key pair (k_i, k_{i+1})   -> next key k_{i+1} -> global
An unseen key has 0 observations and therefore always backs off.

Variants (session score = mean step NLL over the D4 steps; higher = less human-like):
    gauss        N(mu_ctx, sd_ctx) per context (sd = population sd, floored at SD_FLOOR); step NLL = NLL(h) + NLL(p)
    hist         60 equal-width bins over the global 0.1%-99.9% range of the TRAINING values (separately for h and p),
                 add-one smoothing, out-of-range values clipped to the edge bins, density = bin probability / bin width
    gauss-speed  z = (x - mu_ctx) / sd_ctx (gauss contexts), minus the session's own mean z (separately for h and p),
                 scored under N(0, 1)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .phase3b.sequences import Encoder, SeqSet, _ranges

N_MIN = 50
SD_FLOOR = 0.05
N_BINS = 60
RANGE_Q = (0.001, 0.999)
VARIANTS: tuple[str, ...] = ("gauss", "hist", "gauss-speed")
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


class D6DataError(ValueError):
    """D6 was asked to fit on something other than training-split humans."""


@dataclass
class Steps:
    """Flattened D4 steps of a set of sessions."""

    sess: np.ndarray     # session row of every step
    key: np.ndarray      # k_i (vocabulary id)
    nxt: np.ndarray      # k_{i+1}, -1 on the last kept key
    h: np.ndarray        # log1p(hold)
    p: np.ndarray        # log1p(pp), NaN on the last kept key
    n_sessions: int

    @property
    def has_p(self) -> np.ndarray:
        """Steps with a press-to-press target."""
        return self.nxt >= 0


def steps_from_seqset(s: SeqSet, enc: Encoder) -> Steps:
    """The steps D4 scores: first ``enc.max_len`` keys, ``p`` masked on the last kept key."""
    lens = np.minimum(s.lengths, enc.max_len).astype("int64")
    rows = _ranges(s.offsets[:-1], lens)
    sess = np.repeat(np.arange(s.n_sessions), lens)
    pos = rows - s.offsets[:-1][sess]
    has_next = pos < (lens[sess] - 1)
    key = enc.key_ids(s.keycode[rows]).astype("int64")
    nxt = np.where(has_next, np.r_[key[1:], -1], -1).astype("int64")
    h = np.log1p(s.hold[rows])
    with np.errstate(invalid="ignore"):
        p = np.where(has_next, np.log1p(s.pp[rows]), np.nan)
    if not np.isfinite(h).all() or not np.isfinite(p[has_next]).all():
        raise ValueError(f"{s.name}: non-finite hold / press-to-press value on a scored step")
    return Steps(sess, key, nxt, h, p, s.n_sessions)


def assert_train_only(fit_users: Iterable[int], train_users: Iterable[int], other_users: Iterable[int]) -> None:
    """The fitting users are TRAIN users only (a subset of ``train_users``, none of ``other_users``)."""
    fu, tu, ou = set(map(int, fit_users)), set(map(int, train_users)), set(map(int, other_users))
    if not fu <= tu:
        raise D6DataError(f"{len(fu - tu)} fitting users are not train users")
    if fu & ou:
        raise D6DataError(f"{len(fu & ou)} fitting users are validation / test / attacker-pool users")


def _stats(ctx: np.ndarray, x: np.ndarray, n_ctx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.bincount(ctx, minlength=n_ctx).astype("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.bincount(ctx, weights=x, minlength=n_ctx) / n
        var = np.bincount(ctx, weights=(x - mu[ctx]) ** 2, minlength=n_ctx) / n
    return n, mu, np.sqrt(var)


@dataclass
class D6Model:
    """Context tables of the three variants."""

    vocab_size: int
    n_h: np.ndarray          # observations per hold context: keys 0..V-1, then global (index V)
    n_p: np.ndarray          # per press context: pairs 0..V^2-1, next keys V^2..V^2+V-1, global V^2+V
    mu_h: np.ndarray
    sd_h: np.ndarray
    mu_p: np.ndarray
    sd_p: np.ndarray
    edges_h: tuple[float, float]
    edges_p: tuple[float, float]
    hist_h: np.ndarray       # (n contexts, N_BINS) smoothed bin probabilities
    hist_p: np.ndarray

    # ------------------------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, steps: Steps, vocab_size: int) -> "D6Model":
        """Fit every context of every level on the TRAINING steps (all observations count at every level)."""
        V = vocab_size
        g_h, g_p = V, V * V + V
        hp = steps.has_p
        hk, ph, pn, pv = steps.key, steps.key[hp], steps.nxt[hp], steps.p[hp]
        # hold: key level + global
        ctx_h = np.r_[hk, np.full(len(hk), g_h)]
        xh = np.r_[steps.h, steps.h]
        n_h, mu_h, sd_h = _stats(ctx_h, xh, V + 1)
        # press: pair level + next-key level + global
        ctx_p = np.r_[ph * V + pn, V * V + pn, np.full(len(pn), g_p)]
        xp = np.r_[pv, pv, pv]
        n_p, mu_p, sd_p = _stats(ctx_p, xp, V * V + V + 1)
        eh = tuple(float(v) for v in np.quantile(steps.h, RANGE_Q))
        ep = tuple(float(v) for v in np.quantile(pv, RANGE_Q))
        hist_h = cls._hist(ctx_h, xh, eh, V + 1)
        hist_p = cls._hist(ctx_p, xp, ep, V * V + V + 1)
        return cls(V, n_h, n_p, mu_h, np.maximum(np.nan_to_num(sd_h, nan=0.0), SD_FLOOR), mu_p,
                   np.maximum(np.nan_to_num(sd_p, nan=0.0), SD_FLOOR), eh, ep, hist_h, hist_p)

    @staticmethod
    def _bins(x: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
        lo, hi = edges
        w = (hi - lo) / N_BINS
        return np.clip(np.floor((x - lo) / w), 0, N_BINS - 1).astype("int64")

    @classmethod
    def _hist(cls, ctx: np.ndarray, x: np.ndarray, edges: tuple[float, float], n_ctx: int) -> np.ndarray:
        c = np.bincount(ctx * N_BINS + cls._bins(x, edges), minlength=n_ctx * N_BINS).reshape(n_ctx, N_BINS).astype("float64")
        return (c + 1.0) / (c.sum(1, keepdims=True) + N_BINS)

    # ------------------------------------------------------------------------------------ contexts
    def hold_context(self, key: np.ndarray) -> np.ndarray:
        """Key if it has >= N_MIN training holds, else global."""
        V = self.vocab_size
        k = np.asarray(key, dtype="int64")
        ok = (k >= 0) & (k < V)
        n = np.where(ok, self.n_h[np.clip(k, 0, V - 1)], 0)
        return np.where(n >= N_MIN, k, V)

    def press_context(self, key: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        """Pair if it has >= N_MIN training observations, else next key if it has, else global."""
        V = self.vocab_size
        k, n_ = np.asarray(key, dtype="int64"), np.asarray(nxt, dtype="int64")
        ok = (k >= 0) & (k < V) & (n_ >= 0) & (n_ < V)
        pair = np.where(ok, k * V + n_, 0)
        nk = np.where(ok, V * V + n_, 0)
        use_pair = ok & (self.n_p[pair] >= N_MIN)
        use_next = ok & ~use_pair & (self.n_p[nk] >= N_MIN)
        return np.where(use_pair, pair, np.where(use_next, nk, V * V + V))

    # ------------------------------------------------------------------------------------ scoring
    def step_nll(self, steps: Steps, variant: str) -> tuple[np.ndarray, np.ndarray]:
        """``(nll_h, nll_p)`` per step (``nll_p`` = 0 where there is no press-to-press target)."""
        hp = steps.has_p
        ch = self.hold_context(steps.key)
        cp = self.press_context(steps.key[hp], steps.nxt[hp])
        nll_p = np.zeros(len(steps.key))
        if variant == "gauss":
            nll_h = gauss_nll(steps.h, self.mu_h[ch], self.sd_h[ch])
            nll_p[hp] = gauss_nll(steps.p[hp], self.mu_p[cp], self.sd_p[cp])
        elif variant == "hist":
            nll_h = self._hist_nll(steps.h, ch, self.hist_h, self.edges_h)
            nll_p[hp] = self._hist_nll(steps.p[hp], cp, self.hist_p, self.edges_p)
        elif variant == "gauss-speed":
            zh = (steps.h - self.mu_h[ch]) / self.sd_h[ch]
            zp = (steps.p[hp] - self.mu_p[cp]) / self.sd_p[cp]
            zh = zh - _session_mean(zh, steps.sess, steps.n_sessions)[steps.sess]
            sp = steps.sess[hp]
            zp = zp - _session_mean(zp, sp, steps.n_sessions)[sp]
            nll_h = HALF_LOG_2PI + 0.5 * zh * zh
            nll_p[hp] = HALF_LOG_2PI + 0.5 * zp * zp
        else:
            raise ValueError(f"unknown D6 variant {variant!r}")
        return nll_h, nll_p

    def _hist_nll(self, x: np.ndarray, ctx: np.ndarray, table: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
        w = (edges[1] - edges[0]) / N_BINS
        return -np.log(table[ctx, self._bins(x, edges)] / w)

    def session_scores(self, steps: Steps, variant: str) -> np.ndarray:
        """Mean step NLL per session (anomaly score: higher = less human-like)."""
        nh, np_ = self.step_nll(steps, variant)
        tot = np.bincount(steps.sess, weights=nh + np_, minlength=steps.n_sessions)
        cnt = np.bincount(steps.sess, minlength=steps.n_sessions)
        return tot / cnt

    def context_usage(self, steps: Steps) -> dict[str, float]:
        """Share of steps scored at each back-off level (diagnostic)."""
        V = self.vocab_size
        ch = self.hold_context(steps.key)
        cp = self.press_context(steps.key[steps.has_p], steps.nxt[steps.has_p])
        return {"hold_key": float(np.mean(ch < V)), "hold_global": float(np.mean(ch == V)),
                "press_pair": float(np.mean(cp < V * V)), "press_next_key": float(np.mean((cp >= V * V) & (cp < V * V + V))),
                "press_global": float(np.mean(cp == V * V + V))}


def gauss_nll(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """-log N(x; mu, sd^2)."""
    z = (x - mu) / sd
    return HALF_LOG_2PI + np.log(sd) + 0.5 * z * z


def _session_mean(v: np.ndarray, sess: np.ndarray, n: int) -> np.ndarray:
    cnt = np.bincount(sess, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, np.bincount(sess, weights=v, minlength=n) / np.maximum(cnt, 1), 0.0)
