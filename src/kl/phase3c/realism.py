"""Keycode realism of bot sessions vs test humans (docs/PHASE3C_SPEC.md, item 1). Pure numpy on raw sequences."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from ..attacks.a1_gaussian import char_to_key
from ..phase3b.sequences import UNK, SeqSet

SHIFT, BKSP, DEL = 16, 8, 46
ENRICH_RATIO = 10.0          # spec: >= 10x more frequent in one group
MIN_SESSION_PREVALENCE = 0.001   # spec: present in >= 0.1% of sessions (of the group where it is more frequent)

KEY_NAMES: dict[int, str] = {8: "Backspace", 9: "Tab", 13: "Enter", 16: "Shift", 17: "Ctrl", 18: "Alt", 20: "CapsLock", 27: "Esc",
                             32: "Space", 33: "PageUp", 34: "PageDown", 35: "End", 36: "Home", 37: "Left", 38: "Up", 39: "Right",
                             40: "Down", 45: "Insert", 46: "Delete", 91: "Meta", 173: "- (FF)", 186: ";", 187: "=", 188: ",",
                             189: "-", 190: ".", 191: "/", 192: "`", 219: "[", 220: "\\", 221: "]", 222: "'", 229: "IME/unmapped",
                             0: "keycode 0"}


def key_name(code: int) -> str:
    """Readable name of a JavaScript keycode."""
    code = int(code)
    if 65 <= code <= 90:
        return chr(code)
    if 48 <= code <= 57:
        return f"digit {chr(code)}"
    return KEY_NAMES.get(code, f"keycode {code}")


def is_letter_or_space(code: np.ndarray) -> np.ndarray:
    """a-z (uppercase codes 65-90) or space."""
    code = np.asarray(code)
    return ((code >= 65) & (code <= 90)) | (code == 32)


def session_index(s: SeqSet) -> np.ndarray:
    """Session number of every flat keystroke."""
    return np.repeat(np.arange(s.n_sessions), s.lengths)


def bigrams(s: SeqSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(first, second, session)`` of every within-session consecutive keycode pair."""
    sess = session_index(s)
    same = sess[1:] == sess[:-1]
    return s.keycode[:-1][same].astype("int64"), s.keycode[1:][same].astype("int64"), sess[:-1][same]


def _dist(tokens: np.ndarray) -> Counter:
    return Counter(tokens.tolist())


def jsd(p: Mapping, q: Mapping) -> float:
    """Jensen-Shannon DIVERGENCE (base 2, in [0, 1]) between two count dictionaries."""
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0) for k in keys], dtype="float64")
    b = np.array([q.get(k, 0) for k in keys], dtype="float64")
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    return float(jensenshannon(a / a.sum(), b / b.sum(), base=2) ** 2)


def _prevalence(tokens: np.ndarray, sess: np.ndarray, n_sessions: int) -> Counter:
    """Share of sessions containing each token."""
    if not len(tokens):
        return Counter()
    pairs = np.unique(np.stack([sess.astype("int64"), tokens.astype("int64")], 1), axis=0)
    c = Counter(pairs[:, 1].tolist())
    return Counter({k: v / n_sessions for k, v in c.items()})


def enriched(h_tok: np.ndarray, h_sess: np.ndarray, n_h: int, b_tok: np.ndarray, b_sess: np.ndarray, n_b: int,
             label, ratio: float = ENRICH_RATIO, min_prev: float = MIN_SESSION_PREVALENCE) -> pd.DataFrame:
    """Tokens at least ``ratio`` x more frequent (share of all tokens) in one group, present in >= ``min_prev`` of that
    group's sessions. A token absent from one group has an infinite ratio."""
    hc, bc = _dist(h_tok), _dist(b_tok)
    hp, bp = _prevalence(h_tok, h_sess, n_h), _prevalence(b_tok, b_sess, n_b)
    nh, nb = max(len(h_tok), 1), max(len(b_tok), 1)
    rows = []
    for k in set(hc) | set(bc):
        fh, fb = hc.get(k, 0) / nh, bc.get(k, 0) / nb
        r = np.inf if fh == 0 else fb / fh
        if r >= ratio and bp.get(k, 0) >= min_prev:
            rows.append((k, "bots", fh, fb, r, hp.get(k, 0.0), bp.get(k, 0.0)))
        elif (np.inf if fb == 0 else fh / fb) >= ratio and hp.get(k, 0) >= min_prev:
            rows.append((k, "humans", fh, fb, np.inf if fb == 0 else fh / fb, hp.get(k, 0.0), bp.get(k, 0.0)))
    df = pd.DataFrame(rows, columns=["token", "more_in", "freq_humans", "freq_bots", "ratio", "sessions_humans", "sessions_bots"])
    df.insert(1, "name", [label(t) for t in df["token"]])
    return df.sort_values(["more_in", "ratio", "freq_humans", "freq_bots"], ascending=[True, False, False, False]).reset_index(drop=True)


def shift_stats(s: SeqSet, sentences: Sequence[str]) -> dict[str, float]:
    """Shift pattern: presses per capital letter of the target sentence (and per Shift-requiring character), what the
    next non-Shift key is, when Shift is released relative to it, and the Shift hold."""
    lens, starts = s.lengths, s.offsets[:-1]
    pp0 = np.nan_to_num(s.pp, nan=0.0)
    sess = session_index(s)
    csum = np.cumsum(pp0)
    press = csum - pp0 - np.repeat(csum[starts] - pp0[starts], lens)          # press_i - press_first, per session
    release = press + s.hold
    n_caps = sum(sum(ch.isupper() for ch in t) for t in sentences)
    n_shift_chars = sum(sum(char_to_key(ch)[1] for ch in t) for t in sentences)
    is_shift = s.keycode == SHIFT
    idx = np.flatnonzero(is_shift)
    nxt_code, lead, rel_class = [], [], []
    ends = s.offsets[1:]
    for i in idx:                                                              # next non-Shift key of the same session
        j = i + 1
        e = ends[sess[i]]
        while j < e and s.keycode[j] == SHIFT:
            j += 1
        if j >= e:
            nxt_code.append(-1)
            continue
        nxt_code.append(int(s.keycode[j]))
        lead.append(press[j] - press[i])
        rel_class.append(0 if release[i] < press[j] else (1 if release[i] <= release[j] else 2))
    nxt_code = np.asarray(nxt_code)
    rel_class = np.asarray(rel_class)
    n_sh = len(idx)
    holds = s.hold[is_shift]
    frac = (lambda m: float(m.mean()) if len(m) else float("nan"))
    return {
        "n_shift": n_sh, "n_capitals": n_caps, "n_shift_chars": n_shift_chars,
        "shift_per_capital": n_sh / n_caps if n_caps else float("nan"),
        "shift_per_shift_char": n_sh / n_shift_chars if n_shift_chars else float("nan"),
        "sessions_with_shift": float(np.mean(np.add.reduceat(is_shift.astype("int64"), starts) > 0)) if s.n_sessions else float("nan"),
        "next_is_letter": frac((nxt_code >= 65) & (nxt_code <= 90)),
        "next_is_other_key": frac((nxt_code >= 0) & ~((nxt_code >= 65) & (nxt_code <= 90))),
        "shift_is_last_key": frac(nxt_code < 0),
        "released_before_next_press": frac(rel_class == 0),
        "released_during_next_key": frac(rel_class == 1),
        "released_after_next_release": frac(rel_class == 2),
        "lead_ms_median": float(np.median(lead)) if lead else float("nan"),
        "hold_ms_median": float(np.median(holds)) if n_sh else float("nan"),
        "hold_ms_p10": float(np.percentile(holds, 10)) if n_sh else float("nan"),
        "hold_ms_p90": float(np.percentile(holds, 90)) if n_sh else float("nan"),
    }


def group_summary(s: SeqSet, sentences: Sequence[str], unk_of) -> dict[str, float]:
    """Length, UNK share, Backspace / Delete and CapsLock use of one group."""
    sess_counts = lambda code: np.add.reduceat((s.keycode == code).astype("int64"), s.offsets[:-1]) if s.n_sessions else np.zeros(0)
    bk, de, caps = sess_counts(BKSP), sess_counts(DEL), sess_counts(20)
    n_keys = len(s.keycode)
    return {"n_sessions": int(s.n_sessions), "n_keys": int(n_keys),
            "len_mean": float(s.lengths.mean()), "len_median": float(np.median(s.lengths)),
            "len_p5": float(np.percentile(s.lengths, 5)), "len_p95": float(np.percentile(s.lengths, 95)),
            "unk_pct": 100.0 * float(np.mean(unk_of(s.keycode) == UNK)) if n_keys else float("nan"),
            "backspace_per_100_keys": 100.0 * bk.sum() / n_keys, "delete_per_100_keys": 100.0 * de.sum() / n_keys,
            "sessions_with_backspace_or_delete": float(np.mean((bk + de) > 0)),
            "backspace_delete_per_session": float((bk + de).mean()),
            "capslock_per_100_keys": 100.0 * caps.sum() / n_keys,
            "non_letter_space_share": float(np.mean(~is_letter_or_space(s.keycode))),
            **{f"shift_{k}": v for k, v in shift_stats(s, sentences).items()}}


def other_key_rates(groups: Mapping[str, SeqSet], min_prev: float = MIN_SESSION_PREVALENCE) -> pd.DataFrame:
    """Every keycode that is not a letter or space: rate per 1,000 keystrokes in each group (keycodes present in
    >= ``min_prev`` of the sessions of at least one group)."""
    rates, prev = {}, {}
    for g, s in groups.items():
        m = ~is_letter_or_space(s.keycode)
        c = _dist(s.keycode[m].astype("int64"))
        rates[g] = {k: 1000.0 * v / len(s.keycode) for k, v in c.items()}
        prev[g] = _prevalence(s.keycode[m], session_index(s)[m], s.n_sessions)
    codes = sorted({k for g in groups for k, p in prev[g].items() if p >= min_prev})
    df = pd.DataFrame({"keycode": codes, "name": [key_name(k) for k in codes]})
    for g in groups:
        df[g] = [rates[g].get(k, 0.0) for k in codes]
    first = next(iter(groups))
    return df.sort_values(first, ascending=False).reset_index(drop=True)


def realism_tables(humans: SeqSet, bots: Mapping[str, SeqSet], sentence_of: Mapping[str, str], unk_of) -> dict[str, pd.DataFrame]:
    """``summary`` (one row per group), ``jsd`` (per attacker), ``enriched_keys`` / ``enriched_bigrams`` (per attacker),
    ``other_keys`` (non-letter keycode rates)."""
    groups = {"humans": humans, **bots}
    summary = pd.DataFrame([{"group": g, **group_summary(s, [sentence_of[i] for i in s.session_ids], unk_of)} for g, s in groups.items()])
    h_sess = session_index(humans)
    h1, h2, hbs = bigrams(humans)
    h_bi = h1 * 1000 + h2
    jrows, ek, eb = [], [], []
    for a, b in bots.items():
        b1, b2, bbs = bigrams(b)
        b_bi = b1 * 1000 + b2
        jrows.append({"attacker": a, "jsd_keycode": jsd(_dist(humans.keycode.astype("int64")), _dist(b.keycode.astype("int64"))),
                      "jsd_bigram": jsd(_dist(h_bi), _dist(b_bi))})
        k = enriched(humans.keycode.astype("int64"), h_sess, humans.n_sessions, b.keycode.astype("int64"), session_index(b), b.n_sessions, key_name)
        ek.append(k.assign(attacker=a))
        g = enriched(h_bi, hbs, humans.n_sessions, b_bi, bbs, b.n_sessions, lambda t: f"{key_name(t // 1000)} -> {key_name(t % 1000)}")
        eb.append(g.assign(attacker=a))
    return {"summary": summary, "jsd": pd.DataFrame(jrows), "enriched_keys": pd.concat(ek, ignore_index=True),
            "enriched_bigrams": pd.concat(eb, ignore_index=True), "other_keys": other_key_rates(groups)}
