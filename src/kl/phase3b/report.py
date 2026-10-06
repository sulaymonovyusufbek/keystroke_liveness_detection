"""Figures and RESULTS_PHASE3B.md for Phase 3b (reads only the tables / score files written by steps 30-32)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import ROOT, Settings, read_json  # noqa: E402
from ..phase2.evaluation import HEADERS  # noqa: E402

TESTS = ["A1", "A2", "A3", "A4", "A5a", "A5b"]
TRAIN_ATK = ["A1", "A2", "A3", "A4", "A5b"]


# ------------------------------------------------------------------------------------------------ formatting
def pct(v: float) -> str:
    """Percentage with one decimal."""
    return "—" if not np.isfinite(v) else f"{100 * v:.1f}%"


def cell(row: Mapping[str, Any], m: str) -> str:
    """``mean ± sd [seed-42 CI]`` (neural) or ``value [CI]`` (Phase 2); percentages except AUC."""
    mean, sd, lo, hi = (float(row.get(f"{m}_{k}", np.nan)) for k in ("mean", "sd", "ci_lo", "ci_hi"))
    if m == "auc":
        f = lambda x: f"{x:.3f}"  # noqa: E731
    else:
        f = lambda x: f"{100 * x:.1f}"  # noqa: E731
    unit = "" if m == "auc" else "%"
    core = f"{f(mean)}{unit}" + (f" ± {f(sd)}" if np.isfinite(sd) else "")
    return core + (f" [{f(lo)}, {f(hi)}]" if np.isfinite(lo) else "")


def _row(tab: pd.DataFrame, **eq: str) -> pd.Series | None:
    m = np.ones(len(tab), dtype=bool)
    for k, v in eq.items():
        m &= tab[k].astype(str).to_numpy() == v
    return tab.loc[m].iloc[0] if m.any() else None


def md_table(tab: pd.DataFrame, label_cols: Sequence[str], metrics: Sequence[str]) -> list[str]:
    """Markdown table with both naming conventions in the headers."""
    head = "| " + " | ".join(label_cols) + " | " + " | ".join(HEADERS[m] for m in metrics) + " |"
    lines = [head, "|" + "---|" * (len(label_cols) + len(metrics))]
    for _, r in tab.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in label_cols) + " | " + " | ".join(cell(r, m) for m in metrics) + " |")
    return lines + [""]


# ------------------------------------------------------------------------------------------------ figures
def plot_training_curves(curves: pd.DataFrame, path: Path, title: str, ylabel: str) -> None:
    """One panel per (detector, trained_on); train (dashed) and val (solid) loss per epoch for every seed."""
    groups = list(curves.groupby(["detector", "trained_on"], sort=False))
    if not groups:
        return
    n = len(groups)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.3 * rows), squeeze=False)
    for ax, ((det, tr), g) in zip(axes.ravel(), groups):
        for i, (seed, gs) in enumerate(g.groupby("seed")):
            c = f"C{i}"
            ax.plot(gs["epoch"], gs["train_loss"], linestyle="--", color=c, alpha=0.8, label=f"train s{seed}")
            ax.plot(gs["epoch"], gs["val_loss"], linestyle="-", color=c, label=f"val s{seed}")
            best = gs.loc[gs["val_loss"].idxmin()]
            ax.plot(best["epoch"], best["val_loss"], marker="o", color=c)
        ax.set_title(f"{det} ← {tr}", fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, ncol=2)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_d4_hist(z: Mapping[str, np.ndarray], attackers: Sequence[str], path: Path, title: str) -> None:
    """Histograms of the D4-primary anomaly score (mean per-step NLL): test humans vs each attacker; tau10 marked."""
    hum = z["test_m"]
    allv = np.concatenate([hum] + [z[f"bot_m_{a}"] for a in attackers])
    lo, hi = np.quantile(allv, [0.001, 0.995])
    bins = np.linspace(lo, hi, 80)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), squeeze=False)
    for ax, a in zip(axes.ravel(), attackers):
        ax.hist(np.clip(hum, lo, hi), bins=bins, density=True, alpha=0.5, color="C0", label="test humans")
        ax.hist(np.clip(z[f"bot_m_{a}"], lo, hi), bins=bins, density=True, alpha=0.5, color="C3", label=a)
        ax.axvline(-float(z["tau10_primary"]), color="k", linestyle=":", label="τ10 (reject right of it)")
        ax.set_title(a, fontsize=10)
        ax.set_xlabel("mean per-step NLL (higher = more anomalous)")
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def e5_matrices(e5: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(D5 mean over seeds, D3) matrices: rows = trained on, columns = tested on."""
    out = []
    for det in ("D5", "D3"):
        sub = e5[e5["detector"] == det]
        out.append(pd.DataFrame({t: [float(_row(sub, trained_on=r, tested_on=t)[f"{metric}_mean"]) if _row(sub, trained_on=r, tested_on=t) is not None else np.nan
                                     for r in TRAIN_ATK] for t in TESTS}, index=TRAIN_ATK))
    return out[0], out[1]


def plot_e5_heatmaps(e5: pd.DataFrame, path: Path) -> None:
    """D5 (mean over seeds) next to D3 (Phase 2): APCER@tau10 and ROC-AUC."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for i, (metric, lab, vmin, vmax, cmap, pct_) in enumerate((("apcer_fpr_tau10", "APCER / FPR at τ10 (attacks accepted)", 0, 1, "YlOrRd", True),
                                                               ("auc", "ROC-AUC (human vs attack)", 0.3, 1, "YlGnBu", False))):
        d5, d3 = e5_matrices(e5, metric)
        for j, (mat, name) in enumerate(((d5, "D5 (BiGRU, mean of 3 seeds)"), (d3, "D3 (Random Forest, Phase 2)"))):
            ax = axes[i, j]
            im = ax.imshow(mat.to_numpy(dtype="float64"), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(TESTS)), TESTS)
            ax.set_yticks(range(len(TRAIN_ATK)), TRAIN_ATK)
            ax.set_xlabel("tested on")
            ax.set_ylabel("trained on")
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    v = mat.iat[r, c]
                    if np.isfinite(v):
                        dark = (v - vmin) / (vmax - vmin) > 0.6
                        ax.text(c, r, f"{100 * v:.1f}" if pct_ else f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if dark else "black")
            ax.set_title(f"{name}: {lab}", fontsize=9)
            fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ RESULTS_PHASE3B.md
def comparison_table(e4: pd.DataFrame | None, e5: pd.DataFrame | None, e6: pd.DataFrame | None, metric: str) -> list[str]:
    """Session-level (D1, D2, D3) vs sequence-level (D4, D5): one row per detector setting, one column per attacker."""
    spec = [("session", "D1 (OCSVM), attack-free", e4, "D1", "attack-free", None),
            ("session", "D2 (Isolation Forest), attack-free", e4, "D2", "attack-free", None),
            ("sequence", "D4-primary, attack-free", e4, "D4-primary", "attack-free", None),
            ("sequence", "D4-typicality, attack-free", e4, "D4-typicality", "attack-free", None),
            ("session", "D3 trained on the same attacker (E2 diagonal)", e5, "D3", "same", None),
            ("sequence", "D5 trained on the same attacker (E5 diagonal)", e5, "D5", "same", None),
            ("session", "D3, attacker unseen in training (E3)", e6, "D3", "loao", None),
            ("sequence", "D5, attacker unseen in training (E6)", e6, "D5", "loao", None)]
    lines = ["| level | detector / setting | " + " | ".join(TESTS) + " |", "|---|---|" + "---|" * len(TESTS)]
    for level, label, tab, det, tr, _ in spec:
        if tab is None or tab.empty:
            continue
        vals = []
        for a in TESTS:
            trained = {"same": a, "loao": f"all-but-{a}"}.get(tr, tr)
            r = _row(tab, detector=det, trained_on=trained, tested_on=a)
            if r is None:
                vals.append("—")
                continue
            m, sd = float(r[f"{metric}_mean"]), float(r.get(f"{metric}_sd", np.nan))
            txt = f"{m:.3f}" if metric == "auc" else f"{100 * m:.1f}%"
            if np.isfinite(sd):
                txt += f" ± {sd:.3f}" if metric == "auc" else f" ± {100 * sd:.1f}"
            vals.append(txt)
        lines.append(f"| {level} | {label} | " + " | ".join(vals) + " |")
    return lines + [""]


def _num(tab: pd.DataFrame | None, metric: str, **eq: str) -> float:
    if tab is None:
        return float("nan")
    r = _row(tab, **eq)
    return float(r[f"{metric}_mean"]) if r is not None else float("nan")


def write_results_md(st: Settings, out: Path) -> None:
    """RESULTS_PHASE3B.md from the Phase 3b tables and logs (partial results are written as they become available)."""
    rd = lambda n: pd.read_csv(out / n) if (out / n).exists() else None  # noqa: E731
    e4, e5, e6 = rd("e4_open_set.csv"), rd("e5_cross_attack.csv"), rd("e6_loao.csv")
    seq = read_json(out / "sequences_meta.json", {}) or {}
    ver = read_json(out / "bot_verification.json", {}) or {}
    rt = read_json(out / "runtime_log.json", {}) or {}
    tr = rd("truncation_report.csv")
    enc = seq.get("encoder", {})
    dev = next(iter(rt.get("device", {}).values()), {})
    runs = rt.get("runs", {})
    L: list[str] = [
        "# Phase 3b — sequence-level detectors (results)", "",
        f"_Run mode **{'quick' if st.quick else 'full'}**; Phase 1 config hash `{st.config_hash}` (frozen); design pre-registered in "
        "`docs/PHASE3B_PREREGISTRATION.md` (hash verified at the start and end of every step). Terminology: **APCER = FPR = attacks accepted**, "
        "**BPCER = FNR = humans rejected**; neural cells are *mean ± sd over training seeds 42/43/44 [95% user-bootstrap CI of seed 42]*; "
        "Phase 2 rows (D1, D2, D3) are a single deterministic model: *value [95% CI]*. AUC < 0.5 is reported as it is._", "",
        "## Data and set-up", "",
    ]
    hs = seq.get("human_sets", {})
    if hs:
        L.append("Human sessions: " + ", ".join(f"{k} {v['n_sessions']:,} ({v['n_users']:,} users)" for k, v in hs.items()) + ".")
    if enc:
        L.append(f"Sequences are truncated at **max_len = {enc['max_len']}** keystrokes (99th percentile of train-user human session length); "
                 f"vocabulary {enc['vocab_size']} ids ({enc['info']['n_keycodes_kept']} keycodes with ≥ {enc['info']['min_count']} train occurrences + PAD + UNK); "
                 f"standardisation log1p(hold) {enc['mu_h']:.3f} ± {enc['sd_h']:.3f}, log1p(pp) {enc['mu_p']:.3f} ± {enc['sd_p']:.3f}.")
    if ver:
        bots = ver.get("bots", {})
        n_ver = sum(1 for v in bots.values() if v.get("ok") is True)
        L.append(f"Bot keystrokes were regenerated with the frozen Phase 2 attackers and seeds: **{n_ver} of {sum(1 for v in bots.values() if v.get('ok') is not None)} "
                 f"train/test sets reproduce the stored Phase 2 features exactly** (tolerance 1e-9, observed max |diff| "
                 f"{max((v.get('max_abs_diff') or 0.0) for v in bots.values()):.1g}); val bots of A1–A4 and A5b (new, used only for D5 early stopping) were generated with the same scheme.")
    if tr is not None:
        L += ["", "Truncated sessions (first max_len keystrokes kept):", "", "| set | sessions | truncated | fraction |", "|---|---|---|---|"]
        L += [f"| {r.set} | {r.n_sessions:,} | {r.n_truncated:,} | {100 * r.frac_truncated:.2f}% |" for r in tr.itertuples()]
    if dev:
        L += ["", f"Device: **{dev.get('device')}** ({dev.get('gpu_name') or 'CPU'}), torch {dev.get('torch_version')}, CUDA {dev.get('cuda_version')}, "
              f"cuDNN {dev.get('cudnn_version')}; deterministic algorithms on."]
    L.append("")

    # ---- E4
    L += ["## E4 — open set: D4 (one-class, humans only) vs the frozen D1 / D2", ""]
    if e4 is not None and (e4["detector"] == "D4-primary").any():
        order = {d: i for i, d in enumerate(["D1", "D2", "D4-primary", "D4-typicality", "D4-20k", "D4-nokey"])}
        L += md_table(e4.assign(order=e4["detector"].map(order)).sort_values(["tested_on", "order"]), ["detector", "tested_on"], ["apcer_fpr_tau5", "apcer_fpr_tau10", "bpcer_fnr_tau5", "bpcer_fnr_tau10", "auc"])
        par = []
        for a in TESTS:
            d4, d1 = _num(e4, "apcer_fpr_tau10", detector="D4-primary", tested_on=a), _num(e4, "apcer_fpr_tau10", detector="D1", tested_on=a)
            d4t = _num(e4, "apcer_fpr_tau10", detector="D4-typicality", tested_on=a)
            auc4, auc1 = _num(e4, "auc", detector="D4-primary", tested_on=a), _num(e4, "auc", detector="D1", tested_on=a)
            par.append(f"{a}: D4-primary accepts {pct(d4)} of attacks at τ10 (AUC {auc4:.3f}) vs D1 {pct(d1)} (AUC {auc1:.3f}); D4-typicality {pct(d4t)}")
        L += ["At the τ10 operating point (10% of validation humans rejected) — " + "; ".join(par) + ". "
              "D4-20k (data-size control) and D4-nokey (timing only, no key conditioning) are in the table; "
              + "; ".join(f"{a}: D4 {pct(_num(e4, 'apcer_fpr_tau10', detector='D4-primary', tested_on=a))}, D4-20k {pct(_num(e4, 'apcer_fpr_tau10', detector='D4-20k', tested_on=a))}, "
                          f"D4-nokey {pct(_num(e4, 'apcer_fpr_tau10', detector='D4-nokey', tested_on=a))}" for a in TESTS) + ".", ""]
    else:
        L += ["_Not run yet._", ""]

    # ---- E5
    L += ["## E5 — cross-attack matrix: D5 (supervised BiGRU) next to D3 (Random Forest, Phase 2)", ""]
    if e5 is not None and (e5["detector"] == "D5").any():
        for metric, lab in (("apcer_fpr_tau10", "APCER / FPR @τ10 (attacks accepted)"), ("auc", "ROC-AUC")):
            d5, d3 = e5_matrices(e5, metric)
            fmt = (lambda v: f"{v:.3f}") if metric == "auc" else (lambda v: f"{100 * v:.1f}%")
            L += [f"**{lab}** — rows: trained on, columns: tested on; each cell `D5 mean / D3`.", "", "| trained on | " + " | ".join(TESTS) + " |", "|---|" + "---|" * len(TESTS)]
            for r in TRAIN_ATK:
                L.append(f"| {r} | " + " | ".join(f"{fmt(d5.at[r, t])} / {fmt(d3.at[r, t])}" if np.isfinite(d5.at[r, t]) else "—" for t in TESTS) + " |")
            L.append("")
        d5a, d3a = e5_matrices(e5, "apcer_fpr_tau10")
        diag5 = {a: d5a.at[a, a] for a in TRAIN_ATK}
        L += ["Full cells with sd and the seed-42 CI are in `e5_cross_attack.csv`. On the diagonal (trained and tested on the same attacker) D5 accepts "
              + ", ".join(f"{a} {pct(v)} (D3 {pct(d3a.at[a, a])})" for a, v in diag5.items()) + " at τ10. "
              f"Averaged over the off-diagonal cells, D5 accepts {pct(float(np.nanmean([d5a.at[r, t] for r in TRAIN_ATK for t in TESTS if t != r])))} of attacks "
              f"(D3 {pct(float(np.nanmean([d3a.at[r, t] for r in TRAIN_ATK for t in TESTS if t != r])))}); against the verbatim replay A5a D5 accepts "
              + ", ".join(f"{pct(d5a.at[r, 'A5a'])} (←{r})" for r in TRAIN_ATK) + ".", ""]
    else:
        L += ["_Not run yet._", ""]

    # ---- E6
    L += ["## E6 — leave one attacker out: D5 next to D3 (E3)", ""]
    if e6 is not None and (e6["detector"] == "D5").any():
        sub = e6[e6["detector"].isin(["D5", "D3", "D1", "D2"])].copy()
        sub["order"] = sub["detector"].map({"D5": 0, "D3": 1, "D1": 2, "D2": 3})
        L += md_table(sub.sort_values(["tested_on", "order"]), ["detector", "trained_on", "tested_on"], ["apcer_fpr_tau5", "apcer_fpr_tau10", "bpcer_fnr_tau10", "auc"])
        L += ["For each held-out attacker, APCER at τ10 — " + "; ".join(
            f"{a}: D5 {pct(_num(e6, 'apcer_fpr_tau10', detector='D5', tested_on=a))} vs D3 {pct(_num(e6, 'apcer_fpr_tau10', detector='D3', tested_on=a))} "
            f"(AUC {_num(e6, 'auc', detector='D5', tested_on=a):.3f} vs {_num(e6, 'auc', detector='D3', tested_on=a):.3f})" for a in TRAIN_ATK) + ".", ""]
    else:
        L += ["_Not run yet._", ""]

    # ---- comparison
    L += ["## Session-level (D1, D2, D3) vs sequence-level (D4, D5)", "",
          "**APCER / FPR at τ10** (attacks accepted; lower is better; BPCER ≈ 10% by construction for every row):", ""]
    L += comparison_table(e4, e5, e6, "apcer_fpr_tau10")
    L += ["**ROC-AUC** (human vs attack; 0.5 = chance; < 0.5 = attacks look *more* human than humans):", ""]
    L += comparison_table(e4, e5, e6, "auc")

    # ---- runtime + training
    if runs:
        L += ["## Training and runtime", "", "| run | train set | epochs run | best epoch | best val loss | stopped early | train time |", "|---|---|---|---|---|---|---|"]
        for k, v in runs.items():
            ts = v.get("train_set") or f"{v.get('n_train_humans', 0):,} humans + {v.get('n_train_bots', 0):,} bots"
            best = v.get("best_val_nll", v.get("best_val_loss"))
            L.append(f"| {k} | {ts} | {v['epochs_run']} | {v['best_epoch']} | {best:.5f} | {v['stopped_early']} | {v['train_seconds'] / 60:.1f} min |")
        parts = rt.get("parts", {})
        L += ["", "Wall-clock per part: " + ", ".join(f"{k} {v['seconds'] / 60:.1f} min" for k, v in parts.items()) + ".", ""]

    # ---- bug-fix log
    log_path = ROOT / "docs" / "phase3b_bugfixes.md"
    L += ["## Bug fixes and incidents", ""]
    if log_path.exists():
        body = log_path.read_text(encoding="utf-8").splitlines()
        L += [ln.replace("## ", "### ", 1) if ln.startswith("## ") else ln for ln in body if not ln.startswith("# ")]
    (out / "RESULTS_PHASE3B.md").write_text("\n".join(L) + "\n", encoding="utf-8")
