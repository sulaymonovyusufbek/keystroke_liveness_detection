"""PyTorch models of Phase 3b (pre-registered in docs/PHASE3B_PREREGISTRATION.md): D4 and D5, their losses, training
with early stopping, and scoring.

D4  causal GRU over ``[E(key_i), h_{i-1}, p_{i-1}]``; the head sees ``[o_i, E(key_{i+1})]`` and outputs a K-component
    diagonal bivariate Gaussian mixture over ``(h_i, p_i)``; the last step (``p`` masked) uses the marginal over ``h``.
D5  bidirectional GRU over ``[E(key_i), h_i, p_i]``, masked mean pooling, linear -> logit (1 = human).
All recurrences run on packed sequences, so padding never enters them.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede the first CUDA call (deterministic cuBLAS)

import copy  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Any, Callable, Iterator, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  # noqa: E402

from .sequences import EncodedSet  # noqa: E402

LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------------------------- determinism / device
def set_determinism(seed: int) -> None:
    """Seed python / numpy / torch and force deterministic kernels (errors instead of silent non-determinism)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    """CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info(device: torch.device) -> dict[str, Any]:
    """Versions and hardware for the runtime log."""
    info = {"device": str(device), "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(), "gpu_name": None}
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 2 ** 30, 2)
    return info


def to_device(batch: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Numpy batch -> tensors (``lens`` stays on the CPU, as ``pack_padded_sequence`` requires)."""
    out = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in batch.items() if k != "lens"}
    out["lens"] = torch.from_numpy(np.asarray(batch["lens"], dtype="int64"))
    return out


# ---------------------------------------------------------------------------------------------- mixture density
def normal_logpdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Elementwise log N(x; mu, sigma^2)."""
    z = (x - mu) / sigma
    return -0.5 * z * z - torch.log(sigma) - 0.5 * LOG_2PI


def mixture_nll(h: torch.Tensor, p: torch.Tensor, pmask: torch.Tensor, logits: torch.Tensor, mu_h: torch.Tensor,
                mu_p: torch.Tensor, sig_h: torch.Tensor, sig_p: torch.Tensor) -> torch.Tensor:
    """Per-step NLL of a K-component diagonal bivariate Gaussian mixture.

    ``h, p, pmask``: ``(..., )``; mixture parameters ``(..., K)``. Where ``pmask`` is False only ``h`` is scored, with the
    exact marginal of the mixture.
    """
    lp = torch.log_softmax(logits, dim=-1) + normal_logpdf(h.unsqueeze(-1), mu_h, sig_h)
    lp = lp + torch.where(pmask.unsqueeze(-1), normal_logpdf(p.unsqueeze(-1), mu_p, sig_p), torch.zeros_like(lp))
    return -torch.logsumexp(lp, dim=-1)


# ---------------------------------------------------------------------------------------------- models
def _run_gru(gru: nn.GRU, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    packed = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
    out, _ = gru(packed)
    out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
    return out


class D4Model(nn.Module):
    """Autoregressive human-timing model (one-class)."""

    def __init__(self, vocab_size: int, emb: int = 32, hidden: int = 128, layers: int = 2, dropout: float = 0.1,
                 k: int = 5, sigma_floor: float = 0.01, use_keys: bool = True) -> None:
        super().__init__()
        self.k, self.sigma_floor, self.use_keys = k, sigma_floor, use_keys
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0) if use_keys else None
        e = emb if use_keys else 0
        self.gru = nn.GRU(e + 2, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + e, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 5 * k))

    def step_nll(self, b: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """``(nll, valid)``: per-step NLL ``(B, L)`` (0 where not valid) and the validity mask."""
        keys, h, p, pmask, lens = b["keys"], b["h"], b["p"], b["pmask"], b["lens"]
        B, L = h.shape
        valid = torch.arange(L, device=h.device)[None, :] < lens.to(h.device)[:, None]
        zero = torch.zeros(B, 1, device=h.device, dtype=h.dtype)
        prev = torch.stack([torch.cat([zero, h[:, :-1]], 1), torch.cat([zero, p[:, :-1]], 1)], dim=-1)
        if self.use_keys:
            e = self.emb(keys)
            x = torch.cat([e, prev], dim=-1)
            e_next = torch.cat([e[:, 1:], torch.zeros_like(e[:, :1])], dim=1)
            # the next key of the last VALID step is PAD (zero): positions beyond a session's length are PAD already
            e_next = e_next * (torch.arange(L, device=h.device)[None, :, None] < (lens.to(h.device) - 1)[:, None, None])
        else:
            x = prev
        o = _run_gru(self.gru, x, lens)
        z = torch.cat([o, e_next], dim=-1) if self.use_keys else o
        out = self.head(z)
        k = self.k
        logits, mu_h, mu_p = out[..., :k], out[..., k:2 * k], out[..., 2 * k:3 * k]
        sig_h = F.softplus(out[..., 3 * k:4 * k]) + self.sigma_floor
        sig_p = F.softplus(out[..., 4 * k:]) + self.sigma_floor
        nll = mixture_nll(h, p, pmask, logits, mu_h, mu_p, sig_h, sig_p)
        return torch.where(valid, nll, torch.zeros_like(nll)), valid

    def loss(self, b: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        """Token-level mean NLL of the batch and its number of target steps."""
        nll, valid = self.step_nll(b)
        n = int(valid.sum())
        return nll.sum() / n, n

    @torch.no_grad()
    def session_scores(self, b: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Mean per-step NLL of every session (anomaly: higher = more anomalous)."""
        nll, valid = self.step_nll(b)
        return nll.sum(1) / valid.sum(1)


class D5Model(nn.Module):
    """Supervised bidirectional sequence classifier (logit: higher = more human)."""

    def __init__(self, vocab_size: int, emb: int = 32, hidden: int = 128, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.gru = nn.GRU(emb + 2, hidden, num_layers=layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 1)

    def logits(self, b: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``(B,)`` logits."""
        keys, h, p, lens = b["keys"], b["h"], b["p"], b["lens"]
        x = torch.cat([self.emb(keys), h.unsqueeze(-1), p.unsqueeze(-1)], dim=-1)   # masked p is 0 already
        o = _run_gru(self.gru, x, lens)                                              # zeros beyond each length
        pooled = o.sum(1) / lens.to(o.device, o.dtype)[:, None]
        return self.out(pooled).squeeze(-1)


# ---------------------------------------------------------------------------------------------- training
@dataclass
class TrainResult:
    """Outcome of one training run."""

    curve: list[dict[str, float]] = field(default_factory=list)   # epoch, train_loss, val_loss, seconds
    best_epoch: int = 0
    best_val: float = float("inf")
    epochs_run: int = 0
    seconds: float = 0.0            # training wall time without thermal pauses
    paused_seconds: float = 0.0     # time spent in ``between_epochs`` (thermal guard); no computation happens there
    stopped_early: bool = False


def fit(model: nn.Module, epoch_batches: Callable[[int], Iterator[tuple[torch.Tensor, int]]], val_loss: Callable[[], float],
        lr: float, max_epochs: int, patience: int, clip: float, log: Callable[[str], None] | None = None,
        between_epochs: Callable[[int], float] | None = None) -> TrainResult:
    """Adam + gradient clipping; early stopping on ``val_loss`` (patience on strict improvement); best weights restored.

    ``epoch_batches(epoch)`` yields ``(loss, weight)`` for every training batch of that epoch (forward already done in
    train mode); the reported train loss is the weight-averaged batch loss. ``between_epochs(epoch)`` (the thermal guard)
    runs after every epoch and returns the seconds it slept; it must not touch the model, data or RNGs.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    res, best_state, bad, t0 = TrainResult(), None, 0, time.time()
    for epoch in range(1, max_epochs + 1):
        te = time.time()
        model.train()
        tot, wsum = 0.0, 0
        for loss, w in epoch_batches(epoch):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += float(loss.detach()) * w
            wsum += w
        model.eval()
        with torch.no_grad():
            v = float(val_loss())
        res.curve.append({"epoch": epoch, "train_loss": tot / wsum, "val_loss": v, "seconds": time.time() - te})
        res.epochs_run = epoch
        if log:
            log(f"epoch {epoch:2d}: train {tot / wsum:.5f}  val {v:.5f}  ({time.time() - te:.1f}s)")
        if v < res.best_val:
            res.best_val, res.best_epoch, bad = v, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if between_epochs is not None:
            res.paused_seconds += float(between_epochs(epoch))
        if bad >= patience:
            res.stopped_early = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    res.seconds = time.time() - t0 - res.paused_seconds
    return res


def _chunks(order: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for k in range(0, len(order), size):
        yield order[k:k + size]


def d4_epoch_batches(model: D4Model, data: EncodedSet, batch_size: int, gen: torch.Generator, device: torch.device):
    """Epoch iterator for D4: one pass over ``data`` in a seeded random order (last partial batch kept)."""
    def it(epoch: int) -> Iterator[tuple[torch.Tensor, int]]:
        order = torch.randperm(len(data), generator=gen).numpy()
        for idx in _chunks(order, batch_size):
            loss, n = model.loss(to_device(data.batch(idx), device))
            yield loss, n
    return it


@torch.no_grad()
def d4_mean_nll(model: D4Model, data: EncodedSet, batch_size: int, device: torch.device) -> float:
    """Token-level mean NLL over every step of ``data`` (the D4 val loss)."""
    tot, n = 0.0, 0
    for idx in _chunks(np.arange(len(data)), batch_size):
        nll, valid = model.step_nll(to_device(data.batch(idx), device))
        tot += float(nll.sum())
        n += int(valid.sum())
    return tot / n


@torch.no_grad()
def d4_session_scores(model: D4Model, data: EncodedSet, batch_size: int, device: torch.device) -> np.ndarray:
    """Mean per-step NLL per session (float64)."""
    out = [model.session_scores(to_device(data.batch(idx), device)).double().cpu().numpy() for idx in _chunks(np.arange(len(data)), batch_size)]
    return np.concatenate(out) if out else np.zeros(0)


@torch.no_grad()
def d5_logits(model: D5Model, data: EncodedSet, batch_size: int, device: torch.device) -> np.ndarray:
    """Logits per session (float64)."""
    out = [model.logits(to_device(data.batch(idx), device)).double().cpu().numpy() for idx in _chunks(np.arange(len(data)), batch_size)]
    return np.concatenate(out) if out else np.zeros(0)


def concat_encoded(parts: Sequence[EncodedSet], name: str, attacker: str | None = None) -> EncodedSet:
    """Row-wise concatenation (same max_len)."""
    first = parts[0]
    return EncodedSet(name, first.split, first.kind, attacker, np.concatenate([p.session_ids for p in parts]),
                      np.concatenate([p.user_ids for p in parts]), np.concatenate([p.keys for p in parts]),
                      np.concatenate([p.h for p in parts]), np.concatenate([p.p for p in parts]),
                      np.concatenate([p.pmask for p in parts]), np.concatenate([p.lens for p in parts]),
                      sum(p.n_truncated for p in parts), max(p.max_n_keys for p in parts))


def d5_epoch_batches(model: D5Model, humans: EncodedSet, bots: EncodedSet, batch_size: int, gen: torch.Generator, device: torch.device):
    """Class-balanced epochs: all ``N_h`` humans + ``N_h`` bots drawn without replacement (fresh seeded draw each epoch)."""
    n_h, n_b = len(humans), len(bots)
    if n_b < n_h:
        raise ValueError(f"class-balanced sampling needs at least as many bots ({n_b}) as humans ({n_h})")

    def it(epoch: int) -> Iterator[tuple[torch.Tensor, int]]:
        bot_idx = torch.randperm(n_b, generator=gen)[:n_h].numpy()
        src = np.r_[np.zeros(n_h, dtype="int64"), np.ones(n_h, dtype="int64")]       # 0 = human, 1 = bot
        row = np.r_[np.arange(n_h), bot_idx]
        order = torch.randperm(2 * n_h, generator=gen).numpy()
        for sel in _chunks(order, batch_size):
            b = _mixed_batch(humans, bots, src[sel], row[sel])
            y = torch.from_numpy((src[sel] == 0).astype("float32")).to(device)
            loss = F.binary_cross_entropy_with_logits(model.logits(to_device(b, device)), y)
            yield loss, len(sel)
    return it


def _mixed_batch(humans: EncodedSet, bots: EncodedSet, src: np.ndarray, row: np.ndarray) -> dict[str, np.ndarray]:
    hb, bb = src == 0, src == 1
    lens = np.where(hb, humans.lens[np.where(hb, row, 0)], bots.lens[np.where(bb, row, 0)])
    w = int(lens.max())
    out = {}
    for k in ("keys", "h", "p", "pmask"):
        a = np.empty((len(src), w), dtype=getattr(humans, k).dtype)
        a[hb] = getattr(humans, k)[row[hb], :w]
        a[bb] = getattr(bots, k)[row[bb], :w]
        out[k] = a.astype("int64") if k == "keys" else a
    out["lens"] = lens
    return out


@torch.no_grad()
def d5_val_loss(model: D5Model, val_humans: EncodedSet, val_bots: EncodedSet, batch_size: int, device: torch.device) -> float:
    """``0.5 * mean BCE(val humans) + 0.5 * mean BCE(val bots)``."""
    lh = d5_logits(model, val_humans, batch_size, device)
    lb = d5_logits(model, val_bots, batch_size, device)
    bce_h = float(np.mean(np.logaddexp(0.0, -lh)))       # -log sigmoid(l), label 1
    bce_b = float(np.mean(np.logaddexp(0.0, lb)))        # -log(1 - sigmoid(l)), label 0
    return 0.5 * bce_h + 0.5 * bce_b
