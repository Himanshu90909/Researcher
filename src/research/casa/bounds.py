"""
casa.bounds -- deterministic error bounds for renormalized sparse attention.

Notation
--------
``q`` is a query vector in R^d, ``k_i`` keys, ``v_i`` values in R^m,
``s_i = <q, k_i> / sqrt(d)``, ``p = softmax(s)`` and the exact ("dense") output is

    o = sum_i p_i v_i.

A pruner retains a subset ``S`` of the tokens, drops ``D = [N] \\ S``, and
renormalizes over ``S``:

    o_S = sum_{i in S} p^S_i v_i,     p^S_i = e^{s_i} / Z_S,   Z_S = sum_{i in S} e^{s_i}.

Write ``tau = 1 - Z_S / Z`` for the *dropped probability mass*.  Every result in this
module is a statement about ``|| o_S - o ||_2`` that holds for **every** query, key
and value set -- no concentration inequality, no sampling, no assumption about the
data distribution appears anywhere.

Theorem 1 (renormalized-pruning error bound)
--------------------------------------------
For any reference vector ``c`` and with

    D_S = max_{i in S} ||v_i - c||,      D_D = max_{i in D} ||v_i - c||,

we have

    || o_S - o ||_2  <=  tau * ( || o_S - c || + D_D )  <=  tau * ( D_S + D_D )   <= 2 tau D

where ``D = max_i ||v_i - c||``.

Proof.  Because ``Z_S = (1 - tau) Z`` we get ``p^S_i = p_i / (1 - tau)`` for ``i in S``.
Writing ``d_i = v_i - c`` and using ``sum_i p_i = 1``:

    o_S - o = sum_{i in S} p^S_i d_i - sum_i p_i d_i
            = sum_{i in S} p^S_i d_i - (1 - tau) sum_{i in S} p^S_i d_i - sum_{i in D} p_i d_i
            = tau * (o_S - c) - sum_{i in D} p_i d_i.

``||o_S - c|| <= sum_{i in S} p^S_i ||d_i|| <= D_S`` and
``|| sum_{i in D} p_i d_i || <= sum_{i in D} p_i ||d_i|| <= tau D_D``; add them.  []

``o_S`` and ``o`` are both convex combinations, so no bound of this shape can be
smaller than ``tau * D``; the factor 2 is the price of comparing two differently
renormalized mixtures and is attained (docs/CASA.md, Appendix A).

Theorem 2 (tail mass from block summaries)
------------------------------------------
Partition the tokens into blocks ``g`` of size ``n_g`` and let ``u_g`` be any upper
bound on ``max_{i in g} s_i``.  For ``S`` a union of whole blocks with
``m_S = max_{i in S} s_i`` (exact, needs only retained scores),

    Z_D = sum_{g in D} sum_{i in g} e^{s_i} <= sum_{g in D} n_g e^{u_g} =: exp(dropped_logmass)

and since ``Z >= Z_S >= e^{m_S}``,

    tau <= tau_bar := min(1, exp(dropped_logmass - m_S)).                   (sum form)

Dropping block sizes gives the cruder max form
``tau <= min(1, N_D exp(u_D - m_S))`` (:func:`tail_mass_bound`); the sum form
(:func:`tail_mass_bound_lse`) is never worse and is what the certifier uses.

Combining the two gives the certificate this package emits:

    || o_S - o ||_2 <= tau_bar * (D_S + D_all),      tau_bar <= 1,

from (i) the exact scores of the retained tokens, (ii) one scalar per dropped block
(``log n_g + u_g``), and (iii) the scalar value radius ``D_all``.  Dropped keys and
values are never read individually.

Which ``u_g``?
--------------
``scheme="exact"`` takes ``u_g`` to be the exact block maximum, i.e. all scores are
computed.  That is the default because on current hardware the score pass is the
same order as the value pass, and because summary-only envelopes are *vacuous* in
high dimension: a key envelope must hedge a d-dimensional residual, so its slack
grows like ``||q|| r_g`` ~ d while the score spread grows only like sqrt(d).  E5 in
``scripts/run_casa_experiments.py`` measures exactly this (100% retention for
``norm`` and ``centroid`` at d=64).  The envelope schemes are kept because they are
the honest baseline for a future index that can bound ``Z_g`` without scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BlockEnvelope",
    "CertifiedPattern",
    "attention_scores",
    "softmax_weights",
    "dense_attention",
    "renormalized_attention",
    "build_block_envelopes",
    "block_logmass",
    "tail_mass_bound",
    "tail_mass_bound_lse",
    "pruning_error_bound",
    "certified_pattern",
]

SCHEMES = ("norm", "centroid", "exact")
DEFAULT_SCHEME = "exact"
REFERENCES = ("zero", "global_mean")

_OVERFLOW_GUARD = 700.0


def attention_scores(q: np.ndarray, K: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Scaled dot-product scores ``s_i = <q, k_i> / sqrt(d)`` in float64."""
    q = np.asarray(q, dtype=np.float64).ravel()
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2:
        raise ValueError("K must be 2-D")
    if K.shape[1] != q.shape[0]:
        raise ValueError(f"dim mismatch: q.d={q.shape[0]} K.d={K.shape[1]}")
    if scale is None:
        scale = 1.0 / math.sqrt(K.shape[1])
    return K @ q * scale


def softmax_weights(s: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D score vector."""
    s = np.asarray(s, dtype=np.float64).ravel()
    if s.size == 0:
        return s.copy()
    w = np.exp(s - s.max())
    return w / w.sum()


def dense_attention(q: np.ndarray, K: np.ndarray, V: np.ndarray, scores: Optional[np.ndarray] = None):
    """Exact single-head attention. Returns ``(output, scores, weights)``."""
    s = attention_scores(q, K) if scores is None else np.asarray(scores, dtype=np.float64)
    p = softmax_weights(s)
    return p @ np.asarray(V, dtype=np.float64), s, p


def renormalized_attention(
    q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    kept: Sequence[int],
    scores: Optional[np.ndarray] = None,
):
    """Attention restricted to ``kept`` tokens and renormalized over them."""
    idx = np.asarray(kept, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("kept set must be non-empty")
    if scores is None:
        s = attention_scores(q, np.asarray(K, dtype=np.float64)[idx])
    else:
        s = np.asarray(scores, dtype=np.float64)[idx]
    p = softmax_weights(s)
    return p @ np.asarray(V, dtype=np.float64)[idx], s, p


@dataclass(frozen=True)
class BlockEnvelope:
    """A contiguous token block plus a *valid* upper bound on its max score."""

    block: int
    start: int
    stop: int
    upper: float

    @property
    def size(self) -> int:
        return self.stop - self.start


def _block_grid(n: int, block_size: int) -> Tuple[int, np.ndarray, np.ndarray]:
    starts = np.arange(0, n, block_size)
    counts = np.minimum(block_size, n - starts)
    return len(starts), starts, counts


def build_block_envelopes(
    q: np.ndarray,
    K: np.ndarray,
    block_size: int = 16,
    scheme: str = DEFAULT_SCHEME,
    scores: Optional[np.ndarray] = None,
) -> List[BlockEnvelope]:
    """Per-block upper bounds on ``max_{i in block} s_i``, vectorized.

    ``scheme``
        ``"exact"``     exact block maximum of the (computed) scores
        ``"norm"``      ``||q|| * max_i ||k_i|| / sqrt(d)``, needs only key norms
        ``"centroid"``  ``(<q, mu_g> + ||q|| r_g) / sqrt(d)``, needs centroid + radius
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    K = np.asarray(K, dtype=np.float64)
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    n, d = K.shape
    g_count, starts, counts = _block_grid(n, block_size)

    if scheme == "exact":
        s = attention_scores(q, K) if scores is None else np.asarray(scores, dtype=np.float64)
        pad = g_count * block_size - n
        s_pad = np.concatenate([np.asarray(s, dtype=np.float64).ravel(), np.full(pad, -np.inf)])
        upper = s_pad.reshape(g_count, block_size).max(axis=1)
    else:
        pad = g_count * block_size - n
        Kp = np.concatenate([K, np.zeros((pad, d))], axis=0).reshape(g_count, block_size, d)
        if scheme == "norm":
            upper = np.sqrt(np.einsum("gbd,gbd->gb", Kp, Kp)).max(axis=1) * (
                float(np.sqrt(np.einsum("i,i->", q, q))) / math.sqrt(d))
        else:  # centroid + radius: <q, k_i> <= <q, mu_g> + ||q|| * ||k_i - mu_g||
            valid = (np.arange(block_size)[None, :] < counts[:, None]).astype(np.float64)
            mu = np.einsum("gbd,gb->gd", Kp, valid) / counts[:, None]
            diff = (Kp - mu[:, None, :]) * valid[:, :, None]
            radius = np.sqrt(np.einsum("gbd,gbd->gb", diff, diff)).max(axis=1)
            upper = (mu @ q + float(np.linalg.norm(q)) * radius) / math.sqrt(d)

    return [
        BlockEnvelope(i, int(starts[i]), int(starts[i] + counts[i]), float(upper[i]))
        for i in range(g_count)
    ]


def block_logmass(size: int, upper: float) -> float:
    """``log(n_g) + u_g``: a rigorous log upper bound on the block's mass ``Z_g``."""
    if size <= 0:
        return -math.inf
    return math.log(float(size)) + float(upper)


def tail_mass_bound(n_dropped: int, upper_dropped: float, max_kept_score: float) -> float:
    """Theorem 2, max form: ``tau <= min(1, N_D * exp(u_D - m_S))``."""
    if n_dropped <= 0:
        return 0.0
    log_tau = math.log(float(n_dropped)) + float(upper_dropped) - float(max_kept_score)
    return _exp_clamped(log_tau)


def tail_mass_bound_lse(dropped_logmass: float, max_kept_score: float) -> float:
    """Theorem 2, sum form: ``tau <= min(1, exp(dropped_logmass - m_S))``."""
    if dropped_logmass == -math.inf:
        return 0.0
    return _exp_clamped(float(dropped_logmass) - float(max_kept_score))


def _exp_clamped(log_value: float) -> float:
    if log_value >= 0.0:
        return 1.0
    if log_value < -_OVERFLOW_GUARD:
        return 0.0
    return math.exp(log_value)


def pruning_error_bound(tau: float, D_kept: float, D_dropped: float) -> float:
    """Theorem 1: ``|| o_S - o || <= tau * (D_S + D_D)``."""
    return float(tau) * (float(D_kept) + float(D_dropped))


@dataclass
class CertifiedPattern:
    """Outcome of searching for the sparsest certifiable pattern."""

    kept: np.ndarray
    blocks_kept: int
    block_order: List[int]
    tau_bound: float
    error_bound: float
    D_kept: float
    D_all: float
    max_kept_score: float
    n_dropped: int
    dropped_logmass: float
    scheme: str
    reference: str
    feasible: bool
    curve: List[Tuple[int, float]]

    @property
    def n_total(self) -> int:
        return int(self.kept.size) + int(self.n_dropped)

    @property
    def retention(self) -> float:
        return float(self.kept.size) / float(max(self.n_total, 1))


def _value_radii(V: np.ndarray, reference: str) -> Tuple[np.ndarray, float]:
    V = np.asarray(V, dtype=np.float64)
    if reference == "zero":
        norms = np.sqrt(np.einsum("ij,ij->i", V, V))
    elif reference == "global_mean":
        centred = V - V.mean(axis=0)
        norms = np.sqrt(np.einsum("ij,ij->i", centred, centred))
    else:
        raise ValueError(f"unknown reference {reference!r}; expected one of {REFERENCES}")
    return norms, float(norms.max())


def certified_pattern(
    q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 16,
    eps: float = 0.05,
    scheme: str = DEFAULT_SCHEME,
    reference: str = "global_mean",
    max_blocks: Optional[int] = None,
    scores: Optional[np.ndarray] = None,
) -> CertifiedPattern:
    """Fewest blocks whose pruned output is certifiably within ``eps``.

    Blocks are ranked by their envelope bound; the search scans every prefix and
    returns the first one whose certificate passes, so the result is minimal inside
    that ordering (the verifier re-checks this as ``minimal_for_order``).

    Cost: ``O(N d + N m + G log G)`` -- one score pass, one value pass, one sort.
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if K.shape[0] != V.shape[0]:
        raise ValueError("K and V disagree on token count")
    n = K.shape[0]
    if n == 0:
        raise ValueError("empty sequence")
    if eps < 0:
        raise ValueError("eps must be >= 0")

    s = attention_scores(q, K) if scores is None else np.asarray(scores, dtype=np.float64)
    envelopes = build_block_envelopes(q, K, block_size, scheme, scores=s)
    upper_arr = np.array([e.upper for e in envelopes], dtype=np.float64)
    order_idx = np.lexsort((np.arange(len(envelopes)), -upper_arr))
    block_order = [envelopes[i].block for i in order_idx]
    n_keep_blocks = len(envelopes) if max_blocks is None else max(1, min(int(max_blocks), len(envelopes)))
    order_idx = order_idx[:n_keep_blocks]
    limited = [envelopes[i] for i in order_idx]

    starts_arr = np.array([e.start for e in limited], dtype=np.int64)
    counts_arr = np.array([e.size for e in limited], dtype=np.int64)
    perm = (starts_arr[:, None] + np.arange(block_size, dtype=np.int64)[None, :]).ravel()
    perm = perm[perm < n].astype(np.int64)
    uppers = upper_arr[order_idx]
    sizes = counts_arr.astype(np.float64)
    counts = np.cumsum(sizes).astype(np.int64)

    s_perm = s[perm]
    kept_max_score = np.maximum.accumulate(s_perm)
    suffix_lse = np.logaddexp.accumulate((np.log(sizes) + uppers)[::-1])[::-1] if len(limited) else np.empty(0)

    limited_ids = {e.block for e in limited}
    beyond_lse = -math.inf
    for e in envelopes:
        if e.block not in limited_ids:
            beyond_lse = float(np.logaddexp(beyond_lse, block_logmass(e.size, e.upper)))

    norms, D_all = _value_radii(V, reference)
    kept_radius = np.maximum.accumulate(norms[perm])

    G = len(limited)
    # Every candidate prefix is evaluated at once: tau and the bound are elementwise
    # functions of prefix statistics, so there is no Python loop over blocks.
    suffix_full = np.append(suffix_lse, -math.inf)
    dropped_lse = np.logaddexp(suffix_full[1:], beyond_lse)
    n_kept_arr = counts
    m_s_arr = kept_max_score[n_kept_arr - 1]
    D_kept_arr = kept_radius[n_kept_arr - 1]
    n_dropped_arr = n - n_kept_arr
    taus = np.exp(np.minimum(dropped_lse - m_s_arr, 0.0))
    empty = n_dropped_arr <= 0
    taus = np.where(empty, 0.0, taus)
    dropped_lse = np.where(empty, -math.inf, dropped_lse)
    bounds = taus * (D_kept_arr + D_all)

    curve = [(int(k), float(b)) for k, b in zip(range(1, G + 1), bounds)]
    passing = np.flatnonzero(bounds <= eps)

    if passing.size == 0:
        return CertifiedPattern(
            kept=np.arange(n, dtype=np.int64),
            blocks_kept=len(envelopes),
            block_order=block_order,
            tau_bound=0.0,
            error_bound=0.0,
            D_kept=D_all,
            D_all=D_all,
            max_kept_score=float(s.max()),
            n_dropped=0,
            dropped_logmass=-math.inf,
            scheme=scheme,
            reference=reference,
            feasible=False,
            curve=curve,
        )

    k = int(passing[0]) + 1  # 1-based block count, not the array index
    n_kept = int(counts[k - 1])
    return CertifiedPattern(
        kept=np.sort(perm[:n_kept]),
        blocks_kept=k,
        block_order=block_order,
        tau_bound=float(taus[k - 1]),
        error_bound=float(bounds[k - 1]),
        D_kept=float(D_kept_arr[k - 1]),
        D_all=D_all,
        max_kept_score=float(m_s_arr[k - 1]),
        n_dropped=int(n - n_kept),
        dropped_logmass=float(dropped_lse[k - 1]),
        scheme=scheme,
        reference=reference,
        feasible=True,
        curve=curve,
    )
