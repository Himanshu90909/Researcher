"""
casa.attention -- proof-carrying sparse attention.

``certified_sparse_attention`` is a drop-in replacement for dense single-head
attention that returns, next to the output, a :class:`~casa.certificate.Certificate`:
a deterministic upper bound on ``|| out_sparse - out_dense ||_2`` together with the
witness needed to re-check it.  Given a target ``eps`` it keeps the *fewest* blocks
it can certify; if nothing certifies, it falls back to dense attention and says so
(``cert.fallback == True``) rather than silently breaking the promise.

Two parameter-free baselines are included for controlled comparison.  Both are
standard practice and neither returns any guarantee:

* ``topk_attention`` -- keep the ``k`` highest-scoring tokens.
* ``mass_threshold_attention`` -- keep top-scoring tokens until the retained
  softmax mass reaches a budget (nucleus-style pruning).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .bounds import (
    DEFAULT_SCHEME,
    attention_scores,
    certified_pattern,
    dense_attention,
    renormalized_attention,
    softmax_weights,
)
from .certificate import Certificate, digest_array, digest_indices

__all__ = [
    "certified_sparse_attention",
    "topk_attention",
    "mass_threshold_attention",
    "attention_error",
]


def _build_certificate(
    q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    kept: np.ndarray,
    pattern,
    eps: float,
    fallback: bool,
    measured_error: Optional[float],
    bind: bool = True,
) -> Certificate:
    return Certificate(
        eps=float(eps),
        error_bound=float(pattern.error_bound),
        tau_bound=float(pattern.tau_bound),
        n_total=int(K.shape[0]),
        n_retained=int(kept.size),
        n_dropped=int(K.shape[0] - kept.size),
        block_size=int(getattr(pattern, "block_size", 0)) or 0,
        blocks_kept=int(pattern.blocks_kept),
        scheme=str(pattern.scheme),
        reference=str(pattern.reference),
        max_kept_score=float(pattern.max_kept_score),
        D_kept=float(pattern.D_kept),
        D_all=float(pattern.D_all),
        dropped_logmass=float(pattern.dropped_logmass),
        retained=[int(i) for i in kept],
        retained_digest=digest_indices(kept),
        q_digest=digest_array(q.ravel()) if bind else "",
        k_digest=digest_array(K) if bind else "",
        v_digest=digest_array(V) if bind else "",
        feasible=bool(pattern.feasible),
        measured_error=None if measured_error is None else float(measured_error),
        fallback=bool(fallback),
    )


def certified_sparse_attention(
    q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_size: int = 16,
    eps: float = 0.05,
    scheme: str = DEFAULT_SCHEME,
    reference: str = "global_mean",
    audit: bool = False,
    max_blocks: Optional[int] = None,
    bind: bool = True,
) -> Tuple[np.ndarray, Certificate]:
    """Attention over the smallest block set that is certifiably ``eps``-close.

    Parameters
    ----------
    q, K, V
        Query ``(d,)``, keys ``(N, d)``, values ``(N, m)``.
    block_size
        Granularity of pruning (tokens per block).  Must match how the key index is
        bucketed; ``block_size=1`` means per-token pruning.
    eps
        Target ``L2`` distance from the dense output, in the units of ``V``.
    scheme, reference
        Envelope family and value-reference as in :mod:`casa.bounds`.
    audit
        Also compute the exact dense output and record the true error in the
        certificate.  Costs an extra dense pass; use it for validation runs, not
        for production inference.
    bind
        Hash the query/key/value tensors into the certificate so it can only ever be
        verified against the exact inputs it was issued for.  Costs one pass over the
        tensors (a few ms at 16k tokens).  Set ``False`` when the caller already has a
        stable identifier for the key/value block (a cache version, a shard id) and
        wants the certificate to refer to that instead.
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    # One score pass serves certification, the retained output and the audit.
    s = attention_scores(q, K)
    pattern = certified_pattern(
        q, K, V, block_size=block_size, eps=eps, scheme=scheme, reference=reference,
        max_blocks=max_blocks, scores=s,
    )
    kept = pattern.kept

    if pattern.feasible:
        out, _, _ = renormalized_attention(q, K, V, kept, scores=s)
        fallback = False
    else:
        out, _, _ = dense_attention(q, K, V, scores=s)
        fallback = True

    measured = None
    if audit:
        o_full, _, _ = dense_attention(q, K, V, scores=s)
        measured = float(np.linalg.norm(out - o_full))

    cert = _build_certificate(q, K, V, kept, pattern, eps, fallback, measured, bind)
    cert.block_size = int(block_size)
    return out, cert


def topk_attention(q: np.ndarray, K: np.ndarray, V: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` highest-scoring tokens, renormalize. No guarantee."""
    s = attention_scores(q, K)
    k = int(max(1, min(k, s.size)))
    idx = np.argsort(-s, kind="stable")[:k]
    out, _, _ = renormalized_attention(q, K, V, idx)
    return out


def mass_threshold_attention(q: np.ndarray, K: np.ndarray, V: np.ndarray, mass: float = 0.9) -> np.ndarray:
    """Keep top-scoring tokens until ``mass`` of the probability is retained. No guarantee."""
    s = attention_scores(q, K)
    p = softmax_weights(s)
    order = np.argsort(-s, kind="stable")
    csum = np.cumsum(p[order])
    k = int(np.searchsorted(csum, mass) + 1)
    k = int(max(1, min(k, s.size)))
    out, _, _ = renormalized_attention(q, K, V, order[:k])
    return out


def attention_error(o: np.ndarray, o_ref: np.ndarray) -> float:
    """``L2`` distance between two attention outputs."""
    return float(np.linalg.norm(np.asarray(o, dtype=np.float64) - np.asarray(o_ref, dtype=np.float64)))
