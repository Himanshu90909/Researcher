"""
casa.context -- certified pruning of retrieved context.

The same theorem that bounds a pruned attention output bounds the drift of the
context vector a retrieval-augmented model sees.  Put a query embedding ``q``, chunk
embeddings ``c_j`` and chunk *value* vectors ``v_j`` (by default the embeddings
themselves) into the roles of ``q, K, V``: the softmax-attention-weighted context
vector

    g = sum_j softmax(<q, c_j> / sqrt(d))_j v_j

is exactly a single attention output, so ``|| g_pruned - g_dense || <= eps`` is
certified by :func:`casa.certified_sparse_attention`.  What gets dropped here is
*retrieved text*, so ``eps`` becomes a bound on how far the model's context can move
when chunks are removed -- a measurable, auditable alternative to "we pruned 70% of
the prompt and it looked fine".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .attention import certified_sparse_attention, mass_threshold_attention
from .bounds import attention_scores, dense_attention, softmax_weights
from .certificate import Certificate

__all__ = ["Chunk", "certified_context_pruning", "ContextPruningResult"]


@dataclass
class Chunk:
    """A retrieved passage."""

    id: str
    text: str
    embedding: np.ndarray
    value: Optional[np.ndarray] = None
    tokens: int = 0

    def value_vector(self, dim: Optional[int] = None) -> np.ndarray:
        if self.value is not None:
            return np.asarray(self.value, dtype=np.float64).ravel()
        return np.asarray(self.embedding, dtype=np.float64).ravel()[: dim or len(self.embedding)]


@dataclass
class ContextPruningResult:
    kept: List[Chunk]
    dropped: List[Chunk]
    cert: Certificate
    context_dense: np.ndarray
    context_pruned: np.ndarray
    tokens_before: int
    tokens_after: int
    audit: Dict[str, Any]

    @property
    def token_savings(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return 1.0 - self.tokens_after / self.tokens_before


def certified_context_pruning(
    query_embedding: np.ndarray,
    chunks: Sequence[Chunk],
    eps: float = 0.05,
    block_size: int = 1,
    scheme: str = "exact",
    reference: str = "global_mean",
    audit: bool = True,
) -> ContextPruningResult:
    """Drop as many retrieved chunks as ``eps`` provably allows.

    ``block_size=1`` prunes chunk-by-chunk (each chunk is its own block); larger
    blocks prune groups of adjacent-but-unrelated chunks, which is only useful when
    chunks are known to be exchangeable.
    """
    chunks = list(chunks)
    if not chunks:
        raise ValueError("need at least one chunk")
    d = len(np.asarray(query_embedding).ravel())
    K = np.vstack([np.asarray(c.embedding, dtype=np.float64).ravel()[:d] for c in chunks])
    V = np.vstack([c.value_vector(d) for c in chunks])
    q = np.asarray(query_embedding, dtype=np.float64).ravel()

    out, cert = certified_sparse_attention(
        q,
        K,
        V,
        block_size=block_size,
        eps=eps,
        scheme=scheme,
        reference=reference,
        audit=False,
    )
    keep_mask = np.zeros(len(chunks), dtype=bool)
    keep_mask[np.array(cert.retained, dtype=np.int64)] = True
    kept = [c for c, m in zip(chunks, keep_mask) if m]
    dropped = [c for c, m in zip(chunks, keep_mask) if not m]

    context_dense, s, p = dense_attention(q, K, V)
    context_pruned = out
    tokens_before = int(sum(c.tokens for c in chunks)) or len(chunks)
    tokens_after = int(sum(c.tokens for c in kept)) or len(kept)

    audit_info: Dict[str, Any] = {}
    if audit:
        drift = float(np.linalg.norm(context_pruned - context_dense))
        top = int(np.argmax(p))
        audit_info = {
            "measured_drift": drift,
            "certified_drift": cert.error_bound,
            "drift_within_certificate": drift <= cert.error_bound * (1 + 1e-9) + 1e-12,
            "top_chunk_retained": bool(keep_mask[top]),
            "top_chunk_rank_score": float(s[top]),
            "retained_mass": float(p[keep_mask].sum()),
        }

    return ContextPruningResult(
        kept=kept,
        dropped=dropped,
        cert=cert,
        context_dense=context_dense,
        context_pruned=context_pruned,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        audit=audit_info,
    )
