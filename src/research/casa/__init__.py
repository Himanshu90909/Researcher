"""
CASA -- Certified Adaptive Sparse Attention.

Proof-carrying approximation: every pruned attention (or pruned retrieval context)
output ships a deterministic, machine-checkable bound on its distance from the exact
result.  See ``docs/CASA.md`` for the theorems, proofs, algorithms and experiments.
"""

from .attention import (
    attention_error,
    certified_sparse_attention,
    mass_threshold_attention,
    topk_attention,
)
from .bounds import (
    BlockEnvelope,
    CertifiedPattern,
    attention_scores,
    build_block_envelopes,
    certified_pattern,
    dense_attention,
    pruning_error_bound,
    renormalized_attention,
    softmax_weights,
    tail_mass_bound,
)
from .certificate import (
    CERT_VERSION,
    Certificate,
    CertificateLedger,
    VerificationReport,
    verify_certificate,
)
from .context import Chunk, ContextPruningResult, certified_context_pruning

__all__ = [
    "CERT_VERSION",
    "BlockEnvelope",
    "Certificate",
    "CertificateLedger",
    "CertifiedPattern",
    "Chunk",
    "ContextPruningResult",
    "VerificationReport",
    "attention_error",
    "attention_scores",
    "build_block_envelopes",
    "certified_context_pruning",
    "certified_pattern",
    "certified_sparse_attention",
    "dense_attention",
    "mass_threshold_attention",
    "pruning_error_bound",
    "renormalized_attention",
    "softmax_weights",
    "tail_mass_bound",
    "topk_attention",
    "verify_certificate",
]

__version__ = "1.0.0"
