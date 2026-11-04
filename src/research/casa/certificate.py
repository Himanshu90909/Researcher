"""
casa.certificate -- proof-carrying outputs.

A :class:`Certificate` is the *witness* that accompanies a pruned attention output.
It records exactly the quantities a sceptical third party needs in order to
re-derive the guarantee: which tokens were kept, the certified dropped mass, the
value radii, and the envelopes of the dropped blocks.  :func:`verify_certificate`
re-derives everything from the raw tensors and the certificate, without trusting
the code path that produced it, and additionally checks the *measured* error
against the certified bound.

Three levels of assurance, strongest last:

1. **Arithmetic re-check** (:meth:`Certificate.check_arithmetic`) -- offline, no
   tensors.  Recomputes ``tau_bar`` from ``(n_dropped, u_drop, m_S)`` and the bound
   from ``(tau_bar, D_S, D_all)``, and confirms the bound respects ``eps``.
2. **Ownership check** (:meth:`Certificate.matches_inputs`) -- SHA-256 digests bind
   the certificate to the exact query/key/value tensors it was issued for.
3. **Full re-verification** (:func:`verify_certificate`) -- recomputes the retained
   set, the bound and the true error from the raw tensors.

A :class:`CertificateLedger` chains certificates by hash so that a run cannot be
edited after the fact without breaking the chain.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .bounds import (
    attention_scores,
    block_logmass,
    build_block_envelopes,
    dense_attention,
    renormalized_attention,
    tail_mass_bound_lse,
)

__all__ = [
    "CERT_VERSION",
    "Certificate",
    "VerificationReport",
    "verify_certificate",
    "CertificateLedger",
    "digest_array",
    "digest_indices",
]

CERT_VERSION = "casa-1.0"
_FP_TOL = 1e-9  # relative slack for float64 round-off when checking a theorem


def _digest(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=32).hexdigest()


def digest_array(a: np.ndarray) -> str:
    """BLAKE2b over shape, dtype and raw bytes of an array (float64 canonicalised)."""
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return _digest(str(a.shape).encode() + b"|float64|" + a.tobytes())


def digest_indices(idx: Sequence[int]) -> str:
    a = np.asarray(idx, dtype=np.int64)
    if a.size:
        a = np.unique(a)
    return _digest(str(a.shape).encode() + b"|int64|" + a.tobytes())


@dataclass
class Certificate:
    """Machine-checkable guarantee that a pruned output is within ``eps``."""

    eps: float
    error_bound: float
    tau_bound: float
    n_total: int
    n_retained: int
    n_dropped: int
    block_size: int
    blocks_kept: int
    scheme: str
    reference: str
    max_kept_score: float
    D_kept: float
    D_all: float
    dropped_logmass: float
    retained: List[int]
    retained_digest: str
    q_digest: str
    k_digest: str
    v_digest: str
    feasible: bool = True
    measured_error: Optional[float] = None
    fallback: bool = False
    version: str = CERT_VERSION
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # ---------------------------------------------------------------- accessors
    @property
    def retention(self) -> float:
        return self.n_retained / max(self.n_total, 1)

    @property
    def compression(self) -> float:
        """Fraction of tokens discarded."""
        return 1.0 - self.retention

    @property
    def valid(self) -> bool:
        """True when the certificate actually certifies its own target."""
        return bool(self.feasible and not self.fallback and self.error_bound <= self.eps + _FP_TOL)

    # ------------------------------------------------------------- arithmetic
    def check_arithmetic(self, tol: float = 1e-9) -> Dict[str, Any]:
        """Offline re-derivation of the bound from stored quantities only."""
        checks: Dict[str, bool] = {}
        tau = tail_mass_bound_lse(self.dropped_logmass, self.max_kept_score)
        checks["tau_recomputes"] = abs(tau - self.tau_bound) <= tol + 1e-12
        bound = self.tau_bound * (self.D_kept + self.D_all)
        checks["bound_recomputes"] = abs(bound - self.error_bound) <= tol * max(1.0, abs(self.error_bound)) + 1e-12
        checks["bound_within_eps"] = self.error_bound <= self.eps + tol
        checks["retained_digest"] = digest_indices(self.retained) == self.retained_digest
        checks["counts_consistent"] = (
            self.n_retained + self.n_dropped == self.n_total and len(self.retained) == self.n_retained
        )
        checks["tau_in_unit_interval"] = -tol <= self.tau_bound <= 1.0 + tol
        checks["measured_within_bound"] = (
            self.measured_error is None
            or self.measured_error <= self.error_bound * (1.0 + tol) + 1e-12
        )
        return {"ok": all(checks.values()), "checks": checks}

    def matches_inputs(self, q: np.ndarray, K: np.ndarray, V: np.ndarray) -> bool:
        if not (self.q_digest or self.k_digest or self.v_digest):
            return False  # unbound certificate: nothing to bind to
        return (
            digest_array(np.asarray(q).ravel()) == self.q_digest
            and digest_array(K) == self.k_digest
            and digest_array(V) == self.v_digest
        )

    # ----------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Certificate":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, s: str) -> "Certificate":
        return cls.from_dict(json.loads(s))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass
class VerificationReport:
    ok: bool
    checks: Dict[str, bool]
    details: Dict[str, Any]

    def __str__(self) -> str:  # pragma: no cover - formatting only
        status = "PASS" if self.ok else "FAIL"
        failed = [k for k, v in self.checks.items() if not v]
        extra = f" (failed: {', '.join(failed)})" if failed else ""
        return f"[{status}] casa certificate{extra}"


def verify_certificate(
    cert: Certificate,
    q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    tol: float = _FP_TOL,
) -> VerificationReport:
    """Independently re-derive the guarantee from the raw tensors."""
    q = np.asarray(q, dtype=np.float64).ravel()
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {}
    n = K.shape[0]

    bound_inputs = bool(cert.q_digest or cert.k_digest or cert.v_digest)
    checks["ownership"] = cert.matches_inputs(q, K, V) if bound_inputs else True
    checks["shape"] = cert.n_total == n and K.shape[0] == V.shape[0]
    if not (checks["ownership"] and checks["shape"]):
        return VerificationReport(False, checks, {"reason": "certificate does not belong to these inputs"})

    kept = np.array(sorted(set(int(i) for i in cert.retained)), dtype=np.int64)
    checks["retained_digest"] = digest_indices(kept) == cert.retained_digest
    checks["retained_in_range"] = bool(kept.size > 0 and kept[0] >= 0 and kept[-1] < n)
    if not (checks["retained_digest"] and checks["retained_in_range"]):
        return VerificationReport(False, checks, {"reason": "retained set is inconsistent"})

    # --- recompute every quantity the bound depends on, from scratch -----------
    s = attention_scores(q, K)
    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[kept] = True
    n_dropped = int((~kept_mask).sum())

    if cert.reference == "zero":
        radii = np.sqrt(np.einsum("ij,ij->i", V, V))
    elif cert.reference == "global_mean":
        centred = V - V.mean(axis=0)
        radii = np.sqrt(np.einsum("ij,ij->i", centred, centred))
    else:
        return VerificationReport(False, {**checks, "reference": False}, {"reference": cert.reference})
    D_all = float(radii.max())
    D_kept = float(radii[kept].max())
    m_s = float(s[kept].max())

    envelopes = build_block_envelopes(q, K, cert.block_size, cert.scheme)
    dropped_logmass = -math.inf
    for e in envelopes:
        if not kept_mask[e.start:e.stop].all():
            dropped_logmass = float(np.logaddexp(dropped_logmass, block_logmass(e.size, e.upper)))

    tau = tail_mass_bound_lse(dropped_logmass, m_s)
    bound = tau * (D_kept + D_all)

    checks["tau_matches"] = abs(tau - cert.tau_bound) <= tol + 1e-12

    # --- minimality: one fewer block must fail to certify --------------------
    # A certificate may keep more tokens than necessary, but then it is not the
    # certificate the prover claims (and inflating eps after the fact is caught here).
    order = sorted(envelopes, key=lambda e: (-e.upper, e.block))
    minimal = True
    if cert.blocks_kept > 1:
        prev_blocks = order[: cert.blocks_kept - 1]
        prev_kept = np.concatenate([np.arange(e.start, e.stop) for e in prev_blocks]).astype(np.int64)
        if prev_kept.size:
            prev_mask = np.zeros(n, dtype=bool)
            prev_mask[prev_kept] = True
            prev_dlm = -math.inf
            for e in envelopes:
                if not prev_mask[e.start:e.stop].all():
                    prev_dlm = float(np.logaddexp(prev_dlm, block_logmass(e.size, e.upper)))
            prev_m = float(s[prev_kept].max())
            prev_tau = tail_mass_bound_lse(prev_dlm, prev_m)
            prev_bound = prev_tau * (float(radii[prev_kept].max()) + D_all)
            minimal = prev_bound > cert.eps + tol
    checks["minimal_for_order"] = minimal
    checks["bound_matches"] = abs(bound - cert.error_bound) <= tol * max(1.0, abs(cert.error_bound)) + 1e-12
    checks["bound_within_eps"] = bound <= cert.eps + tol
    if dropped_logmass == -math.inf:
        checks["dropped_logmass_matches"] = cert.dropped_logmass == -math.inf
    else:
        checks["dropped_logmass_matches"] = (
            abs(dropped_logmass - cert.dropped_logmass) <= 1e-9 * max(1.0, abs(dropped_logmass))
        )

    # --- the actual error, measured against dense attention -------------------
    o_full, _, _ = dense_attention(q, K, V)
    o_kept, _, _ = renormalized_attention(q, K, V, kept)
    measured = float(np.linalg.norm(o_kept - o_full))
    checks["measured_within_certified"] = measured <= bound * (1.0 + tol) + 1e-12
    if cert.measured_error is not None:
        checks["measured_reported_honestly"] = abs(measured - cert.measured_error) <= 1e-9 * max(1.0, measured)

    details.update(
        {
            "input_binding_checked": bound_inputs,
            "measured_error": measured,
            "certified_bound": bound,
            "reported_bound": cert.error_bound,
            "eps": cert.eps,
            "tau": tau,
            "retention": len(kept) / n,
            "n_dropped": n_dropped,
        }
    )
    return VerificationReport(all(checks.values()), checks, details)


class CertificateLedger:
    """Append-only, hash-chained log of certificates (JSON Lines)."""

    def __init__(self, path: str):
        self.path = path

    def append(self, cert: Certificate, **extra: Any) -> str:
        prev = self._last_hash()
        entry = {"prev": prev, "cert": cert.to_dict(), **extra}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()

    def _last_hash(self) -> str:
        last, _ = self.read()
        return last

    def read(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]
        except FileNotFoundError:
            return "", []
        last = ""
        for entry in lines:
            last = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        return last, lines

    @staticmethod
    def verify(path: str) -> Dict[str, Any]:
        """Check chain integrity and re-derive every entry's arithmetic offline."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = [json.loads(line) for line in fh if line.strip()]
        except FileNotFoundError:
            return {"ok": False, "entries": 0, "reason": "ledger not found"}

        prev = ""
        broken = 0
        arithmetic_failures = 0
        for entry in lines:
            cert = Certificate.from_dict(entry["cert"])
            if entry.get("prev", "") != prev:
                broken += 1
            if not cert.check_arithmetic()["ok"]:
                arithmetic_failures += 1
            prev = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()

        return {
            "ok": broken == 0 and arithmetic_failures == 0,
            "entries": len(lines),
            "broken_links": broken,
            "arithmetic_failures": arithmetic_failures,
        }
