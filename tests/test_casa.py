"""
Tests for CASA -- certified adaptive sparse attention.

Every test here is a *claim from the paper* turned into an executable check:
the theorem (E1), the guarantee (E2), the verifier's teeth (E8), the ledger, the
brute-force agreement of the tail-mass bound, and the two applications.

Run with:  pytest tests/test_casa.py -q
       or:  python tests/test_casa.py
"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.research import casa  # noqa: E402
from src.research.casa.bounds import (  # noqa: E402
    attention_scores,
    build_block_envelopes,
    dense_attention,
    pruning_error_bound,
    renormalized_attention,
    softmax_weights,
    tail_mass_bound_lse,
)

TOL = 1e-9


def problem(seed=0, n=512, d=32, m=16, query_scale=8.0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=d) * query_scale
    K = rng.normal(size=(n, d))
    V = rng.normal(size=(n, m)) / np.sqrt(m)
    return q, K, V


# --------------------------------------------------------------- the theorem
def test_theorem_1_bound_holds_for_every_configuration():
    """E1: the certified bound is never violated, whatever the knob settings."""
    checked = 0
    for seed in range(24):
        q, K, V = problem(seed, n=256, query_scale=float(2 + 3 * (seed % 6)))
        for scheme in ("exact", "centroid", "norm"):
            for block_size in (1, 8, 64):
                for eps in (0.5, 0.1, 0.02):
                    o, cert = casa.certified_sparse_attention(
                        q, K, V, block_size=block_size, eps=eps, scheme=scheme, audit=True
                    )
                    o_full, _, _ = dense_attention(q, K, V)
                    err = float(np.linalg.norm(o - o_full))
                    assert err <= cert.error_bound * (1 + TOL) + 1e-12, (seed, scheme, block_size, eps)
                    checked += 1
    assert checked == 24 * 3 * 3 * 3


def test_theorem_1_bound_against_exhaustive_tau():
    """Theorem 1 with the *exact* tail mass must bound the true error, always."""
    q, K, V = problem(7, n=400, d=24, m=12, query_scale=5.0)
    s = attention_scores(q, K)
    p = softmax_weights(s)
    o_full, _, _ = dense_attention(q, K, V)
    c = V.mean(axis=0)
    for kept_block in (0, 3, 9, 40):
        kept = np.arange(0, (kept_block + 1) * 8, dtype=np.int64)
        kept = kept[kept < K.shape[0]]
        o_kept, _, _ = renormalized_attention(q, K, V, kept, scores=s)
        tau = float(1.0 - p[kept].sum())
        radius = np.linalg.norm(V - c, axis=1)
        bound = pruning_error_bound(tau, float(radius[kept].max()), float(radius.max()))
        assert float(np.linalg.norm(o_kept - o_full)) <= bound + 1e-12


def test_factor_two_is_the_worst_case():
    """The 2 in `2 tau D` is real: a dropped token pointing the other way attains it."""
    D = 1.0
    # two-token sequence: one kept at score 0, one dropped with score log(mass ratio)
    # Construct: p_dropped = tau, values v_kept = +D/2 * u, v_dropped = -D/2 * u
    u = np.array([1.0, 0.0])
    tau = 0.5
    K = np.array([[0.0], [math.log(tau / (1 - tau))]])  # scores s0=0, s1=log(tau/(1-tau))
    V = np.array([[0.5 * D, 0.0], [-0.5 * D, 0.0]])
    q = np.array([0.0])
    # scale=1 for d=1, so scores are exactly the values in K
    o_full, s, p = dense_attention(q, K, V, scores=K[:, 0])
    assert abs(p[1] - tau) < 1e-12
    o_kept, _, _ = renormalized_attention(q, K, V, [0], scores=K[:, 0])
    err = float(np.linalg.norm(o_kept - o_full))
    radius = np.linalg.norm(V, axis=1)
    bound = pruning_error_bound(tau, float(radius[0]), float(radius[1]))
    assert err <= bound + 1e-12
    assert err > 0.9 * bound  # the bound is essentially attained, not vacuous


def test_tail_mass_bound_dominates_true_mass():
    """Theorem 2: the certified tail mass is an upper bound on the real one."""
    q, K, V = problem(3, n=600, d=32, query_scale=6.0)
    s = attention_scores(q, K)
    p = softmax_weights(s)
    for block_size in (1, 4, 32):
        envelopes = build_block_envelopes(q, K, block_size, "exact")
        order = sorted(envelopes, key=lambda e: (-e.upper, e.block))
        for n_blocks in (1, 2, 5, 20):
            kept = np.concatenate([np.arange(e.start, e.stop) for e in order[:n_blocks]]).astype(np.int64)
            dropped_lse = math.log(sum(math.exp(e.upper) * e.size for e in order[n_blocks:])) \
                if n_blocks < len(order) else -math.inf
            tau_bar = tail_mass_bound_lse(dropped_lse, float(s[kept].max()))
            tau_true = float(1.0 - p[kept].sum())
            assert tau_bar >= tau_true - 1e-12, (block_size, n_blocks, tau_bar, tau_true)


def test_envelopes_are_upper_bounds_for_every_scheme():
    q, K, V = problem(5, n=257, d=16, query_scale=4.0)  # 257 is not a multiple of 8
    s = attention_scores(q, K)
    for scheme in ("exact", "norm", "centroid"):
        for block_size in (1, 8, 16):
            for e in build_block_envelopes(q, K, block_size, scheme):
                assert e.upper >= float(s[e.start:e.stop].max()) - 1e-12, (scheme, block_size, e)


# -------------------------------------------------------------- the guarantee
def test_eps_is_never_exceeded_when_certified():
    q, K, V = problem(11, n=1024, query_scale=12.0)
    o_full, _, _ = dense_attention(q, K, V)
    for eps in (0.5, 0.2, 0.05, 0.01):
        o, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=eps, audit=True)
        if cert.feasible:
            assert float(np.linalg.norm(o - o_full)) <= eps * (1 + TOL) + 1e-12


def test_eps_zero_falls_back_to_exact_attention():
    q, K, V = problem(13, n=300, query_scale=4.0)
    o, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=0.0, audit=True)
    o_full, _, _ = dense_attention(q, K, V)
    assert cert.retention == 1.0
    assert np.allclose(o, o_full, atol=1e-12)


def test_tighter_budget_never_prunes_more():
    q, K, V = problem(17, n=1024, query_scale=10.0)
    previous = 0.0
    for eps in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        _, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=eps)
        assert cert.retention >= previous - 1e-12
        previous = cert.retention


def test_certificate_reports_its_own_honesty():
    q, K, V = problem(19, n=512, query_scale=9.0)
    o, cert = casa.certified_sparse_attention(q, K, V, block_size=8, eps=0.05, audit=True)
    assert cert.valid is True
    assert cert.measured_error is not None
    assert cert.measured_error <= cert.error_bound * (1 + TOL) + 1e-12
    assert cert.n_retained + cert.n_dropped == cert.n_total


# ---------------------------------------------------------------- verifier
def test_verifier_accepts_honest_certificate():
    q, K, V = problem(23, n=512, query_scale=8.0)
    _, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=0.05, audit=True)
    report = casa.verify_certificate(cert, q, K, V)
    assert report.ok, report
    assert report.checks["minimal_for_order"]


def test_verifier_rejects_every_tampering():
    q, K, V = problem(29, n=512, query_scale=8.0)
    _, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=0.05, audit=True)

    def tampered(mutate):
        c = casa.Certificate.from_dict(cert.to_dict())
        mutate(c)
        return casa.verify_certificate(c, q, K, V).ok

    assert not tampered(lambda c: setattr(c, "error_bound", c.error_bound * 0.1))
    assert not tampered(lambda c: setattr(c, "tau_bound", c.tau_bound * 0.5))
    assert not tampered(lambda c: c.retained.pop())
    assert not tampered(lambda c: c.retained.__setitem__(0, (c.retained[0] + 1) % c.n_total))
    assert not tampered(lambda c: setattr(c, "error_bound", 0.0))
    assert not tampered(lambda c: setattr(c, "eps", c.eps * 10))  # no longer minimal


def test_verifier_rejects_a_certificate_for_other_inputs():
    q, K, V = problem(31, n=256)
    _, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=0.05)
    q2, K2, V2 = problem(32, n=256)
    assert casa.verify_certificate(cert, q2, K2, V2).ok is False
    K3 = K.copy()
    K3[0, 0] += 1e-12  # even a 1e-12 edit breaks the binding
    assert casa.verify_certificate(cert, q, K3, V).ok is False


def test_unbound_certificate_is_marked_as_such():
    q, K, V = problem(37, n=256)
    _, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=0.05, bind=False)
    report = casa.verify_certificate(cert, q, K, V)
    assert report.ok
    assert report.details["input_binding_checked"] is False


# ------------------------------------------------------------------ ledger
def test_ledger_chains_and_rechecks_arithmetic():
    q, K, V = problem(41, n=256)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "certs.jsonl")
        ledger = casa.CertificateLedger(path)
        for eps in (0.2, 0.1, 0.05):
            _, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=eps, audit=True)
            ledger.append(cert, tag=f"eps={eps}")
        report = casa.CertificateLedger.verify(path)
        assert report["ok"], report
        assert report["entries"] == 3

        # edit one entry after the fact -> the chain must break
        import json as _json

        with open(path) as fh:
            entries = [_json.loads(line) for line in fh if line.strip()]
        entries[1]["cert"]["eps"] = 0.9
        with open(path, "w") as fh:
            for entry in entries:
                fh.write(_json.dumps(entry, sort_keys=True) + "\n")
        result = casa.CertificateLedger.verify(path)
        assert result["ok"] is False
        assert result["broken_links"] >= 1


def test_certificate_roundtrips_through_json():
    q, K, V = problem(43, n=256)
    _, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=0.05, audit=True)
    again = casa.Certificate.from_json(cert.to_json())
    assert again.to_dict() == cert.to_dict()
    assert casa.verify_certificate(again, q, K, V).ok
    assert again.check_arithmetic()["ok"]


# ------------------------------------------------------- baselines & context
def test_baselines_produce_no_guarantee():
    """Sanity: the heuristics run, and mass pruning is a poor proxy for error."""
    q, K, V = problem(47, n=1024, query_scale=8.0)
    o_full, s, p = dense_attention(q, K, V)
    o_topk = casa.topk_attention(q, K, V, 32)
    o_mass = casa.mass_threshold_attention(q, K, V, 0.9)
    assert o_topk.shape == o_full.shape and o_mass.shape == o_full.shape
    mass_kept = int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), 0.9) + 1)
    assert mass_kept < 64  # 90% of the mass sits in very few tokens
    assert float(np.linalg.norm(o_mass - o_full)) > 0.0


def test_context_pruning_bounds_drift_and_keeps_the_gold_chunk():
    rng = np.random.default_rng(53)
    emb = rng.normal(size=(200, 32))
    query = rng.normal(size=32) * 6.0
    gold = int(np.argmax(emb @ query))
    chunks = [casa.Chunk(id=f"c{i}", text=f"t{i}", embedding=emb[i], tokens=100) for i in range(200)]
    res = casa.certified_context_pruning(query, chunks, eps=0.1, block_size=1)
    assert f"c{gold}" in {c.id for c in res.kept}
    assert res.audit["drift_within_certificate"]
    assert res.token_savings > 0.5
    assert res.tokens_after < res.tokens_before
    assert res.cert.check_arithmetic()["ok"]


def test_torch_head_matches_numpy_and_keeps_gradients():
    torch = __import__("torch")
    q, K, V = problem(59, n=256, query_scale=8.0)
    head = casa_torch_head(block_size=16, eps=0.05)
    qt = torch.tensor(q, requires_grad=False)
    Kt = torch.tensor(K, requires_grad=True)
    Vt = torch.tensor(V, requires_grad=True)
    out, cert = head(qt, Kt, Vt, return_certificate=True)
    o_np, cert_np = casa.certified_sparse_attention(q, K, V, block_size=16, eps=0.05)
    assert torch.allclose(out.detach(), torch.tensor(o_np), atol=1e-9)
    assert cert.retained == cert_np.retained
    out.sum().backward()
    assert Kt.grad is not None and Vt.grad is not None
    kept = np.array(cert.retained)
    dropped = np.setdiff1d(np.arange(K.shape[0]), kept)
    assert torch.all(Vt.grad[torch.tensor(dropped)] == 0).item()  # dropped tokens get no gradient


def casa_torch_head(**kwargs):
    from src.research.casa.torch_head import CertifiedAttentionHead

    return CertifiedAttentionHead(**kwargs)


if __name__ == "__main__":
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:  # pragma: no cover
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # pragma: no cover
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} CASA tests passed")
    sys.exit(1 if failures else 0)
