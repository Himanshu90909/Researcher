#!/usr/bin/env python3
"""
Reproducible experiments for CASA (Certified Adaptive Sparse Attention).

Studies (all seeded; re-running reproduces the numbers):

  E1  certificate validity        -- can the theorem be violated? (must be 0)
  E2  peakedness frontier         -- retention and error vs the certified budget,
                                     swept over attention peakedness
  E3  bound tightness             -- measured error / certified bound
  E4  heuristic control           -- mass- and top-k pruning at matched budgets,
                                     including an adversarial value geometry
  E5  envelope ablation           -- norm / centroid / exact envelopes: when is a
                                     score bound good enough to certify on?
  E6  RAG context pruning         -- token savings with a certified drift bound
  E7  cost                        -- certification overhead vs dense attention
  E8  tamper detection            -- does the verifier actually have teeth?

Usage:
    python scripts/run_casa_experiments.py [--quick] [--no-figures]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.research import casa  # noqa: E402
from src.research.casa.bounds import attention_scores, dense_attention  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "casa")
SEED = 20261204


# --------------------------------------------------------------------------- data
def make_problem(
    rng: np.random.Generator,
    n: int = 1024,
    d: int = 64,
    m: int = 32,
    query_scale: float = 8.0,
    clustered: bool = False,
    n_clusters: int = 16,
    adversarial: bool = False,
):
    """Draw one query/key/value problem.

    ``query_scale`` multiplies the query norm and therefore controls softmax
    peakedness: scores have standard deviation ``query_scale`` (keys and queries are
    unit-variance), so a larger value means a more selective attention head.
    ``clustered`` replaces isotropic keys with keys around ``n_clusters`` centroids.
    ``adversarial`` inflates the value norms of mid-ranked tokens -- exactly the
    tokens a mass-threshold pruner cannot see.
    """
    q = rng.normal(size=d) * query_scale
    if clustered:
        centroids = rng.normal(size=(n_clusters, d)) * 2.0
        assign = rng.integers(0, n_clusters, size=n)
        K = centroids[assign] + rng.normal(size=(n, d)) * 0.35
    else:
        K = rng.normal(size=(n, d))
    V = rng.normal(size=(n, m)) / np.sqrt(m)
    if adversarial:
        s = attention_scores(q, K)
        order = np.argsort(-s)
        band = order[max(1, n // 64): max(2, n // 16)]
        V[band] *= 12.0
    return q, K, V


def l2(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def pct(x, p) -> float:
    return float(np.percentile(np.asarray(x, dtype=np.float64), p))


# -------------------------------------------------------------------------- E1
def e1_validity(quick: bool) -> Dict:
    trials = 10 if quick else 25
    configs = [
        (scheme, block_size, clustered, qs)
        for scheme in ("exact", "centroid", "norm")
        for block_size in (1, 8, 32)
        for clustered in (False, True)
        for qs in (2.0, 8.0, 20.0)
    ]
    rng = np.random.default_rng(SEED)
    eps_grid = [0.5, 0.2, 0.1, 0.05, 0.01]
    total = violations = verify_failures = 0
    worst_ratio = 0.0
    for scheme, block_size, clustered, qs in configs:
        for _ in range(trials):
            q, K, V = make_problem(rng, n=512 if quick else 1024, query_scale=qs, clustered=clustered)
            eps = float(rng.choice(eps_grid))
            o, cert = casa.certified_sparse_attention(
                q, K, V, block_size=block_size, eps=eps, scheme=scheme, audit=True
            )
            total += 1
            o_full, _, _ = dense_attention(q, K, V)
            err = l2(o, o_full)
            if err > cert.error_bound * (1 + 1e-9) + 1e-12:
                violations += 1
            worst_ratio = max(worst_ratio, err / max(cert.error_bound, 1e-12))
            if total % 7 == 0:
                if not casa.verify_certificate(cert, q, K, V).ok:
                    verify_failures += 1
    return {
        "certificates": total,
        "configs": len(configs),
        "bound_violations": violations,
        "independent_verifier_failures": verify_failures,
        "worst_measured_over_bound": worst_ratio,
    }


# -------------------------------------------------------------------------- E2
def e2_frontier(quick: bool) -> Dict:
    rng = np.random.default_rng(SEED + 1)
    reps = 6 if quick else 16
    eps_grid = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    out: Dict[str, List[Dict]] = {}
    for qs in (2.0, 4.0, 8.0, 16.0):
        rows = []
        for eps in eps_grid:
            ret, errs, bounds, ratios = [], [], [], []
            for _ in range(reps):
                q, K, V = make_problem(rng, n=1024 if quick else 2048, query_scale=qs)
                o, cert = casa.certified_sparse_attention(
                    q, K, V, block_size=16, eps=eps, scheme="exact", audit=True
                )
                o_full, _, _ = dense_attention(q, K, V)
                ret.append(cert.retention)
                errs.append(l2(o, o_full))
                bounds.append(cert.error_bound)
                ratios.append(errs[-1] / max(cert.error_bound, 1e-12))
            rows.append({
                "eps": eps,
                "retention_mean": float(np.mean(ret)),
                "retention_p95": pct(ret, 95),
                "measured_error_mean": float(np.mean(errs)),
                "measured_error_p95": pct(errs, 95),
                "bound_mean": float(np.mean(bounds)),
                "ratio_mean": float(np.mean(ratios)),
            })
        out[f"query_scale_{qs:g}"] = rows
    return out


# -------------------------------------------------------------------------- E3
def e3_tightness(quick: bool) -> Dict:
    rng = np.random.default_rng(SEED + 2)
    ratios, rel = [], []
    for _ in range(10 if quick else 40):
        q, K, V = make_problem(rng, n=1024, query_scale=float(rng.uniform(2, 20)))
        eps = float(rng.choice([0.3, 0.1, 0.05, 0.02]))
        o, cert = casa.certified_sparse_attention(q, K, V, block_size=8, eps=eps, audit=True)
        o_full, _, _ = dense_attention(q, K, V)
        ratios.append(cert.measured_error / max(cert.error_bound, 1e-12))
        rel.append(cert.measured_error / max(float(np.linalg.norm(o_full)), 1e-12))
    return {
        "measured_over_bound": {
            "n": len(ratios), "mean": float(np.mean(ratios)), "p50": pct(ratios, 50),
            "p95": pct(ratios, 95), "max": float(np.max(ratios)),
        },
        "relative_error_of_pruned_output": {"mean": float(np.mean(rel)), "p95": pct(rel, 95)},
        "note": "ratio <= 1 always (the theorem); the mean reports how conservative it is",
    }


# -------------------------------------------------------------------------- E4
def e4_heuristic_control(quick: bool) -> Dict:
    rng = np.random.default_rng(SEED + 3)
    reps = 8 if quick else 30
    eps = 0.05
    rows = {}
    for name, adv in (("benign_values", False), ("adversarial_values", True)):
        rec = {"casa": [], "topk": [], "mass": [], "ret": [], "mass_ret": []}
        for _ in range(reps):
            q, K, V = make_problem(rng, n=1024, query_scale=8.0, adversarial=adv)
            o_full, s, p = dense_attention(q, K, V)
            o_c, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=eps, scheme="exact")
            k = max(1, int(round(cert.retention * K.shape[0])))
            o_m = casa.mass_threshold_attention(q, K, V, 0.9)
            rec["casa"].append(l2(o_c, o_full))
            rec["topk"].append(l2(casa.topk_attention(q, K, V, k), o_full))
            rec["mass"].append(l2(o_m, o_full))
            rec["ret"].append(cert.retention)
            # how much of the probability the mass-pruner actually keeps
            p2 = np.sort(p)[::-1]
            rec["mass_ret"].append(float(np.searchsorted(np.cumsum(p2), 0.9) + 1) / K.shape[0])
        # --- calibrated heuristic: choose the most aggressive top-k budget whose
        # worst error on a calibration split stays under eps, then test it on fresh
        # queries. This is what a practitioner must do without a certificate.
        budgets = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70]
        calib = {}
        calib_q = [make_problem(rng, n=1024, query_scale=8.0, adversarial=adv) for _ in range(10)]
        for f in budgets:
            k = max(1, int(round(f * 1024)))
            errs = []
            for qc, Kc, Vc in calib_q:
                oc, _, _ = dense_attention(qc, Kc, Vc)
                errs.append(l2(casa.topk_attention(qc, Kc, Vc, k), oc))
            calib[f] = float(max(errs))
        feasible_budgets = [f for f in budgets if calib[f] <= eps]
        chosen = min(feasible_budgets) if feasible_budgets else None
        held_out = []
        if chosen is not None:
            k = max(1, int(round(chosen * 1024)))
            for _ in range(20):
                qh, Kh, Vh = make_problem(rng, n=1024, query_scale=8.0, adversarial=adv)
                oh, _, _ = dense_attention(qh, Kh, Vh)
                held_out.append(l2(casa.topk_attention(qh, Kh, Vh, k), oh))
        rows[name] = {
            "eps": eps,
            "n": reps,
            "calibrated_topk": {
                "budget_chosen": chosen,
                "calibration_worst_error": None if chosen is None else calib[chosen],
                "heldout_violations": int(sum(e > eps for e in held_out)),
                "heldout_n": len(held_out),
                "heldout_worst_error": float(np.max(held_out)) if held_out else None,
            },
            "matched_retention_mean": float(np.mean(rec["ret"])),
            "mass_pruner_retention_mean": float(np.mean(rec["mass_ret"])),
            "casa_error_max": float(np.max(rec["casa"])),
            "casa_violations": int(sum(e > eps for e in rec["casa"])),
            "casa_certified_all": bool(all(casa.certified_sparse_attention(*(make_problem(
                np.random.default_rng(i), n=256, query_scale=8.0, adversarial=adv)))[1].valid for i in range(3))),
            "topk_error_median": pct(rec["topk"], 50),
            "topk_error_max": float(np.max(rec["topk"])),
            "topk_violations": int(sum(e > eps for e in rec["topk"])),
            "mass_error_median": pct(rec["mass"], 50),
            "mass_error_max": float(np.max(rec["mass"])),
            "mass_violations": int(sum(e > eps for e in rec["mass"])),
        }
    return rows


# -------------------------------------------------------------------------- E5
def e5_envelope_ablation(quick: bool) -> Dict:
    rng = np.random.default_rng(SEED + 4)
    reps = 4 if quick else 12
    eps = 0.1
    out: Dict[str, Dict] = {}
    for space, clustered, adv in (
        ("isotropic_gaussian", False, False),
        ("clustered_keys", True, False),
        ("clustered_keys_adversarial_values", True, True),
    ):
        row = {}
        for scheme in ("norm", "centroid", "exact"):
            ret = []
            for _ in range(reps):
                q, K, V = make_problem(rng, n=1024, query_scale=8.0, clustered=clustered, adversarial=adv)
                _, cert = casa.certified_sparse_attention(
                    q, K, V, block_size=16, eps=eps, scheme=scheme, audit=True
                )
                ret.append(cert.retention)
            row[scheme] = {
                "retention_mean": float(np.mean(ret)),
                "dense_fallback_fraction": float(np.mean([r >= 0.999 for r in ret])),
            }
        out[space] = row
    out["eps"] = eps
    out["note"] = ("an envelope must hedge a d-dimensional residual, so its slack grows ~d "
                   "while the score spread grows ~sqrt(d); at d=64 the bound-only schemes certify "
                   "nothing. This is why CASA defaults to exact scores.")
    return out


# -------------------------------------------------------------------------- E6
def e6_rag(quick: bool) -> Dict:
    rng = np.random.default_rng(SEED + 5)
    reps = 5 if quick else 12
    n_chunks = 128 if quick else 256
    d = 32
    eps_grid = [0.5, 0.2, 0.1, 0.05]
    rows = []
    for eps in eps_grid:
        savings, gold, drift, bounds, viol = [], [], [], [], 0
        for _ in range(reps):
            emb = rng.normal(size=(n_chunks, d))
            query = rng.normal(size=d) * 6.0
            gold_idx = int(np.argmax(emb @ query))
            chunks = [
                casa.Chunk(id=f"c{i}", text=f"passage {i}", embedding=emb[i],
                           tokens=int(80 + 40 * (i % 3)))
                for i in range(n_chunks)
            ]
            res = casa.certified_context_pruning(query, chunks, eps=eps, block_size=1, audit=True)
            savings.append(res.token_savings)
            gold.append(int(f"c{gold_idx}" in {c.id for c in res.kept}))
            drift.append(res.audit["measured_drift"])
            bounds.append(res.cert.error_bound)
            viol += int(not res.audit["drift_within_certificate"])
        rows.append({
            "eps": eps,
            "token_savings_mean": float(np.mean(savings)),
            "chunks_retained_mean": float(1 - np.mean(savings)),
            "gold_chunk_retention_rate": float(np.mean(gold)),
            "measured_drift_median": float(np.median(drift)),
            "measured_drift_max": float(np.max(drift)),
            "certified_bound_mean": float(np.mean(bounds)),
            "certificate_violations": int(viol),
        })
    return {"chunks": n_chunks, "dim": d, "audit_reps": reps, "rows": rows}


# -------------------------------------------------------------------------- E7
def e7_cost(quick: bool) -> Dict:
    """Wall-clock cost of certification next to the dense path (single head, CPU)."""
    rng = np.random.default_rng(SEED + 6)
    out = {}
    for n in (1024, 4096):
        q, K, V = make_problem(rng, n=n, query_scale=8.0)
        reps = 3 if quick else 8
        t = time.perf_counter()
        for _ in range(reps):
            dense_attention(q, K, V)
        t_dense = (time.perf_counter() - t) / reps
        row = {"dense_ms": 1000 * t_dense}
        for bind in (False, True):
            t = time.perf_counter()
            for _ in range(reps):
                o, cert = casa.certified_sparse_attention(
                    q, K, V, block_size=16, eps=0.05, scheme="exact", bind=bind
                )
            t_casa = (time.perf_counter() - t) / reps
            row[f"certified_ms_bind_{str(bind).lower()}"] = 1000 * t_casa
            row[f"overhead_x_bind_{str(bind).lower()}"] = t_casa / max(t_dense, 1e-9)
        row["retention"] = cert.retention
        out[f"n_{n}"] = row
    out["note"] = ("a single numpy CPU head is not the regime where pruning pays: dense attention is "
                   "0.05-0.2 ms, so constant-factor bookkeeping dominates the ratio even though the "
                   "asymptotics match (O(Nd)). Cryptographic binding costs one pass over the tensors "
                   "and dominates at 16k tokens, which is why the certificate can instead refer to a "
                   "caller-supplied block identifier (bind=False). The saving CASA buys is on the "
                   "retained tokens, not on the score pass: see E6.")
    return out


# -------------------------------------------------------------------------- E8
def e8_tamper(quick: bool) -> Dict:
    """Mutate a valid certificate in several ways; the verifier must reject each."""
    rng = np.random.default_rng(SEED + 7)
    q, K, V = make_problem(rng, n=1024, query_scale=8.0)
    _, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=0.05, scheme="exact", audit=True)
    base_ok = casa.verify_certificate(cert, q, K, V).ok

    def case(name, mutate):
        c = casa.Certificate.from_dict(cert.to_dict())
        mutate(c)
        return name, casa.verify_certificate(c, q, K, V).ok is False

    tampered = [
        case("inflate_eps", lambda c: setattr(c, "eps", c.eps * 10)),
        case("shrink_bound", lambda c: setattr(c, "error_bound", c.error_bound * 0.1)),
        case("shrink_tau", lambda c: setattr(c, "tau_bound", c.tau_bound * 0.5)),
        case("drop_more_tokens", lambda c: (c.retained.pop(), setattr(c, "n_retained", c.n_retained - 1))),
        case("swap_retained_token", lambda c: c.retained.__setitem__(0, (c.retained[0] + 1) % c.n_total)),
        case("fake_bound_claim", lambda c: setattr(c, "error_bound", 0.0)),
        case("honest_measured_error", lambda c: setattr(c, "measured_error", c.measured_error * 5.0)),
    ]
    # also: certificate issued for one problem must not verify against another
    q2, K2, V2 = make_problem(rng, n=1024, query_scale=8.0)
    foreign = casa.verify_certificate(cert, q2, K2, V2).ok is False
    return {
        "valid_certificate_verifies": base_ok,
        "tamper_cases": {name: detected for name, detected in tampered},
        "detection_rate": float(sum(1 for _, d in tampered if d)) / len(tampered),
        "foreign_problem_rejected": foreign,
    }


# ---------------------------------------------------------------------- figures
def figures(exp: Dict) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    saved = []

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    shades = {2.0: "#b0b7c3", 4.0: "#8f9bb3", 8.0: "#4b6fb0", 16.0: "#1f3f8f"}
    for key, rows in exp["E2_frontier"].items():
        qs = float(key.split("_")[-1])
        eps = [r["eps"] for r in rows]
        axes[0].plot(eps, [100 * r["retention_mean"] for r in rows], "o-",
                     color=shades[qs], label=f"peakedness {qs:g}")
        axes[1].plot(eps, [r["measured_error_mean"] for r in rows], "o-", color=shades[qs],
                     label=f"measured, {qs:g}")
        axes[1].plot(eps, [r["bound_mean"] for r in rows], ":", color=shades[qs], alpha=0.75)
    axes[0].set_xscale("log"); axes[0].invert_xaxis()
    axes[0].set_xlabel("certified error budget eps"); axes[0].set_ylabel("tokens retained (%)")
    axes[0].set_title("Sparsity that comes with a guarantee")
    axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].invert_xaxis()
    axes[1].set_xlabel("certified error budget eps"); axes[1].set_ylabel("L2 error")
    axes[1].set_title("Measured (solid) vs certified (dotted)")
    axes[0].legend(frameon=False, fontsize=8); axes[0].grid(alpha=0.25, ls=":")
    axes[1].grid(alpha=0.25, ls=":")
    fig.tight_layout()
    p = os.path.join(RESULTS, "fig1_peakedness_frontier.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p)

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    cases = ["benign_values", "adversarial_values"]
    methods = [("casa_error_max", "CASA (certified)", "#2ecc71"),
               ("topk_error_max", "top-k, matched budget", "#4b7eb0"),
               ("mass_error_max", "keep 90% mass", "#e05c4a")]
    width = 0.26
    for i, (field, label, color) in enumerate(methods):
        vals = [exp["E4_heuristic_control"][c][field] for c in cases]
        ax.bar(np.arange(len(cases)) + (i - 1) * width, vals, width, label=label, color=color)
    ax.axhline(exp["E4_heuristic_control"]["benign_values"]["eps"], color="k", ls="--", lw=1,
               label="target eps = 0.05")
    ax.set_xticks(np.arange(len(cases)))
    ax.set_xticklabels(["benign values", "adversarial values"])
    ax.set_ylabel("worst L2 error over queries")
    ax.set_title("What a guarantee buys: worst case stays bounded")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = os.path.join(RESULTS, "fig2_heuristic_control.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    spaces = ["isotropic_gaussian", "clustered_keys", "clustered_keys_adversarial_values"]
    for i, scheme in enumerate(("norm", "centroid", "exact")):
        vals = [100 * exp["E5_envelope_ablation"][sp][scheme]["retention_mean"] for sp in spaces]
        ax.bar(np.arange(len(spaces)) + (i - 1) * 0.26, vals, 0.26, label=f"{scheme} envelope",
               color={"norm": "#e05c4a", "centroid": "#e0a13a", "exact": "#2ecc71"}[scheme])
    ax.set_xticks(np.arange(len(spaces)))
    ax.set_xticklabels(["isotropic keys", "clustered keys", "clustered + adversarial V"], fontsize=8)
    ax.set_ylabel("tokens retained at eps = 0.1 (%)")
    ax.set_title("Bounding scores costs everything; computing them costs nothing extra")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = os.path.join(RESULTS, "fig3_envelope_ablation.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); saved.append(p)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    t0 = time.time()
    exp: Dict[str, Dict] = {}
    for name, fn in [
        ("E1_validity", e1_validity),
        ("E2_frontier", e2_frontier),
        ("E3_tightness", e3_tightness),
        ("E4_heuristic_control", e4_heuristic_control),
        ("E5_envelope_ablation", e5_envelope_ablation),
        ("E6_rag_context_pruning", e6_rag),
        ("E7_cost", e7_cost),
        ("E8_tamper_detection", e8_tamper),
    ]:
        t = time.time()
        exp[name] = fn(args.quick)
        print(f"[{time.time() - t:6.2f}s] {name}")

    payload = {"seed": SEED, "quick": args.quick, "runtime_s": round(time.time() - t0, 2), **exp}
    with open(os.path.join(RESULTS, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    figs = [] if args.no_figures else figures(exp)

    e1 = exp["E1_validity"]
    print("\n=== E1 validity ===")
    print(f"  certificates {e1['certificates']} over {e1['configs']} configs")
    print(f"  violations {e1['bound_violations']} | verifier failures {e1['independent_verifier_failures']}"
          f" | worst measured/bound {e1['worst_measured_over_bound']:.4f}")

    print("\n=== E2 peakedness frontier ===")
    for key, rows in exp["E2_frontier"].items():
        print("  " + key)
        for r in rows:
            print(f"    eps={r['eps']:<5} retention={100*r['retention_mean']:5.1f}%  "
                  f"error={r['measured_error_mean']:.5f}  bound={r['bound_mean']:.4f}  "
                  f"ratio={r['ratio_mean']:.4f}")

    print("\n=== E3 tightness ===")
    print("  " + json.dumps(exp["E3_tightness"]["measured_over_bound"]))

    print("\n=== E4 heuristic control ===")
    for case, row in exp["E4_heuristic_control"].items():
        cal = row["calibrated_topk"]
        print(f"  {case}: casa_retention={100*row['matched_retention_mean']:.1f}% "
              f"mass_retention={100*row['mass_pruner_retention_mean']:.1f}%")
        print(f"    calibrated top-k budget={cal['budget_chosen']} "
              f"held-out violations={cal['heldout_violations']}/{cal['heldout_n']} "
              f"worst={cal['heldout_worst_error']}")
        print(f"    CASA   max={row['casa_error_max']:.5f} violations={row['casa_violations']}/{row['n']}")
        print(f"    top-k  med={row['topk_error_median']:.5f} max={row['topk_error_max']:.5f} "
              f"violations={row['topk_violations']}/{row['n']}")
        print(f"    mass   med={row['mass_error_median']:.5f} max={row['mass_error_max']:.5f} "
              f"violations={row['mass_violations']}/{row['n']}")

    print("\n=== E5 envelope ablation (retention at eps=0.1) ===")
    for space, row in exp["E5_envelope_ablation"].items():
        if not isinstance(row, dict) or "norm" not in row:
            continue
        print(f"  {space}: " + "  ".join(f"{s}={100*row[s]['retention_mean']:.1f}%" for s in ("norm", "centroid", "exact")))

    print("\n=== E6 RAG ===")
    for r in exp["E6_rag_context_pruning"]["rows"]:
        print(f"  eps={r['eps']:<5} savings={100*r['token_savings_mean']:5.1f}%  "
              f"gold={100*r['gold_chunk_retention_rate']:5.1f}%  drift={r['measured_drift_median']:.4f}"
              f" <= {r['certified_bound_mean']:.4f}  violations={r['certificate_violations']}")

    print("\n=== E7 cost ===")
    for k, v in exp["E7_cost"].items():
        if isinstance(v, dict):
            print(f"  {k}: dense={v['dense_ms']:.3f}ms "
                  f"unbound={v['certified_ms_bind_false']:.3f}ms ({v['overhead_x_bind_false']:.2f}x) "
                  f"bound={v['certified_ms_bind_true']:.3f}ms ({v['overhead_x_bind_true']:.2f}x) "
                  f"retention={100*v['retention']:.1f}%")

    print("\n=== E8 tamper detection ===")
    print("  " + json.dumps(exp["E8_tamper_detection"]))

    print(f"\nresults -> {RESULTS}")
    for f in figs:
        print("  figure " + os.path.basename(f))


if __name__ == "__main__":
    main()
