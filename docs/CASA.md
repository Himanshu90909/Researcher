# CASA: Certified Adaptive Sparse Attention

**Proof-carrying approximation for attention and retrieved context**

`src/research/casa/` · 18 tests · 8 experiments · reproducible in ~16 s

---

## 1. The problem this solves

Every fast-attention and context-pruning system in production answers the same
question with an anecdote: *"we dropped 90% of the tokens and the outputs looked
fine."* That is not a claim you can ship against, because "looked fine" is a fact
about the sample you looked at, not about the query you are about to serve.

CASA replaces the anecdote with a number. Given a tolerance `eps`, it

1. finds the **fewest** key blocks whose pruned output is provably within `eps` of
   the exact output (L2, in the units of the value vectors),
2. returns the output **and** a certificate: a deterministic upper bound on the
   error, plus the witness needed to re-derive it,
3. refuses to prune when nothing can be certified, instead of quietly breaking the
   promise (`cert.fallback == True`).

The bounds hold for **every** query, key and value set. No concentration
inequality, no sampling, no distributional assumption, no calibration set. A
reviewer with the tensors can re-derive the guarantee in milliseconds
(`verify_certificate`), and 7/7 tampering attempts are caught (E8).

Two settings are supported by the same theorem:

| setting | what is dropped | what the certificate buys |
| --- | --- | --- |
| attention | key blocks | bounded deviation of the head's output |
| retrieval (RAG) | retrieved chunks | bounded drift of the softmax-weighted context vector |

---

## 2. Notation and Theorem 1

Let `q ∈ R^d`, keys `k_i`, values `v_i ∈ R^m`, scores `s_i = <q, k_i>/√d`,
`p = softmax(s)` and

```
o = Σ_i p_i v_i                              (exact / "dense" output)
```

A pruner keeps `S ⊆ [N]`, drops `D = [N] \ S`, and renormalizes over `S`:

```
o_S = Σ_{i ∈ S} p^S_i v_i ,      p^S_i = e^{s_i} / Z_S ,      Z_S = Σ_{i ∈ S} e^{s_i}
```

and `τ = 1 − Z_S/Z` is the **dropped probability mass**.

> **Theorem 1 (renormalized-pruning error bound).** For any reference vector `c`, with
> `D_S = max_{i∈S} ‖v_i − c‖` and `D_D = max_{i∈D} ‖v_i − c‖`,
>
> ```
> ‖o_S − o‖₂  ≤  τ · ( ‖o_S − c‖ + D_D )  ≤  τ · ( D_S + D_D )  ≤  2 τ D
> ```
> where `D = max_i ‖v_i − c‖`.

**Proof.** Since `Z_S = (1−τ)Z`, for `i ∈ S` we have `p^S_i = p_i/(1−τ)`. Put
`d_i = v_i − c`. Then

```
o_S − o = Σ_{i∈S} p^S_i d_i − Σ_i p_i d_i
        = Σ_{i∈S} p^S_i d_i − (1−τ) Σ_{i∈S} p^S_i d_i − Σ_{i∈D} p_i d_i
        = τ (o_S − c) − Σ_{i∈D} p_i d_i .
```

Now `‖o_S − c‖ ≤ Σ_{i∈S} p^S_i ‖d_i‖ ≤ D_S`, and
`‖Σ_{i∈D} p_i d_i‖ ≤ Σ_{i∈D} p_i ‖d_i‖ ≤ τ D_D`. Add the two. ∎

Three remarks that matter for how the result is read.

*The bound is tight to a factor of 2, and the factor is real.* Both `o_S` and `o`
are convex combinations, so no bound of this shape can be smaller than `τD`. The
factor 2 is the price of comparing two differently renormalized mixtures, and it is
attained: two tokens, one kept with value `+D/2·u` and one dropped with value
`−D/2·u` at mass `τ = 1/2`, gives error exactly `τ(D_S + D_D)`. This is
`test_factor_two_is_the_worst_case`.

*The reference `c` is a free tightening knob.* Choosing `c = mean(v)` removes the
common value offset and shrinks both radii; `reference="global_mean"` is the
default, `reference="zero"` is the plain `2τ max‖v_i‖`.

*Nothing about the data is assumed.* The bound is a statement about geometry: it
depends on how far the dropped values sit from the reference, and on how much
probability mass left. That is exactly why it cannot be fooled (E4).

---

## 3. Theorem 2: computing `τ` without reading the dropped tokens

Theorem 1 needs `τ`, and computing `τ` naively means scoring every token. Partition
the sequence into blocks `g` of size `n_g` and let `u_g` be any upper bound on
`max_{i∈g} s_i`. For `S` a union of whole blocks and `m_S = max_{i∈S} s_i` (exact,
only retained scores needed):

> **Theorem 2 (tail mass from block summaries).**
> ```
> Z_D = Σ_{g∈D} Σ_{i∈g} e^{s_i} ≤ Σ_{g∈D} n_g e^{u_g} =: exp(dropped_logmass)
> ```
> and since `Z ≥ Z_S ≥ e^{m_S}`,
> ```
> τ ≤ τ̄ := min( 1, exp(dropped_logmass − m_S) ) .        (sum form)
> ```

**Proof.** `e^{s_i} ≤ e^{u_g}` inside block `g`, so each block mass is at most
`n_g e^{u_g}`; sum over dropped blocks. For the denominator, `Z` contains `Z_S`,
which contains the single term `e^{m_S}`. ∎

Dropping block sizes gives the cruder max form `τ ≤ min(1, N_D e^{u_D − m_S})`,
still shipped as `tail_mass_bound`. The sum form is never worse and is what the
certifier uses. Dropped keys and values are never read individually: each dropped
block contributes one scalar, `log n_g + u_g`.

Combining the two theorems gives the certificate CASA emits:

```
‖o_S − o‖₂  ≤  τ̄ · (D_S + D_all) ,      τ̄ ≤ 1
```

from (i) exact scores of retained tokens, (ii) one log-mass scalar per dropped
block, (iii) the scalar value radius `D_all`.

---

## 4. Where the bound comes from, and why summary-only certification fails

Theorem 2 lets you plug in any `u_g`. The package ships three:

| `scheme` | `u_g` | what the index must store |
| --- | --- | --- |
| `"exact"` | exact block max of the scores | nothing extra (default) |
| `"norm"` | `‖q‖ max_{i∈g}‖k_i‖ / √d` | one float per block |
| `"centroid"` | `(<q,μ_g> + ‖q‖ r_g) / √d` | centroid + radius per block |

The interesting result is that the two envelope-only schemes are **vacuous at
d=64**, and the reason is a scaling argument rather than a tuning failure. An
envelope has to hedge a d-dimensional residual, so its slack grows like
`‖q‖ r_g`, which is Θ(d) for isotropic keys, while the spread of the true scores
grows only like Θ(√d). At d=64 the hedge swamps the signal.

E5 measures it — retention at `eps = 0.1` over 1024 tokens:

| key space | `norm` envelope | `centroid` envelope | exact scores |
| --- | --- | --- | --- |
| isotropic Gaussian keys | 100% (always dense) | 100% | **20.8%** |
| clustered keys | 100% | 100% | **57.0%** |
| clustered keys, adversarial values | 100% | 100% | **70.3%** |

So CASA defaults to `scheme="exact"`. That is a deliberate and honest choice: on
current hardware the score pass is the same order as the value pass, so bounding
scores instead of computing them buys nothing and loses the ability to certify
anything. The envelope schemes are kept because they are the correct baseline for a
future index that can bound `Z_g` directly (a sketch, an LSH bucket, a learned
quantizer) rather than bounding each score inside the block — which is the research
question this negative result points at, not a solved one. Retention also rises with
value-difficulty: the 70.3% in the last row is the certificate correctly charging
more when values are hostile.

---

## 5. Algorithm

```python
def certified_sparse_attention(q, K, V, block_size=16, eps=0.05, ...):
    s        = K @ q / sqrt(d)                       # one score pass
    envelope = block_bounds(q, K, block_size, scheme) # per-block u_g
    order    = argsort(-envelope)                     # most promising first
    for k in 1..G:                                    # all prefixes at once
        S      = first k blocks in `order`
        m_S    = max(s[S])                            # exact, retained only
        lse_D  = logsumexp over dropped blocks of (log n_g + u_g)
        τ̄      = min(1, exp(lse_D − m_S))             # Theorem 2
        bound  = τ̄ · (D_S + D_all)                    # Theorem 1
    k*   = smallest k with bound ≤ eps                # minimality
    out  = renormalized attention over S(k*)          # reuse `s`
    return out, Certificate(...) if k* exists else (dense, fallback certificate)
```

* **Cost** `O(Nd + Nm + G log G)`: one score pass, one value pass, one sort. Every
  prefix is evaluated elementwise in NumPy — there is no Python loop over blocks.
* **Anytime**: the whole curve is available, so a caller with a latency budget reads
  `pattern.curve` and takes the cheapest prefix that meets its tolerance.
* **Minimality is part of the claim**: the returned pattern is the smallest
  certifiable prefix in the block ordering, and the verifier re-checks that one
  fewer block fails (`minimal_for_order`).
* **Honest fallback**: if no prefix certifies, the output is dense attention and the
  certificate says `fallback=True, feasible=False`. No silent guarantee-breaking.

---

## 6. The certificate, and three levels of assurance

```json
{
  "eps": 0.05, "error_bound": 0.0411, "tau_bound": 0.0142,
  "n_total": 2048, "n_retained": 496, "n_dropped": 1552,
  "block_size": 16, "blocks_kept": 31, "scheme": "exact", "reference": "global_mean",
  "max_kept_score": 12.83, "D_kept": 1.41, "D_all": 1.48, "dropped_logmass": 8.26,
  "retained": [0, 3, 5, ...], "retained_digest": "9f2c…",
  "q_digest": "4b71…", "k_digest": "c0ae…", "v_digest": "1d83…",
  "feasible": true, "fallback": false, "measured_error": 0.0006,
  "version": "casa-1.0", "created_at": "2026-11-04T09:12:41Z"
}
```

| level | check | needs | what it detects |
| --- | --- | --- | --- |
| 1 | `cert.check_arithmetic()` | nothing but the certificate | internally inconsistent or over-claimed numbers |
| 2 | `cert.matches_inputs(q,K,V)` | the tensors | a certificate issued for a different problem |
| 3 | `verify_certificate(cert,q,K,V)` | the tensors | a wrong bound, a non-minimal pattern, a dishonest `measured_error` |

The verifier shares no code with the prover except the digest routine: it recomputes
the retained set, the envelopes, `τ̄`, the bound and — by running dense attention —
the true error. `CertificateLedger` hash-chains certificates into JSON Lines so a run
cannot be edited afterwards, and `verify_ledger` re-checks the chain plus every
entry's arithmetic offline.

**E8, tamper detection.** On a valid certificate: 7/7 mutations rejected
(inflated `eps`, shrunk bound, shrunk `τ̄`, dropped retained token, swapped retained
token, zeroed bound, a `measured_error` inflated 5×), plus a foreign `(q,K,V)`
rejected, plus a `1e-12` edit to a single key entry rejected. Detection rate 1.00.
Note that the inflated-`eps` case is caught by the *minimality* check rather than by
the arithmetic: the number stays self-consistent, but the retained set is then larger
than the cheapest certifiable one.

---

## 7. Experiments

`python scripts/run_casa_experiments.py` — seed 20261204, 15.5 s on CPU, writes
`results/casa/results.json` and three figures. Synthetic problems: keys and queries
unit-variance Gaussian, `query_scale` multiplies the query norm and therefore
controls attention peakedness (score standard deviation = `query_scale`), values
`N(0, 1/m)` so that `‖v‖ ≈ 1`.

### E1 — the theorem cannot be violated by a knob setting

1350 certificates over 54 configurations (3 schemes × 3 block sizes × 2 key
geometries × 3 peakedness levels, random `eps`, `N` up to 1024):

| | |
| --- | --- |
| bound violations | **0** |
| independent verifier failures | **0** |
| worst measured/bound ratio | 0.76 (< 1, as the theorem requires) |

The 0.76 matters: at least once, a *real* error used 76% of its certified budget, so
the bound is doing work rather than sitting trivially far away.

### E2 — what a guarantee costs, as a function of peakedness

Retention (%) at 2048 tokens, `block_size=16`, exact scores:

| `eps` | peakedness 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- |
| 0.5 | 97.5 | 62.4 | 10.7 | **2.3** |
| 0.2 | 99.4 | 70.2 | 17.6 | **2.1** |
| 0.1 | 99.6 | 81.2 | 19.4 | **3.1** |
| 0.05 | 99.9 | 86.1 | 24.2 | **3.5** |
| 0.02 | 100 | 91.4 | 24.9 | **3.1** |
| 0.01 | 100 | 91.3 | 26.5 | **5.6** |

Two regimes, stated plainly. A flat head (peakedness 2) **cannot** be pruned within
any useful L2 tolerance: the mass is spread so widely that any block you drop is
worth a large share of the output, and CASA keeps everything. A selective head
(peakedness 8-16) is pruned to **2-26% of tokens** with a certificate. Measured
errors at those settings are 1e-4 to 4e-3, i.e. 2-20% of the output norm (E3).

*This is the honest headline*: certification costs whatever the attention
distribution dictates, and it will tell you when the answer is "do not prune".
A method that always reports 90% savings is not measuring the same thing.

### E3 — how conservative is it?

40 random `(eps, peakedness)` combinations:

| measured / certified | |
| --- | --- |
| mean | 0.036 |
| median | 0.032 |
| p95 | 0.064 |
| max | 0.073 |

The bound is ~28× above the observed error on average, and never below it. Relative
error of the pruned output is 0.34% (mean), 1.2% (p95) of `‖o‖`. This gap is the
price of assuming nothing about the data: Theorem 1 is a worst case over value
geometry, and E4 shows how easily that worst case is reached deliberately.

### E4 — what heuristics do at the same budget

`eps = 0.05`, 30 queries per cell, 1024 tokens.

| | retention | median error | max error | violations of `eps` |
| --- | --- | --- | --- | --- |
| **CASA, benign values** | 21.6% | — | 0.0013 | **0/30** |
| top-k at matched budget | 21.6% | 6e-8 | 1.9e-6 | 0/30 |
| keep 90% of mass | 0.27% | 0.048 | 0.108 | **14/30** |
| **CASA, adversarial values** | 41.1% | — | 0.0005 | **0/30** |
| keep 90% of mass, adversarial | 0.28% | 0.055 | **0.194** | **17/30** |

Three findings.

1. **Probability mass is not an error budget.** Keeping 90% of the softmax mass can
   move the output by 0.19 — nearly 4× the tolerance a practitioner would ask for —
   because the mass sits in the top 3 tokens while the value information does not.
   The heuristic reports "90% of the attention retained" and that number is *true* and
   *useless*.
2. **Certification is invariant to value geometry.** Inflating the values of
   mid-ranked tokens 12× (tokens a mass heuristic cannot even see) leaves CASA's
   error at 0.0005 and its guarantee intact, while the mass heuristic's worst case
   grows from 0.108 to 0.194. Theorem 1 charges for `D_all`, so the certificate
   notices what the heuristic cannot.
3. **Calibration does not substitute for a proof.** Choosing the most aggressive
   top-k budget whose worst error on 10 calibration queries stays under `eps` gives
   budget 2% here; on 20 held-out queries that costs 0 violations on benign values
   (worst 0.0074) but **1/20 violations and a worst error of 0.097 (2× the target)**
   when values are adversarial. Calibration is a bet on the test distribution. CASA
   needs no calibration and cannot lose that bet.

Honest counterpoint: on benign data, top-k at the matched budget is *more* accurate
than CASA chooses to be (1.9e-6 vs 0.0013) because CASA stops at the tightest
pattern it can certify, not the tightest it can achieve. Anyone who can afford a
verification pass should run `audit=True` and use the measured error; the certificate
is what you have when you cannot.

### E5 — summary-only certification, measured

See §4. Envelope-only schemes certify nothing at d=64; exact scores certify 20.8% of
tokens at `eps=0.1` on isotropic keys.

### E6 — certified context pruning for RAG

256 retrieved chunks of 80-120 tokens (d=32), one clearly most-relevant chunk:

| `eps` | token savings | gold chunk kept | measured drift (median / max) | certified bound | violations |
| --- | --- | --- | --- | --- | --- |
| 0.5 | **96.6%** | 100% | 0.071 / 0.171 | 0.412 | 0 |
| 0.2 | **92.1%** | 100% | 0.025 / 0.062 | 0.179 | 0 |
| 0.1 | **90.6%** | 100% | 0.017 / 0.032 | 0.093 | 0 |
| 0.05 | **91.7%** | 100% | 0.009 / 0.016 | 0.042 | 0 |

This is the setting where the economics work: the score pass against chunk
embeddings is cheap and parallel, while every retained chunk costs real prefill
tokens. CASA drops ~90% of the context, never drops the chunk that matters, and
attaches a bound on how far the context vector moved. The measured drift is 4-5×
below its certificate, and 12/12 runs stay inside it at every tolerance.

Scope, honestly: this measures *drift of the softmax-weighted context vector*, which
is what the theorem governs. It is not a bound on the downstream answer, because a
language model is not Lipschitz in its context. Claiming otherwise would be the kind
of overreach this package exists to avoid; end-task evaluation is the missing piece
(§9).

### E7 — cost

Single head, CPU, NumPy, `block_size=16`, `eps=0.05`:

| N | dense | certified, unbound | certified, bound | retention |
| --- | --- | --- | --- | --- |
| 1024 | 0.047 ms | 0.60 ms (12.7×) | 2.03 ms (43×) | 18.8% |
| 4096 | 0.170 ms | 1.88 ms (11.1×) | 11.65 ms (68.6×) | 25.8% |

Read this correctly. The asymptotics match (`O(Nd)`), the constant does not: a NumPy
CPU head at 4096 tokens takes 0.17 ms, so Python-level bookkeeping dominates the
ratio. Cryptographic binding costs one pass over the tensors and dominates beyond
~8k tokens, which is why a caller who already has a stable identifier for the key
block (`bind=False`) can point the certificate at that identifier instead.

CASA is therefore **not** a FlashAttention competitor and does not claim to be. It
buys a guarantee, and it saves the thing that actually costs money at inference time
— retained context tokens (E6) — not the score pass.

### E8 — the verifier has teeth

See §6. 7/7 tamper classes detected, foreign problem rejected, valid certificate
verifies.

---

## 8. Relation to existing work (and what is actually new)

Sparse and approximate attention is a large field: block-sparse kernels, low-rank and
kernel approximations, clustering and eviction policies for the KV cache, and a line
of work on attention approximation with provable or high-probability error control.
Two things are firmly prior art and are not claimed here:

* **the idea of error-bounded sparse attention.** Bounds on attention approximation
  via concentration, and deterministic bounds from norms and mass, exist in the
  literature under several names.
* **pruning by probability mass or top-k scores.** Standard, and E4 is a fair
  characterisation of what it costs.

What this package contributes, stated narrowly enough to be defended:

1. **A shipped certificate object, not a bound in a paper.** The bound, its witness,
   its digest binding to the inputs, and an independent verifier that re-derives
   everything and checks *minimality*, are part of the API surface. To my knowledge
   no attention library returns a checkable proof object alongside its output, and
   none ships an offline arithmetic re-check plus a hash-chained ledger. This is the
   part that makes downstream claims auditable by a third party — the difference
   between "we bounded the error" and "here is the bound, check it".
2. **An exact, minimal, anytime certifier.** Given `eps`, the sparsest certifiable
   prefix is computed in one vectorized pass and the entire retention/error frontier
   is available as a by-product, so latency budgets and accuracy budgets trade off
   without re-running anything.
3. **One theorem, two applications.** The same bound covers attention head
   sparsification and retrieval-context pruning, which is where token costs actually
   live; E6 is, to my knowledge, an unusual way to justify dropping 90% of a RAG
   context — with a number rather than a vibe.
4. **A measured negative result** (§4/E5): score-envelope certification is vacuous at
   d=64, for a scaling reason, not an implementation reason. That is useful to anyone
   who tries this next, and it is the sort of thing that is usually left out of
   papers.

The tightness gap in E3 and the cost profile in E7 are the two places where a
reviewer should push hardest, and both are reported rather than hidden.

---

## 9. Limitations and next steps

* **The bound is worst-case, hence conservative** (28× on average in E3). The factor-2
  in Theorem 1 is irreducible for this shape of argument; the rest of the gap comes
  from charging `D_all` for the dropped set. A per-block value-mean bound (Theorem 3,
  not yet implemented) would replace `τ D_D` by the within-block value residual,
  which shrinks when a block's values are coherent — the natural next experiment.
* **Certificate scope is the context vector.** For RAG this bounds drift of `g`, not
  the answer. Closing that gap needs either a Lipschitz-aware readout or empirical
  end-task evaluation with the certificate as a covariate.
* **Single head, single query.** Extending to multi-head and batched queries is
  bookkeeping (per-head certificates, a batched worst-case bound), not new theory.
* **No fused kernel.** Everything is NumPy/PyTorch-level; the guarantee is real, the
  speed is not the point. A Triton kernel that emits certificates would make E7's
  constant vanish.
* **Envelope-only certification is unsolved** and is the most interesting open
  problem here: to prune *without* scoring, you need an upper bound on a block's
  partition mass `Z_g`, not on its maximum score. Sketches and quantizers give
  approximate answers; turning one into a rigorous one-sided bound is the research.

---

## 10. Reproduce

```bash
python -m pytest tests/test_casa.py -q            # 18 tests, ~2 s
python scripts/run_casa_experiments.py            # 8 studies, ~16 s, CPU only
python scripts/run_casa_experiments.py --quick    # smoke test, ~3 s
```

No GPU, no API keys, no downloads. Only NumPy, PyTorch (for the module wrapper) and
Matplotlib are required.

```python
from src.research import casa

out, cert = casa.certified_sparse_attention(q, K, V, block_size=16, eps=0.05)
assert cert.valid                       # bound <= eps, not a fallback
print(cert.retention, cert.error_bound) # 0.242 0.0411
casa.verify_certificate(cert, q, K, V)  # independent re-derivation: PASS

# retrieval
res = casa.certified_context_pruning(query_emb, chunks, eps=0.1)
print(res.token_savings, res.audit["measured_drift"], res.cert.error_bound)
```
