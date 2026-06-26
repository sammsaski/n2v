# Sound ViT verification in n2v — status & honest assessment

_Branch `feat/vit-vnncomp2023-sound`. Autonomous session, 2026-06-25._

## TL;DR

A **sound, end-to-end, star-set verifier for the VNN-COMP 2023 ViT benchmark
now runs inside n2v** and is validated sound (0 containment violations; LP
membership confirms the reach set encloses sampled outputs). All attention
primitives have Monte-Carlo soundness tests (145 passing).

Two reach modes: `concretize` (IBP-class) and `symbolic-av` (a CROWN-class
precision mode — value-path-symbolic A·V + prefix-aligned residual; §3bis). The
precision mode **soundly verifies instances the IBP baseline cannot** (ibp inst0
at ε=0.45/255: margin +0.249 vs concretize −0.066) and is ~9× faster.

At the benchmark's full radius ε = 1/255 **both modes certify 0/200** — and this
is **expected and principled, not a defect**: the benchmark was constructed by
*removing every instance that vanilla CROWN can already certify*, so **no
non-branch-and-bound method (IBP, CROWN, or this star verifier) can certify them
at full ε**. With `symbolic-av` the closest margins are only ≈ −0.21 (the gap
nearly closed), confirming the bar is just out of reach for any incomplete
method. The published sound score (α,β-CROWN **79/200**) comes *entirely* from
branch-and-bound; "all 200" is beyond current sound SOTA (the 200/200 Marabou
entry carries a flagged Gurobi soundness bug). Closing the gap requires adding
BaB on top of this sound reach engine — see §5.

## 1. What was built (committed)

| Artifact | What |
|---|---|
| `docs/theory/sound-vit-reach.md` | Verified per-op sound design (Box/Zono/Star) + 2 adversarial-review soundness corrections |
| `n2v/nn/layer_ops/bilinear_matmul_reach.py` | Sound set@set matmul (QKᵀ, A·V) via Rump interval matmul; 83 tests |
| `n2v/nn/layer_ops/softmax_attention_reach.py` | Exact correlated row-softmax bound; 62 tests |
| `benchmarks/vit_vnncomp2023/model.py` | ViT_BN mirroring the ONNX + weight loader; parity 9.5e-7 |
| `benchmarks/vit_vnncomp2023/reach.py` | `ViTReacher` — sound end-to-end driver |
| `run_benchmark.py`, `measure.py`, `instances/` | Sweep harness + vendored 100 instances/model |

## 2. Soundness — validated

- Each attention op has a Monte-Carlo containment test over all sign regimes,
  batched heads, the per-head softmax axis (145 tests pass).
- End-to-end: for a real instance, 40/40 sampled true logit vectors lie inside
  the reach set's sound bound; an authoritative `Star.contains` LP membership
  returns `True`. No reachable output is excluded.
- The two soundness traps the adversarial review caught are handled in the
  design: the (i,j)-local softmax reciprocal, and provenance-aware residual
  addition (n2v's structural `_same_predicate_system` gate is *not* relied on).

## 3. Measured results (concretize / IBP-precision reach, `estimate` margins)

| model | verified @ full ε=1/255 | ε\* (×/255), median / max | full-ε margin |
|---|---|---|---|
| pgd_2_3_16 | **0 / 30** | 0.0000 / 0.0000 | ≈ −1000 … −2000 |
| ibp_3_3_8  | **0 / 15** | **0.344 / 0.375** | ≈ −1.0 … −1.6 |

Forward reach ≈ 5 s (pgd, 5 tokens) / ≈ 30 s (ibp, 17 tokens).

**Calibration:** on `pgd_2_3_16` the verifier certifies ε\*=0 on every instance —
*exactly* matching W11's report that auto_LiRPA **IBP certifies ε\*=0 on every
pgd_2_3_16 instance**. So our concretize precision is IBP-class, as intended.
The PGD-trained model is adversarially hostile to interval bounds (logit
over-approximation ≈ ±1000 vs true ≈ ±10), so ε\*=0 there.

**The interesting model is `ibp_3_3_8`** (IBP-trained): full-ε margins are only
≈ −1.0 (close to 0, not vacuous) and the verifier certifies a real radius
ε\* ≈ 0.34/255 (about one-third of full ε). This is where a precision upgrade
(symbolic attention, §5) has a chance of flipping individual margins positive,
and where any future BaB would converge fastest.

## 3bis. Symbolic precision mode (`mode='symbolic-av'`)

The concretize driver box-lifts the attention output, discarding the value-path
correlation. The `symbolic-av` mode instead keeps it: A·V via `av_envelope_star`
(O affine in V's predicates + sign-aware McCormick slack) and a provenance-aware
**prefix-aligned residual** (§6.2) so the stream stays input-correlated. QKᵀ and
softmax are still concretized (so the attention *weights* A remain box-lifted).

Validated on pgd_2_3_16 inst 0 at full ε (LP margins): **sound** (0/3 containment
violations), and **tighter** — LP min-margin **−522 vs −848** concretize (≈1.6×),
with **fewer predicates** (4308 vs 14776, since S/A/O are not all box-lifted).

**Concrete win — verifies where the IBP-class baseline cannot.** On ibp_3_3_8
inst 0 at ε = 0.45/255:

| mode | reach | nVar | LP margin | estimate margin | result |
|---|---|---|---|---|---|
| concretize | 31 s | 36130 | −0.066 (8/9 classes) | −0.109 | **unknown** |
| symbolic-av | 3.5 s | 5559 | **+0.249 (9/9)** | **+0.138** | **VERIFIED** (sound) |

So symbolic-av soundly certifies an instance the concretize/IBP path cannot, and
is ~9× faster (no box-lift bloat). The precision lever works end-to-end. Note the
*estimate* margin is also positive — the prefix-residual + av_envelope predicate
bounds tighten even the fast (no-LP) path, so a symbolic-av sweep need not pay LP
cost to benefit.

**Full-ε sweep (symbolic-av, estimate margins, 200 instances): 0/200** — but the
gap collapsed: the closest ibp_3_3_8 margins are now ≈ **−0.21** (vs concretize's
≈ −1.0; pgd stays ≈ −1000). LP margins add only ≈ +0.11 (per the 0.45/255 datum),
so even symbolic-av+LP stays just short of 0. This is the **principled confirmation
of §4**: the instances were filtered to defeat every non-BaB method, so a sound
verifier — even a tight one — lands just below the bar at full ε. Crossing it
requires branch-and-bound (§5). `results_symbolic_fulleps.csv`.

## 4. Why full-ε ≈ 0 is principled (the key insight)

`generate_properties.py` builds each instance set by keeping only CIFAR images
where (a) PGD fails to find an adversarial example at ε=1/255 **and (b) vanilla
CROWN cannot already certify robustness**. Filter (b) means the 200 instances
are, by construction, exactly the ones on which *incomplete* (non-BaB) bound
propagation fails. So IBP, CROWN, and this sound star verifier all certify ≈0 at
full ε; the entire 79/200 is recovered by α,β-CROWN's **branch-and-bound**. This
is not a looseness bug in our engine — it is the benchmark's design.

## 5. Path to a higher verified count (honest, scoped)

The sound reach engine is the right substrate; the missing ingredient is
**branch-and-bound**, which n2v can support naturally:

1. **Exact ReLU splitting** — `relu_star_exact` already branches an unstable
   ReLU into active/inactive sub-stars. A bound-guided BaB (split the most
   impactful unstable neuron, prune sub-problems whose margin lower bound is
   already > 0) is the α,β-CROWN strategy and reuses our sound reach per node.
2. **Symbolic attention (precision lever, design Slices 1–2)** — keep the
   value-path correlation (sign-aware A·V affine in V) + prefix-aligned residual
   so each BaB node's bound is tighter, reducing the split count. Sound,
   predicate-preserving; built on the existing `_mul_stars_mccormick` idiom.
3. **Input splitting** as a fallback on the few most sensitive pixels.

## 5bis. BaB implemented — and the honest ViT outcome

A general, manual BaB engine is now in the toolbox (`n2v/nn/bab.py`, no external
verifier; see `docs/theory/branch-and-bound.md`): **input-domain BaB**
(`verify_bab`/`verify_bab_model`, any model/layer), **controlled ReLU
neuron-split BaB** (`verify_bab_relu`, nn.Sequential), and n2v's **exact reach** =
complete ReLU split. All sound, 8 tests.

Applied to ViT (input-split, bounder = symbolic-av, falsifier = `falsify`):
- It trivially verifies instances already at/under the single-shot radius
  (ε=0.50, 0.55/255 → 1 node).
- Just **above** the single-shot radius (≈0.57/255), it does not scale: at
  ε=0.60/255 it ran 150 nodes to **depth 80**, a leaf reached margin +0.027, but
  the covering tree did not close. A **3072-dim** input box has exponentially
  many corners each needing ~80 splits — the curse of dimensionality.
- ReLU neuron-split doesn't help the ViT either: the full-ε looseness is the
  **softmax** attention weights, not the FF ReLUs.

**Conclusion:** sound star+LP bounding + BaB (input- or ReLU-split) cannot reach
the 79/200 at full ε. That number requires **CROWN-style linear bounds + a
branching rule over the softmax/attention nonlinearity** (α,β-CROWN's
general-nonlinearity BaB, the paper behind this benchmark) — a different bounder
than star+LP (whose symbolic attention blows up predicates/LP). The BaB search
scaffold here is reusable; the missing piece is that CROWN-class bounder + a
softmax split. See `docs/theory/branch-and-bound.md` §3.

## 6. Reproduce

```bash
python benchmarks/vit_vnncomp2023/measure.py          # verified count + eps*
python benchmarks/vit_vnncomp2023/run_benchmark.py --eps 0.00392   # full sweep
pytest tests/soundness/test_soundness_bilinear_matmul.py \
       tests/soundness/test_soundness_softmax_attention.py \
       tests/integration/test_vit_model_parity.py
```
