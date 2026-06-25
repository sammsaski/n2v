# Soundly verifying the VNN-COMP 2023 ViT benchmark in n2v

Status: design (branch `feat/vit-vnncomp2023-sound`). Goal: maximize the number
of verified instances on the VNN-COMP 2023 ViT benchmark using **sound**
reachability — every reach op must produce a true over-approximation of the
reachable set (no reachable output is ever excluded). Box, Zonotope and Star
representations are all supported. The approach translates nnVLA's CROWN
soundness math into n2v's set semantics.

This document was produced from a multi-agent design + adversarial-soundness
review pass; the two soundness corrections in §6 came out of that review and are
load-bearing.

---

## 1. Benchmark op inventory & critical path

Both models load at `ir_version=4, opset=9`, input `[b,3,32,32]`, output
`[b,10]`. **Build to the exported ONNX, not the `vit.py` training source** — the
export replaced LayerNorm with **BatchNormalization** and pools with
**ReduceMean over tokens** (not a cls-token gather).

Models (`…/ViT_vnncomp2023/models/{pgd_2_3_16,ibp_3_3_8}/model.onnx`):

| model | patch | tokens | depth | heads | dim | dim_head |
|---|---|---|---|---|---|---|
| PGD_2_3_16 | 16 | 4+1 = 5  | 2 | 3 | 48 | 16 |
| IBP_3_3_8  | 8  | 16+1 = 17 | 3 | 3 | 48 | 16 |

Ordered ops (PGD_2_3_16; IBP is the same pattern, 3 blocks, 17 tokens):

1. `Conv` patch-embed `[3,32,32]·[48,3,16,16]` stride 16 → `[48,2,2]`
2. `Reshape`→`Transpose[0,2,1]` → `[4,48]`
3. `+ cls_token` (`Concat` axis=1) → `[5,48]`
4. `+ positions[5,48]`
5. **Block ×depth — Attn (PreNorm+Residual):** `BatchNorm` (per-token, 48-dim) →
   `q,k,v = MatMul+Add` (W`[48,48]`,b`[48]`) → reshape `[3,5,16]` →
   `S = MatMul(q,kᵀ)` → `Mul 0.25` (=16^-0.5) → `Softmax axis=-1` →
   `MatMul(attn,v)` → reshape `[5,48]` → out-proj `MatMul+Add` → `Add` residual
6. **Block — MLP (PreNorm+Residual):** `BatchNorm` → `MatMul+Add[48,96]` →
   `ReLU` → `MatMul+Add[96,48]` → `Add` residual
7. `ReduceMean axis=1` (tokens) → `[48]`
8. final `BatchNorm[48]`
9. `Gemm` head (transB=1, W`[10,48]`,b`[10]`) → `[10]`

| op | class |
|---|---|
| Conv patch-embed | exact/affine (ImageStar 4-D path exists) |
| MatMul(set, const) + Add bias; Gemm | exact/affine (`Linear`/`affine_map`) |
| BatchNorm (eval) | exact/affine — per-channel `γ̂x+β̂` |
| Add (set+const), Concat cls, +pos | exact/affine |
| Add (set+set) residual | exact, **multi-input** (see §6.2) |
| ReduceMean (tokens), Mul const | exact/affine (`(1/L)Σ`, scalar map) |
| ReLU | piecewise-linear — exact split or triangle approx |
| **MatMul(set, set)** — QKᵀ, A·V | **bilinear — sound relaxation** |
| **Softmax** — exp/rowsum/recip/normalize | **nonlinear — sound relaxation** |

**Critical path:** everything is exact-affine except ReLU (piecewise-linear,
solvable with existing ops) and **attention = QKᵀ → softmax → A·V**. That bilinear
+ softmax block is the *entire* nonlinear deliverable. Notably **absent / not
required**: LayerNorm-with-variance, Div, GELU, Tanh, Sigmoid, cls-gather pool.
This makes the benchmark far more tractable than a general ViT.

---

## 2. What already exists in n2v (reuse map)

Upstream `main` (the branch base) already has, in `n2v/nn/reach.py`:

- `_mul_stars_mccormick` (≈L2031): **element-wise** McCormick star product with
  the 4 envelope inequalities, fresh predicate per dim, LP bounds, and
  shared-predicate vs block-diagonal-join handling. **Reusable primitive.**
- `_mul_boxes` / `_mul_zonos`: four-corner interval element-wise products.
- `_add_sets` (≈L1844), `_join_predicates` (≈L1733), `_join_star_systems`
  (≈L1772), `_same_predicate_system` (≈L1702): residual/branch joining.
- `_handle_onnx_matmul` (≈L1527): handles `set @ const` only — **returns `None`
  for set@set** (the wiring gap to fill).
- `_handle_onnx_binary_op` (≈L1160): residual Add(set,set) → `_add_sets`.
- Conv2d ImageStar reach (`conv2d_reach._conv2d_imagestar_4d`), `batchnorm`,
  `relu`, `linear`, plain `softmax` (its **Star path is a box-lift** — replace).

`n2v/sets/star.py`: `Star = {c + V[:,1:]·α | C·α≤d, pred_lb≤α≤pred_ub}`, col 0 of
`V` is the center. `affine_map(W,b)` exact. `get_ranges()`/`get_range(i)` =
**LP-exact** per-dim bounds (honor `C·α≤d`). `estimate_ranges()` = fast, **ignores
`C`** (predicate-box only) — sound but loose. `contains(x, method='lp')` = the
authoritative soundness oracle. New predicate variables are added by appending a
generator column to `V`, zero-padding existing `C` rows, vstacking constraint
rows, and extending `pred_lb/ub` (idiom: see `convex_hull`/`minkowski_sum`).

**To build:** set@set matmul (QKᵀ/A·V), softmax relaxation, sign-aware A·V
tightening, provenance-aware residual, and the graph wiring.

---

## 3. Sound reach design — per op, per set-rep

Conventions: token-state is flattened token-major `(L·C,)`, row `= token·C + c`,
matching `ImageStar.to_star()` after the patch-embed conv. **Global invariant:**
the output set encloses `{f(x) : x ∈ input set}`; for Star, every concrete
`y=f(x)` admits `α` with `C·α≤d`, `pred_lb≤α≤pred_ub`, `y=c_out+V_out[:,1:]·α`.

### 3.1 Affine ops (conv, BatchNorm, Linear/QKV/out-proj/MLP/Gemm, +cls, +pos, scale, ReduceMean)
Exact in all reps via `affine_map`. Per-token Linear `W` on the `(L·C,)` state =
`affine_map(kron(I_L, W), tile(b, L))`. BatchNorm eval = per-channel
`y=γ̂x+β̂`, `γ̂=γ/√(var+ε)`, `β̂=β−γ̂·mean` → same `kron(I_L, diag(γ̂))` broadcast
(⚠ the stock `batchnorm_star` builds a `C×C` diag and will crash on the `L·C`
state — wire the per-token broadcast). ReduceMean over tokens = `affine_map(
kron((1/L)·1_Lᵀ, I_C))`. Conv patch-embed uses the existing ImageStar 4-D path
then `to_star()` (a pure row permutation — predicate system preserved exactly).

### 3.2 Scores S = scale·QKᵀ  [bilinear]
`S_ij = scale·Σ_m Q_im·K_jm`, scale = 1/√d_head = 0.25 > 0. This is the only place
Q,K uncertainty is consumed.

- **concretize (default, = CROWN):** take interval bounds of Q,K
  (`get_range`/`get_ranges` LP for Star, `get_bounds` for Zono, `lb/ub` for Box),
  do a sound interval matmul (Rump midpoint-radius — see §5), scale, →
  `[S_lo,S_hi]`. For Star: `Star.from_bounds(S_lo,S_hi)` (box-lift of the *scores
  only*; softmax then keeps its own relaxation symbolic). Drops the
  ‖q‖/‖k‖ correlation — sound, loose.
- **mccormick (precision lever, §7 Slice 2):** per product term introduce a
  fresh predicate `z=Q_im·K_jm` with the 4 McCormick rows; `S_ij=scale·Σ_m z`
  is then exact-affine in the z's. Keeps the QK correlation concretize drops.
  ⚠ McCormick bounds **must** be live `get_range` (LP) on the current star,
  never stale/`estimate`.

### 3.3 Softmax = exp / rowsum / reciprocal / normalize  [nonlinear]
Per row over the key axis. `E=exp(S)`, `T_i=Σ_j E_ij`, `A_ij=E_ij/T_i`.

- **exp (convex):** lower = tangent at `mid=(s_lo+s_hi)/2`:
  `E≥exp(mid)·(s−mid)+exp(mid)`; upper = chord between `(s_lo,exp s_lo)` and
  `(s_hi,exp s_hi)`. Box interval is the exact monotone `[exp s_lo, exp s_hi]`.
- **row sum** `T_i=Σ_j E_ij`: exact-affine in the E's. Use the **per-target
  correlated** denominator (CROWN ratio trick) — for the cell `A_ij` being
  bounded, hold its own numerator at the ratio-extremising endpoint while the
  rest sit opposite. ⚠ see §6.1 — this stays **(i,j)-local**.
- **reciprocal 1/T (convex, T>0):** upper = chord, lower = tangent at midpoint;
  clamp denom ≥ 1e-300. Slopes ≤ 0 (sign care).
- **normalize** `A_ij=E_ij·(1/T_i)`: McCormick product of two boxed factors →
  `A_ij∈[a_lb,a_ub]⊆[0,1]`, **`a_lb≥0` enforced**.
- **Box / IBP path:** correlated row-softmax bound (monotone): for the upper of
  `A_ij`, put logit `j` at `s_hi` and the rest at `s_lo`, softmax; mirror for the
  lower. Tightest box. Always **intersect Star bounds with IBP** for
  non-regression: `a_lb=max(a_lb^shi,a_lb^ibp)`, `a_ub=min(…)`.

### 3.4 O = A·V  [bilinear, sign-aware — the part to get right]
`O_id=Σ_j A_ij·V_jd`, `A_ij∈[a_lb,a_ub]`, `a_lb≥0` invariant; `V_jd` arbitrary
sign. Envelope linear in V, slopes ≥0:
- **V_jd≥0:** `up=a_ub·V`, `low=a_lb·V`.
- **V_jd≤0:** `up=a_lb·V`, `low=a_ub·V` (**multipliers swap** — the historically
  wrong branch; getting it backwards excludes true outputs).
- **mixed (v_lo<0<v_hi):** McCormick chords, `width=v_hi−v_lo`:
  `up_slope=(a_ub·v_hi−a_lb·v_lo)/width`, `up_bias=a_lb·v_lo−up_slope·v_lo`;
  `low_slope=(a_lb·v_hi−a_ub·v_lo)/width`, `low_bias=a_ub·v_lo−low_slope·v_lo`.
  Floor slopes at 0 (FP). Sum facets over j.
- **Star:** since A is concretized to `[a_lb,a_ub]` (Q,K already absorbed), the
  envelope is affine **in V only**. Encode lower+upper facets as **two `C·α≤d`
  rows on a fresh `O_id` predicate** (V_jd expressed via its generator rows).
  ⚠ Do **not** use `affine_map` here — that treats the relaxation as exact and is
  unsound (CROWN's two affine maps cannot be one Star basis; see §6.3).
- **Generic fallback:** A·V via the existing element-wise McCormick contraction
  is sound for arbitrary `A` sign; the sign-aware envelope is the tighter
  refinement when `a_lb≥0`. Assert `a_lb≥0` on entry (fail loud).

### 3.5 ReLU / pool / head / output property
ReLU: existing exact-split or triangle. Pool/Gemm: affine. **Property** (argmax
preservation, robustness disjunction): for each `i≠label` prove margin
`Y_label−Y_i>0`. Star: `get_min` of the margin over the polytope; or
`intersect_half_space(e_i−e_label, 0)` then `is_empty_set()`. Robust ⟺ all i.
Intersect LP margin with IBP margin (take the max).

---

## 4. Soundness invariants summary (where naive nnVLA→n2v translation breaks)

1. **A·V branch V≤0:** multipliers swap (upper←a_lb, lower←a_ub).
2. **CROWN dual envelopes → single Star V:** lower+upper become `C·α≤d` rows on a
   fresh predicate, never an `affine_map`.
3. **McCormick / envelope bounds must be live LP `get_range`**, not stale/estimate.
4. **exp/recip tangent-lower assume convexity on the whole interval** — valid for
   exp (global) and 1/T (T>0 only). Do **not** reuse for GELU / non-monotone ops.
5. **Residual Add provenance** — see §6.2.
6. **softmax row-sum=1 is not a usable equality** under relaxation; only the
   sound inequality `Σ_j a_lb ≤ 1 ≤ Σ_j a_ub` may be added.

---

## 5. Sound interval matmul (Rump midpoint–radius)

For `C = A@B`, `A∈[Al,Au]`, `B∈[Bl,Bu]` (batched over leading dims):
`Am=(Al+Au)/2, Ar=(Au−Al)/2`, `Bm,Br` likewise.
`Cm = Am@Bm`, `Cr = |Am|@Br + Ar@|Bm| + Ar@Br`, `C∈[Cm−Cr, Cm+Cr]`.
Proof: each term's deviation `|A_im B_mj − Am Bm| ≤ |Am|Br+Ar|Bm|+ArBr`; summing
over the contraction index gives `Cr`. Sound, batches with `np.matmul`. Negative
`scale` handled by `min/max(scale·Cm±scale·Cr)`.

---

## 6. Soundness corrections from the adversarial review (load-bearing)

### 6.1 Reciprocal predicate must stay (i,j)-local
The per-target correlated denominator `T_i^{(j)}` is valid only for the single
matched cell `A_ij`. `1/T_i` is **row-shared** (j-independent); its true range can
far exceed `1/T_i^{(j)}`. **Never** materialize a shared row-reciprocal predicate
`r_i` from per-target denominators — that excludes reachable `1/T_i` for any
non-matched consumer. Either (a) build the reciprocal interval+facets fresh per
target cell and use them **only inside that one cell's product**, then discard;
or (b) if a single shared `r_i` predicate is wanted, bound it with the **true**
row-sum extremes `r_i∈[1/ΣE_ub, 1/ΣE_lb]` and apply the per-target tightening only
inside each cell's product, never to `r_i`'s box. A §8 test must check `1/T_i`
itself (not just `A_ij`) lies in the predicate bounds for all samples.

### 6.2 Residual Add must be provenance-aware
n2v's `_same_predicate_system` decides "shared α" by **structural equality** of
`(nVar,C,d,pred_lb,pred_ub)` — there is no predicate provenance on a `Star`. Two
**independent** box-lifted stars with coincidentally identical bounds are then
mis-identified as correlated, and `_add_sets` takes the exact `V1+V2` branch,
cancelling generators and **excluding true outputs** (CE: `x∈[-1,1]` and
`branch=(-1)·x'` with `x'` an *independent* `[-1,1]`; structural check passes,
result collapses to `{0}` vs true `[-2,2]`). This is a latent *soundness* concern
in the exact-branch gating.

Two distinct issues, do not conflate: (a) the block-diagonal **fallback** in
`_add_sets`/`_join_predicates` is **sound but loose** — tightening it to a
prefix-aligned join is the tracked **precision** future-work **I-35** (referenced
in the `_add_sets` docstring; lives in `.claude/issues.md`, which is gitignored
and not in-clone, so the issue text is unavailable here). (b) The
structural-equality **exact-branch gate** above is a separate latent
**unsoundness**. The repo already found & fixed a prior prefix-align
under-approximation in `Concat` (`TestConcatMismatchedPredicates` in
`tests/soundness/test_soundness_mul_div_concat.py`) — prefix-align must be done
carefully.

For our attention residual `y = x + attn(x)`, `attn(x)` is built **from** `x` by
only **appending** predicate columns, so `x`'s α is a genuine **prefix** of
`attn(x)`'s α. The sound **and tight** op is **provenance-aware prefix-aligned
addition**: pad `x`'s generators with zeros for the fresh columns and add over
`attn(x)`'s constraint system (which already contains `x`'s). We use provenance
we control (never the structural fast path), which delivers I-35's tightness
*and* avoids the gate's false-positive. `relational.py::_relational_join` already
implements this prefix-aligned construction (input predicates identified,
block-diagonal only over the appended tails) and is directly adaptable — it
stacks states, the residual needs element-wise add. Where provenance can't be
proven, fall back to the block-diagonal join (sound, looser).

### 6.3 Other guardrails
- A·V: assert `a_lb≥0` (and `a_ub≥a_lb`) on entry; the slope FP-clamp is then a
  no-op (so it can't hide an upstream softmax bug).
- Zono A·V error generator radius = **max of the envelope gap over the V-box
  vertices**, never the gap at the box center (center-eval under-sizes the radius
  and excludes true outputs).
- exp monotone clamp: tighten to `[max(line_lb,exp s_lo), min(line_ub,exp s_hi)]`
  (the loose `min/max` form is still sound).
- Patch-embed: assert the chosen flatten (HWC vs CHW) matches the ONNX
  reshape/transpose so the per-token Linear acts on the feature axis.

---

## 7. Implementation order (vertical slices, each kept soundness-green)

- **Slice 0 — affine backbone + Box end-to-end.** Conv/BN/Linear/Gemm/ReduceMean/
  scale/residual/ReLU + Box softmax (IBP correlated row bound) + Box bilinear
  (Rump). Wire set@set matmul. Verifies the easy instances; proves the graph
  traverses end-to-end. Ship the ONNX-oracle harness here.
- **Slice 1 — Star attention.** Star softmax (exp tangent/chord + per-target
  correlated rowsum + reciprocal + sign-aware A·V), Q,K concretized; residual via
  prefix-align; intersect every A and the margin with IBP. The big precision win.
- **Slice 2 — symbolic McCormick QKᵀ.** Replace concretized scores with fresh
  McCormick predicates (keeps QK correlation). Gated behind a per-instance budget.
- **Slice 3 — Zono path.** Bilinear + A·V via error-generator; softmax via
  box→zono. A faster mid-precision tier and a soundness cross-check.
- **Slice 4 — exact ReLU + tuning.** Exact ReLU split on the small MLP; LP-vs-
  estimate bound tuning.

## 8. Soundness test plan

Follow `tests/soundness/`. Per op: seed RNG, N≈200 samples, tol ±1e-5; build the
torch/np op as independent ground truth; sample the **input** set, run the op,
assert each output `contains` (Star LP) / within bounds (Box/Zono). Hammer the
historically-wrong cases: A·V branch B (V≤0) and C (mixed); McCormick sign rows;
the `1/T_i` containment (§6.1); residual with independent-but-identical operands
(§6.2 regression). End-to-end: load each ONNX, encode the per-channel input box
(eps `(1/255)/std_c`), reach, check the margin; oracle = forward on N sampled
inputs, assert (i) every sampled logit ∈ reach enclosure, (ii) no "verified"
instance has a sample that violates argmax.
