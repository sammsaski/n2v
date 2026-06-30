# LP-free CROWN ViT verifier — translated from NNV `ViTCrown` (no NNV dependency)

This is the n2v translation of NNV's **now-working** ViT attention verifier
(`code/nnv/examples/Transformer/ViT_VNNCOMP2023/ViTCrown.m`). The decisive change
from every prior n2v attention track (`reach.py` symbolic-av, `attn_score_bab.py`,
the FF-ReLU-split BaB — all star+LP) is that **there is no LP solver anywhere**:
the certificate is dense backward linear-bound propagation (α,β-CROWN style). This
is exactly why it verifies an instance the star+LP route could not — the binding
margin LP was a 23k-variable solve that did not scale.

## Result — VNN-COMP 2023 `ibp_3_3_8`, ε = 1/255

**NNV "instance 11" (0-based index 10) is VERIFIED robust, LP-free, sound.** Every
intermediate number matches NNV's `ViTCrown` to 3–4 decimals:

| stage | n2v `vit_crown` | NNV `ViTCrown` |
|---|---|---|
| single-shot backward CROWN | −0.33963 | −0.3395 |
| + refine 1 pass | −0.00276 | −0.0029 |
| + refine converged (2) | −0.00116 | −0.0011 |
| + α-optimization | **+0.00250 ✅** | +0.00257 ✅ |

certified margins (LB): `5.659 6.712 0.504 +0.0025 2.482 3.160 1.339 6.317 3.503`
(≈ NNV's `5.660 6.711 0.504 +0.00257 2.482 3.159 1.339 6.317 3.503`).
**Soundness:** 0 / 20 000 random + 300-step PGD violations of the certified
margin; the certificate is a true lower bound (mirrors NNV's 0 / 100 000 MC, no CE).
Runtime ≈ 35 s, no LP.

> ⚠️ **Indexing.** NNV's "instance 11" is **1-based** (`M.images(11)`), i.e. n2v's
> **0-based index 10** (the closest instance, single-shot −0.0627 in the old
> star+LP track). Use `--instances 10` for the verified one.

### Generalization (refine + α, ~30 s each, no BaB) — reproduces NNV's table

| idx (NNV#) | n2v | NNV | | idx (NNV#) | n2v | NNV |
|---|---|---|---|---|---|---|
| 10 (#11) | **+0.00250 ✅** | +0.00257 | | 45 (#46) | −0.01145 | −0.01145 |
| 28 (#29) | −0.00591 | −0.00586 | | 34 (#35) | −0.01605 | −0.01601 |
| 0 (#1)  | −0.00780 | −0.00787 | | 86 (#87) | −0.03311 | −0.03317 |
| 88 (#89) | −0.01249 | −0.01111 | | | | |

The 6 near-misses are −0.006…−0.033 — exactly NNV's documented GenBaB regime (a
few score-branch nodes close them). The completeness step is the next increment.

## Why this works where the LP route walled out

Three sound mechanisms, each ported faithfully (NNV's own diagnosis):

1. **Score-carrying softmax backward** (the missing piece). Softmax is *decomposed*
   into `exp → reduce_sum (affine) → reciprocal → broadcast (affine) → eprod`, each
   with a sound tangent/chord or McCormick backward relaxation, so a real
   coefficient flows backward to the score — restoring the Q–K input correlation
   that the old constant-box softmax zeroed.
2. **CROWN intermediate-bound refinement** (the tightness driver). Replacing the
   loose IBP boxes at every nonlinearity input with bounds from a backward CROWN
   pass collapses the looseness (−0.3396 → −0.0012). This is what tightens the FF
   ReLU pre-activation boxes (Track A's binding looseness): block-2 FF pre-act
   width 6.44 → 1.78.
3. **α-optimization** (crosses zero). Projected-gradient ascent (Adam, torch
   autograd) on the ReLU lower slopes **and** the attention-McCormick plane
   interpolation `aL,aU` on every bilinear, with **sound double re-evaluation** —
   any α ∈ [0,1] is sound, so the verdict never depends on the autodiff arithmetic.

## Code

- `n2v/nn/crown_reach.py` — the toolbox-general LP-free CROWN op-DAG engine:
  `forward_ibp`, `backward_crown` (relu / exp / reciprocal / eprod / bmatmul
  relaxations), `crown_bounds` / `refine_bounds`, `optimize_alpha`. The same
  backward pass runs in numpy (the sound certificate) and, with `tensor=True`, in
  torch (gradients for α only). No benchmark assumption — keyed on the op DAG.
- `benchmarks/vit_vnncomp2023/vit_crown.py` — the ViT lowering (`to_ops`, mirroring
  `ViTCrown.toOps` + `ViTReach.patchEmbed`), `eps_box`, `verify_instance`.
- `benchmarks/vit_vnncomp2023/run_vit_crown.py` — the runnable driver / sweep.
- Tests: `tests/soundness/test_soundness_crown_reach.py` (engine, self-contained
  synthetic attention DAG), `tests/integration/test_vit_crown_lpfree.py`
  (orientation gate, single-shot soundness, slow inst-11 verify + MC gate).

## Run

```bash
cd benchmarks/vit_vnncomp2023
python run_vit_crown.py --model ibp_3_3_8 --instances 10 --alpha          # VERIFIED
python run_vit_crown.py --model ibp_3_3_8 --instances 10 28 0 88 45 34 86 --alpha
```

## Soundness

Every relaxation is a sound over-approximation, so the verdict gate (all margin
lower bounds > 0) cannot produce a false VERIFIED. The orientation gate validates
the lowering against the torch model to ~1e-15 (build the DAG from a `.double()`
model so the BatchNorm affine is computed in float64). Lower the model in float64;
the verdict path is pure numpy.
