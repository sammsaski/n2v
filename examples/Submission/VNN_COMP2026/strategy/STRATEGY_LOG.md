# VNN-COMP 2026 strategy optimization log

Goal: maximize n2v's VNN-COMP score under the **+10 correct / 0 unknown / −150
false-UNSAT** model. User decisions driving this work:

- **Zero-risk / integrity-first**: never emit an unsound (probabilistic) UNSAT.
  Points come only from (a) ORT-revalidated falsification SAT and (b) sound-reach
  UNSAT — neither can score −150.
- **Defer HW timeout-tuning**: optimize verdict-correctness now (sound-vs-prob,
  falsification method/budget); fine timeout tuning waits until the competition
  machine is known.

Working dir: `examples/Submission/VNN_COMP2026/strategy/` (prototypes &
analysis; nothing here changes the shipped tool until we promote it).
Baseline = the 723-instance "different"-mode sweep in
`../sweep/results/sweep/results.csv`.

## Tooling built

- `analyze_gold.py` — builds VNN-COMP 2025 ground truth (true=SAT iff any tool
  found a valid CEX, else UNSAT) from `vnncomp2025_results/`, joins our sweep on
  (category, onnx, vnnlib) basenames, classifies vs gold, flags probabilistic
  (unsound) UNSATs and missed SATs.
- `falsify_bakeoff.py` — for a set of target instances (default = the gold-SAT
  instances we missed), runs a menu of falsification methods/budgets, ORT-
  revalidates each CE, reports which crack each instance and how fast. This is
  how we pick per-benchmark falsify configs (zero soundness risk).

## Baseline cross-reference (492 of 723 instances matched 2025 gold)

- **0 false-UNSAT landmines** and **0 false-SAT** in the baseline — the ORT
  witness gate and `verify_specification` UNSAT invariants are holding.
- **7 UNSATs came from probabilistic reach** (ml4acopf 6, yolo 1). Correct on
  2025 gold but UNSOUND → on unseen 2026 instances each is a −150 gamble.
  Zero-risk: concede to `unknown` (−70 pts here) unless sound reach recovers them.
- **28 missed SATs** (gold=SAT, we said unknown/timeout) = the prime +10 pool,
  zero risk. Falsification runs before reach, so most are falsification misses.
- 341 missed UNSATs (gold=UNSAT, we timed out) — sound-reach/timing-bound,
  deferred with HW tuning. Inflated by the bounded-parallel sweep (18 cores/inst).

## Falsification bake-off results (what cracks the missed SATs)

Random and PGD are useless on flat-gradient / combinatorial nets; **gradient-free
Square is the key tool.**

| benchmark | missed | best falsifier | result |
|---|---|---|---|
| **sat_relu** | 14 | `random+square` (n_iters 20k) | **14/14 in ~0.1s each** — random/PGD get 0/14 |
| cersyve | 1 | `random` 5k / `square` | 1/1 (current n_rand=3000 just misses) |
| cgan_2023 small_transformer | 1 | any (random 100 finds it) | not a strategy gap — **272 MB model file missing** in cgan_2023/2.0/onnx |
| cora_2024 cifar10-trades | 1 | none | hard — needs completeness (BaB); stays unknown |
| lsnc_relu quadrotor2d | 1 | none | hard; stays unknown |
| relusplitter mnist_fc ×3 | 3 | none (square/strong/50k all miss) | hard (RS-robust); stays unknown |
| soundnessbench model | 1 | none @ ≤5k | planted CE; needs more budget or BaB |
| traffic_signs / metaroom / cctsdb | 6 | (bake-off running) | TBD |

### Confirmed config wins so far
- **sat_relu → `random+square`**: +14 SAT, ~0.1s each (was 0 via random+pgd).
  sat_relu's nets encode Boolean SAT → flat gradients → Square is the right tool.
- **cersyve**: bump falsification (n_rand 3000→5000 or add square) → +1.

### Confirmed non-strategy findings
- cgan_2023 `small_transformer` missed = missing model file (present in cgan2026);
  needs the large-model download/copy, not a config change.
- Hard SAT instances (cora-trades, lsnc-quadrotor, relusplitter-mnist_fc,
  soundnessbench) are falsification-resistant → n2v's lack of branch-and-bound is
  the ceiling. They remain sound `unknown` (0 pts, no risk).

### Heavy-category bake-off (traffic_signs / metaroom / cctsdb)
**0/6 cracked by any falsifier** (random/PGD/APGD/Square/strong, up to 50k/20k
budget). These missed SATs are falsification-resistant — n2v's lack of complete
branch-and-bound is the ceiling. They stay sound `unknown` (0 pts, no risk).

### Sound-reach recovery of probabilistic holds
`sound_reach_probe.py` on the 6 ml4acopf holds (14_ieee prop9, all three model
variants): sound `approx` returns **`unknown` in ~7-9s** — the McCormick
over-approx is too loose to prove safety. So the conceded probabilistic holds are
NOT recoverable soundly. Cost of zero-risk on ml4acopf is real (−6 here), but the
−150 exposure is removed.

## Validated changes (in `benchmark_configs_proposed.py`)

**Change 1 — drop ALL probabilistic reach (9 configs):** cersyve, cgan(transformer),
cifar100, nn4sys(_default), tinyimagenet, ml4acopf, vggnet16, vit, yolo. Small nets
→ sound `approx` (chance at sound holds); frontier/large nets → `[]` falsify-only
(save budget; holds unprovable anyway). Eliminates every false-UNSAT (−150) path.

**Change 2 — falsification:**
- **sat_relu → `random+square` (n_iters 20k)**: 35→**50** CEs on the full 100-instance
  corpus (validated, strict superset, ~0.1s each). **+15 SAT.**
- **cersyve n_rand 3000→5000**: +1 SAT (pendulum_pretrain_con).

## Net score estimate (VNN-COMP +10/0/−150)

On the 492 gold-matched instances of the baseline sweep:

| | correct SAT | correct UNSAT (sound) | correct UNSAT (prob) | false UNSAT | score |
|---|---|---|---|---|---|
| **baseline** | 89 | 27 | 7 (unsound) | 0 | 1230 |
| **proposed** | 104 | 27 | 0 | 0 | **1310** |

**+80 pts on the matched sample, and 0 unsound-UNSAT paths** (baseline's 7
prob-UNSATs were correct on 2025 gold but are −150 landmines on unseen 2026
instances — one wrong wipes 15 correct answers). The sat_relu Square win is **+15
on the full corpus** (validated), so the real-competition gain is larger than the
matched-subset +80; the zero-risk derisking is the dominant benefit.

## Benchmarks still needing added support (orthogonal to strategy)

| benchmark | gap | type |
|---|---|---|
| cgan upsample (×2) | opset-9 `Upsample` → onnx2torch `ValueType.UNKNOWN` load fail | **fixable** — loader opset-upgrade shim could catch this RuntimeError |
| cgan_2023 small_transformer | 272 MB model file absent (present in cgan2026) | **packaging** — download/copy |
| smart_turn | quantized transformer (`DequantizeLinear` + QDQ + Erf/Sqrt) | extended-track frontier (I-38) |
| vit / vggnet / cctsdb | bilinear attention / ImageNet scale / discrete control-flow | sound-reach frontiers (load+falsify only) |
| cora-trades, lsnc-quadrotor, relusplitter-mnist_fc, soundnessbench | gold-SAT but falsification-resistant | **branch-and-bound gap** — the biggest single capability missing |

## BaB experiment (tested — corrects the earlier unvalidated claim)

`bab_falsify.py` = best-first **input-split** BaB falsifier (bisect input box,
score subboxes by sampled falsification loss, expand most-promising first, run
APGD inside; reuses the validated objective). Tested head-to-head vs
budget-matched heavy plain falsification on the cases where input-split is
tractable:

| instance | dim | heavy APGD (nr20·ns200) | Square 50k | input-split BaB |
|---|---|---|---|---|
| lsnc quadrotor2d | 6 | miss | miss | **miss** (400 nodes, 108s) |
| soundnessbench | 128 | miss | miss | **miss** (54 nodes, 121s) |

**Input-split BaB finds 0 additional SATs.** And both of these are dubious gold
anyway (below). So the "build a light BaB" recommendation is NOT supported — input
splitting doesn't help where tractable, and is infeasible for the high-dim image
nets (mnist_fc 784, cora 3072) that need it.

### Gold-map contamination found (sharpens the real miss count)
Naive "any tool says sat → SAT" is corrupted by unsound-tool false CEs and
sound-tool disagreements. Re-classified the 28 missed SATs requiring a complete
tool or ≥2 tools AND no sound-tool (abCROWN/neuralsat) conflict:

- **2 SUSPECT** (not real misses): `lsnc quadrotor2d` (only rover says sat;
  **abCROWN proves UNSAT** → rover CE almost certainly invalid) and
  `soundnessbench model` (abCROWN sat **vs** neuralsat unsat — direct conflict).
  n2v returning `unknown`/safe here is correct; no falsifier *should* find them.
- **26 GENUINE**, of which the proposed config already solves **16**: sat_relu 14
  (Square, validated), cersyve 1 (n_rand bump), cgan small_transformer 1 (once the
  file is present — random cracks it).
- **~10 genuinely-SAT misses remain** (cora-trades, metaroom ×2, relusplitter
  mnist_fc ×3, traffic_signs ×3, cctsdb ×1). **Every one was solved ONLY by
  complete tools** (abCROWN / neuralsat / cora) — no sampler/incomplete tool got
  them. That is the signature of **complete neuron-split BaB** (abCROWN's core),
  which the input-split prototype is not and cannot cheaply become.

**Bottom line on BaB:** the cheap version (input-split) buys nothing (tested). The
~10 remaining genuine misses need a full neuron-split BaB + LP-bounding engine — a
major build, ~+10 instances (~+100 pts) upside, with 2 of the "misses" being noisy
gold. High cost, modest and uncertain payoff. The validated falsification wins
(sat_relu +15) and the zero-risk derisking are far better $/effort.

## Recommendation / next steps
1. **Promote `benchmark_configs_proposed.py`** to the shipped config (the two
   changes are validated; the probabilistic drop is the user's zero-risk call).
2. Highest-leverage capability investment = **a light branch-and-bound / input-
   splitting falsifier** for the hard gold-SATs (relusplitter, cora, lsnc,
   soundnessbench) — the only missed SATs no falsification method cracks.
3. Loader: extend the opset-upgrade shim to catch the cgan `Upsample` RuntimeError;
   obtain the missing cgan_2023 transformer file.
4. Re-run a SERIAL full-core sweep with the proposed config to measure the
   sound-UNSAT recovery (the 341 missed UNSATs were inflated by bounded-parallel
   starvation) once the competition HW is known.
