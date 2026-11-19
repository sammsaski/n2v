"""Soundness and unit cells of the ViT lane of the LP-free CROWN verifier,
``n2v/nn/vit_crown.py``.

The component: ``load_vit_onnx`` rebuilds a VNN-COMP ViT (``ViT_BN``) from the
initializers of its ONNX file (``_infer_arch``, ``_build_torch_vit``);
``to_ops`` lowers a ``ViT_BN`` to the flat operation list of
``n2v.nn.crown_reach`` (``_patch_embed``, ``_per_token_op``, ``_bn_op``,
``_slice_op``, ``_affine_op``, ``_lin``); ``norm_img`` and ``eps_box`` turn a
raw image into the model input and its eps-ball; ``margin_spec`` states the
robustness property; ``crown_margins``, ``verify_instance`` and
``verify_halfspace_group_safe`` give the verdict. The competition runner calls
``load_vit_onnx``, ``to_ops`` and ``verify_halfspace_group_safe`` in that order.

The promise, link by link, and the oracle of each:

- Parity. The model ``load_vit_onnx`` returns computes what the ONNX file
  computes, and the operation list of ``to_ops`` computes what the model
  computes. Oracle: a ViT written here with no n2v code and with the VNN-COMP
  initializer names, exported to ONNX. Its float32 weights are the file's
  initializers, so its float64 forward pass is the file's network in float64;
  onnxruntime replays the file to confirm the export. Slack against that
  reference: 1e-9 times the logit scale (the lowered network agrees to about
  1e-15; a float32 BatchNorm moves a logit by about 1e-7, each listed lowering
  error by 1e-2 or more).
- The input box. ``eps_box`` is the set of raw images within eps of the given
  one in the infinity norm, clipped to [0, 1] and normalised, in (c, h, w)
  row-major order; a box that is too small is a false 'safe'. Oracle: the
  clipped ball normalised by a formula written here, in float64 and, for the
  rounding case, in rationals.
- The verdict. ``verify_instance`` says robust, and
  ``verify_halfspace_group_safe`` says safe, only when no input of the box
  violates the property (the unsafe region of a group is the OR of its rows
  ``G_k y <= g_k``); ``crown_margins`` returns lower bounds of
  ``C y - offset``. Oracle: operation lists whose true answer is exact: an
  affine network (a linear function is smallest at a corner of the box, in
  rationals), a ReLU network (its linear regions enumerated exactly,
  ``tests/_oracles/rational.py``), a linear classifier on raw images (closed
  form over the independent box), and one or two ReLU neurons whose CROWN
  relaxation is computed by hand in the docstrings, for the cases that check
  that an option reaches the engine. On the ViT itself the bound is compared
  one-sided with the reference model on sampled inputs of the box.

The rounding to nearest of the CROWN margin is registry VERDICT-1, marked in
``tests/contracts/test_verdicts.py``; the cases here stay clear of it (dyadic
data, or margins far from zero). Defects marked here:
a file with another head count is rebuilt silently as three heads; a file the
loader cannot rebuild crashes (KeyError, torch RuntimeError) instead of being
refused; ``to_ops`` lowers a float32 model with its BatchNorm in float32, and
crashes later on a model whose attention width is not its embedding width;
``eps_box`` rounds the ends of the box to nearest and answers for a flat
image; ``margin_spec`` crashes on a label outside 0 to 9; a group whose
thresholds are fewer than its rows is answered; ``crown_margins`` returns NaN
when an exponential of the engine overflows.
"""
import copy
import importlib.util
import os
import warnings
from fractions import Fraction

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

from n2v.nn import crown_reach as cr
from n2v.nn import vit_crown as vc
from tests._oracles import rational as rat
from tests.known_bugs import DID_NOT_RAISE, known_bug

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_MODEL = os.path.join(REPO, "benchmarks", "vit_vnncomp2023", "model.py")
if not os.path.exists(BENCH_MODEL):
    # A tracked file: a missing one is a broken checkout, never a skip.
    raise FileNotFoundError(f"vendored ViT model missing: {BENCH_MODEL}")

# The CIFAR-10 statistics the raw-image entries normalise with, written here
# (not read from n2v) so that the box oracle is independent.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

# The widths of the VNN-COMP ViTs, which load_vit_onnx assumes.
DIM, HEADS, DIM_HEAD, MLP = 48, 3, 16, 96


def _scale_pad(y):
    """Slack against a float64 reference of a logit vector: 1e-9 times its
    scale. The lowered network agrees to about 1e-15."""
    return 1e-9 * (1.0 + float(np.max(np.abs(y))))


def _pad(v):
    """Slack of the 1e-12 class against one float64 reference value."""
    return 1e-12 * (1.0 + abs(float(v)))


def _benchmark_model_module():
    """The benchmark's ``model.py`` (the ``ViT_BN`` class that ``to_ops``
    duck-types), loaded under its own name so that no other ``model`` module
    on the path is picked up."""
    spec = importlib.util.spec_from_file_location("vit_bench_model_for_crown", BENCH_MODEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
#  An independent ViT with the VNN-COMP initializer names                      #
# --------------------------------------------------------------------------- #
class _BatchNormOverFeatures(nn.Module):
    """BatchNorm1d over the feature axis of a (batch, tokens, features) tensor."""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.BatchNorm1d(width)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _Embedding(nn.Module):
    def __init__(self, patch, width):
        super().__init__()
        patches = (32 // patch) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.positions = nn.Parameter(torch.zeros(patches + 1, width))
        self.projection = nn.Conv2d(3, width, patch, stride=patch)

    def forward(self, x):
        z = self.projection(x).flatten(2).transpose(1, 2)
        z = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), z], dim=1)
        return z + self.positions


class _Attention(nn.Module):
    """Multi-head self-attention, scores divided by sqrt(dim_head). With
    ``extra`` the block holds a seventh weight MatMul (``mix``)."""

    def __init__(self, width, heads, dim_head, extra=False):
        super().__init__()
        self.heads, self.dim_head, self.extra = heads, dim_head, extra
        inner = heads * dim_head
        self.query = nn.Linear(width, inner)
        self.key = nn.Linear(width, inner)
        self.value = nn.Linear(width, inner)
        self.out = nn.Linear(inner, width)
        if extra:
            self.mix = nn.Linear(width, width)

    def forward(self, x):
        b, n, _ = x.shape

        def split(t):
            return t.reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)

        q, k, v = split(self.query(x)), split(self.key(x)), split(self.value(x))
        a = torch.softmax(q @ k.transpose(-1, -2) / self.dim_head ** 0.5, dim=-1)
        o = (a @ v).permute(0, 2, 1, 3).reshape(b, n, self.heads * self.dim_head)
        o = self.out(o)
        return self.mix(o) if self.extra else o


class _PreNormResidual(nn.Module):
    def __init__(self, width, inner):
        super().__init__()
        self.fn = nn.Sequential(_BatchNormOverFeatures(width), inner)

    def forward(self, x):
        return x + self.fn(x)


class _TokenMean(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


def _reference_vit(patch, depth, seed, width=DIM, heads=HEADS, dim_head=DIM_HEAD,
                   mlp=MLP, extra=False):
    """``nn.Sequential(embedding, blocks, head)``, so that the parameter names
    are the VNN-COMP ones: ``0.*`` the embedding, ``1.<i>.0.fn.*`` the
    attention of block i, ``1.<i>.1.fn.*`` its feed-forward, ``2.1`` and
    ``2.2`` the head BatchNorm and Linear. Every parameter and BatchNorm
    statistic is random: nonzero class token and positions, BatchNorm scales
    other than 1 and different per feature, nonzero running means, and query
    and key weights scaled so that the attention is far from uniform."""
    torch.manual_seed(seed)
    blocks = nn.Sequential(*[nn.Sequential(
        _PreNormResidual(width, _Attention(width, heads, dim_head, extra)),
        _PreNormResidual(width, nn.Sequential(
            nn.Linear(width, mlp), nn.ReLU(), nn.Dropout(0.0), nn.Linear(mlp, width))))
        for _ in range(depth)])
    net = nn.Sequential(_Embedding(patch, width), blocks,
                        nn.Sequential(_TokenMean(), nn.BatchNorm1d(width), nn.Linear(width, 10)))
    with torch.no_grad():
        for name, buf in net.named_buffers():
            if name.endswith("running_var"):
                buf.copy_(0.25 + 2.0 * torch.rand_like(buf))
            elif name.endswith("running_mean"):
                buf.copy_(0.5 * torch.randn_like(buf))
        for name, prm in net.named_parameters():
            if "cls_token" in name or "positions" in name:
                prm.copy_(0.5 * torch.randn_like(prm))
            elif ".norm." in name or name.startswith("2.1."):
                prm.copy_(1.0 + 0.5 * torch.randn_like(prm) if name.endswith("weight")
                          else 0.5 * torch.randn_like(prm))
            elif (".query." in name or ".key." in name) and name.endswith("weight"):
                prm.mul_(3.0)
    return net.eval()


def _export(net, path):
    """Export to ONNX (opset 13, batch 1, input name ``input``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(net, torch.zeros(1, 3, 32, 32), str(path), opset_version=13,
                          dynamo=False, input_names=["input"], output_names=["output"])


def _images(n, seed):
    """n inputs of the model (normalised images), (n, 3, 32, 32), float64."""
    return np.random.default_rng(seed).standard_normal((n, 3, 32, 32))


def _replay(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float64)).numpy()


# key: (patch, depth, seed). p16-depth2 has a 2x2 patch grid, two blocks and
# five tokens; p8-depth1 a 4x4 grid and 17 tokens; p16-depth1 is the small
# network the CROWN cases bound.
_VITS = {"p16-depth2": (16, 2, 0), "p8-depth1": (8, 1, 1), "p16-depth1": (16, 1, 2)}


@pytest.fixture(scope="module")
def vit_files(tmp_path_factory):
    """``key -> (onnx path, float64 reference model, patch, depth)``. The
    export is checked against onnxruntime here, so that a broken export fails
    as a setup error and never as a finding of the tests."""
    folder = tmp_path_factory.mktemp("vit_crown")
    out = {}
    for key, (patch, depth, seed) in _VITS.items():
        net = _reference_vit(patch, depth, seed)
        path = folder / f"{key}.onnx"
        _export(net, path)
        ref = copy.deepcopy(net).double().eval()
        X = _images(2, seed=99)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        y_ort = np.vstack([sess.run(None, {"input": X[i:i + 1].astype(np.float32)})[0]
                           for i in range(len(X))])
        y_ref = _replay(ref, X)
        if np.max(np.abs(y_ort - y_ref)) > 1e-4 * (1.0 + np.max(np.abs(y_ref))):
            pytest.fail(f"{key}: the exported file and the reference disagree by "
                        f"{np.max(np.abs(y_ort - y_ref)):.3e}")
        out[key] = (str(path), ref, patch, depth)
    return out


# --------------------------------------------------------------------------- #
#  Parity: load_vit_onnx and to_ops                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["p16-depth2", "p8-depth1"])
def test_load_vit_onnx_computes_what_the_file_computes(vit_files, key):
    """The rebuilt model, in float64, computes the file's network on eight
    inputs. Catches a weight read for the wrong module (query and key
    swapped, every block reading block 0's matrices, the feed-forward
    BatchNorm read from the attention one, any other name mapped wrongly), an
    attention scale other than 1/sqrt(dim_head), a patch size or depth
    inferred wrongly from the initializers (two patch grids, two depths), a
    BatchNorm or pooling of the forward pass other than the file's, and a
    model returned in float32."""
    path, ref, patch, depth = vit_files[key]
    model = vc.load_vit_onnx(path)
    assert (model.patch_size, len(model.blocks), model.n_tokens) == \
        (patch, depth, (32 // patch) ** 2 + 1)
    X = _images(8, seed=3)
    y_ref = _replay(ref, X)
    y = _replay(model, X)
    err = float(np.max(np.abs(y - y_ref)))
    assert err <= _scale_pad(y_ref), f"{key}: |loaded - file| = {err:.3e}"


@pytest.mark.parametrize("key", ["p16-depth2", "p8-depth1"])
def test_to_ops_of_the_loaded_model_computes_what_the_file_computes(vit_files, key):
    """The runner's chain ``to_ops(load_vit_onnx(file))``: the operation list,
    evaluated on eight inputs, gives the file's logits. Catches every
    lowering error (patch embedding: class-token position, kernel layout,
    patch order, conv bias; per-token maps: weight and bias layout;
    BatchNorm: scale layout and the shift ``bias - scale * mean``; head
    slices; score scale, softmax sums and broadcast; residuals; the
    feed-forward and head BatchNorms; token mean) and every loader error that
    reaches the lowering, on two blocks with a 2x2 patch grid and on one
    block with a 4x4 grid."""
    path, ref, _, _ = vit_files[key]
    ops = vc.to_ops(vc.load_vit_onnx(path))
    X = _images(4, seed=4)
    y_ref = _replay(ref, X)
    for i, x in enumerate(X):
        y = cr.eval_ops(ops, x.reshape(-1))[0]
        err = float(np.max(np.abs(y - y_ref[i])))
        assert err <= _scale_pad(y_ref[i]), f"{key} input {i}: |ops - file| = {err:.3e}"


def _benchmark_vit(dtype, **widths):
    """The benchmark's ``ViT_BN`` (the class ``to_ops`` duck-types besides its
    own), patch 16, two blocks, in ``dtype``: torch's random initialisation of
    the linear and conv layers, and random BatchNorms (scales other than 1,
    nonzero means), class token and positions."""
    module = _benchmark_model_module()
    torch.manual_seed(7)
    m = module.ViT_BN(patch_size=16, depth=2, **widths)
    with torch.no_grad():
        for name, prm in m.named_parameters():
            if "norm" in name:
                prm.copy_(1.0 + 0.5 * torch.randn_like(prm) if name.endswith("weight")
                          else 0.5 * torch.randn_like(prm))
            elif name in ("cls_token", "positions"):
                prm.copy_(0.5 * torch.randn_like(prm))
        for name, buf in m.named_buffers():
            if name.endswith("running_var"):
                buf.copy_(0.25 + 2.0 * torch.rand_like(buf))
            elif name.endswith("running_mean"):
                buf.copy_(0.5 * torch.randn_like(buf))
    return m.to(dtype).eval()


@pytest.mark.parametrize("case", [
    "dim32-heads2-float64",
    pytest.param("float32", marks=known_bug(
        19, "to_ops lowers a float32 ViT_BN with its BatchNorm scale and shift "
            "computed in float32 (the model's affine()), so the operation list is "
            "another network than the stored weights: logits off by about 1e-7, "
            "and a margin that small can be certified on the wrong side")),
    pytest.param("inner-width-not-dim", marks=known_bug(
        23, "to_ops returns an operation list for a ViT_BN whose heads * "
            "dim_head is not dim, whose head slices and concat assume the "
            "embedding width; evaluating or bounding it crashes with numpy's "
            "matmul ValueError instead of to_ops refusing the model",
        raises=ValueError)),
])
def test_to_ops_computes_the_benchmark_class(case):
    """``to_ops`` on the benchmark's ``ViT_BN`` (its docstring names it), with
    the reference the model's own forward pass in float64 on the stored
    weights. ``dim32-heads2-float64``: widths other than the VNN-COMP ones
    (embedding 32, two heads of 16, hidden 40): catches a lowering that
    assumes 48 features, three heads or 16 per head. ``float32``: the model
    as ``build_model`` returns it. ``inner-width-not-dim``: two heads of 8 on
    an embedding of 24; the promise is a correct list or a refusal
    (``NotImplementedError``) from ``to_ops``."""
    if case == "dim32-heads2-float64":
        m = _benchmark_vit(torch.float64, dim=32, heads=2, dim_head=16, mlp_dim=40)
    elif case == "float32":
        m = _benchmark_vit(torch.float32)
    else:
        m = _benchmark_vit(torch.float64, dim=24, heads=2, dim_head=8, mlp_dim=40)
    try:
        ops = vc.to_ops(m)
    except NotImplementedError:
        if case == "inner-width-not-dim":
            return
        raise
    X = _images(3, seed=5)
    y_ref = _replay(copy.deepcopy(m).double(), X)
    for i, x in enumerate(X):
        y = cr.eval_ops(ops, x.reshape(-1))[0]
        err = float(np.max(np.abs(y - y_ref[i])))
        assert err <= _scale_pad(y_ref[i]), f"{case} input {i}: |ops - model| = {err:.3e}"


# --------------------------------------------------------------------------- #
#  The loader on files it must refuse                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", [
    "seven-weight-matmuls",
    pytest.param("not-a-vit", marks=known_bug(
        23, "load_vit_onnx on an ONNX file without the VNN-COMP ViT initializer "
            "names crashes with KeyError '0.positions' instead of refusing it",
        raises=KeyError)),
    pytest.param("other-widths", marks=known_bug(
        23, "load_vit_onnx on a ViT file whose widths are not 48/96 crashes with "
            "a torch RuntimeError (a view of the class token, or the strict "
            "load_state_dict) instead of refusing it",
        raises=RuntimeError)),
])
def test_load_vit_onnx_refuses_a_file_it_cannot_rebuild(tmp_path, case):
    """A file the loader cannot rebuild as the module's ``ViT_BN`` is refused
    with ``ValueError`` (cannot be read) or ``NotImplementedError`` (not
    supported). ``seven-weight-matmuls``: a block with one more weight
    MatMul than the six the loader maps (refused today by its count check;
    catches the check removed, which maps the first six MatMuls and answers
    for another network). ``not-a-vit``: a linear classifier.
    ``other-widths``: embedding 32, hidden 64, two heads of 16."""
    path = tmp_path / f"{case}.onnx"
    if case == "seven-weight-matmuls":
        _export(_reference_vit(16, 1, seed=3, extra=True), path)
    elif case == "not-a-vit":
        torch.manual_seed(3)
        _export(nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10)).eval(), path)
    else:
        _export(_reference_vit(16, 1, seed=3, width=32, heads=2, dim_head=16, mlp=64), path)
    with pytest.raises((ValueError, NotImplementedError)):
        vc.load_vit_onnx(str(path))


@known_bug(23, "load_vit_onnx rebuilds a ViT file with four heads of 12 (inner "
               "width 48) as three heads of 16 without a word, so the runner "
               "verifies another network than the file (logits off by about 0.2)")
def test_load_vit_onnx_with_another_head_count_refuses_or_computes_the_file(tmp_path):
    """The head count is not in the initializers (only in the graph's Reshape
    constants), so a file whose query, key and value widths are 48 but split
    into four heads loads strictly into the three-head model. The promise:
    refused, or rebuilt so that it computes the file's network (the same
    reference and slack as ``test_load_vit_onnx_computes_what_the_file_computes``)."""
    net = _reference_vit(16, 1, seed=4, heads=4, dim_head=12)
    path = tmp_path / "four_heads.onnx"
    _export(net, path)
    try:
        model = vc.load_vit_onnx(str(path))
    except (ValueError, NotImplementedError):
        return
    X = _images(4, seed=6)
    y_ref = _replay(copy.deepcopy(net).double(), X)
    err = float(np.max(np.abs(_replay(model, X) - y_ref)))
    assert err <= _scale_pad(y_ref), f"|loaded - file| = {err:.3e}"


# --------------------------------------------------------------------------- #
#  The input side: norm_img and eps_box                                         #
# --------------------------------------------------------------------------- #
def _raw_image():
    """A raw (3, 32, 32) image in [0, 1]: different per channel and pixel,
    with pixels at 0 and at 1, within eps of 0 and of 1, and interior ones."""
    img = np.random.default_rng(11).uniform(0.05, 0.95, (3, 32, 32))
    img[0, 0, 0] = 0.0
    img[1, 7, 30] = 0.0
    img[2, 31, 2] = 1.0
    img[0, 12, 5] = 1.0
    img[1, 3, 3] = 0.002
    img[2, 20, 9] = 0.998
    return img


def _normalised(img):
    """(img - mean) / std per channel, flattened channel by channel, row by
    row: the model input in (c, h, w) row-major order."""
    return np.concatenate([((img[c] - MEAN[c]) / STD[c]).ravel() for c in range(3)])


def _ball_box(img, eps):
    """The eps-ball of a raw image in the infinity norm, clipped to [0, 1],
    normalised: (lower, upper) in the model's input order."""
    return (_normalised(np.minimum(np.maximum(img - eps, 0.0), 1.0)),
            _normalised(np.minimum(np.maximum(img + eps, 0.0), 1.0)))


def test_norm_img_is_the_normalised_image_in_chw_order():
    """Catches a normalisation with another channel's statistics and a
    flatten in another order (Fortran order gives the (w, h, c) layout)."""
    img = _raw_image()
    x = vc.norm_img(img)
    ref = _normalised(img)
    assert x.shape == ref.shape
    assert np.all(np.abs(x - ref) <= 1e-12 * (1.0 + np.abs(ref)))


@pytest.mark.parametrize("eps", [None, 0.03, 0.0], ids=["default", "0.03", "0"])
def test_eps_box_is_the_normalised_clipped_ball(eps):
    """The box is the clipped ball, end by end: no end inside the ball's (a
    false 'safe'), none outside (a box that leaves the image domain). The
    default radius is 1/255 (``eps`` not passed); 0 gives the image itself.
    Catches half the radius, the lower end clipped before eps is subtracted
    (a pixel at 0 reaches -eps), the upper end not clipped, and a box laid
    out in another order than the model input."""
    img = _raw_image()
    lb, ub = vc.eps_box(img) if eps is None else vc.eps_box(img, eps)
    ref_lo, ref_hi = _ball_box(img, 1.0 / 255 if eps is None else eps)
    assert lb.shape == ref_lo.shape and ub.shape == ref_hi.shape
    tol_lo, tol_hi = 1e-12 * (1.0 + np.abs(ref_lo)), 1e-12 * (1.0 + np.abs(ref_hi))
    assert np.all(lb <= ref_lo + tol_lo) and np.all(ub >= ref_hi - tol_hi), \
        "the box misses part of the ball"
    assert np.all(lb >= ref_lo - tol_lo) and np.all(ub <= ref_hi + tol_hi), \
        "the box is larger than the ball"


@pytest.mark.parametrize("slack", [
    "float64",
    pytest.param("exact", marks=known_bug(
        7, "eps_box computes the normalised ends (x -+ eps - mean) / std rounded "
           "to nearest, so about half of them lie one ulp inside the exact "
           "normalised ball and the box misses its boundary")),
])
def test_eps_box_holds_the_exact_normalised_ball(slack):
    """On a dyadic image (multiples of 1/256) with eps = 1/64 the clipped
    ball's ends are exact, and so is their normalisation in rationals with
    the stored mean and std. Every lower end must be at or below the exact
    one and every upper end at or above, with no slack (``exact``) or a pad
    of 1e-12 of the value (``float64``, the passing neighbour that shows the
    box is right up to rounding)."""
    img = np.random.default_rng(12).integers(0, 257, (3, 32, 32)) / 256.0
    eps = 1.0 / 64
    lb, ub = vc.eps_box(img, eps)
    worst = Fraction(0)
    for c in range(3):
        mean, std = Fraction(MEAN[c]), Fraction(STD[c])
        for h in range(32):
            for w in range(32):
                x, i = Fraction(img[c, h, w]), (c * 32 + h) * 32 + w
                lo = (min(max(x - Fraction(eps), Fraction(0)), Fraction(1)) - mean) / std
                hi = (min(max(x + Fraction(eps), Fraction(0)), Fraction(1)) - mean) / std
                pad_lo = Fraction(_pad(float(lo))) if slack == "float64" else Fraction(0)
                pad_hi = Fraction(_pad(float(hi))) if slack == "float64" else Fraction(0)
                worst = max(worst, Fraction(lb[i]) - lo - pad_lo, hi - Fraction(ub[i]) - pad_hi)
    assert worst <= 0, f"an end lies {float(worst):.3e} inside the exact ball"


@pytest.mark.parametrize("shape", [
    "hwc",
    pytest.param("flat", marks=known_bug(
        23, "eps_box on a flat image of 3072 values answers with a box of 9216 "
            "values (every value normalised with all three channel statistics) "
            "instead of refusing the shape", raises=DID_NOT_RAISE)),
])
def test_eps_box_refuses_an_image_that_is_not_channels_first(shape):
    """The documented input is a raw (3, 32, 32) image. An image in (h, w, c)
    order or flattened is not one, and must raise ``ValueError``."""
    img = _raw_image()
    bad = img.transpose(1, 2, 0) if shape == "hwc" else img.reshape(-1)
    with pytest.raises(ValueError):
        vc.eps_box(bad, 0.01)


# --------------------------------------------------------------------------- #
#  The property: margin_spec                                                    #
# --------------------------------------------------------------------------- #
def test_margin_spec_has_one_row_per_other_class():
    """For every label, the nine rows are exactly ``e_label - e_other`` for
    the nine other classes (in any order). Catches a class left out (the
    last class for every label but 9), a row repeated, and the sign
    reversed (``other - label`` is positive where the label loses)."""
    for label in range(10):
        C = vc.margin_spec(label)
        assert C.shape == (9, 10)
        expected = set()
        for other in range(10):
            if other != label:
                row = [0.0] * 10
                row[label], row[other] = 1.0, -1.0
                expected.add(tuple(row))
        assert {tuple(r) for r in np.asarray(C, dtype=float)} == expected, f"label {label}"


@known_bug(23, "margin_spec of a label outside 0 to 9 crashes with IndexError "
               "instead of refusing the label", raises=IndexError)
def test_margin_spec_refuses_a_label_outside_the_ten_classes():
    """Label 10 does not exist: the property cannot be stated."""
    with pytest.raises(ValueError):
        vc.margin_spec(10)


# --------------------------------------------------------------------------- #
#  Small operation lists                                                        #
# --------------------------------------------------------------------------- #
def _input(n):
    return dict(type="input", **{"in": []}, dim=n, mat=None)


def _relu(k, n):
    return dict(type="relu", **{"in": k}, dim=n, mat=None)


def _add(a, b, n):
    return dict(type="add", **{"in": [a, b]}, dim=n, mat=None)


def _one_neuron():
    """y = relu(x) on x in [-1, 2]. The CROWN lower bound of y with lower
    slope a is min over the box of a * x = -a. The area rule starts at a = 1
    (u = 2 > -l = 1); the best slope is a = 0, where the bound is 0, the true
    minimum."""
    ops = [_input(1), vc._affine_op(0, [[1.0]], [0.0], None), _relu(1, 1),
           vc._affine_op(2, [[1.0]], [0.0], None)]
    return ops, np.array([-1.0]), np.array([2.0])


def _two_neurons():
    """y = relu(relu(x) - x) on x in [-1, 1], whose maximum is 1 (at x = -1).
    The inner sum z = relu(x) - x has the interval bounds [-1, 2]; its CROWN
    bounds are [-1, 1] (relu(x) <= x/2 + 1/2 gives z <= 1/2 - x/2). The upper
    chord of the outer relu over [-1, 2] is y <= 2/3 z + 2/3, which gives
    max y <= 4/3; over [-1, 1] it is y <= z/2 + 1/2, which gives 1."""
    ops = [_input(1), vc._affine_op(0, [[1.0]], [0.0], None), _relu(1, 1),
           vc._affine_op(0, [[-1.0]], [0.0], None), _add(2, 3, 1), _relu(4, 1)]
    return ops, np.array([-1.0]), np.array([1.0])


@pytest.mark.parametrize("options, expected", [
    ({}, -1.0),
    ({"alpha": True, "alpha_lr": 0.0}, -1.0),
    ({"alpha": True, "alpha_iter": 0}, -1.0),
    ({"alpha": True}, 0.0),
    ({"alpha": True, "refine": False}, 0.0),
    ({"alpha": True, "offset": [0.25]}, -0.25),
    ({"offset": [0.25]}, -1.25),
], ids=["default", "alpha-lr0", "alpha-iter0", "alpha", "alpha-ibp",
        "alpha-offset", "offset"])
def test_crown_margins_options_reach_the_relu_slope(options, expected):
    """``crown_margins`` on ``_one_neuron`` (values worked out in its
    docstring): with ``alpha_lr=0`` or ``alpha_iter=0`` nothing moves and the
    margin is the starting bound -1; with the default optimisation it is the
    best bound 0; an offset is subtracted on both branches. Catches
    ``alpha_lr`` or ``alpha_iter`` not handed to the optimiser, the offset
    not subtracted after the alpha branch (margin 0 for g = 0.25), and the
    alpha option dropped."""
    ops, lb, ub = _one_neuron()
    margins = vc.crown_margins(ops, lb, ub, np.array([[1.0]]), **options)
    assert abs(float(margins[0]) - expected) <= 1e-9, f"{float(margins[0])!r} != {expected}"


@pytest.mark.parametrize("options, expected", [
    ({}, -1.0),
    ({"refine": False}, -4.0 / 3.0),
    ({"refine_iters": 0}, -4.0 / 3.0),
    ({"refine_max_dim": 0}, -4.0 / 3.0),
], ids=["default", "ibp", "iters0", "max-dim0"])
def test_crown_margins_refinement_options(options, expected):
    """``crown_margins`` of -y on ``_two_neurons`` (values in its docstring):
    refined, the margin is -1; without refinement, with zero refinement
    passes or with no op narrow enough to refine (``refine_max_dim=0``) the
    interval bounds stand and it is -4/3. Catches ``refine_iters`` or
    ``refine_max_dim`` not handed to ``refine_bounds``, and ``refine=False``
    refining anyway."""
    ops, lb, ub = _two_neurons()
    margins = vc.crown_margins(ops, lb, ub, np.array([[-1.0]]), **options)
    assert abs(float(margins[0]) - expected) <= 1e-9, f"{float(margins[0])!r} != {expected}"


# A ReLU network: 2 inputs, 3 hidden neurons, 2 outputs, dyadic data. The
# wide box is wider than 2 and beyond -1 and 1, and every neuron changes
# phase on it; the degenerate box fixes the second input.
_W1 = [[1.0, -0.5], [-0.75, 1.0], [0.5, 0.25]]
_C1 = [0.25, -0.5, 0.0]
_W2 = [[1.0, -1.0, 0.5], [-0.5, 0.75, 1.0]]
_C2 = [0.0, 0.25]
_RELU_BOXES = {"wide": ([-1.5, -0.75], [1.25, 2.0]),
               "degenerate": ([-1.5, 0.5], [1.25, 0.5])}


def _relu_ops():
    return [_input(2), vc._affine_op(0, _W1, _C1, None), _relu(1, 3),
            vc._affine_op(2, _W2, _C2, None)]


def _relu_exact_min(G, lb, ub):
    """The exact minimum of each row of ``G y`` over the box, in rationals:
    every linear region of the network (``tests/_oracles/rational.py``) at
    every vertex."""
    pieces = rat.affine(rat.relu(rat.affine(rat.box(lb, ub), _W1, _C1)), _W2, _C2)
    images = [v for piece in pieces for v in piece.image()]
    return [min(rat.dot(rat.vector(row), v) for v in images) for row in G]


@pytest.mark.parametrize("box, options", [
    ("wide", {}), ("wide", {"refine": False}), ("wide", {"alpha": True, "alpha_iter": 10}),
    ("wide", {"refine_iters": 1, "refine_max_dim": 2}), ("degenerate", {}),
], ids=["default", "ibp", "alpha", "iters1-maxdim2", "degenerate"])
def test_crown_margins_are_lower_bounds_on_a_relu_network(box, options):
    """Every margin is at most the exact minimum of ``G_r y - g_r`` over the
    box, for rows of both signs and offsets of both signs, on a box where
    every neuron changes phase and on one with a zero-width coordinate.
    Catches the interval pass of ``refine=False`` run on a wrong box, an
    offset added or left out, and any unsound relaxation the options reach.
    The reference is exact; the pad of the 1e-12 class is the rounding of
    the CROWN arithmetic itself (the relu slopes u / (u - l) are not
    dyadic), whose margin rounded to nearest is registry VERDICT-1, marked
    in ``tests/contracts/test_verdicts.py`` and not marked again here."""
    lb, ub = _RELU_BOXES[box]
    G = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [-0.5, 1.0]])
    g = np.array([0.5, -0.25, 0.75, -1.0, 0.125, 0.0])
    exact = _relu_exact_min(G, lb, ub)
    margins = vc.crown_margins(_relu_ops(), np.array(lb), np.array(ub), G,
                               offset=g, **options)
    assert margins.shape == (len(G),)
    for r in range(len(G)):
        true_margin = exact[r] - Fraction(g[r])
        assert Fraction(float(margins[r])) <= true_margin + Fraction(_pad(float(true_margin))), \
            f"row {r}: margin {float(margins[r])!r} above the exact {float(true_margin)!r}"


def _softmax_of_one_key(top):
    """The softmax of a single key as ``to_ops`` lowers attention (exp, row
    sum, reciprocal, product): y = exp(s) * (1 / exp(s)) with
    s = (top / 2) (x + 1) on x in [-1, 1], so the scores range over
    [0, top]. y = 1 for every x. The sum never falls below 1, so the
    reciprocal's floor at 1e-12 (VERDICT-17) is not reached."""
    a = top / 2.0
    ops = [_input(1), vc._affine_op(0, [[a]], [a], None),
           dict(type="exp", **{"in": 1}, dim=1, mat=None),
           vc._affine_op(2, [[1.0]], [0.0], None),
           dict(type="reciprocal", **{"in": 3}, dim=1, mat=None),
           dict(type="eprod", **{"in": [2, 4]}, dim=1, mat=None)]
    return ops, np.array([-1.0]), np.array([1.0])


@pytest.mark.parametrize("top", [
    600.0,
    pytest.param(800.0, marks=known_bug(
        19, "when a score's upper bound passes about 709.8, exp overflows to inf in "
            "the interval pass (crown_reach.forward_ibp), its reciprocal is 0 and the "
            "product inf * 0 is NaN, so crown_margins returns NaN instead of a lower "
            "bound with every option (the verdicts read NaN as not safe)")),
], ids=["scores-to-600", "scores-to-800"])
def test_crown_margins_bound_a_softmax_whose_exponential_overflows(top):
    """``crown_margins`` of y and of -y on ``_softmax_of_one_key``, with the
    default options, the interval pass and alpha: the first margin must be
    at most 1 and the second at most -1 (one-sided, a pad of the 1e-12
    class). A NaN is not a lower bound and fails both. Scores up to 600:
    exp stays finite (the passing neighbour); up to 800: exp overflows.
    Wide interval bounds of attention scores reach this on real ViTs (a
    two-block ViT with sharp attention on a box of half-width 1e-3). Catches
    a relaxation of exp, reciprocal or product that excludes the true value,
    and a bound that is not a number."""
    ops, lb, ub = _softmax_of_one_key(top)
    for options in ({}, {"refine": False}, {"alpha": True, "alpha_iter": 3}):
        with np.errstate(over="ignore", invalid="ignore"):
            margins = vc.crown_margins(ops, lb, ub, np.array([[1.0], [-1.0]]), **options)
        assert margins[0] <= 1.0 + _pad(1.0) and margins[1] <= -1.0 + _pad(1.0), \
            f"{options}: bounds of y = 1 are [{margins[0]!r}, {-margins[1]!r}]"


# --------------------------------------------------------------------------- #
#  The verdict on an affine network: verify_halfspace_group_safe               #
# --------------------------------------------------------------------------- #
_AW = [[1.0, -2.0], [0.5, 1.5], [-1.0, 0.25]]
_AB = [3.5, 2.0, -0.5]
_A_LB, _A_UB = [-1.0, -0.5], [2.0, 1.5]


def _affine_ops():
    return [_input(2), vc._affine_op(0, _AW, _AB, None)]


def _affine_min(row):
    """Exact minimum of ``row . (W x + b)`` over the box: at a corner."""
    return min(sum(Fraction(r) * (Fraction(w0) * Fraction(x0) + Fraction(w1) * Fraction(x1)
                                  + Fraction(b0))
                   for r, (w0, w1), b0 in zip(row, _AW, _AB))
               for x0 in (_A_LB[0], _A_UB[0]) for x1 in (_A_LB[1], _A_UB[1]))


def _groups():
    """Groups (rows ORed) with thresholds derived from the exact minima m of
    their rows: unreachable rows get g = m - 1, reachable ones g = m + 1/2,
    the boundary row g = m (the minimum point satisfies G y <= g with
    equality, so it is a counterexample). The reachable row of
    ``reachable-g-positive`` has m >= 1/2, so g > 0 and m + g > 0: a margin
    computed as G y + g calls it safe."""
    r0, r1, r2 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]
    m0, m1, m2 = _affine_min(r0), _affine_min(r1), _affine_min(r2)
    if not m1 >= Fraction(1, 2):
        pytest.fail(f"setup: row r1 needs a minimum of at least 1/2, has {m1}")
    half = Fraction(1, 2)
    return {
        "unreachable-rows": ([r0, r1], [m0 - 1, m1 - 1], True),
        "reachable-g-positive": ([r1], [m1 + half], False),
        "second-row-reachable": ([r0, r2], [m0 - 1, m2 + half], False),
        "boundary": ([r2], [m2], False),
        "one-row-1d": (r0, [m0 - 1], True),
    }


@pytest.mark.parametrize("options", [{}, {"refine": False}, {"alpha": True, "alpha_iter": 3}],
                         ids=["default", "ibp", "alpha"])
def test_verify_halfspace_group_safe_on_an_affine_network(options):
    """On an affine network CROWN is exact, so the verdict must be the true
    one for every group of ``_groups`` and every margin at most the exact
    ``min G_r y - g_r`` (dyadic data: no rounding). Catches g subtracted with
    the wrong sign, a group declared safe when one row is unreachable (the
    rows are ORed), a margin of exactly 0 taken as safe, the offset not
    subtracted on the alpha branch, and a one-row G given flat (one margin
    per row). The same
    operation list and box serve every group in turn, and must come back
    unchanged."""
    ops, lb, ub = _affine_ops(), np.array(_A_LB), np.array(_A_UB)
    for name, (G, g, expected) in _groups().items():
        safe, margins = vc.verify_halfspace_group_safe(
            ops, lb, ub, G, [float(v) for v in g], **options)
        assert safe == expected, f"{name}: safe={safe}, margins {margins}"
        rows = np.atleast_2d(G)
        assert np.shape(margins) == (len(rows),), f"{name}: margins {np.shape(margins)}"
        for r, row in enumerate(rows):
            true_margin = _affine_min(row) - g[r]
            assert Fraction(float(margins[r])) <= true_margin, \
                f"{name} row {r}: margin {float(margins[r])!r} above {float(true_margin)!r}"
    assert np.array_equal(lb, _A_LB) and np.array_equal(ub, _A_UB), "the box was changed"
    assert np.array_equal(ops[1]["W"], _AW) and np.array_equal(ops[1]["b"], _AB), \
        "the operation list was changed"


@pytest.mark.parametrize("length", [
    "longer",
    pytest.param("shorter", marks=known_bug(
        23, "verify_halfspace_group_safe (crown_margins) broadcasts a threshold "
            "vector shorter than the rows of G, so a group of two rows with one "
            "threshold is answered with that threshold for both rows instead of "
            "refused", raises=DID_NOT_RAISE)),
])
def test_a_group_whose_thresholds_do_not_match_its_rows_is_refused(length):
    """G has two rows; g has three (``longer``) or one (``shorter``) entries.
    The property cannot be read, so ``ValueError``."""
    ops, lb, ub = _affine_ops(), np.array(_A_LB), np.array(_A_UB)
    g = [0.0, 0.0, 0.0] if length == "longer" else [0.0]
    with pytest.raises(ValueError):
        vc.verify_halfspace_group_safe(ops, lb, ub, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], g)


def test_verify_halfspace_group_safe_hands_its_options_to_the_engine():
    """On ``_one_neuron`` (y = relu(x) >= 0) the unsafe region y <= -1/2 is
    unreachable. Without alpha the margin is -1 + 1/2 < 0 (not proved); with
    alpha the best slope gives 0 + 1/2 > 0 (proved). Catches the keyword
    options not handed on to ``crown_margins``."""
    ops, lb, ub = _one_neuron()
    safe, margins = vc.verify_halfspace_group_safe(ops, lb, ub, [[1.0]], [-0.5], alpha=True)
    assert safe and abs(float(margins[0]) - 0.5) <= 1e-9, margins
    safe, margins = vc.verify_halfspace_group_safe(ops, lb, ub, [[1.0]], [-0.5])
    assert not safe and abs(float(margins[0]) + 0.5) <= 1e-9, margins


# --------------------------------------------------------------------------- #
#  The verdict on raw images: verify_instance                                   #
# --------------------------------------------------------------------------- #
_LABEL = 3
# pixels (channel, row, column) the classifiers read; _raw_image has 0 at
# (1, 7, 30) and 1 at (2, 31, 2), and interior values at the others.
_P_INTERIOR, _P_ZERO, _P_ONE, _P_OTHER = (0, 4, 21), (1, 7, 30), (2, 31, 2), (2, 9, 14)


def _index(p):
    c, h, w = p
    return (c * 32 + h) * 32 + w


def _linear_classifier(rows):
    """``rows``: {other class: (constant, {pixel: weight})}, the margin
    ``y_label - y_other = constant + sum weight * x[pixel]`` on the
    normalised input. The logit of the label is 0; every class not listed has
    the margin 5. Returns the operation list (one affine op on 3072 inputs)."""
    W, b = np.zeros((10, 3072)), np.zeros(10)
    for other in range(10):
        if other == _LABEL:
            continue
        const, weights = rows.get(other, (5.0, {}))
        b[other] = -const
        for p, w in weights.items():
            W[other, _index(p)] = -w
    return [_input(3072), vc._affine_op(0, W, b, None)]


def _true_min_margins(rows, img, eps):
    """The exact minimum over the independent clipped, normalised ball of
    each margin ``y_label - y_other`` (a linear function: each weight takes
    the end of its sign)."""
    lo, hi = _ball_box(img, eps)
    out = {}
    for other in range(10):
        if other == _LABEL:
            continue
        const, weights = rows.get(other, (5.0, {}))
        out[other] = const + sum(w * (lo[_index(p)] if w > 0 else hi[_index(p)])
                                 for p, w in weights.items())
    return out


def _instances():
    """name -> (rows, eps passed to verify_instance or None for the
    default). The zero crossing of ``radius`` is at a raw radius of 0.02:
    robust at 1/255 and at 0.015, not at 0.03."""
    img = _raw_image()
    x_int = (img[_P_INTERIOR] - MEAN[0]) / STD[0]
    zero_lo = (0.0 - MEAN[1]) / STD[1]           # the clipped lower end of _P_ZERO
    one_hi = (1.0 - MEAN[2]) / STD[2]            # the clipped upper end of _P_ONE
    a = 2.0
    radius_row = (-a * ((img[_P_INTERIOR] - 0.02) - MEAN[0]) / STD[0], {_P_INTERIOR: a})
    return {
        # every margin positive only because the ball is clipped at 0 and 1,
        # class 9 (the last row) with its own margin
        "robust-clipped": ({2: (0.05 - zero_lo, {_P_ZERO: 1.0}),
                            4: (0.05 + one_hi, {_P_ONE: -1.0}),
                            9: (0.5 - 0.8 * x_int, {_P_INTERIOR: 0.8, _P_OTHER: 0.1})},
                           0.03),
        # positive at the image, lost to class 1 inside the ball
        "one-row-lost": ({1: (0.1 - x_int, {_P_INTERIOR: 1.0})}, 0.03),
        # the label is the lowest logit everywhere
        "label-lowest": ({o: (-2.0 - o, {_P_OTHER: 0.5}) for o in range(10) if o != _LABEL},
                         0.03),
        "radius-at-0.03": ({1: radius_row}, 0.03),
        "radius-default-eps": ({1: radius_row}, None),
    }


@pytest.mark.parametrize("name", ["robust-clipped", "one-row-lost", "label-lowest",
                                  "radius-at-0.03", "radius-default-eps"])
def test_verify_instance_is_robust_only_when_every_margin_holds_on_the_ball(name):
    """``verify_instance`` on a raw image and a linear classifier: robust
    exactly when every true margin is positive on the clipped ball (CROWN is
    exact on an affine list; no margin here is within 1e-3 of 0), every
    margin at most its true minimum, and the box it reports is the ball.
    Catches a verdict on ANY row, the margin rows reversed or missing a
    class, eps not handed to ``eps_box`` (``radius-at-0.03`` is robust at
    the default 1/255), half the radius, the clip of either end, and the box
    laid out in another order."""
    rows, eps = _instances()[name]
    img = _raw_image()
    radius = 1.0 / 255 if eps is None else eps
    true_min = _true_min_margins(rows, img, radius)
    if min(abs(v) for v in true_min.values()) < 1e-3:
        pytest.fail(f"setup: {name} has a true margin within 1e-3 of 0: {true_min}")
    expected = all(v > 0 for v in true_min.values())
    ops = _linear_classifier(rows)
    if eps is None:
        robust, margins, info = vc.verify_instance(ops, img, _LABEL)
    else:
        robust, margins, info = vc.verify_instance(ops, img, _LABEL, eps=eps)
    assert robust == expected, f"{name}: robust={robust}, true minima {true_min}"
    ref_lo, ref_hi = _ball_box(img, radius)
    assert np.all(np.abs(info["lb"] - ref_lo) <= 1e-12 * (1.0 + np.abs(ref_lo)))
    assert np.all(np.abs(info["ub"] - ref_hi) <= 1e-12 * (1.0 + np.abs(ref_hi)))
    for r, row in enumerate(np.asarray(info["C"])):
        others = [k for k in range(10) if row[k] < 0]
        if row[_LABEL] == 1.0 and len(others) == 1 and np.count_nonzero(row) == 2:
            bound = true_min[others[0]]
            assert margins[r] <= bound + _pad(bound), f"{name} row {r}: {margins[r]} > {bound}"


def test_verify_instance_hands_its_options_to_the_engine():
    """A classifier whose label logit is relu(x) for one normalised pixel x
    whose range [l, u] over the ball has l < 0 < u and u > -l, and every
    other logit -1/10: the true margins are relu(x) + 1/10 > 0. CROWN without
    alpha bounds relu(x) below by l (slope 1), so the margin l + 1/10 is
    negative; with alpha the best slope 0 gives 1/10 > 0. Catches the
    keyword options not handed on to ``crown_margins``."""
    img = _raw_image().copy()
    eps = 0.1
    img[_P_INTERIOR] = MEAN[0] + eps / 2              # l = -eps/(2 std), u = 3 eps/(2 std)
    l_norm = -eps / 2 / STD[0]
    if not l_norm + 0.1 < -0.05:
        pytest.fail(f"setup: the default bound {l_norm + 0.1} must be clearly negative")
    W1 = np.zeros((1, 3072))
    W1[0, _index(_P_INTERIOR)] = 1.0
    W2 = np.zeros((10, 1))
    W2[_LABEL, 0] = 1.0
    b2 = np.full(10, -0.1)
    b2[_LABEL] = 0.0
    ops = [_input(3072), vc._affine_op(0, W1, [0.0], None), _relu(1, 1),
           vc._affine_op(2, W2, b2, None)]
    robust, margins, _ = vc.verify_instance(ops, img, _LABEL, eps=eps, alpha=True)
    assert robust and np.all(np.abs(margins - 0.1) <= 1e-9), margins
    robust, margins, _ = vc.verify_instance(ops, img, _LABEL, eps=eps)
    assert not robust and np.all(np.abs(margins - (l_norm + 0.1)) <= 1e-9), margins


# --------------------------------------------------------------------------- #
#  The chain on a real ViT: CROWN margins of the lowered network               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("options", [
    {}, {"refine": False}, {"refine_iters": 2, "alpha": True, "alpha_iter": 5},
], ids=["default", "ibp", "runner"])
def test_the_vit_verdict_is_not_safe_when_a_sampled_input_is_unsafe(vit_files, options):
    """The runner's chain ``load_vit_onnx`` -> ``to_ops`` ->
    ``verify_halfspace_group_safe`` on a one-block ViT, over a box of
    half-width 1e-4 around an input. Each of the nine margin rows of label 3
    gets the threshold g just above its smallest value over 128 sampled
    inputs of the box (both extreme corners among them), so each row's
    unsafe region holds a true output: the group must not be safe, and every
    margin must be at most the sampled minimum of ``G_r y - g_r``. With the
    runner's options (refinement and alpha) and without. Catches a lowering
    or a loader that moves the logits by more than the bound's slack (about
    1e-3 here), and an unsound bound on the ViT's operations."""
    path, ref, _, _ = vit_files["p16-depth1"]
    ops = vc.to_ops(vc.load_vit_onnx(path))
    x0 = _images(1, seed=8)[0].reshape(-1)
    lb, ub = x0 - 1e-4, x0 + 1e-4
    rng = np.random.default_rng(9)
    X = np.vstack([lb, ub, lb + (ub - lb) * rng.integers(0, 2, (62, lb.size)),
                   lb + (ub - lb) * rng.random((64, lb.size))])
    y = _replay(ref, X.reshape(-1, 3, 32, 32))
    G = vc.margin_spec(3)
    sampled_min = (y @ G.T).min(axis=0)
    g = sampled_min + 1e-9 * (1.0 + np.abs(sampled_min))
    safe, margins = vc.verify_halfspace_group_safe(ops, lb, ub, G, g, **options)
    assert not safe, f"safe on margins {margins}"
    for r in range(len(G)):
        assert margins[r] <= sampled_min[r] - g[r] + _pad(sampled_min[r]), \
            f"row {r}: margin {margins[r]!r} above the sampled {sampled_min[r] - g[r]!r}"
