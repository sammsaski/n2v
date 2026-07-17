"""Flow-based per-group population model for group fairness, built on n2v's
probabilistic flow module.

Each group's population P_{V|a} is a **normalizing flow** trained on that group's
real input rows with n2v's flow matching (`train_flow` + `FlowODE`) -- the kind
of learned density model VeriFair prescribes as a population. Drawing a synthetic
group member is sampling the flow. Both group-fairness notions build on this:
`verify_parity` (demographic parity) fits one flow per group on all rows;
`verify_eqodds` (equalized odds) passes a true-label `mask` so each flow models a
per-(group, label) cell. The verdict then bounds the relevant per-group rate with
Clopper-Pearson.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_FAIRN2V_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_FAIRN2V_DIR))

from adapter import DatasetAdapter
from n2v.probabilistic.flow import FlowODE, VelocityField, train_flow


def make_favorable_classifier(adapter: DatasetAdapter):
    """Batched ``V -> bool`` favorable-outcome classifier from the trained net:
    argmin/argmax of the logits (per ``class_type``) compared to ``positive_class``."""
    net = adapter.net
    pos = adapter.positive_class
    take_min = adapter.class_type == 'min'

    def classify(V) -> np.ndarray:
        x = torch.as_tensor(np.asarray(V), dtype=torch.float32)
        logits = net.forward(x)
        preds = logits.argmin(dim=1) if take_min else logits.argmax(dim=1)
        return (preds == pos).cpu().numpy()

    return classify


def _group_labels(adapter: DatasetAdapter) -> np.ndarray:
    """Group index per sample: the 0/1 sensitive value (binary) or the argmax
    over the sensitive one-hot block."""
    cols = adapter.sensitive_features
    if adapter.sensitive_encoding == 'binary':
        return (adapter.X[cols[0]] > 0.5).astype(int)
    return adapter.X[cols, :].argmax(axis=0)


class FlowGroupModel:
    """Per-group normalizing-flow population model.

    Fits one flow per group to that group's standardized input rows via n2v's
    conditional flow matching (random pairing; no OT coupling). ``sample(a, batch)``
    draws Gaussian latents, maps them to data space with ``FlowODE.inverse``,
    de-standardizes, clamps to ``[0, 1]``, and pins the sensitive columns to group ``a``.

    ``mask`` (optional boolean array over samples) restricts training to a subset
    of rows -- e.g. a true-label cell, so each group's flow models
    ``P(features | group=a, Y=label)`` for the equalized-odds construction.
    """

    def __init__(self, adapter: DatasetAdapter, *, mask=None, hidden: int = 128,
                 n_layers: int = 4, n_epochs: int = 300, batch_size: int = 256,
                 coupling: str = 'none', n_ode_steps: int = 30, seed: int = 0):
        self.cols = list(adapter.sensitive_features)
        self.encoding = adapter.sensitive_encoding
        self.k = 2 if self.encoding == 'binary' else len(self.cols)
        self.n_ode_steps = n_ode_steps
        self.dim = adapter.X.shape[0]
        torch.manual_seed(seed)
        self._gen = torch.Generator().manual_seed(seed)

        Xt = adapter.X.T                               # (n_samples, n_features)
        labels = _group_labels(adapter)
        if mask is not None:                           # restrict to a label cell
            Xt = Xt[mask]
            labels = labels[mask]
        self.flows: list[FlowODE] = []
        self.stats: list[tuple[np.ndarray, np.ndarray]] = []   # (mean, std) per group
        self.group_sizes: list[int] = []
        for a in range(self.k):
            Xa = Xt[labels == a]
            if Xa.shape[0] < 2:
                raise ValueError(f"group {a} has {Xa.shape[0]} sample(s); need >= 2 to fit a flow")
            mean = Xa.mean(axis=0)
            std = Xa.std(axis=0) + 1e-6
            Z = torch.tensor((Xa - mean) / std, dtype=torch.float32)
            vf = VelocityField(dim=self.dim, hidden=hidden, n_layers=n_layers)
            vf, _ = train_flow(vf, Z, n_epochs=n_epochs, batch_size=batch_size,
                               coupling=coupling)
            self.flows.append(FlowODE(vf))
            self.stats.append((mean, std))
            self.group_sizes.append(int(Xa.shape[0]))

    def sample(self, a: int, batch: int) -> np.ndarray:
        """Draw ``batch`` synthetic members of group ``a``, shape (batch, dim)."""
        mean, std = self.stats[a]
        z = torch.randn(batch, self.dim, generator=self._gen)
        with torch.no_grad():
            X = self.flows[a].inverse(z, n_steps=self.n_ode_steps, method='rk4').numpy()
        X = np.clip(X * std + mean, 0.0, 1.0)
        if self.encoding == 'binary':
            X[:, self.cols[0]] = float(a)
        else:
            X[:, self.cols] = 0.0
            X[:, self.cols[a]] = 1.0
        return X
