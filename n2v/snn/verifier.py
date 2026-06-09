"""
SNN verification high-level interface.

Contains Monte Carlo sampling utilities and SNNVerifier, which trains and verifies
F2F SNN models. Modified from external_snnv/snn_comparison.py:
  - Global flag mutations (e.g. _USE_EQ_CONSTRAINTS) target n2v.snn.lp's namespace
    via module-attribute assignment rather than the 'global' statement, which only
    affects the local module's namespace after the code was split.
  - train() saves snn_checkpoint.pt (state dict); load via load_checkpoint()
    or reconstruct F2FMLP manually and call load_state_dict().
"""

from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import n2v.snn.lp as _lp_module
from n2v.snn.model import F2FMLP
from n2v.snn.encoding import encode_batch, latency_from_values, spike_train_from_latencies
from n2v.snn.lp import (
    make_bounds, feasible_latencies,
    build_symbolic_relaxation_lp,
    build_symbolic_relaxation_lp_split,
    verify_symbolic_sample,
    _LP_CONTEXT,
)


# ---------------------------------------------------------------------------
# Monte Carlo helpers
# ---------------------------------------------------------------------------

def bounds_cover_outputs(row: dict, outputs: np.ndarray, tol: float = 1e-6) -> float:
    """Check what fraction of Monte Carlo output samples fall within the LP bounds.

    Used as a sanity check: if the LP bounds are sound, every MC sample should
    be covered. A coverage below 100% indicates an LP soundness violation.

    Returns the coverage percentage (should be ~100% for correct bounds).
    """
    lb = np.array(row["lb"], dtype=float)
    ub = np.array(row["ub"], dtype=float)
    covered = np.logical_and(outputs >= lb[None, :] - tol, outputs <= ub[None, :] + tol)
    return 100.0 * float(np.all(covered, axis=1).mean())


def monte_carlo_outputs(model, image_flat: np.ndarray, indices: np.ndarray,
                        epsilon: float, num_steps: int, num_samples: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Sample random perturbations within the ε-ball and collect model outputs.

    For each sample, a random input value is drawn uniformly from [lb[i], ub[i]]
    for each perturbed dimension i, converted to a spike train, and simulated.

    Returns shape (num_samples, num_classes) of output scores.
    """
    lb, ub = make_bounds(image_flat, indices, epsilon)
    outputs = []
    for _ in range(num_samples):
        point = image_flat.copy()
        if len(indices) > 0:
            point[indices] = rng.uniform(lb[indices], ub[indices])
        lat = latency_from_values(torch.from_numpy(point).float(), num_steps).numpy()
        spike_train = spike_train_from_latencies(lat, num_steps)
        score, _, _ = model.simulate_with_patterns(spike_train)
        outputs.append(score)
    return np.stack(outputs, axis=0)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize(rows, epsilons, ks, mode="full", split_strategy=None):
    """Aggregate per-sample verification rows into summary statistics.

    Groups rows by (epsilon, k) and computes certification rates, median/mean
    gaps, timing statistics, and depth statistics.
    """
    summary = []
    for epsilon in epsilons:
        for k in ks:
            group = [r for r in rows if r["mode"] == mode
                     and math.isclose(r["epsilon"], epsilon) and r["k"] == k
                     and (split_strategy is None or r.get("split_strategy") == split_strategy)]
            if not group:
                continue
            cert = sum(r["certified"] for r in group)
            gaps = np.array([r["gap"] for r in group], dtype=float)
            times = np.array([r["runtime_s"] for r in group], dtype=float)
            widths = np.array([r["bound_width"] for r in group], dtype=float)
            lambdas = np.array([r.get("n_lambda", 0) for r in group], dtype=float)
            patterns = np.array([r.get("n_patterns", 0) for r in group], dtype=float)
            explicit_coverage = (
                100.0 * sum(bool(r.get("explicit_coverage") or False) for r in group) / len(group)
                if any("explicit_coverage" in r for r in group) else 100.0
            )
            mc_coverage = (
                float(np.mean([r.get("mc_coverage_pct", np.nan) for r in group]))
                if any("mc_coverage_pct" in r for r in group) else 100.0
            )
            if any("depth_reached" in r for r in group):
                depths = [r.get("depth_reached", 0) for r in group]
                depth0_count = int(sum(d == 0 for d in depths))
                split_count = int(sum(d > 0 for d in depths))
                median_depth = float(np.median(depths))
            else:
                depths = [0] * len(group)
                depth0_count = split_count = 0
                median_depth = 0.0
            entry = {
                "epsilon": epsilon, "k": k, "samples": len(group),
                "mc_coverage_pct": mc_coverage,
                "explicit_coverage_pct": explicit_coverage,
                "certified_pct": 100.0 * cert / len(group),
                "unknown_pct": 100.0 * (len(group) - cert) / len(group),
                "median_gap": float(np.median(gaps)),
                "median_time_s": float(np.median(times)),
                "median_width": float(np.median(widths)),
                "median_lambda": float(np.median(lambdas)),
                "median_patterns": float(np.median(patterns)),
                "depth0_count": depth0_count,
                "split_count": split_count,
                "median_depth": median_depth,
            }
            if mode == "symbolic":
                certs = [int(r.get("certified", False)) for r in group]
                entry["time_depth"] = [[float(t), int(d), c]
                                       for t, d, c in zip(times.tolist(), depths, certs)]
            else:
                entry["times_s"] = times.tolist()
            summary.append(entry)
    return summary


def summarize_depth0_exhaustive(rows, epsilons, ks):
    """Aggregate results from the two-stage depth-0 + exhaustive-fallback strategy."""
    summary = []
    for epsilon in epsilons:
        for k in ks:
            sym_rows = {
                r["image_idx"]: r for r in rows
                if r["mode"] == "symbolic" and r.get("split_strategy") == "depth0"
                and math.isclose(r["epsilon"], epsilon) and r["k"] == k
            }
            full_rows = {
                r["image_idx"]: r for r in rows
                if r["mode"] == "full"
                and math.isclose(r["epsilon"], epsilon) and r["k"] == k
            }
            all_idxs = set(sym_rows) | set(full_rows)
            if not all_idxs:
                continue

            cert_depth0 = cert_exhaust = cert_neither = 0
            times_depth0 = []
            times_exhaust = []

            for idx in sorted(all_idxs):
                sym = sym_rows.get(idx)
                full = full_rows.get(idx)
                if sym:
                    times_depth0.append(sym["runtime_s"])
                if sym and sym["certified"]:
                    cert_depth0 += 1
                elif full:
                    times_exhaust.append(full["runtime_s"])
                    if full["certified"]:
                        cert_exhaust += 1
                    else:
                        cert_neither += 1
                else:
                    cert_neither += 1

            n = len(all_idxs)
            summary.append({
                "epsilon": epsilon, "k": k, "samples": n,
                "cert_depth0_pct": 100.0 * cert_depth0 / n,
                "cert_exhaustive_pct": 100.0 * cert_exhaust / n,
                "cert_total_pct": 100.0 * (cert_depth0 + cert_exhaust) / n,
                "uncertified_pct": 100.0 * cert_neither / n,
                "median_time_depth0_s": float(np.median(times_depth0)) if times_depth0 else float("nan"),
                "median_time_exhaust_s": float(np.median(times_exhaust)) if times_exhaust else float("nan"),
            })
    return summary


# ---------------------------------------------------------------------------
# Row caching helpers
# ---------------------------------------------------------------------------

def _row_cache_key(image_idx, epsilon, k, mode, split_strategy=None):
    """Compute a hashable cache key for one verification result row."""
    return (image_idx, round(float(epsilon), 10), int(k), mode, split_strategy)


def load_existing_rows(output_dir: Path) -> dict:
    """Load previously computed verification rows from disk into a lookup dict.

    Checks both results.json (final output) and rows_checkpoint.json (partial
    checkpoint from an interrupted run). Returns a dict keyed by _row_cache_key.
    """
    cache = {}
    for fname in ("results.json", "rows_checkpoint.json"):
        path = output_dir / fname
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            for r in data.get("rows", []):
                key = _row_cache_key(r["image_idx"], r["epsilon"], r["k"],
                                     r["mode"], r.get("split_strategy"))
                cache[key] = r
        except Exception as exc:
            print(f"warning: could not load cache from {path} ({exc}); skipping")
    return cache


# ---------------------------------------------------------------------------
# SNNVerifier — public class interface
# ---------------------------------------------------------------------------

class SNNVerifier:
    """Train and verify an F2F SNN. Pixel selection is handled externally by the caller.

    Typical usage:
        verifier = SNNVerifier(hidden_sizes=[128, 64], num_steps=20, beta=0.9,
                               threshold=1.0, output_dir="experiments/outputs/snn")
        verifier.train(train_ds, test_ds, epochs=5, lr=5e-4, ...)
        result = verifier.verify(image_flat, indices, epsilon=0.05, label=3, ...)
    """

    def __init__(self, hidden_sizes: list[int], num_steps: int, beta: float,
                 threshold: float, output_dir: str | Path):
        self.hidden_sizes = list(hidden_sizes)
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold
        self.output_dir = Path(output_dir)
        self.model: F2FMLP | None = None
        self._input_size: int | None = None
        self._num_classes: int | None = None

    def load_checkpoint(self) -> bool:
        """Load the trained model from the saved checkpoint file.

        Returns True if the checkpoint existed and was loaded, False otherwise.
        Supports both the new format (with 'config' key) and the legacy format.
        """
        ckpt = self.output_dir / "snn_checkpoint.pt"
        if not ckpt.exists():
            return False
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "config" in data:
            input_size = data["config"]["input_size"]
            num_classes = data["config"]["num_classes"]
        else:
            # Legacy checkpoint: infer input_size and num_classes from the state dict.
            sd = data["model_state_dict"]
            input_size = sd["fcs.0.weight"].shape[1]
            last_w = sorted([k for k in sd if k.startswith("fcs.") and k.endswith(".weight")],
                            key=lambda k: int(k.split(".")[1]))[-1]
            num_classes = sd[last_w].shape[0]
        model = F2FMLP(
            input_size=input_size, hidden_sizes=self.hidden_sizes,
            num_classes=num_classes, beta=self.beta,
            threshold=self.threshold, num_steps=self.num_steps,
        )
        model.load_state_dict(data["model_state_dict"])
        self.model = model.eval()
        self._input_size = input_size
        self._num_classes = num_classes
        return True

    def train(self, train_ds, test_ds, epochs: int, lr: float,
              train_limit: int, batch_size: int) -> dict:
        """Train the SNN. Saves checkpoint to output_dir. Returns train_summary.

        Saves snn_checkpoint.pt (state-dict checkpoint). Use load_checkpoint()
        to reload, or reconstruct F2FMLP manually and call load_state_dict().

        Infers input_size and num_classes from the first batch of training data.
        Runs one evaluation pass on the full test set after each epoch.
        """
        input_size = train_ds[0][0].numel()   # total number of pixels (flattened)

        # Infer num_classes by scanning a small batch of labels.
        sample_loader = DataLoader(Subset(train_ds, list(range(min(256, len(train_ds))))),
                                   batch_size=256)
        labels_seen = set()
        for _, lbls in sample_loader:
            for l in lbls:
                labels_seen.add(int(l))
        num_classes = max(labels_seen) + 1

        self._input_size = input_size
        self._num_classes = num_classes
        ckpt = self.output_dir / "snn_checkpoint.pt"

        model = F2FMLP(
            input_size=input_size, hidden_sizes=self.hidden_sizes,
            num_classes=num_classes, beta=self.beta,
            threshold=self.threshold, num_steps=self.num_steps,
        )

        # Cap the training set at train_limit samples to allow quick experiments.
        subset = Subset(train_ds, list(range(min(train_limit, len(train_ds)))))
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        summary = {"epochs": epochs, "train_limit": len(subset)}

        for epoch in range(epochs):
            model.train()
            correct = total = 0
            loss_sum = 0.0
            for images, labels in loader:
                spikes = encode_batch(images, self.num_steps)
                scores = model(spikes)
                loss = criterion(scores, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                correct += (scores.argmax(dim=1) == labels).sum().item()
                total += labels.numel()
            summary[f"epoch_{epoch+1}_loss"] = loss_sum / max(1, len(loader))
            summary[f"epoch_{epoch+1}_train_acc"] = 100.0 * correct / max(1, total)

            model.eval()
            t_correct = t_total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    spikes = encode_batch(images, self.num_steps)
                    scores = model(spikes)
                    t_correct += (scores.argmax(dim=1) == labels).sum().item()
                    t_total += labels.numel()
            summary[f"epoch_{epoch+1}_test_acc"] = 100.0 * t_correct / max(1, t_total)
            print(f"epoch {epoch+1}: loss={summary[f'epoch_{epoch+1}_loss']:.4f}"
                  f"  train_acc={summary[f'epoch_{epoch+1}_train_acc']:.1f}%"
                  f"  test_acc={summary[f'epoch_{epoch+1}_test_acc']:.1f}%")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "train_summary": summary,
            "config": {"input_size": input_size, "num_classes": num_classes},
        }, ckpt)
        self.model = model.eval()
        return summary

    @torch.no_grad()
    def test_accuracy(self, test_ds, batch_size: int = 256) -> float:
        """Evaluate classification accuracy on the test set. Returns percentage."""
        assert self.model is not None, "call train() or load_checkpoint() first"
        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        correct = total = 0
        for images, labels in loader:
            spikes = encode_batch(images, self.num_steps)
            scores = self.model(spikes)
            correct += (scores.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
        return 100.0 * correct / max(1, total)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> int:
        """Return the predicted class label for a single image tensor."""
        assert self.model is not None
        spikes = encode_batch(image.unsqueeze(0), self.num_steps)
        return int(self.model(spikes)[0].argmax().item())

    @torch.no_grad()
    def scores(self, image: torch.Tensor) -> np.ndarray:
        """Return raw F2F scores for a single image (shape: num_classes)."""
        assert self.model is not None
        spikes = encode_batch(image.unsqueeze(0), self.num_steps)
        return self.model(spikes)[0].cpu().numpy()

    def verify(self, image_flat: np.ndarray, indices: np.ndarray, epsilon: float,
               label: int, split_depth: int = -1,
               strategy: str = "choice-influence",
               parallel_workers: int = 1,
               parallel_backend: str = "thread",
               mc_samples: int = 0,
               singleton_bounds: bool = False,
               tight_bounds: bool = False,
               track_depth: bool = False,
               image_idx: int | None = None,
               eq_constraints: bool = True,
               debug_lp: bool = False,
               amo: bool = False) -> dict:
        """Certify one sample. indices and epsilon are passed in by the caller.

        split_depth=-1 (default): Two-stage strategy.
          Stage 1: depth-0 LP (single LP, all k pixels symbolic).
          Stage 2: exhaustive fallback if Stage 1 fails.

        split_depth=0: Depth-0 LP only (no fallback).

        split_depth=N>0: Symbolic split to depth N.

        Returns a dict with keys 'symbolic' (Stage 1 result) and 'exhaustive'
        (Stage 2 result, or None if Stage 1 certified or split_depth != -1).
        """
        assert self.model is not None, "call train() or load_checkpoint() first"
        # Seed the failure-diagnostic context with the image index.
        _LP_CONTEXT["image_idx"] = image_idx
        # Set global LP flags in the lp module's namespace (not this module's).
        _lp_module._USE_EQ_CONSTRAINTS = eq_constraints
        _lp_module._DEBUG_LP = debug_lp
        _lp_module._USE_AMO_CONSTRAINTS = amo
        k = len(indices)
        rng = np.random.default_rng(0)

        # Optional Monte Carlo sanity check.
        mc_outputs = None
        if mc_samples > 0:
            mc_outputs = monte_carlo_outputs(
                self.model, image_flat, indices, epsilon,
                self.num_steps, mc_samples, rng,
            )

        if split_depth == -1:
            # Stage 1: depth-0 LP.
            symbolic = verify_symbolic_sample(
                    self.model, image_flat, label, epsilon, k, self.num_steps,
                    tight_bounds=tight_bounds, split_depth=0, max_depth_cap=0,
                    parallel_workers=parallel_workers, parallel_backend=parallel_backend,
                    split_strategy=strategy, track_depth=track_depth,
                    pixel_indices=indices, singleton_bounds=singleton_bounds,
                )
            symbolic["split_strategy"] = "depth0"
            if mc_outputs is not None:
                symbolic["mc_coverage_pct"] = bounds_cover_outputs(symbolic, mc_outputs)
                symbolic["mc_samples"] = mc_samples

            if not symbolic["certified"]:
                # Stage 2: exhaustive fallback.
                exh = self._verify_exhaustive(image_flat, indices, epsilon, label)
                if mc_outputs is not None:
                    exh["mc_coverage_pct"] = bounds_cover_outputs(exh, mc_outputs)
                    exh["mc_samples"] = mc_samples
                return {"symbolic": symbolic, "exhaustive": exh}
            return {"symbolic": symbolic, "exhaustive": None}

        else:
            # Symbolic split: no exhaustive fallback.
            result = verify_symbolic_sample(
                self.model, image_flat, label, epsilon, k, self.num_steps,
                tight_bounds=tight_bounds, split_depth=split_depth,
                max_depth_cap=0 if split_depth == 0 else None,
                parallel_workers=parallel_workers, parallel_backend=parallel_backend,
                split_strategy=strategy, track_depth=track_depth,
                pixel_indices=indices, singleton_bounds=singleton_bounds,
            )
            result["split_strategy"] = strategy
            if mc_outputs is not None:
                result["mc_coverage_pct"] = bounds_cover_outputs(result, mc_outputs)
                result["mc_samples"] = mc_samples
            return {"symbolic": result, "exhaustive": None}

    def _verify_exhaustive(self, image_flat: np.ndarray, indices: np.ndarray,
                           epsilon: float, label: int) -> dict:
        """Exhaustive fallback: enumerate ALL feasible spike-timing combinations.

        For each combination of latency assignments in the Cartesian product
            product(feasible_latencies(lb[i], ub[i], T) for i in indices)
        simulate the network exactly and collect the output scores.

        Certification: label score minus max competitor score > 0 for every combo.
        """
        lb, ub = make_bounds(image_flat, indices, epsilon)
        base_lat = latency_from_values(torch.from_numpy(image_flat).float(), self.num_steps).numpy()

        choices = [feasible_latencies(float(lb[i]), float(ub[i]), self.num_steps) for i in indices]

        n_lambda = int(np.prod([len(c) for c in choices], dtype=np.int64)) if choices else 1

        outputs = []
        t0 = time.perf_counter()
        for combo in itertools.product(*choices):
            lat = base_lat.copy()
            lat[indices] = np.array(combo, dtype=np.int64)
            spike_train = spike_train_from_latencies(lat, self.num_steps)
            score, _, _ = self.model.simulate_with_patterns(spike_train)
            outputs.append(score)
        runtime = time.perf_counter() - t0

        Y = np.stack(outputs, axis=0)
        lb_y = Y.min(axis=0)
        ub_y = Y.max(axis=0)

        gap = float(lb_y[label] - np.max(np.delete(ub_y, label)))
        return {
            "epsilon": float(epsilon),
            "k": int(len(indices)),
            "label": int(label),
            "mode": "full",
            "n_lambda": n_lambda,
            "gap": gap,
            "certified": gap > 0.0,
            "bound_width": float(np.mean(ub_y - lb_y)),
            "runtime_s": runtime,
            "lb": lb_y.tolist(),
            "ub": ub_y.tolist(),
        }
