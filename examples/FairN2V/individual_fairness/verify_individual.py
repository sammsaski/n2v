"""
Exact Fairness Verification of a fairness classifier (NN), dataset-agnostic.
The dataset (and thus its model, data, and fairness declaration) is selected
via config['dataset'] / the DatasetAdapter loader registry in adapter.py.
Generates results for: (1) Counterfactual fairness table
                       (2) Individual fairness area plot
                       (3) Comprehensive timing table

This script can be run standalone or called from run_individual_fairness.py
Standalone: default paths under the FairN2V dir (../models, ../data,
../results/individual_fairness/<ts>)
Runner-driven: paths come from the `config` dict passed by the runner
"""

import sys
import time
import datetime
from pathlib import Path

import numpy as np
import torch

from n2v.sets import Star, HalfSpace
from n2v.utils.verify_specification import verify_specification

# data/, models/, results/ and adapter.py live in the FairN2V dir, one level up.
_FAIRN2V_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_FAIRN2V_DIR))

from adapter import LOADERS

# Per-sample verdict codes stored in the result matrix.
_FAIR, _UNFAIR, _UNKNOWN = 1, 0, 2


def perturbation_box(x, epsilon, perturbable_features):
    """Build the input Star: an epsilon-box around a (already counterfactual) sample.

    The sensitive attribute is expected to be ALREADY set by the caller via
    adapter.counterfactuals(); this function only widens the perturbable
    numerical features by +/- epsilon and clamps to the valid feature domain.

    The data is min-max normalized upstream, so every feature lives in [0, 1];
    that -- not the raw per-feature min/max -- is the correct clamp domain.
    (Clamping normalized values against raw bounds can invert lb/ub when a
    feature's raw range is not exactly [0, 1].)

    Args:
        x:          1-D feature vector, shape (n,) -- the counterfactual sample
        epsilon:    perturbation radius (0.0 -> counterfactual fairness, no widening;
                    >0.0 -> individual fairness, widen numerical features)
        perturbable_features: numerical columns epsilon is allowed to move

    Returns:
        Star: the input set (box) for reachability
    """
    x = np.asarray(x, dtype=np.float64)

    disturbance = np.zeros_like(x)
    if epsilon > 0:
        disturbance[perturbable_features] = epsilon

    # Clamp to the normalized feature domain [0, 1]
    lb = np.clip(x - disturbance, 0.0, 1.0)
    ub = np.clip(x + disturbance, 0.0, 1.0)

    # float32 at the boundary to match ONNX models' input dtype
    lb = lb.reshape(-1, 1).astype(np.float32)
    ub = ub.reshape(-1, 1).astype(np.float32)
    return Star.from_bounds(lb, ub)


def robustness_set(target, output_size, class_type):
    """
    Create unsafe/not robust region from a target label of a classification NN

    Args:
        target:      label idx of the given input set
        output_size: number of output classes of the NN
        class_type:  assume max, but could also be min like in ACAS Xu ('min', 'max')

    Returns:
        Hs: unsafe/not robust region as a list of HalfSpace objects
    """
    if target >= output_size:
        raise ValueError("Target idx must be less than the output size of the NN.")

    G = np.eye(output_size)
    G = np.delete(G, target, axis=0)
    if class_type == 'max':
        # predicted class is the largest logit: unsafe if a competitor is >= target
        G = -G
        G[:, target] = 1
    elif class_type == 'min':
        # predicted class is the smallest logit: unsafe if a competitor is <= target
        G[:, target] = -1
    else:
        raise ValueError(f"class_type must be 'min' or 'max', got {class_type!r}")

    return [HalfSpace(G[i, :], np.zeros((1, 1))) for i in range(G.shape[0])]


def model_accuracy(adapter):
    """Fraction of the full dataset the (softmax-stripped) net classifies correctly.

    A sanity check that the model loaded correctly: prediction is argmin for
    class_type 'min', argmax otherwise.
    """
    net, X, y = adapter.net, adapter.X, adapter.y
    total_corr = 0
    for i in range(X.shape[1]):
        x_t = torch.tensor(X[:, i], dtype=torch.float32).reshape(1, -1)
        logits = net.forward(x_t)  # (1, n_classes)
        pred = int(logits.argmin()) if adapter.class_type == 'min' else int(logits.argmax())
        total_corr += int(pred == y[i])
    return total_corr / X.shape[1]


def _select_indices(total_obs, num_obs, sample_indices, random_seed):
    """The rows to verify: explicit indices (to match an external tool) or a
    seeded random sample without replacement."""
    if sample_indices is not None:
        return np.asarray(sample_indices, dtype=int)
    rng = np.random.default_rng(random_seed)
    return rng.choice(total_obs, size=num_obs, replace=False)


def is_sample_fair(adapter, x_sample, target, epsilon):
    """Fair iff the prediction is preserved across EVERY counterfactual of the
    sensitive attribute (one for binary, k-1 for a k-category one-hot). A single
    counterfactual whose epsilon-box can reach an unsafe region marks it unfair."""
    for cf in adapter.counterfactuals(x_sample):
        IS = perturbation_box(cf, epsilon, adapter.perturbable_features)
        R = adapter.net.reach(IS, method='exact')
        spec = robustness_set(target, R[0].dim, adapter.class_type)
        if verify_specification(R, spec).verdict != 'UNSAT':
            return False
    return True


def verify_model(adapter, model_name, epsilon, *, num_obs, sample_indices,
                 timeout, random_seed):
    """Verify one model across the epsilon grid; returns per-epsilon result rows.

    Each row is {epsilon, fair, unfair, unknown, total_time, avg_time}. Within an
    epsilon, verification stops once `timeout` seconds elapse and the remaining
    samples are marked unknown.
    """
    X = adapter.X
    total_obs = X.shape[1]
    indices = _select_indices(total_obs, num_obs, sample_indices, random_seed)

    rows = []
    for eps in epsilon:
        res = np.zeros(num_obs)
        times = np.zeros(num_obs)
        t_epsilon_start = time.time()

        for i in range(num_obs):
            idx = indices[i]
            t = time.time()
            fair = is_sample_fair(adapter, X[:, idx], adapter.y[idx], eps)
            res[i] = _FAIR if fair else _UNFAIR
            times[i] = time.time() - t

            if time.time() - t_epsilon_start > timeout:
                print(f"Timeout reached for epsilon = {eps}: stopping verification "
                      f"for this epsilon.")
                res[i + 1:] = _UNKNOWN  # mark remaining as unknown
                break

        rob = int(np.sum(res == _FAIR))
        not_rob = int(np.sum(res == _UNFAIR))
        unk = int(np.sum(res == _UNKNOWN))
        total_time = float(np.sum(times))
        avg_time = total_time / num_obs

        print(f"Model: {model_name}")
        print(f"======= FAIRNESS RESULTS e: {eps} ==========")
        print(" ")
        print(f"Number of fair samples = {rob}, equivalent to {100 * rob / num_obs}% of the samples.")
        print(f"Number of non-fair samples = {not_rob}, equivalent to {100 * not_rob / num_obs}% of the samples.")
        print(f"Number of unknown samples = {unk}, equivalent to {100 * unk / num_obs}% of the samples.")
        print(" ")
        print(f"It took a total of {total_time} seconds to compute the verification results, "
              f"an average of {avg_time} seconds per sample")

        rows.append({
            'epsilon': eps,
            'fair_pct': 100 * rob / num_obs,
            'unfair_pct': 100 * not_rob / num_obs,
            'unknown_pct': 100 * unk / num_obs,
            'total_time': total_time,
            'avg_time': avg_time,
        })
    return rows


def _default_config():
    ts = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    return {
        'models_dir': _FAIRN2V_DIR / 'models',
        'data_dir': _FAIRN2V_DIR / 'data',
        'output_dir': _FAIRN2V_DIR / 'results' / 'individual_fairness' / ts,
        'dataset': 'adult',
        'data_file': 'adult_data.npz',
        'model_list': ['AC-1', 'AC-3'],
        'num_obs': 100,
        'random_seed': 500,
        'timeout': 600,
        'epsilon_counterfactual': [0.0],
        'epsilon_individual': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
    }


def main(config=None):
    ## Setup
    # Config is set by the runner script; fall back to defaults when standalone.
    if config is None:
        config = _default_config()
    config['output_dir'].mkdir(parents=True, exist_ok=True)

    loader = LOADERS[config.get('dataset', 'adult')]
    onnx_files = sorted(config['models_dir'].glob('*.onnx'))
    model_list = config['model_list']
    # Counterfactual fairness (epsilon = 0) and individual fairness (epsilon > 0)
    # are verified over one combined epsilon grid.
    epsilon = config['epsilon_counterfactual'] + config['epsilon_individual']
    sample_indices = config.get('sample_indices')

    ## Loop through each model
    results_counterfactual, results_individual, results_timing = [], [], []
    for onnx_path in onnx_files:
        model_name = onnx_path.stem
        if model_name not in model_list:
            continue

        # Build the dataset adapter: loads + normalizes data and wraps the model.
        # Everything dataset-specific lives behind this single call.
        adapter = loader(config['data_dir'], onnx_path, config['data_file'])
        print(f"Model: {model_name}")
        print(f"Accuracy of Model: {model_accuracy(adapter)}")

        # Cap requested sample count to what the dataset actually has (German has
        # 150 rows vs Adult's 9769); explicit sample_indices are respected as-is.
        total_obs = adapter.X.shape[1]
        num_obs = len(sample_indices) if sample_indices is not None else config['num_obs']
        if sample_indices is None and num_obs > total_obs:
            print(f"Requested num_obs={num_obs} exceeds dataset size {total_obs}; "
                  f"using {total_obs}.")
            num_obs = total_obs

        rows = verify_model(
            adapter, model_name, epsilon,
            num_obs=num_obs, sample_indices=sample_indices,
            timeout=config['timeout'], random_seed=config['random_seed'])

        for row in rows:
            if row['epsilon'] == 0.0:
                results_counterfactual.append({
                    'model': model_name,
                    'fair_pct': row['fair_pct'],
                    'unfair_pct': row['unfair_pct'],
                })
            else:
                results_individual.append({
                    'model': model_name,
                    'epsilon': row['epsilon'],
                    'fair_pct': row['fair_pct'],
                    'unfair_pct': row['unfair_pct'],
                    'unknown_pct': row['unknown_pct'],
                })
            results_timing.append({
                'model': model_name,
                'epsilon': row['epsilon'],
                'total_time': row['total_time'],
                'avg_time': row['avg_time'],
            })

    ## Save results to CSV files
    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    csv_counterfactual = config['output_dir'] / f"counterfactual_{timestamp}.csv"
    with open(csv_counterfactual, "w", encoding="utf-8") as file:
        file.write("Model,FairPercent,UnfairPercent\n")
        for row in results_counterfactual:
            file.write(f"{row['model']},{row['fair_pct']},{row['unfair_pct']}\n")
    print(f"Counterfactual results saved to {csv_counterfactual}")

    csv_individual = config['output_dir'] / f"individual_{timestamp}.csv"
    with open(csv_individual, "w", encoding="utf-8") as file:
        file.write("Model,Epsilon,FairPercent,UnfairPercent,UnknownPercent\n")
        for row in results_individual:
            file.write(f"{row['model']},{row['epsilon']},{row['fair_pct']},{row['unfair_pct']},{row['unknown_pct']}\n")
    print(f"Individual results saved to {csv_individual}")

    csv_timing = config['output_dir'] / f"timing_{timestamp}.csv"
    with open(csv_timing, "w", encoding="utf-8") as file:
        file.write("Model,Epsilon,TotalTime,AvgTimePerSample\n")
        for row in results_timing:
            file.write(f"{row['model']},{row['epsilon']},{row['total_time']},{row['avg_time']}\n")
    print(f"Timing results saved to {csv_timing}")

    print(" ")
    print("======= FairN2V VERIFICATION COMPLETE ==========")
    print("Generated files:")
    print(f"  1. {csv_counterfactual} (for counterfactual fairness table)")
    print(f"  2. {csv_individual} (for individual fairness area plot)")
    print(f"  3. {csv_timing} (for comprehensive timing table)")


if __name__ == "__main__":
    main()
