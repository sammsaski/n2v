"""
FairN2V - Individual / Counterfactual Fairness Runner
Runs the exact-reachability verification pipeline (counterfactual + individual
fairness) on a dataset selected via --dataset (default: adult).

USAGE (run from the FairN2V dir):
  python run_individual_fairness.py                          # Adult (default)
  python run_individual_fairness.py --dataset adult_debiased
  python run_individual_fairness.py --dataset german
  python run_individual_fairness.py --dataset bank
  python run_individual_fairness.py --dataset folktables
  python run_individual_fairness.py --dataset folktables_race

OUTPUTS (under FairN2V/results/individual_fairness/<timestamp>/):
  - CSV files with verification results
  - PNG/PDF figures
  - LaTeX tables (counterfactual, timing)

REQUIREMENTS:
  - n2v toolbox installed (import n2v)
  - the chosen dataset's ONNX models in FairN2V/models/ and .npz in
    FairN2V/data/ (named by its adapter.RUN_PROFILES entry)
"""

import argparse
import datetime
import sys
import warnings
from pathlib import Path

# This runner sits at the FairN2V top; the individual-fairness library modules
# live in individual_fairness/. Put both that dir and the FairN2V dir on the path.
_FAIRN2V_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_FAIRN2V_DIR / 'individual_fairness'))  # verify_individual, plot_individual
sys.path.insert(0, str(_FAIRN2V_DIR))                          # find adapter

import n2v
import verify_individual
import plot_individual

from adapter import RUN_PROFILES

## ================== CONFIGURATION ==================
# Run knobs; the CLI exposes the common ones (--dataset, --num-obs, --models).
# Edit here to change the rest.
NUM_OBS = 100          # test samples per model (auto-capped to dataset size)
RANDOM_SEED = 500      # RNG seed for reproducibility
TIMEOUT = 600          # per-epsilon timeout, seconds
# 0.0 -> counterfactual fairness (flip sensitive attribute only);
# >0.0 -> individual fairness (flip SA + perturb numerical features).
EPSILON_COUNTERFACTUAL = [0.0]
EPSILON_INDIVIDUAL = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
SAVE_PNG = True
SAVE_PDF = True
## ================== END CONFIGURATION ==================


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the FairN2V fairness-verification pipeline on a dataset.')
    parser.add_argument('--dataset', default='adult', choices=list(RUN_PROFILES),
                        help='Dataset profile to verify (default: adult).')
    parser.add_argument('--num-obs', type=int, default=NUM_OBS,
                        help=f'Number of test samples, auto-capped to dataset size '
                             f'(default: {NUM_OBS}).')
    parser.add_argument('--models', nargs='+', default=None, metavar='MODEL',
                        help='Override the profile model list (filenames without .onnx).')
    return parser.parse_args(argv)


def build_config(args):
    """Assemble the config dict the step modules (verify/plot) consume."""
    profile = RUN_PROFILES[args.dataset]
    ts = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    return {
        'models_dir': _FAIRN2V_DIR / 'models',
        'data_dir': _FAIRN2V_DIR / 'data',
        'output_dir': _FAIRN2V_DIR / 'results' / 'individual_fairness' / ts,
        'dataset': args.dataset,
        'data_file': profile['data_file'],
        'model_list': args.models or profile['model_list'],
        'num_obs': args.num_obs,
        'random_seed': RANDOM_SEED,
        'timeout': TIMEOUT,
        'epsilon_counterfactual': EPSILON_COUNTERFACTUAL,
        'epsilon_individual': EPSILON_INDIVIDUAL,
        'save_png': SAVE_PNG,
        'save_pdf': SAVE_PDF,
    }


def validate_config(config):
    """Fail early on a missing dir / data file / model set."""
    if not config['models_dir'].is_dir():
        raise FileNotFoundError(f"Models directory not found: {config['models_dir']}")
    if not config['data_dir'].is_dir():
        raise FileNotFoundError(f"Data directory not found: {config['data_dir']}")
    data_file_path = config['data_dir'] / config['data_file']
    if not data_file_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    model_found = False
    for model_name in config['model_list']:
        if (config['models_dir'] / f"{model_name}.onnx").is_file():
            model_found = True
            print(f"  Found model: {model_name}")
        else:
            warnings.warn(f"Model not found: {config['models_dir'] / f'{model_name}.onnx'}")
    if not model_found:
        raise FileNotFoundError(f"No models found in: {config['models_dir']}")

    config['output_dir'].mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {config['output_dir']}")


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)

    print("======= FairN2V Pipeline ==========\n")
    print("Validating configuration...")
    validate_config(config)
    print("\nConfiguration validated successfully.\n")

    print("======= STEP 1: Running Verification ==========\n")
    verify_individual.main(config)

    print("\n======= STEP 2: Generating Figures ==========\n")
    plot_individual.main(config)

    print("\n======= FairN2V Pipeline Complete ==========")
    print(f"All results saved to: {config['output_dir']}")
    return config


if __name__ == "__main__":
    main()
