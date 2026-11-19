"""
FairN2V - Group Fairness Runner (flow population)
Runs the group-fairness notions on a dataset selected via --dataset (default:
adult), each certified against per-group normalizing flows with Clopper-Pearson
intervals:
  - demographic parity  (verify_parity):  the 80%-rule on the marginal favorable rate
  - equal opportunity   (verify_eqodds):  TPR gap within tolerance tau
  - equalized odds      (verify_eqodds):  TPR and FPR gaps within tolerance tau
The group-fairness counterpart to run_individual_fairness.py; follows the n2v
example house style (main() + argparse, as in FlowConformal / ACASXu).

USAGE (run from the FairN2V dir):
  python run_group_fairness.py                          # Adult (default)
  python run_group_fairness.py --dataset adult_debiased
  python run_group_fairness.py --dataset german
  python run_group_fairness.py --dataset bank
  python run_group_fairness.py --dataset folktables
  python run_group_fairness.py --dataset folktables_race

OUTPUTS (under FairN2V/results/group_fairness/<timestamp>/):
  - parity_<dataset>.csv         demographic-parity verdict, flow rates + CP intervals, runtime
  - parity_<dataset>.png / .pdf  per-group favorable-rate bar chart
  - eqodds_<dataset>.csv         equal-opportunity + equalized-odds verdicts, per-group TPR/FPR + CP intervals, runtime
  - eqodds_<dataset>.png / .pdf  per-group TPR/FPR bar chart

REQUIREMENTS:
  - n2v toolbox installed (import n2v.probabilistic.flow)
  - the chosen dataset's ONNX models in FairN2V/models/ and .npz in
    FairN2V/data/ (named by its adapter.RUN_PROFILES entry)
"""

import argparse
import datetime
import sys
import warnings
from pathlib import Path

# This runner sits at the FairN2V top; the flow group-fairness library modules
# live in group_fairness/. Put both that dir and the FairN2V dir on the path.
_FAIRN2V_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_FAIRN2V_DIR / 'group_fairness'))  # verify_parity/eqodds, plot_parity/eqodds
sys.path.insert(0, str(_FAIRN2V_DIR))                          # find adapter

import verify_parity
import verify_eqodds
import plot_parity
import plot_eqodds

from adapter import RUN_PROFILES

## ================== CONFIGURATION ==================
# Statistical knobs; the CLI exposes only the common run knobs (--dataset,
# --models, --n-samples). Edit here to change the rest.
C = 0.8            # demographic-parity threshold, the 80%-rule constant
TAU = 0.1          # equal-opp / equalized-odds tolerance: max allowed TPR/FPR gap between groups
DELTA = 0.05       # spec-level failure budget (verdict holds with prob >= 1 - DELTA)
N_SAMPLES = 50_000 # flow draws per group for the Clopper-Pearson interval
N_EPOCHS = 300     # training epochs per per-group flow
RANDOM_SEED = 0    # RNG seed for reproducibility
SAVE_PNG = True
SAVE_PDF = True
## ================== END CONFIGURATION ==================


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the FairN2V flow group-fairness pipeline '
                    '(demographic parity + equalized odds) on a dataset.')
    parser.add_argument('--dataset', default='adult', choices=list(RUN_PROFILES),
                        help='Dataset profile to verify (default: adult).')
    parser.add_argument('--n-samples', type=int, default=N_SAMPLES,
                        help=f'Flow draws per group for the CP interval (default: {N_SAMPLES}).')
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
        'output_dir': _FAIRN2V_DIR / 'results' / 'group_fairness' / ts,
        'dataset': args.dataset,
        'data_file': profile['data_file'],
        'model_list': args.models or profile['model_list'],
        'n_samples': args.n_samples,
        'n_epochs': N_EPOCHS,
        'random_seed': RANDOM_SEED,
        'c': C,
        'tau': TAU,
        'delta': DELTA,
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

    print("======= FairN2V Group Fairness Pipeline ==========\n")
    print("Validating configuration...")
    validate_config(config)
    print("\nConfiguration validated successfully.\n")

    print("======= STEP 1: Demographic parity ==========\n")
    verify_parity.main(config)

    print("\n======= STEP 2: Equal opportunity + equalized odds ==========\n")
    verify_eqodds.main(config)

    print("\n======= STEP 3: Generating figures ==========\n")
    plot_parity.main(config)
    plot_eqodds.main(config)

    print("\n======= FairN2V Group Fairness Pipeline Complete ==========")
    print(f"All results saved to: {config['output_dir']}")
    return config


if __name__ == "__main__":
    main()
