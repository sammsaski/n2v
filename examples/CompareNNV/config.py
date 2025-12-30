"""
Configuration for MATLAB NNV vs n2v comparison experiments.

This module defines:
- Experiment configurations (model + method + parameters)
- Method mappings between Python n2v and MATLAB NNV
- Default settings
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


# =============================================================================
# Method Mappings
# =============================================================================

# Maps experiment method names to n2v API parameters
N2V_METHOD_CONFIG = {
    'exact': {
        'method': 'exact',
        'kwargs': {},
    },
    'approx': {
        'method': 'approx',
        'kwargs': {},
    },
    'relax-star-area': {
        'method': 'approx',
        'kwargs': {'relax_method': 'area'},
    },
    'relax-star-range': {
        'method': 'approx',
        'kwargs': {'relax_method': 'range'},
    },
}

# Maps experiment method names to NNV reachMethod names
NNV_METHOD_CONFIG = {
    'exact': 'exact-star',
    'approx': 'approx-star',
    'relax-star-area': 'relax-star-area',
    'relax-star-range': 'relax-star-range',
}


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single verification experiment."""
    id: int
    model: str
    method: str
    set_type: str
    epsilon: float = 1/255
    relax_factor: Optional[float] = None
    sample_id: int = 0

    def get_output_dir(self, base_dir: Path, tool: str = 'n2v') -> Path:
        """Get output directory for this experiment."""
        # For toy models, use set_type as subdirectory (matches MATLAB NNV output structure)
        if self.model.startswith('toy'):
            return base_dir / 'outputs' / tool / self.model / self.set_type

        method_str = self.method
        if self.relax_factor is not None:
            # Use underscore for decimal (e.g., 0_25 instead of 0.25)
            rf_str = f"{self.relax_factor:.2f}".replace('.', '_')
            method_str = f"{self.method}_{rf_str}"
        return base_dir / 'outputs' / tool / self.model / method_str

    def get_result_filename(self) -> str:
        """Get result filename for this experiment."""
        # For toy models, use set_type in filename (matches MATLAB NNV output structure)
        if self.model.startswith('toy'):
            return f"results_{self.set_type}.mat"

        method_str = self.method.replace('-', '_')
        if self.relax_factor is not None:
            # Use underscore for decimal (e.g., 0_25 instead of 0.25)
            rf_str = f"{self.relax_factor:.2f}".replace('.', '_')
            method_str = f"{method_str}_{rf_str}"
        return f"results_{method_str}.mat"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'model': self.model,
            'method': self.method,
            'set_type': self.set_type,
            'epsilon': self.epsilon,
            'relax_factor': self.relax_factor,
            'sample_id': self.sample_id,
        }


# =============================================================================
# Experiment Definitions
# =============================================================================

def generate_experiments() -> List[ExperimentConfig]:
    """Generate all experiment configurations."""
    experiments = []
    exp_id = 0

    # Default epsilon for MNIST
    eps = 1/255

    # Relaxation factors to test
    relax_factors = [0.25, 0.50, 0.75]

    # ==========================================================================
    # FC MNIST Models with Star sets
    # ==========================================================================
    for model in ['fc_mnist', 'fc_mnist_small']:
        # Exact and approx methods
        for method in ['exact', 'approx']:
            exp_id += 1
            experiments.append(ExperimentConfig(
                id=exp_id,
                model=model,
                method=method,
                set_type='star',
                epsilon=eps,
            ))

        # Relaxed methods (only for fc_mnist, not small)
        if model == 'fc_mnist':
            for method in ['relax-star-area', 'relax-star-range']:
                for rf in relax_factors:
                    exp_id += 1
                    experiments.append(ExperimentConfig(
                        id=exp_id,
                        model=model,
                        method=method,
                        set_type='star',
                        epsilon=eps,
                        relax_factor=rf,
                    ))

    # ==========================================================================
    # CNN Models with ImageStar sets
    # ==========================================================================
    for model in ['cnn_conv_relu', 'cnn_avgpool', 'cnn_maxpool']:
        # Exact and approx methods
        for method in ['exact', 'approx']:
            exp_id += 1
            experiments.append(ExperimentConfig(
                id=exp_id,
                model=model,
                method=method,
                set_type='imagestar',
                epsilon=eps,
            ))

        # Relaxed methods
        for method in ['relax-star-area', 'relax-star-range']:
            for rf in relax_factors:
                exp_id += 1
                experiments.append(ExperimentConfig(
                    id=exp_id,
                    model=model,
                    method=method,
                    set_type='imagestar',
                    epsilon=eps,
                    relax_factor=rf,
                ))

    # ==========================================================================
    # Toy Models with Zono and Box sets
    # ==========================================================================
    for model in ['toy_fc_4_3_2', 'toy_fc_8_4_2']:
        for set_type in ['zono', 'box']:
            exp_id += 1
            experiments.append(ExperimentConfig(
                id=exp_id,
                model=model,
                method='approx',
                set_type=set_type,
                epsilon=0.1,  # Larger epsilon for toy models
            ))

    return experiments


# =============================================================================
# Experiment Registry
# =============================================================================

ALL_EXPERIMENTS = generate_experiments()


def get_experiment(exp_id: int) -> ExperimentConfig:
    """Get experiment by ID."""
    for exp in ALL_EXPERIMENTS:
        if exp.id == exp_id:
            return exp
    raise ValueError(f"Experiment ID {exp_id} not found")


def get_experiments_by_model(model: str) -> List[ExperimentConfig]:
    """Get all experiments for a specific model."""
    return [exp for exp in ALL_EXPERIMENTS if exp.model == model]


def get_experiments_by_method(method: str) -> List[ExperimentConfig]:
    """Get all experiments for a specific method."""
    return [exp for exp in ALL_EXPERIMENTS if exp.method == method]


def get_experiments_by_set_type(set_type: str) -> List[ExperimentConfig]:
    """Get all experiments for a specific set type."""
    return [exp for exp in ALL_EXPERIMENTS if exp.set_type == set_type]


def list_experiments():
    """Print all experiments."""
    print(f"{'ID':<4} {'Model':<20} {'Method':<20} {'Set Type':<12} {'Relax Factor':<12}")
    print("-" * 70)
    for exp in ALL_EXPERIMENTS:
        rf = f"{exp.relax_factor:.2f}" if exp.relax_factor else "-"
        print(f"{exp.id:<4} {exp.model:<20} {exp.method:<20} {exp.set_type:<12} {rf:<12}")


# =============================================================================
# Default Settings
# =============================================================================

DEFAULT_EPSILON = 1/255
DEFAULT_SAMPLE_ID = 0
DEFAULT_NUM_CLASSES = 10

# Paths (relative to CompareNNV directory)
MODELS_DIR = Path('models')
SAMPLES_DIR = Path('samples')
OUTPUTS_DIR = Path('outputs')


if __name__ == '__main__':
    print("Experiment Configurations:")
    print("=" * 70)
    list_experiments()
    print(f"\nTotal experiments: {len(ALL_EXPERIMENTS)}")
