"""
Benchmark configuration for n2v performance testing.

This module defines benchmark experiments using the same models as CompareNNV
but focused on measuring n2v performance rather than comparing to NNV.
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark experiment."""
    id: int
    name: str
    model: str
    method: str
    set_type: str
    epsilon: float
    relax_factor: Optional[float] = None
    category: str = "misc"
    skip_by_default: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'model': self.model,
            'method': self.method,
            'set_type': self.set_type,
            'epsilon': self.epsilon,
            'relax_factor': self.relax_factor,
            'category': self.category,
        }


# Method configuration for n2v API
METHOD_CONFIG = {
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


def generate_benchmarks() -> List[BenchmarkConfig]:
    """Generate all benchmark configurations."""
    benchmarks = []
    bench_id = 0

    # Default epsilon for MNIST
    eps = 1/255

    # Relaxation factors
    relax_factors = [0.25, 0.50, 0.75]

    # =========================================================================
    # FC MNIST Benchmarks
    # =========================================================================
    for model in ['fc_mnist', 'fc_mnist_small']:
        # Exact method
        bench_id += 1
        benchmarks.append(BenchmarkConfig(
            id=bench_id,
            name=f"{model}_exact",
            model=model,
            method='exact',
            set_type='star',
            epsilon=eps,
            category='fc',
        ))

        # Approx method
        bench_id += 1
        benchmarks.append(BenchmarkConfig(
            id=bench_id,
            name=f"{model}_approx",
            model=model,
            method='approx',
            set_type='star',
            epsilon=eps,
            category='fc',
        ))

        # Relaxed methods (only for fc_mnist)
        if model == 'fc_mnist':
            for method in ['relax-star-area', 'relax-star-range']:
                for rf in relax_factors:
                    bench_id += 1
                    method_short = method.replace('relax-star-', '')
                    benchmarks.append(BenchmarkConfig(
                        id=bench_id,
                        name=f"{model}_{method_short}_{rf:.2f}",
                        model=model,
                        method=method,
                        set_type='star',
                        epsilon=eps,
                        relax_factor=rf,
                        category='fc',
                    ))

    # =========================================================================
    # CNN Benchmarks
    # =========================================================================
    for model in ['cnn_conv_relu', 'cnn_avgpool', 'cnn_maxpool']:
        # Exact method (skip by default - too slow)
        bench_id += 1
        benchmarks.append(BenchmarkConfig(
            id=bench_id,
            name=f"{model}_exact",
            model=model,
            method='exact',
            set_type='imagestar',
            epsilon=eps,
            category='cnn',
            skip_by_default=True,
        ))

        # Approx method
        bench_id += 1
        benchmarks.append(BenchmarkConfig(
            id=bench_id,
            name=f"{model}_approx",
            model=model,
            method='approx',
            set_type='imagestar',
            epsilon=eps,
            category='cnn',
        ))

        # Relaxed methods
        for method in ['relax-star-area', 'relax-star-range']:
            for rf in relax_factors:
                bench_id += 1
                method_short = method.replace('relax-star-', '')
                benchmarks.append(BenchmarkConfig(
                    id=bench_id,
                    name=f"{model}_{method_short}_{rf:.2f}",
                    model=model,
                    method=method,
                    set_type='imagestar',
                    epsilon=eps,
                    relax_factor=rf,
                    category='cnn',
                ))

    # =========================================================================
    # Toy Model Benchmarks
    # =========================================================================
    for model in ['toy_fc_4_3_2', 'toy_fc_8_4_2']:
        for set_type in ['zono', 'box']:
            bench_id += 1
            benchmarks.append(BenchmarkConfig(
                id=bench_id,
                name=f"{model}_{set_type}",
                model=model,
                method='approx',
                set_type=set_type,
                epsilon=0.1,
                category='toy',
            ))

    return benchmarks


# All benchmarks
ALL_BENCHMARKS = generate_benchmarks()


def get_benchmark(bench_id: int) -> BenchmarkConfig:
    """Get benchmark by ID."""
    for bench in ALL_BENCHMARKS:
        if bench.id == bench_id:
            return bench
    raise ValueError(f"Benchmark ID {bench_id} not found")


def get_benchmarks_by_model(model: str) -> List[BenchmarkConfig]:
    """Get all benchmarks for a specific model."""
    return [b for b in ALL_BENCHMARKS if b.model == model]


def get_benchmarks_by_method(method: str) -> List[BenchmarkConfig]:
    """Get all benchmarks for a specific method."""
    return [b for b in ALL_BENCHMARKS if b.method == method]


def get_benchmarks_by_category(category: str) -> List[BenchmarkConfig]:
    """Get all benchmarks for a specific category."""
    return [b for b in ALL_BENCHMARKS if b.category == category]


def list_benchmarks():
    """Print all benchmarks."""
    print(f"{'ID':<4} {'Name':<35} {'Category':<8} {'Skip':<6}")
    print("-" * 60)
    for bench in ALL_BENCHMARKS:
        skip = "Yes" if bench.skip_by_default else "No"
        print(f"{bench.id:<4} {bench.name:<35} {bench.category:<8} {skip:<6}")


if __name__ == '__main__':
    print("Benchmark Configurations:")
    print("=" * 60)
    list_benchmarks()
    print(f"\nTotal benchmarks: {len(ALL_BENCHMARKS)}")
    print(f"  FC:  {len(get_benchmarks_by_category('fc'))}")
    print(f"  CNN: {len(get_benchmarks_by_category('cnn'))}")
    print(f"  Toy: {len(get_benchmarks_by_category('toy'))}")
