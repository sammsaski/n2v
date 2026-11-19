"""
DatasetAdapter: a dataset-agnostic container for everything the fairness
verification loop needs to know about a dataset.

The adapter holds *nouns* (facts about the data): the samples, the trained
network, per-feature clamps, and a declaration of which feature is sensitive
and how the model picks its class. The verification loop and the fairness
*definitions* (the verbs) read these facts instead of hardcoding them.

This module provides the DatasetAdapter container, a per-dataset loader for
each supported dataset (load_adult, load_german, ...), and two registries the
pipeline selects from via config['dataset']: LOADERS (what the dataset is) and
RUN_PROFILES (its data file + default model list). Adding a dataset in the
shared npz X/y format is a thin loader plus one entry in each registry.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from n2v.nn import NeuralNetwork
from n2v.utils.model_loader import load_onnx
from n2v.utils.model_preprocessing import strip_final_softmax


@dataclass
class DatasetAdapter:
    name: str                       # human-readable dataset name, e.g. 'adult'

    # --- data (loaded + normalized) ---
    X: np.ndarray                   # features, shape (n_features, n_samples)
    y: np.ndarray                   # int labels, shape (n_samples,)
    min_values: np.ndarray          # per-feature lower clamp, shape (n_features,)
    max_values: np.ndarray          # per-feature upper clamp, shape (n_features,)

    # --- trained model ---
    net: NeuralNetwork              # softmax already stripped, ready for reach()

    # --- fairness declaration ---
    sensitive_features: list        # column indices of the protected attribute
    perturbable_features: list      # numerical columns epsilon is allowed to move
    sensitive_encoding: str         # 'binary' | 'onehot'
    output_size: int                # number of output classes
    class_type: str                 # 'min' or 'max' -- how this net picks the class
    positive_class: int             # output index counted as the favorable outcome

    # Human-readable names for the sensitive groups, in group-index order
    # (binary: [value-0 name, value-1 name]; onehot: per sensitive column). Used
    # for plot legends; None falls back to "group 0", "group 1", ... Only set
    # where the raw encoding is known -- left None for datasets whose codebook
    # isn't documented, to avoid mislabeling.
    group_names: Optional[list] = None

    def counterfactuals(self, x):
        """Return the counterfactual versions of x w.r.t. the sensitive attribute.

        A counterfactual is x with ONLY the protected attribute changed to an
        alternative value; non-sensitive features are untouched.

        - 'binary'  -> exactly one counterfactual: the flipped value.
        - 'onehot'  -> one counterfactual per *other* category (k-1 of them),
                       each a valid one-hot vector over the sensitive columns.

        The number of counterfactuals therefore depends on the encoding; the
        caller treats a sample as fair only if the prediction is preserved
        across ALL returned counterfactuals.

        Args:
            x: 1-D feature vector, shape (n_features,)

        Returns:
            list[np.ndarray]: counterfactual copies of x (float64)
        """
        x = np.asarray(x, dtype=np.float64)
        cols = list(self.sensitive_features)

        if self.sensitive_encoding == 'binary':
            cf = x.copy()
            cf[cols] = 1.0 - cf[cols]
            return [cf]

        if self.sensitive_encoding == 'onehot':
            out = []
            for c in cols:
                if x[c] == 1.0:
                    continue  # this is the sample's actual category, not a counterfactual
                cf = x.copy()
                cf[cols] = 0.0  # clear the one-hot block...
                cf[c] = 1.0     # ...and activate the alternative category
                out.append(cf)
            return out

        raise ValueError(
            f"sensitive_encoding must be 'binary' or 'onehot', got {self.sensitive_encoding!r}"
        )


def _load_npz_adapter(name, data_dir, model_path, data_file, **declaration):
    """Shared loader for datasets in the common npz `X`/`y` format.

    Loads + min-max normalizes the data and wraps the ONNX model, then stamps
    the per-dataset fairness `declaration` onto a DatasetAdapter. All datasets
    that share this layout (Adult, German, ...) reuse this; each only differs
    in the declaration. Adding such a dataset is therefore one thin wrapper.

    Args:
        name:        dataset name
        data_dir:    directory containing the .npz data file
        model_path:  path to the .onnx model to verify
        data_file:   name of the .npz file inside data_dir
        declaration: the fairness fields (sensitive_features, perturbable_features,
                     sensitive_encoding, output_size, class_type)

    Returns:
        DatasetAdapter: populated, ready for the verification loop
    """
    data_dir = Path(data_dir)

    # --- load data ---
    data = np.load(data_dir / data_file)
    X = data['X']
    y = data['y']

    X_test_loaded = X.T
    y_test_loaded = y[:, 0].astype(int)

    # --- normalize ---
    min_values = X_test_loaded.min(axis=1)
    max_values = X_test_loaded.max(axis=1)

    # Ensure no division by zero for constant features
    variable_features = max_values - min_values > 0
    min_values[~variable_features] = 0.0  # avoids changing constant features
    max_values[~variable_features] = 1.0  # avoids division by zero

    X_test_loaded = (X_test_loaded - min_values[:, None]) / (max_values - min_values)[:, None]

    # --- load + wrap the model ---
    netONNX = load_onnx(model_path)
    netONNX = strip_final_softmax(netONNX)  # drop trailing softmax for Star reachability
    net = NeuralNetwork(netONNX)

    return DatasetAdapter(
        name=name,
        X=X_test_loaded,
        y=y_test_loaded,
        min_values=min_values,
        max_values=max_values,
        net=net,
        **declaration,
    )


def load_adult(data_dir, model_path, data_file='adult_data.npz'):
    """Adult-Income (UCI). Sensitive attribute: sex (binary, column 8)."""
    return _load_npz_adapter(
        'adult', data_dir, model_path, data_file,
        sensitive_features=[8],
        perturbable_features=[0, 9, 10, 11],
        sensitive_encoding='binary',
        output_size=2,
        class_type='min',
        positive_class=0,  # index 0 = income > $50k (~24% base rate)
        group_names=['Female', 'Male'],  # group 0 is the ~33% minority = female
    )


def load_adult_debiased(data_dir, model_path, data_file='adult_data.npz'):
    """Adult-Income (UCI), debiased models. Same data + fairness declaration as
    load_adult (sensitive column 8); only the verified networks differ -- these
    are the fairness-trained ACD-* models, the debiased half of the paper's
    biased-vs-debiased comparison. Ported from adult_debiased_verify.m.
    """
    return _load_npz_adapter(
        'adult_debiased', data_dir, model_path, data_file,
        sensitive_features=[8],
        perturbable_features=[0, 9, 10, 11],
        sensitive_encoding='binary',
        output_size=2,
        class_type='min',
        positive_class=0,  # index 0 = income > $50k (~24% base rate)
        group_names=['Female', 'Male'],  # group 0 is the ~33% minority = female
    )


def load_german(data_dir, model_path, data_file='german_data.npz'):
    """German Credit. Sensitive attribute: sex/marital status (binary, column 19).

    Indices ported from german_verify.m (1-indexed -> 0-indexed):
    sensitive [20]->[19]; perturbable [2,5,8,10,12,15,16]->[1,4,7,9,11,14,15].
    """
    return _load_npz_adapter(
        'german', data_dir, model_path, data_file,
        sensitive_features=[19],
        perturbable_features=[1, 4, 7, 9, 11, 14, 15],
        sensitive_encoding='binary',
        output_size=2,
        class_type='min',
        positive_class=0,  # index 0 = good credit (~70% base rate)
    )


# Folktables ACSIncome (US Census, CA 2018 1-Year): predicts income > $50k.
# Data + FT-* nets are built in a separate prep repo (dataset-prep-fairn2v);
# the FT-* nets are trained for argmin, so class_type='min'.
# The 13-feature representation is shared by the sex and race profiles below
# (same noun, different fairness verb -- only the sensitive declaration differs):
#     [AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RACE_*x4]
#       0    1     2    3    4     5     6     7    8     9..12
# Perturbable = the continuous numerical features age and hours-worked
# ([0]=AGEP, [7]=WKHP); the rest (incl. the race one-hot) are left fixed.
_FOLKTABLES_PERTURBABLE = [0, 7]


def load_folktables(data_dir, model_path, data_file='folktables_data.npz'):
    """Folktables ACSIncome, SEX profile: sex is binary (column 8)."""
    return _load_npz_adapter(
        'folktables', data_dir, model_path, data_file,
        sensitive_features=[8],
        perturbable_features=_FOLKTABLES_PERTURBABLE,
        sensitive_encoding='binary',
        output_size=2,
        class_type='min',
        positive_class=1,  # index 1 = income > $50k (~41% base rate)
        group_names=['Male', 'Female'],  # ACS SEX: 1=Male->0, 2=Female->1
    )


def load_folktables_race(data_dir, model_path, data_file='folktables_data.npz'):
    """Folktables ACSIncome, RACE profile: race is a one-hot block (columns
    9..12 = White/Black/Asian/Other). Same data file + FT-* nets as the sex
    profile; only the sensitive declaration changes. Exercises the one-hot
    counterfactual path (k-1 = 3 counterfactuals per sample)."""
    return _load_npz_adapter(
        'folktables_race', data_dir, model_path, data_file,
        sensitive_features=[9, 10, 11, 12],
        perturbable_features=_FOLKTABLES_PERTURBABLE,
        sensitive_encoding='onehot',
        output_size=2,
        class_type='min',
        positive_class=1,  # index 1 = income > $50k (~41% base rate)
        group_names=['White', 'Black', 'Asian', 'Other'],  # one-hot cols 9..12
    )


def load_bank(data_dir, model_path, data_file='bank_data.npz'):
    """Bank Marketing (UCI). Sensitive attribute: age (binary, column 0).

    Indices ported from bank_verify.m (1-indexed -> 0-indexed):
    sensitive [1]->[0]; perturbable [6,12,13,14,15]->[5,11,12,13,14].
    """
    return _load_npz_adapter(
        'bank', data_dir, model_path, data_file,
        sensitive_features=[0],
        perturbable_features=[5, 11, 12, 13, 14],
        sensitive_encoding='binary',
        output_size=2,
        class_type='min',
        positive_class=0,  # index 0 = subscribed (~13% base rate)
    )


# Registry: dataset key -> loader. The verification driver picks one via
# config['dataset'], so adding a dataset = write a loader + add one line here.
LOADERS = {
    'adult': load_adult,
    'adult_debiased': load_adult_debiased,
    'german': load_german,
    'bank': load_bank,
    'folktables': load_folktables,
    'folktables_race': load_folktables_race,
}


# Run profiles: the per-dataset "how to run it" companion to LOADERS' "what it
# is". The runner selects one via config['dataset'] / --dataset to fill in the
# data file and which models to verify. Adding a dataset = one LOADERS entry +
# one RUN_PROFILES entry. (data_file repeats the loader's own default on
# purpose, so the runner can check the file exists before loading anything.)
RUN_PROFILES = {
    'adult':           {'data_file': 'adult_data.npz',  'model_list': ['AC-1', 'AC-3', 'AC-4']},
    'adult_debiased':  {'data_file': 'adult_data.npz',  'model_list': ['ACD-1', 'ACD-3', 'ACD-4']},
    'german':          {'data_file': 'german_data.npz', 'model_list': ['GC-1', 'GC-2', 'GC-3']},
    'bank':            {'data_file': 'bank_data.npz',   'model_list': ['BM-5', 'BM-6', 'BM-7']},
    'folktables':      {'data_file': 'folktables_data.npz', 'model_list': ['FT-1', 'FT-2', 'FT-3']},
    'folktables_race': {'data_file': 'folktables_data.npz', 'model_list': ['FT-1', 'FT-2', 'FT-3']},
}
