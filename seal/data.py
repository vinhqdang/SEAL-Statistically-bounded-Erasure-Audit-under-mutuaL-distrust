"""Dataset loading and the population split used throughout the experiments.

Every dataset is split into three disjoint pools:
  * training pool   -- partitioned across federation clients (banks / FL
    participants), this is the actual sensitive "retained set".
  * calibration pool -- a public reference corpus, never used in training,
    that the server uses to characterize the null distribution of honest
    unlearning effect sizes (SEAL's calibration step).
  * holdout pool     -- a disjoint, never-trained-on pool used only to
    *evaluate* how much an audit channel leaks about training-set
    membership (the standard non-member side of a membership-inference
    evaluation). Never touched by any mechanism or by SEAL itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, fetch_covtype
from sklearn.preprocessing import StandardScaler


@dataclass
class Population:
    name: str
    X: np.ndarray
    y: np.ndarray
    n_features: int
    n_classes: int


def load_german_credit(seed: int = 0) -> Population:
    """UCI Statlog (German Credit Data), fetched as OpenML `credit-g`.
    A real bank credit-scoring dataset -- used here as the bank-consortium
    RTBF instantiation described in the SEAL protocol."""
    bunch = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    y = (df.pop("class") == "bad").astype(int).to_numpy()
    cat_cols = [c for c in df.columns if df[c].dtype.name in ("category", "object")]
    num_cols = [c for c in df.columns if c not in cat_cols]
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])
    X = pd.get_dummies(df, columns=cat_cols, drop_first=True).to_numpy(dtype=float)
    return Population("german_credit", X, y, n_features=X.shape[1], n_classes=2)


def load_mnist_subset(n_samples: int = 9000, seed: int = 0) -> Population:
    """A subsample of MNIST -- the standard image-classification benchmark
    used throughout the machine-unlearning literature (SISA, membership
    inference, etc.), included so SEAL is evaluated on the same class of
    benchmark as the SOTA it is compared to, not only on the fintech case."""
    bunch = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_full = bunch.data.astype(float) / 255.0
    y_full = bunch.target.astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_full), size=n_samples, replace=False)
    return Population("mnist", X_full[idx], y_full[idx], n_features=X_full.shape[1], n_classes=10)


def load_covertype(n_samples: int | None = None, seed: int = 0) -> Population:
    """UCI Forest Covertype -- 581,012 real rows, 54 features, 7-class
    cartographic classification. A genuinely large-scale benchmark (two
    orders of magnitude bigger than German Credit's 1,000 rows and the
    9,000-row MNIST subsample used elsewhere in this project), included so
    the certificate's calibration and detection-power numbers are checked
    at a scale where per-client shards are themselves tens of thousands of
    rows, not a few hundred. `n_samples=None` uses the full dataset."""
    bunch = fetch_covtype()
    X_full = StandardScaler().fit_transform(bunch.data.astype(float))
    y_full = bunch.target.astype(int) - 1  # labels are 1..7 -> 0..6
    if n_samples is not None and n_samples < len(X_full):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_full), size=n_samples, replace=False)
        X_full, y_full = X_full[idx], y_full[idx]
    return Population("covertype", X_full, y_full, n_features=X_full.shape[1], n_classes=7)


@dataclass
class SplitPopulation:
    clients_X: list
    clients_y: list
    calib_X: np.ndarray
    calib_y: np.ndarray
    holdout_X: np.ndarray
    holdout_y: np.ndarray


def split_population(
    pop: Population,
    k_clients: int,
    calib_frac: float,
    holdout_frac: float,
    seed: int,
) -> SplitPopulation:
    rng = np.random.default_rng(seed)
    n = len(pop.X)
    perm = rng.permutation(n)
    n_calib = int(n * calib_frac)
    n_holdout = int(n * holdout_frac)
    calib_idx = perm[:n_calib]
    holdout_idx = perm[n_calib : n_calib + n_holdout]
    train_idx = perm[n_calib + n_holdout :]

    train_idx = rng.permutation(train_idx)
    client_splits = np.array_split(train_idx, k_clients)
    clients_X = [pop.X[idx] for idx in client_splits]
    clients_y = [pop.y[idx] for idx in client_splits]

    return SplitPopulation(
        clients_X=clients_X,
        clients_y=clients_y,
        calib_X=pop.X[calib_idx],
        calib_y=pop.y[calib_idx],
        holdout_X=pop.X[holdout_idx],
        holdout_y=pop.y[holdout_idx],
    )


def nearest_neighbor_pool(query_X: np.ndarray, pool_X: np.ndarray, pool_y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The k nearest (by Euclidean distance) points in `pool` to each row of
    `query_X`, used to build position-matched member / non-member sets for
    the near-neighbor leakage evaluation. Uses a tree-based / chunked
    nearest-neighbor search rather than a dense (query x pool) distance
    matrix, which would be O(n_query * n_pool * n_dims) in memory --
    negligible for a 1,000-row dataset but hundreds of gigabytes for a
    pool the size of Covertype's retained set."""
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(pool_X))
    nn = NearestNeighbors(n_neighbors=k).fit(pool_X)
    _, idx = nn.kneighbors(query_X)
    idx = np.unique(idx.reshape(-1))
    return pool_X[idx], pool_y[idx]
