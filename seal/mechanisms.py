"""Server-side unlearning "mechanisms" -- one honest, one honest-but-approximate,
and two dishonest strategies a model owner might run when it receives an
erasure request but is not trusted to comply.

Every mechanism has the signature
    mechanism(model, clients_X, clients_y, forget_client, forget_mask, params, seed) -> Params
so the audit code never needs to know which one produced the model it is
checking -- that is exactly the point of the audit.
"""
from __future__ import annotations

import numpy as np

from .flmodel import FederatedSoftmax, Params


def _drop_forgotten(clients_X, clients_y, forget_client: int, forget_mask: np.ndarray):
    out_X = list(clients_X)
    out_y = list(clients_y)
    keep = ~forget_mask
    out_X[forget_client] = clients_X[forget_client][keep]
    out_y[forget_client] = clients_y[forget_client][keep]
    return out_X, out_y


def honest_retrain_from_scratch(
    model: FederatedSoftmax,
    clients_X, clients_y,
    forget_client: int,
    forget_mask: np.ndarray,
    params: Params,
    seed: int,
    rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
) -> Params:
    """Gold-standard honest unlearning: retrain the whole federation from a
    fresh initialization with the forgotten records removed."""
    fX, fy = _drop_forgotten(clients_X, clients_y, forget_client, forget_mask)
    fresh = FederatedSoftmax(model.n_features, model.n_classes, model.l2)
    fresh.init(seed=seed)
    return model.fedavg_train(fX, fy, rounds, local_epochs, lr, batch_size, seed=seed, init_params=fresh.params)


def honest_finetune_unlearn(
    model: FederatedSoftmax,
    clients_X, clients_y,
    forget_client: int,
    forget_mask: np.ndarray,
    params: Params,
    seed: int,
    extra_rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
) -> Params:
    """Cheaper, still-honest unlearning: warm-start FedAvg from the current
    global model and continue training with the forgotten records removed.
    Approximate (does not exactly match retrain-from-scratch) but genuinely
    updates the model in the direction retraining would."""
    fX, fy = _drop_forgotten(clients_X, clients_y, forget_client, forget_mask)
    return model.fedavg_train(fX, fy, extra_rounds, local_epochs, lr, batch_size, seed=seed, init_params=params)


def dishonest_lazy(
    model: FederatedSoftmax,
    clients_X, clients_y,
    forget_client: int,
    forget_mask: np.ndarray,
    params: Params,
    seed: int,
    extra_rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
) -> Params:
    """Dishonest server that ignores the erasure request and just keeps
    training on the data it already has -- including the records it was
    asked to forget -- then returns the result as if compliant. This is
    the adversary the parameter-inspection impossibility result is
    actually about: it produces a parameter shift of the SAME kind and
    magnitude as honest unlearning (both are "a few more rounds of
    FedAvg"), so distance-from-the-original-checkpoint alone cannot tell
    them apart. A server that instead literally freezes its parameters
    would be caught by parameter distance trivially -- that isn't the
    interesting case."""
    return model.fedavg_train(clients_X, clients_y, extra_rounds, local_epochs, lr, batch_size, seed=seed, init_params=params)


def dishonest_spoof(
    model: FederatedSoftmax,
    clients_X, clients_y,
    forget_client: int,
    forget_mask: np.ndarray,
    params: Params,
    seed: int,
    epochs: int,
    lr: float,
) -> Params:
    """Targeted spoofing attack: gradient-ASCENT fine-tuning restricted to
    exactly the forgotten records, with nothing else touched. This drives
    the loss up on the audited points -- fooling a naive own-data-only
    behavioral test -- while leaving the model's behavior on every nearby
    (but not literally forgotten) point unchanged, because the influence of
    the forgotten records on the rest of the decision boundary was never
    actually removed."""
    Xf = clients_X[forget_client][forget_mask]
    yf = clients_y[forget_client][forget_mask]
    return model.local_sgd(Xf, yf, params, lr=lr, epochs=epochs, batch_size=max(len(Xf), 1), seed=seed, ascent=True)
