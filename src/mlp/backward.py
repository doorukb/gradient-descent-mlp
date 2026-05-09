"""
§5.2 — Backpropagation.

Generalises the §1.2 / §1.3 derivations to arbitrary depth. Walks layers from
last to first, computing dW_l = (augmented A_l)^T · δ at each step, then
propagates δ backward through (a) the linear part W^T (with the bias row
dropped), then (b) the elementwise sigmoid derivative.
"""

from __future__ import annotations
import numpy as np

from mlp.loss import mse_loss_grad


def backprop(
    my_mlp: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    label: np.ndarray,
    pred: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute ∂L/∂W_l for every weight matrix in the model.

    Returns a dict with keys 'dW0', 'dW1', ... matching the model's W keys,
    where each dW{l} has the same shape as W{l}.
    """
    total_layers = len(my_mlp)
    dcache: dict[str, np.ndarray] = {}

    # Seed: gradient of loss w.r.t. the network's output. The output layer is
    # linear, so no sigmoid factor here.
    delta = mse_loss_grad(label, pred)

    for layer in range(total_layers - 1, -1, -1):
        A = cache[f"A{layer}"]
        ones_column = np.ones((A.shape[0], 1))
        A_augmented = np.hstack([A, ones_column])

        # The §1.2 result, applied at every layer.
        dcache[f"dW{layer}"] = A_augmented.T @ delta

        # Propagate delta backward into the previous layer (if any).
        if layer > 0:
            W = my_mlp[f"W{layer}"]
            W_without_bias = W[:-1, :]
            # A_{layer} is post-sigmoid, so σ'(z) = A * (1 - A).
            sigmoid_gradient = A * (1 - A)
            delta = (delta @ W_without_bias.T) * sigmoid_gradient

    return dcache
