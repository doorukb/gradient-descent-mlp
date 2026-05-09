"""§5.3 — Vanilla full-batch gradient descent training loop."""

from __future__ import annotations
import numpy as np

from mlp.forward import mlp_forward
from mlp.backward import backprop
from mlp.loss import mse_loss


def grad_descent(
    data: np.ndarray,
    my_mlp: dict[str, np.ndarray],
    iterations: int,
    learning_rate: float,
) -> tuple[list[float], dict[str, np.ndarray]]:
    """
    Full-batch gradient descent on the synthetic (X, Y, Z) dataset.

    Each iteration:
        1. forward pass
        2. compute MSE loss (recorded after the update)
        3. backprop
        4. update each W_l ← W_l - lr * dW_l

    The returned list also contains the loss BEFORE any updates, so its
    length is iterations + 1.

    Parameters
    ----------
    data : ndarray of shape (N, 3) — columns (X, Y, Z)
    my_mlp : dict from init_mlp; mutated in place
    iterations : number of update steps
    learning_rate : step size η
    """
    # Split into features (X, Y) and targets (Z); keep Z 2D for safe broadcasting.
    inputs = data[:, :2]
    targets = data[:, 2:3]

    _, predictions = mlp_forward(my_mlp, inputs)
    losses = [mse_loss(targets, predictions)]

    for _ in range(iterations):
        cache, predictions = mlp_forward(my_mlp, inputs)
        gradients = backprop(my_mlp, cache, targets, predictions)
        for layer in my_mlp:
            my_mlp[layer] -= learning_rate * gradients[f"d{layer}"]

        _, predictions = mlp_forward(my_mlp, inputs)
        losses.append(mse_loss(targets, predictions))

    return losses, my_mlp
