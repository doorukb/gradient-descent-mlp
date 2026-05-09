"""
§1.4 + §4.2 — Implicit-bias trick + full forward pass.

The forward pass stores each layer's post-activation output in a cache as
A{l} (with A0 = X), so backprop can reuse them without recomputing.
"""

from __future__ import annotations
import numpy as np

from mlp.activations import sigmoid_forward


def modify_x_w(
    x: np.ndarray, w: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    §1.4 — Return (x', w') such that x' · w' == x · w + b.

    x is promoted to 2D internally so the augmentation works for both single
    samples and batches. The output is always 2D — that's intentional, since
    the matmul X' · W' is always at least 2D anyway.
    """
    x = np.atleast_2d(x)
    ones_column = np.ones((x.shape[0], 1))
    x_new = np.hstack([x, ones_column])
    w_new = np.vstack([w, np.atleast_2d(b)])
    return x_new, w_new


def mlp_forward(
    my_mlp: dict[str, np.ndarray], x: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    §4.2 — Forward pass through an arbitrary-depth MLP.

    Sigmoid is applied on every layer except the final one (the output layer
    is linear, since we are doing regression on a continuous target).

    Returns
    -------
    cache  : dict with keys 'A0', 'A1', ..., 'AL'  (A0 == x, AL == output)
    output : ndarray, identical to cache['AL']
    """
    x = np.atleast_2d(x)
    cache = {"A0": x}
    n_layers = len(my_mlp)
    A = x

    for l in range(n_layers):
        W = my_mlp[f"W{l}"]
        A_aug = np.hstack([A, np.ones((A.shape[0], 1))])
        Z = A_aug @ W
        A = sigmoid_forward(Z) if l < n_layers - 1 else Z
        cache[f"A{l + 1}"] = A

    return cache, A
