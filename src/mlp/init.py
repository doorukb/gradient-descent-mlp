"""
§3 — Model initialization.

Weights are sampled from N(mean=1, std=0.25). The MLP is stored as a dict
{'W0': ndarray, 'W1': ndarray, ...} to match the notebook's convention.

The implicit-bias trick (§1.4) is baked in here: each weight matrix has shape
(input_dim + 1, output_dim), where the extra row absorbs the bias. The forward
pass is responsible for augmenting x with a column of ones.
"""

from __future__ import annotations
import numpy as np


def init_weight_matrix(In: int = 2, Out: int = 1) -> np.ndarray:
    """Single weight matrix of shape (In, Out), drawn from N(1, 0.25)."""
    return np.random.normal(loc=1.0, scale=0.25, size=(In, Out))


def init_mlp(output_sizes: list[int] = [2, 5, 1]) -> dict[str, np.ndarray]:
    """
    Build an MLP whose layer sizes follow ``output_sizes``.

    The first element of ``output_sizes`` is the input dimension; each subsequent
    element is the output dim of that layer (and the input dim of the next).

    Each W{i} has shape (output_sizes[i] + 1, output_sizes[i + 1]) — the +1 is
    the bias row from the implicit-bias trick (§1.4).
    """
    model: dict[str, np.ndarray] = {}
    n_layers = len(output_sizes) - 1
    for i in range(n_layers):
        in_dim = output_sizes[i] + 1   # +1 for the implicit bias row
        out_dim = output_sizes[i + 1]
        model[f"W{i}"] = init_weight_matrix(in_dim, out_dim)
    return model
