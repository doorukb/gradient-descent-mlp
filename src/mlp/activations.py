"""
§4.1 — Sigmoid activation.

The backward function uses the closed-form derivative derived in §1.1:

    dσ/dx = σ(x) * (1 - σ(x))
"""

from __future__ import annotations
import numpy as np


def sigmoid_forward(x: np.ndarray) -> np.ndarray:
    """Elementwise sigmoid: σ(x) = 1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-x))


def sigmoid_backward(x: np.ndarray) -> np.ndarray:
    """Elementwise derivative of sigmoid w.r.t. its input, σ(x) * (1 - σ(x))."""
    s = sigmoid_forward(x)
    return s * (1 - s)
