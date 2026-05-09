"""§5.1 — Mean squared error loss and its gradient."""

from __future__ import annotations
import numpy as np


def mse_loss(label: np.ndarray, pred: np.ndarray) -> float:
    """Mean squared error between predictions and labels."""
    return float(np.mean((pred - label) ** 2))


def mse_loss_grad(label: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE w.r.t. ``pred`` — used as the backprop seed.

        d_loss / d_pred = 2 * (pred - label) / N
    """
    return 2 * (pred - label) / pred.size
