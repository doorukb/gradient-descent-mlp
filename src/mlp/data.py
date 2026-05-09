"""
§2 — Synthetic dataset.

Sampling from:
    X, Y ~ U[-1, 1]
    err  ~ N(0, 0.5)
    Z    = X^2 - Y^2 + 1.2 + err
"""

from __future__ import annotations
import numpy as np


def sample_points(n: int) -> np.ndarray:
    """Sample n points from the saddle distribution; returns shape (n, 3)."""
    x = np.random.uniform(-1, 1, size=n)
    y = np.random.uniform(-1, 1, size=n)
    err = np.random.normal(loc=0.0, scale=0.5, size=n)
    z = x ** 2 - y ** 2 + 1.2 + err
    return np.column_stack([x, y, z])


def create_train_and_test(
    train_size: int = 100, test_size: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Independently sample a training set and a test set."""
    return sample_points(train_size), sample_points(test_size)
