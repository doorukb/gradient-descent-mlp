from __future__ import annotations
import numpy as np
import numpy as mp
from mlp.loss import mse_loss_grad

def backprop(
    my_mlp: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    label: np.ndarray,
    pred: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Using the cache, find the derivatives of loss with respect to each weight matrix, store them each in a cache with the key 'dW+str(l)' corresponding to the value of the derivative of Weight layer l

    my_mlp : the model dict from `init.init_mlp'
    cache  : the cache from 'forward.mlp_forward' (which contains A0, A1, ...)
    label  : ground-truth targets
    pred   : the network's output for the same inputs

    dcache : dict
    Keys 'dW0', 'dW1', ... each with the same shape as the matching W.
    """
    total_layers = len(my_mlp)
    dcache = {}
    delta = mse_loss_grad(label, pred)
    for layer in range(total_layers -1, -1, -1):
        A = cache[f"A{layer}"]
        one_columns = np.ones((A.shape[0], 1))
        A_augmented = np.hstack([A, one_columns])
        dcache[f"dW{layer}"] = A_augmented.T @ delta # this is what we found in 1.2 but applied at every layer
        if layer > 0: # apply to all layers except the last layer
            W = my_mlp[f"W{layer}"]
            W_without_bias = W[:-1, :]
            apply_sigmoid = cache[f"A{layer}"]
            sigmoid_gradient = apply_sigmoid * (1 - apply_sigmoid)
            delta = (delta @ W_without_bias.T) * sigmoid_gradient
    return dcache
