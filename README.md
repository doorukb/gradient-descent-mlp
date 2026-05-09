# Gradient Descent MLP — From Scratch

A from-scratch NumPy implementation of a Multi-Layer Perceptron trained with
gradient descent, built alongside the
`Bonus_Gradient_descent_3stars.ipynb` problem set.

The notebook is the assignment; this package is the supporting library that
keeps each concept (data, init, forward, loss, backward, optimizer, tuning)
in its own clearly-named module so it can be tested in isolation.

## Project layout

```
gradient-descent-mlp/
├── notebooks/                    original assignment notebook
├── src/mlp/                      library code, one concept per file
├── tests/                        unit tests + numerical gradient checks
├── experiments/                  runnable scripts for §6.1 / §6.2 / §6.3
└── figures/                      saved plots (optional)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to use this repo

1. **The notebook** (`notebooks/Bonus_Gradient_descent_3stars.ipynb`) is the
   primary surface — the assignment expects answers there.
2. **The library** (`src/mlp/`) mirrors every working function from the
   notebook. The two are kept in sync by hand: solve in the notebook, port
   into `src/mlp/`, run tests.
3. **The experiments** in `experiments/` are scripted, reproducible runs of
   §6.1 / §6.2 / §6.3. They import from `src/mlp/` only — useful for
   reproducing results without a notebook environment.

### Notebook → library mapping

| Notebook section            | Module                                |
|-----------------------------|---------------------------------------|
| §1.1 sigmoid derivative     | `src/mlp/activations.py` (notebook §1.1 markdown) |
| §1.2–1.3 derivations        | notebook §1.2–1.3 markdown cells      |
| §1.4 `modify_x_w`           | `src/mlp/forward.py`                  |
| §2 dataset + plots          | `src/mlp/data.py`, `plotting.py`      |
| §3 init                     | `src/mlp/init.py`                     |
| §4.1 sigmoid                | `src/mlp/activations.py`              |
| §4.2 forward pass           | `src/mlp/forward.py`                  |
| §5.1 loss                   | `src/mlp/loss.py`                     |
| §5.2 backprop               | `src/mlp/backward.py`                 |
| §5.3 gradient descent       | `src/mlp/optimizer.py`                |
| §5.4 prediction surface     | `src/mlp/plotting.py`                 |
| §6 tuning                   | `src/mlp/tuning.py` + `experiments/`  |

## Running tests

```bash
pytest tests/ -v
```

The most important test is `tests/test_backward.py`, which compares the
analytical gradient from `backward.py` against numerical finite differences.
If that test passes, backprop is mathematically correct.

`tests/test_loss.py` and `tests/test_activations.py` include the same kind
of numerical gradient check for MSE and sigmoid respectively.

## Running the experiments

From the project root:

```bash
python experiments/01_one_vs_two_layer.py        # §6.1
python experiments/02_with_validation.py         # §6.2
python experiments/03_hyperparameter_search.py   # §6.3
```

Each script is self-contained — it sets its own seeds and runs end-to-end.

## Conventions

- **Model dict.** `{'W0': ndarray, 'W1': ndarray, ...}` — one weight matrix
  per layer, indexed from zero.
- **Forward cache.** `{'A0': X, 'A1': ..., 'AL': output}`, where `Al` is the
  post-activation output of layer `l` (and `A0` is the raw input).
- **Implicit bias (§1.4).** Bias is folded into `W` by giving each weight
  matrix shape `(d_in + 1, d_out)`. The forward pass augments inputs with
  a column of ones; backprop drops the last row of `W` when propagating
  gradients backward.
- **Train / val / test split.** 100 / 20 outer split (in `data.py`); a
  further 80 / 20 train/val split inside training (in `tuning.py`). The
  test set is sealed until the very end of §6.3.

## Assignment status

All seven sections complete; 19 unit tests passing. The §6.3 hyperparameter
search produces an unbiased final test-set evaluation under strict
methodological discipline (test set inspected only after every
hyperparameter choice has been locked in).
