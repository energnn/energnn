# EnerGNN

[![PyPI Latest Release](https://img.shields.io/pypi/v/energnn.svg)](https://pypi.org/project/energnn/)
[![Documentation Status](https://readthedocs.org/projects/energnn/badge/?version=latest)](https://energnn.readthedocs.io/en/latest/?badge=latest)
[![MPL-2.0 License](https://img.shields.io/badge/license-MPL_2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)
[![Actions Status](https://github.com/energnn/energnn/actions/workflows/python-package.yml/badge.svg)](https://github.com/energnn/energnn/actions/workflows/python-package.yml)

A Graph Neural Network library based on [JAX](https://docs.jax.dev/) and [Flax](https://flax.readthedocs.io/), specifically
designed for real-life energy networks and large complex industrial infrastructures.

EnerGNN provides:

- A **Hyper Heterogeneous Multi Graph** (H2MG) data representation.
- A **Graph Neural Network** (GNN) library robust to structure variations (outages, reconfigurations, etc.).
- A clear interface to apply GNNs to custom use-cases (optimization, simulation, etc.).

## Documentation

You can find the full documentation on [ReadTheDocs](https://energnn.readthedocs.io/).

## Installation

EnerGNN is available on [PyPI](https://pypi.org/project/energnn/) for Python >= 3.11.

```shell
pip install energnn
```

If you want to install the extra GPU dependencies, use:

```shell
pip install energnn[gpu]
```

## Quick Start

This example shows how to train a small GNN to solve a linear system (DC Power Flow) modeled as a graph.

```python
import optax
from energnn.problem.example import LinearSystemProblemLoader
from energnn.model.ready_to_use import TinyRecurrentEquivariantGNN
from energnn.trainer import Trainer

# 1. Load a problem (DC Power Flow linear systems)
problem_loader = LinearSystemProblemLoader(seed=1)

# 2. Initialize a model
model = TinyRecurrentEquivariantGNN(
    in_structure=problem_loader.context_structure,
    out_structure=problem_loader.decision_structure,
)

# 3. Train the model
trainer = Trainer(model=model, gradient_transformation=optax.adam(1e-3))
trainer.train(train_loader=problem_loader, n_epochs=10)

# 4. Use the model
for problem_batch in problem_loader:
    context_batch, _ = problem_batch.get_context()
    decision_batch, _ = model.forward_batch(graph=context_batch)
    break
```

## Development

To build this package locally from sources, we recommend using [uv](https://github.com/astral-sh/uv):

```shell
uv sync
# Or for GPU support
uv sync --extra gpu
```

## Supporting Institutions

| RTE                                                                                                                                                | Université de Liège                                                                                                                                | INRIA                                                                                                                                                |
|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="docs/_static/rte_white.png#gh-dark-mode-only" height="100px"/> <img src="docs/_static/rte_black.png#gh-light-mode-only" height="100px"/> | <img src="docs/_static/ulg_white.png#gh-dark-mode-only" height="100px"/> <img src="docs/_static/ulg_black.png#gh-light-mode-only" height="100px"/> | <img src="docs/_static/inria_white.png#gh-dark-mode-only" width="160px"/> <img src="docs/_static/inria_black.png#gh-light-mode-only" width="160px"/> |

## Cite Us

```bibtex
@software{energnn,
    author = {{Committers of EnerGNN}},
    title = {{EnerGNN: A Graph Neural Network library for real-life Energy networks.}},
    url = {https://github.com/energnn},
}
```