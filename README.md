# EnerGNN

[![PyPI Latest Release](https://img.shields.io/pypi/v/energnn.svg)](https://pypi.org/project/energnn/)
[![Documentation Status](https://readthedocs.org/projects/energnn/badge/?version=latest)](https://energnn.readthedocs.io/en/latest/?badge=latest)
[![MPL-2.0 License](https://img.shields.io/badge/license-MPL_2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)
[![Actions Status](https://github.com/energnn/energnn/actions/workflows/python-package.yml/badge.svg)](https://github.com/energnn/energnn/actions/workflows/python-package.yml)


A Graph Neural Network library for real-life energy networks.

## Documentation

You can find the documentation to date with the last release [here](https://energnn.readthedocs.io/) on ReadTheDocs.

## Installation

EnerGNN is available on [PyPi](https://pypi.org/project/energnn/) for Python >= 3.11.
To install it, run the following command:

```shell
pip install energnn
```

If you want to install the extra GPU dependencies, use this instead :

```shell
pip install energnn[gpu]
```

## Build from sources

To build this package locally, you can use one of the following commands at the root of the project:

```cmd
uv sync
```

```cmd
uv sync --extra gpu
```

The first one will install the `energnn` package with only CPU support.

Use the second one to also install the GPU extra dependencies (obtained from `jax[cuda12]`).

## Build the documentation

To build and access the documentation, run the following:

```shell
cd docs
make html
open _build/html/index.html
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