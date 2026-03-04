# EnerGNN

A Graph Neural Network library for real-life energy networks.

## Build

To build this package locally, you can use one of the following commands at the root of the project:

```cmd
uv sync
```

```cmd
uv sync --extra gpu
```

The first one will install the `energnn` package with only CPU support.

Use the second one to also install the GPU extra dependencies (obtained from `jax[cuda12]`).

## Documentation

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
