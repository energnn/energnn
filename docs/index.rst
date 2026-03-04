=======
EnerGNN
=======

*A Graph Neural Network library for real-life Energy networks.*

-----

**EnerGNN** is a python package based on `JAX <https://docs.jax.dev/en/latest/index.html>`_ and
`Flax <https://flax.readthedocs.io/en/latest/>`_, that provides :

- A **Hyper Heterogeneous Multi Graph** (H2MG) data representation, especially designed for large complex industrial networks
  (such as an Electrical Power Transmission System);
- A compatible **Graph Neural Network** (GNN) library, robust to structure variations
  (outages, construction of new infrastructure, renaming / reordering, etc.);
- An clear interface to help users apply **energnn** to their own custom use-cases.

It is currently being used in multiple full-scale and real-life use-cases at Réseau de Transport d'Électricité (RTE).
If you wish to either contribute or use it for your use cases, feel free to email us at balthazar.donon@rte-france.com.

-----

Installation
============

To install the CPU version.

.. code-block:: bash

    pip install energnn

Or to install the GPU version.

.. code-block:: bash

    pip install energnn --extra gpu

------------

Basic Usage
===========

This package considers the training of GNN models to solve distributions of optimization problems
(which encompasses traditional supervised learning).

.. code-block:: python

    from energnn.tests.utils import TestProblemLoader
    from energnn.model.ready_to_use import TinyRecurrentEquivariantGNN
    from energnn.trainer import SimpleTrainer
    import optax

    problem_loader = TestProblemLoader(seed=1)
    model = TinyRecurrentEquivariantGNN(
        in_structure=problem_loader.context_structure,
        out_structure=problem_loader.decision_structure,
    )
    trainer = SimpleTrainer(model=model, gradient_transformation=optax.adam(1e-3))
    trainer.train(train_loader=problem_loader, n_epochs=10)

- The loader `TestProblemLoader` allows to iterate over multiple instances of the optimization problem class,
  which encapsulate the business logic (input, output, objective, etc.).
- The model processes graph data whose structure should match `in_structure` and returns graphs whose structure is
  defined as `out_structure`.
- The `trainer` iterates over the loader and updates the model weights to improve the model performance.

Once trained, the model can be applied on new problem instances as follows.

.. code-block:: python

    test_loader = TestProblemLoader(seed=3)
    for problem_batch in test_loader:
        context_batch, _ = problem_batch.get_context()                  # Extract input
        decision_batch, _ = model.forward_batch(graph=context_batch)    # Infer decisions
        metrics, _ = problem_batch.get_metrics(decision=decision_batch) # Compute metrics

-------------

User guides
===========

.. toctree::
    :maxdepth: 2

    basics
    tutorial_notebook
    custom_use_case

-------------

API Reference
=============

For detailed description of energnn classes and methods, check out the API reference documentation.

.. toctree::
   :maxdepth: 2
   :titlesonly:

   reference/index