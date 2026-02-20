EnerGNN Basics
==============

This page introduces the general framework of the **EnerGNN** library.

Amortized Optimization
----------------------

Consider an optimization problem formulated as follows:

.. math::

    \begin{align}
        y^\star(x) \in \arg \min _y \ f(y;x),
    \end{align}

where:

- :math:`x` is a **context** graph (input data),
- :math:`y` is a **decision** graph (output data),
- :math:`f` is the **objective** function to minimize.

Here, :math:`f` is common to an entire class of problems, while :math:`x` is specific to a given instance.
We seek to solve this problem for a distribution of contexts :math:`x \sim p`, using a trainable GNN model :math:`\hat{y}_\theta`, parameterized by :math:`\theta`.
This leads to the following **Amortized Optimization** problem:

.. math::

    \begin{align}
        \theta^\star \in \arg \min _\theta \ \mathbb{E}_{x \sim p} [f(\hat{y}_\theta;x)].
    \end{align}

This problem is addressed via the following training loop:

.. math::

    \begin{align}
        x &\sim p & & (1) \text{Context sampling}\\
        \hat{y} &\gets \hat{y}_\theta(x) & & (2) \text{Decision inference} \\
        \hat{g} &\gets \nabla_y f(\hat{y};x) & & (3) \text{Gradient estimation} \\
        \theta &\gets \theta - \alpha J_\theta[\hat{y}_\theta]^\top.\hat{g} & & (4) \text{Back-propagation}
    \end{align}

**EnerGNN** handles steps (2) and (4), which are independent of the use case. Steps (1) and (3) are handled by specific implementations of :class:`energnn.problem.Problem`.

-------------------------

Data Representation using H2MG
-------------------------

Contexts :math:`x`, decisions :math:`\hat{y}`, and gradients :math:`\hat{g}` are all represented as graphs within the **H2MG** (*Hyper Heterogeneous Multi Graph*) structure.

- **Hyper**: Edges can connect more than two entities (hyper-edges) using an address system.
- **Heterogeneous**: The graph can contain different types of edges, each with its own features.
- **Multi**: Multiple edges can exist between the same entities.

In practice, an :class:`energnn.graph.Graph` is a dictionary of :class:`energnn.graph.Edge` objects. Each edge type defines:
- `address_list`: The entities connected by this edge.
- `feature_list`: The features associated with this edge.

.. code-block:: python

    from energnn.graph import Graph, Edge

    # Simplified example of H2MG structure
    graph = Graph.from_dict(edge_dict={
        "nodes": Edge(features=node_features),
        "lines": Edge(features=line_features, addresses=line_indices),
    })

For computations with JAX, we use :class:`energnn.graph.JaxGraph`, which is an optimized version compatible with automatic differentiation.

--------------------------


Implementing a Problem
----------------------

The :mod:`energnn.problem` API provides the interface to integrate your own use cases.

Problem
.......

The :class:`energnn.problem.Problem` class defines a single instance of the problem. It must implement:
- :meth:`get_context`: Returns the input graph :math:`x`.
- :meth:`get_gradient`: Computes the gradient :math:`\nabla_y f` for a given decision. Depending on the use-case, the
  gradient can either be straightforward to compute, or require more expensive Monte-Carlo computations.
- :meth:`get_metrics`: Evaluates the quality of a decision (e.g., value of the cost function).

Il faut aussi déclarer les propriétés context_structure et decision_structure, qui permettent de construire un GNN
compatible.

ProblemBatch & Loader
.....................

For training, problems are grouped into a :class:`energnn.problem.ProblemBatch`. The :class:`energnn.problem.ProblemLoader` is the iterator that provides these batches to the training engine.

.. code-block:: python

    for batch in train_loader:
        context, _ = batch.get_context()
        # The batch efficiently handles GPU computations
        ...

------------------------

Graph Neural Network Models
----------

**EnerGNN** provides GNN models designed to natively process H2MG structures.
The main model, :class:`energnn.model.SimpleGNN`, follows a modular pipeline:

1. **Normalizer**: Adjusts the distribution of input features (e.g., uniformly distributed between -1 and 1).
2. **Encoder**: Embeds features into a latent space.
3. **Coupler**: Handles information propagation (e.g., via Message Passing or Neural ODE) over the graph structure.
4. **Decoder**: Produces the final decision from latent representations.

All modules inherit from `flax.nnx.Module`, allowing great flexibility and perfect integration with the JAX ecosystem.

Pour pouvoir instancier un modèle, il est nécessaire de savoir quelle est la structure des contextes (input) et
quelle est la structure des décisions (output).

Quelques implémentations standards sont disponibles dans TODO, et peuvent s'instancier comme suit:
Code snippet pour montrer comment le créer.

-------------------

Trainer
-------

The :class:`energnn.trainer.SimpleTrainer` orchestrates the learning process.
It takes as input the model, a gradient transformation (via `optax`), and handles:

- The training and evaluation loop.
- Checkpoint saving.
- Metric tracking (via :mod:`energnn.tracker`).

.. code-block:: python

    trainer = SimpleTrainer(model=model, gradient_transformation=optax.adam(1e-3))
    trainer.train(train_loader=train_loader, val_loader=val_loader, n_epochs=10)

-----------------------------

Experiment Tracking & Storage
-----------------------------

EnerGNN provides interfaces to track experiments and persist data.

Tracking
........

The :class:`energnn.tracker.Tracker` interface allows logging metrics, configurations, and artifacts to external platforms (e.g., Neptune). You can pass a tracker to the :meth:`SimpleTrainer.train` method.

.. code-block:: python

    from energnn.tracker import MyCustomTracker # hypothetical implementation

    tracker = MyCustomTracker()
    trainer.train(..., tracker=tracker)

Storage & Checkpointing
.......................

Models and datasets can be persisted using the :class:`energnn.storage.Storage` interface. The :class:`SimpleTrainer` also supports checkpointing via `Orbax`:

- **Checkpoints**: Use :meth:`save_checkpoint` and :meth:`load_checkpoint` to manage model states.
- **Storage**: The :class:`energnn.storage.Storage` interface (e.g., local or S3) handles the physical upload/download of these artifacts.

-----------------------------

Next Steps
----------

Now that you are familiar with the basics, you can:

- Follow the :doc:`tutorial` for a hands-on example.
- Learn how to implement a :doc:`custom_use_case`.
- Explore the :doc:`reference/index` for detailed API information.
