Glossary
========

This glossary defines key concepts used throughout the **EnerGNN** documentation and library.

Graph & Data Representation
---------------------------

.. glossary::

    H2MG
        Hyper Heterogeneous Multi Graph.
        The core data representation in EnerGNN.
        It extends traditional graphs to support complex industrial network topologies:

        *   **Hyper Graph**: Edges (Hyper-edges) can connect more than two entities.
        *   **Heterogeneous Graph**: Supports multiple types of components (e.g., buses, lines, transformers).
        *   **Multi Graph**: Multiple components can be connected to the same set of entities.

    Hyper-edge
        The fundamental building block of an :term:`H2MG`. Unlike traditional edges that connect exactly two nodes,
        a hyper-edge can connect to any number of addresses via its **ports**.

    Address
        The interface points between :term:`Hyper-edges <Hyper-edge>`. In EnerGNN, addresses do not carry numerical features;
        they only define the connectivity (topology) of the graph.

    Port
        A named connection point on a :term:`Hyper-edge` that links it to an :term:`Address`.
        All hyper-edges of the same type share the same port names.

Optimization Concepts
---------------------

.. glossary::

    Context
        Denoted by :math:`x`.
        The input data of an optimization problem, represented as an H2MG graph.
        It typically contains the parameters of the problem (e.g., network topology, physical constraints).

    Decision
        Denoted by :math:`y`.
        The output produced by the :abbr:`GNN (Graph Neural Network)` model, represented as an H2MG graph.
        It represents the variables to be optimized or predicted (e.g., phase angles, voltage setpoints, etc.).

    Objective Function
        A function :math:`f` that evaluates the quality of a :term:`Decision` :math:`y` given a :term:`Context` :math:`x`
        (i.e. :math:`f(y;x)`).
        In EnerGNN, models are trained to minimize this objective function (or its expectation).

    Gradient of the Objective Function
        A direction of improvement in the decision space for a given :term:`Decision` :math:`y`
        given a :term:`Context` :math:`x` (i.e. :math:`\nabla_y f(y;x)`).
        This gradient is backpropagated into the GNN model during training,
        in order to improve the model performance w.r.t. the objective function.

    Amortized Optimization
        A framework where a model (like a :abbr:`GNN (Graph Neural Network)`) is trained to predict the solution
        to an optimization problem directly,
        rather than solving each instance from scratch using iterative solvers. For a detailed introduction, see [Amos2022]_.

Deep Learning & Framework
-------------------------

.. glossary::

    Permutation Equivariance
        A property of a function where permuting the input (e.g., reordering buses in a power grid)
        leads to an equivalent permutation of the output. :abbr:`GNNs (Graph Neural Networks)` are by design permutation-equivariant.

    Problem
        An abstraction representing a single instance of an optimization or learning task.
        In EnerGNN, a :class:`~energnn.problem.Problem` provides the :term:`Context`,
        evaluates the :term:`Objective Function`, and computes its :term:`Gradient <Gradient of the Objective Function>`.

    Problem Batch
        A collection of :term:`Problem` instances grouped together for efficient processing.
        Training a :abbr:`GNN (Graph Neural Network)` typically involves computing gradients over a batch
        to stabilize learning and leverage parallel computation (e.g., on GPUs).

    Problem Loader
        An iterator (implementing the :class:`~energnn.problem.ProblemLoader` interface) that yields
        batches of optimization problems for training or evaluation.

    Trainer
        The component (:class:`~energnn.trainer.Trainer`) that orchestrates the :abbr:`GNN (Graph Neural Network)` training loop,
        handling back-propagation, optimization steps (via Optax), and evaluation.

References
----------

.. [Amos2022] Brandon Amos. "Tutorial on Amortized Optimization". *Foundations and Trends in Machine Learning*, 2022.
