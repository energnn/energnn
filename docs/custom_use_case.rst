Use Case Implementation
=======================

This page explains how **EnerGNN** can be leveraged to address your own custom use cases.
But first, make sure to check the :doc:`basics` and :doc:`tutorial_notebook` pages.

Each use case implementation shall encompass their underlying business logic:

- How input data (i.e., **contexts**) are defined and loaded in memory,
- How output data (i.e., **decisions**) should look like,
- How the gradient shall be estimated (closed-form for supervised learning vs.
  Monte Carlo estimation for more complex cases).

Depending on the use case, very different implementation choices can be made, but all should respect the interface
defined in :mod:`energnn.problem`.

The following guide walks you through the major steps in implementing your own **EnerGNN** use case.

Overview
--------

Implementing your custom use case requires the following class implementations.

1. :class:`~energnn.problem.Problem` -- Implements the logic for a single problem instance (context, gradient, metrics).
2. :class:`~energnn.problem.ProblemBatch` -- Handles batching of multiple problem instances together. The implementation
   can be optimized for parallel computation, and even leverage GPU parallelization.
3. :class:`~energnn.problem.ProblemLoader` -- Iterates over a whole dataset of problems,
   by returning a different :class:`~energnn.problem.ProblemBatch` at every iteration.

All three classes should however share two common properties, :attr:`~energnn.problem.Problem.context_structure`
and :attr:`~energnn.problem.Problem.decision_structure`, which define the name of the object classes and of their
repective ports and features appearing in **contexts** and **decisions**.

----------

Step 0 — Define Graph Structures
--------------------------------

**EnerGNN** uses the class :class:`~energnn.graph.GraphStructure` to understand the format of your data.
You must define a structure
for your **contexts** and your **decisions**.
They are mandatory properties for your :class:`~energnn.problem.Problem`, :class:`~energnn.problem.ProblemBatch`
and :class:`~energnn.problem.ProblemLoader` implementations.

.. code-block:: python

    from energnn.graph import EdgeStructure, GraphStructure

    # Example: a context graph with lines, switches, generators and loads
    CONTEXT_STRUCTURE: GraphStructure = GraphStructure.from_dict(edge_structure_dict={
        "lines": EdgeStructure.from_list(address_list=["bus1", "bus2"], feature_list=["r", "x"]),
        "switches": EdgeStructure.from_list(address_list=["bus1", "bus2"], feature_list=None),
        "generators": EdgeStructure.from_list(address_list=["bus"], feature_list=["p0", "q0"]),
        "loads": EdgeStructure.from_list(address_list=["bus"], feature_list=["p", "q"]),
    })

    # Let us say that we wish to predict a log-probability for each switch to be open,
    # along with a generation variation.
    DECISION_STRUCTURE: GraphStructure = GraphStructure.from_dict(edge_structure_dict={
        "switches": EdgeStructure.from_list(address_list=None, feature_list=["log_prob"]),
        "generators": EdgeStructure.from_list(address_list=None, feature_list=["delta_p"]),
    })

**Important constraints:**

1. All classes in the **context** shall be at least of order 1 (i.e., have 1 or more ports);
2. All classes appearing in the **decision** shall also be appearing in the **context**;
3. No port can be predicted by the GNN, so all attributes :code:`address_list` in the decision structure shall be None;

For now, there is no support for global features (i.e., that would not be borne by a specific object),
but feel free to reach out if that's something you would like to see included.

------------------

Step 1 — Implement the Problem Interface
----------------------------------------

Your implementation of the class :class:`~energnn.problem.Problem` should represent a single problem instance.
You must implement the following properties:

- :attr:`~energnn.problem.Problem.context_structure`,
- :attr:`~energnn.problem.Problem.decision_structure`,

And the following methods:

- :meth:`~energnn.problem.Problem.get_context`: Returns the **context** graph :math:`x`,
  instantiated as a :class:`~energnn.graph.JaxGraph`,
- :meth:`~energnn.problem.Problem.get_gradient`: Computes :math:`\nabla_y f(y;x)` for a given **decision** :math:`y`,
  instantiated as a :class:`~energnn.graph.JaxGraph`,
- :meth:`~energnn.problem.Problem.get_metrics`: Computes :math:`f(y;x)` for a given **decision** :math:`y` as a :code:`float`.

**Tracking relevant quantities**:
All three methods have a key word argument :attr:`get_info` to trigger an optional behavior.
If :code:`True`, these methods return optional dictionaries that are passed to your experiment tracker.
It's useful for debugging and tracking, but not necessary in your first implementation.

**Data representation**:
Contexts, decisions and gradients are all instantiated as :class:`~energnn.graph.JaxGraph`, which is a version of
:class:`~energnn.graph.Graph` designed to work seamlessly with JAX.

**Decoupling gradients and metrics**:
You can use different objective functions in :meth:`~energnn.problem.Problem.get_metrics`
and in :meth:`~energnn.problem.Problem.get_gradient`.
For instance, you can use a non-differentiable function :math:`f` as a metrics, and a differentiable function :math:`f'`
in :meth:`~energnn.problem.Problem.get_gradient`.
Loosely speaking, :meth:`~energnn.problem.Problem.get_metrics` just has to return a scalar value that quantifies how good
a decision is, and :meth:`~energnn.problem.Problem.get_gradient` just has to return the opposite of a direction of
improvement for a decision.

.. code-block:: python

    from typing import Any
    import jax.numpy as jnp
    from energnn.graph import Graph, JaxGraph, GraphStructure
    from energnn.problem import Problem
    from energnn.problem.metadata import ProblemMetadata

    class MyProblem(Problem):
        def __init__(self, path: Any):
            # Implement your own data import, and store relevant state data
            self.context, self.state: tuple[JaxGraph, Any] = self._import_from_pypowsybl(path)

        @property
        def context_structure(self) -> GraphStructure:
            return CONTEXT_STRUCTURE

        @property
        def decision_structure(self) -> GraphStructure:
            return DECISION_STRUCTURE

        def get_context(self, get_info: bool = False) -> tuple[JaxGraph, dict[str, Any]]:
            return self.context, {}

        def get_gradient(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict[str, Any]]:
            # Implement your own gradient estimation method
            grad: JaxGraph = self._estimate_gradient(decision, self.state)
            return grad, {}

        def get_metrics(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[float, dict[str, Any]]:
            # Implement your own metrics estimation method
            grad: float = self._estimate_metrics(decision, self.state)
            return grad, {}

        def save(self, path):
            # Implement your own save method
            pass

        @classmethod
        def load(cls, path):
            # Implement your own load method
            pass

Step 2 — Handle Batching (ProblemBatch)
---------------------------------------

To train efficiently on GPUs, multiple problems are grouped together into a :class:`energnn.problem.ProblemBatch`.
The batch interface mirrors the :class:`~energnn.problem.Problem` interface but operates on concatenated graphs.

It is very common that the different problem instances within a batch have a different amount of objects for a given class.
For instance, consider a batch with 2 instances, where:

- The first **context** has 5 switches,
- The second **context** has 7 switches.

To collate them together, we have to pad the first **context** with 2 fictitious switches, so that the two **contexts**
end up having the same number of switches.
The following code snippet shows how to do so.

.. code-block::

    from energnn.graph import Graph, GraphShape, collate_graphs, max_shape, separate_graphs

    context_1: Graph = ...
    context_2: Graph = ...

    # Step 1 : Get the two graph shapes
    # The true_shape property computes the number of non fictitious objects of each class
    shape_1 = context_1.true_shape
    shape_2 = context_2.true_shape

    # Step 2 : Compute the largest shape
    # It computes for each object class the maximum number of objects in the list
    max_shape = max_shape([shape_1, shape_2])

    # Step 3 : Pad all contexts with fictitious objects if required
    context_1.pad(max_shape)
    context_2.pad(max_shape)

    # The true_shape property is not altered by the padding, but the current_shape is.

    # Step 4 : Collate contexts together
    context_batch = collate_graphs([context_1, context_2])

    # Et voilà, you have a context batch filled with fictitious objects if necessary.
    # We can pass it to a model to compute a batch of decisions.
    # The EnerGNN models are implemented to return 0 values on fictitious objects,
    # And to keep track of which objects actually are fictitious or not.
    decision_batch = my_model.forward_batch(context_batch)

    # Wait... What if we need to split this batch of decisions,
    # and get rid of fictitious objects ?

    # Step 5 : Split decisions apart
    decision_1, decision_2 = separate_graphs(decision_batch)

    # Step 6 : Unpad decisions
    decision_1.unpad()
    decision_2.unpad()

    # And now we have decisions without any fictitious object!

The following :class:`~energnn.problem.ProblemBatch` implementation assumes that :

- Single :class:`~energnn.problem.Problem` instances have been generated and saved beforehand,
- The gradient computation can be performed in batch,
- A :code:`max_shape` has been computed over the whole dataset.

.. code-block:: python

    from typing import Any
    from energnn.problem import ProblemBatch, Problem
    from energnn.graph import JaxGraph, Graph, GraphStructure, collate_graphs

    class MyBatch(ProblemBatch):
        def __init__(self, path_list: list[str], max_shape: GraphShape):

            self.problems: list[MyProblem] = [MyProblem.load(path) for path in path_list]

            # Get all contexts, pad them and collate them together.
            context_list, _ = zip([pb.get_context() for pb in self.problems])
            np_context_list = [context.to_numpy_graph() for context in context_list]
            [np_context.pad(max_shape) for np_context in np_context_list]
            np_context_batch = collate_graphs(np_context_list)
            self.context_batch = JaxGraph.from_numpy_graph(np_context_batch)

        @property
        def context_structure(self) -> GraphStructure:
            return CONTEXT_STRUCTURE
        
        @property
        def decision_structure(self) -> GraphStructure:
            return DECISION_STRUCTURE

        def get_context(self, get_info: bool = False) -> tuple[JaxGraph, dict[str, Any]]:
            return self.context_batch, {}

        def get_gradient(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict[str, Any]]:
            batch_grad = self._compute_batch_grad(decision, self.problem_list)
            return batch_grad, {}

        def get_metrics(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[list[float], dict[str, Any]]:
            metrics_list = self._compute_metrics_list(decision, self.problem_list)
            return batch_grad, {}

Step 3 — Data Loading (ProblemLoader)
-------------------------------------

The :class:`energnn.problem.ProblemLoader` is an iterator that yields batches.

.. code-block:: python

    from typing import Any, Iterator
    from energnn.problem import ProblemLoader, ProblemBatch

    class MyLoader(ProblemLoader):
        def __init__(self, dataset: list[str], batch_size: int, shuffle: bool = False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self._current_idx = 0

        def __iter__(self) -> Iterator[ProblemBatch]:
            # Handle shuffling if needed
            self._current_idx = 0
            return self

        def __next__(self) -> ProblemBatch:
            if self._current_idx >= len(self.dataset):
                raise StopIteration
            
            # Slice dataset and return a MyBatch instance
            path_list = self.dataset[self._current_idx:self._current_idx+self.batch_size]
            self._current_idx += self.batch_size
            return MyBatch(path_list)

        def __len__(self) -> int:
            return len(self.dataset) // self.batch_size

Interface Checklist
-------------------

When implementing your custom use case, ensure these requirements are met:

- ``context_structure`` and ``decision_structure`` properties are defined.
- ``get_context()`` returns a :class:`energnn.graph.JaxGraph`.
- ``get_gradient()`` returns a :class:`energnn.graph.JaxGraph` with the same topology as the decision.
- ``get_metrics()`` returns a scalar (for :class:`~energnn.problem.Problem`)
  or a list of scalars (for :class:`~energnn.problem.ProblemBatch`).
- Graphs are correctly converted between :class:`~energnn.graph.Graph` (NumPy-based, useful for building/collating)
  and :class:`~energnn.graph.JaxGraph` (JAX-based, used by the models).

Summary
-------
By implementing these interfaces, your problem becomes fully compatible with EnerGNN's models and trainers.
You can find more practical examples in the :doc:`tutorial` or by looking at the ``tests/utils.py`` file in the repository.

Next steps
----------
- See :doc:`basics` for more details on H2MG graphs.
- Visit the :doc:`reference/index` for the full API specification of the problem module.