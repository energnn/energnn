Custom Use Case
===============

This page explains how to plug your own optimization or learning problem into EnerGNN.
To integrate a new use case, you need to define the graph data structures and implement the interfaces for single problem instances, batched computations, and data loading.

Overview
--------

In EnerGNN, a use case is defined by three main components:

1. **Graph Structures**: Define the topology and features of your inputs (**context**) and outputs (**decision**).
2. **Problem**: Implements the logic for a single instance (context, gradient, metrics).
3. **ProblemBatch & Loader**: Handle batching and iteration over multiple instances for training and evaluation.

The following sections guide you through implementing these components.

Step 1 — Define Graph Structures
--------------------------------

EnerGNN uses :class:`energnn.graph.GraphStructure` to understand the format of your data. You must define a structure
for your context and for your decisions.

.. code-block:: python

    from energnn.graph import EdgeStructure, GraphStructure

    # Example: a context graph with nodes and edges
    # "nodes" have features ["x", "b"]
    # "edges" connect "src" to "dst" and have feature ["w"]
    CONTEXT_STRUCTURE: GraphStructure = GraphStructure.from_dict(edge_structure_dict={
        "nodes": EdgeStructure.from_list(address_list=None, feature_list=["x", "b"]),
        "edges": EdgeStructure.from_list(address_list=["src", "dst"], feature_list=["w"]),
    })

    # Decisions often have a simpler structure, e.g., only node outputs
    DECISION_STRUCTURE: GraphStructure = GraphStructure.from_dict(edge_structure_dict={
        "nodes": EdgeStructure.from_list(address_list=None, feature_list=["y"]),
    })

Important constraints:

- **Gradient matching**: The gradient graph (returned by ``get_gradient``) must have the exact same structure
  (edge types and address lists) as the decision graph.
- **Consistency**: All instances in a dataset must adhere to the same structures.

Step 2 — Implement the Problem Interface
----------------------------------------

The :class:`energnn.problem.Problem` class represents a single optimization instance. You must implement the following methods:

- ``get_context()``: Returns the input graph :math:`x`, referred to as the **context**.
- ``get_gradient(decision)``: Computes :math:`\nabla_y f(y;x)` for a given **decision** :math:`y`.
- ``get_metrics(decision)``: Evaluates the **decision** (e.g., returns the objective value).
- ``get_metadata()``: Provides metadata like problem name and shapes.
- ``get_zero_decision()`` (Optional but recommended): Returns an initial decision (usually zeros).

.. code-block:: python

    from typing import Any
    import jax.numpy as jnp
    from energnn.graph import Graph, JaxGraph, GraphStructure
    from energnn.problem import Problem
    from energnn.problem.metadata import ProblemMetadata

    class MyProblem(Problem):
        def __init__(self, data_params: Any):
            self.params = data_params

        @property
        def context_structure(self) -> GraphStructure:
            return CONTEXT_STRUCTURE

        @property
        def decision_structure(self) -> GraphStructure:
            return DECISION_STRUCTURE

        def get_context(self, get_info: bool = False) -> tuple[JaxGraph, dict[str, Any]]:
            # 1. Create a standard Graph (NumPy-based)
            g: Graph = Graph.from_dict(edge_dict={
                "nodes": {"features": {"x": self.params.x, "b": self.params.b}},
                "edges": {
                    "addresses": {"src": self.params.src, "dst": self.params.dst},
                    "features": {"w": self.params.w}
                }
            })
            # 2. Convert to JaxGraph for the engine
            return JaxGraph.from_numpy_graph(g), {}

        def get_gradient(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict[str, Any]]:
            # Example: gradient for MSE 0.5 * (y - target)^2 => y - target
            # Convert decision back to NumPy for easier manipulation if needed,
            # OR manipulate JAX arrays directly on JaxGraph edges.
            
            y: jnp.ndarray = decision["nodes"].feature_array[..., 0] # Example: first feature
            target: jnp.ndarray = ... # Compute target from context
            grad_val: jnp.ndarray = y - target
            
            # Reconstruct or update gradient graph
            # A common pattern is to convert to numpy for feature update
            grad_np: Graph = decision.to_numpy_graph()
            grad_np["nodes"].feature_array[..., 0] = grad_val
            
            return JaxGraph.from_numpy_graph(grad_np), {}

        def get_metrics(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[float, dict[str, Any]]:
            y: jnp.ndarray = decision["nodes"].feature_array
            mse: jnp.ndarray = jnp.mean((y - target)**2)
            return float(mse), {}

        def get_metadata(self) -> ProblemMetadata:
            return ProblemMetadata(
                name="my_problem",
                config_id="v1",
                code_version=1,
                context_shape={}, # Add relevant shapes
                decision_shape={}
            )

        def save(self, *, path: str) -> None:
            # Implement persistence logic
            pass

Step 3 — Handle Batching (ProblemBatch)
---------------------------------------

To train efficiently on GPUs, multiple problems are grouped into a :class:`energnn.problem.ProblemBatch`. The batch interface mirrors the :class:`~energnn.problem.Problem` interface but operates on concatenated graphs.

While you can implement your own, a common pattern is to use :func:`energnn.graph.collate_graphs` to merge multiple :class:`~energnn.graph.Graph` objects into one.

.. code-block:: python

    from typing import Any
    from energnn.problem import ProblemBatch, Problem
    from energnn.graph import JaxGraph, Graph, GraphStructure, collate_graphs

    class MyBatch(ProblemBatch):
        def __init__(self, problems: list[Problem]):
            self.problems: list[Problem] = problems
            # Collate individual context graphs into one large graph
            # 1. Extract context as JaxGraph, then to NumPy Graph for collation
            ctx_list: list[Graph] = [p.get_context()[0].to_numpy_graph() for p in problems]
            # 2. Collate and convert back to JaxGraph
            self.batched_context: JaxGraph = JaxGraph.from_numpy_graph(collate_graphs(ctx_list))

        @property
        def context_structure(self) -> GraphStructure:
            return CONTEXT_STRUCTURE
        
        @property
        def decision_structure(self) -> GraphStructure:
            return DECISION_STRUCTURE

        def get_context(self, get_info: bool = False) -> tuple[JaxGraph, dict[str, Any]]:
            return self.batched_context, {}

        def get_gradient(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict[str, Any]]:
            # Compute gradients for each instance in the batch
            # 'decision' is a batched JaxGraph
            # One can use JAX transformations (like vmap) or manual collation
            ...
            return batched_grad, {}

        def get_metrics(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[list[float], dict[str, Any]]:
            # Return a list of metrics, one per instance
            # JaxGraph.get_item(i) can be used to extract a single instance from a batch
            metrics: list[float] = [
                p.get_metrics(decision=decision.get_item(i))[0] 
                for i, p in enumerate(self.problems)
            ]
            return metrics, {}

Step 4 — Data Loading (ProblemLoader)
-------------------------------------

The :class:`energnn.problem.ProblemLoader` is an iterator that yields batches. It typically takes a :class:`energnn.problem.ProblemDataset` as input.

.. code-block:: python

    from typing import Any, Iterator
    from energnn.problem import ProblemLoader, ProblemDataset, ProblemBatch

    class MyLoader(ProblemLoader):
        def __init__(self, dataset: ProblemDataset, batch_size: int, shuffle: bool = False):
            self.dataset: ProblemDataset = dataset
            self.batch_size: int = batch_size
            self.shuffle: bool = shuffle
            self._current_idx: int = 0

        def __iter__(self) -> Iterator[ProblemBatch]:
            # Handle shuffling if needed
            self._current_idx = 0
            return self

        def __next__(self) -> ProblemBatch:
            if self._current_idx >= len(self.dataset):
                raise StopIteration
            
            # Slice dataset and return a MyBatch instance
            batch_problems: list[ProblemMetadata] = ... 
            self._current_idx += self.batch_size
            return MyBatch(batch_problems)

        def __len__(self) -> int:
            return len(self.dataset) // self.batch_size

Interface Checklist
-------------------

When implementing your custom use case, ensure these requirements are met:

- [ ] ``context_structure`` and ``decision_structure`` properties are defined.
- [ ] ``get_context()`` returns a :class:`energnn.graph.JaxGraph`.
- [ ] ``get_gradient()`` returns a :class:`energnn.graph.JaxGraph` with the same topology as the decision.
- [ ] ``get_metrics()`` returns a scalar (for :class:`~energnn.problem.Problem`) or a list of scalars (for :class:`~energnn.problem.ProblemBatch`).
- [ ] Graphs are correctly converted between :class:`~energnn.graph.Graph` (NumPy-based, useful for building/collating) and :class:`~energnn.graph.JaxGraph` (JAX-based, used by the models).

TODO : dire un truc sur le fait que la logique de chargement / stockage des données est cachée dans votre implem.
+ Il est encouragé d'optimiser l'implem du batch pour limiter le nombre d'opération sur les données.

Summary
-------

By implementing these interfaces, your problem becomes fully compatible with EnerGNN's models and trainers. You can find more practical examples in the :doc:`tutorial` or by looking at the ``tests/utils.py`` file in the repository.

Next steps
----------
- See :doc:`basics` for more details on H2MG graphs.
- Visit the :doc:`reference/index` for the full API specification of the problem module.