#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from copy import deepcopy

import numpy as np
from omegaconf import DictConfig

from energnn.graph import Edge, EdgeStructure, Graph, GraphShape, GraphStructure, collate_graphs
from energnn.graph.jax import JaxGraph
from energnn.problem import Problem, ProblemBatch, ProblemLoader, ProblemMetadata

TEST_CONTEXT_STRUCTURE = GraphStructure(
    edges={
        "arrow": EdgeStructure(address_list=["from", "to"], feature_list=["value"]),
        "source": EdgeStructure(address_list=["id"], feature_list=["value"]),
    }
)
TEST_DECISION_STRUCTURE = GraphStructure(edges={"source": EdgeStructure(address_list=None, feature_list=["value"])})


class TestProblemBatch(ProblemBatch):
    __test__ = False

    def __init__(self, *, context: Graph, oracle: Graph):
        self.context = context
        self.oracle = oracle
        self.jax_context = JaxGraph.from_numpy_graph(context)
        self.jax_oracle = JaxGraph.from_numpy_graph(oracle)

    @property
    def decision_structure(self) -> GraphStructure:
        return TEST_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return TEST_CONTEXT_STRUCTURE

    def get_context(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.jax_context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)` computed by the AC Power Flow solver."""
        return deepcopy(self.jax_oracle), {}

    def get_gradient(self, decision: JaxGraph, cfg: DictConfig | None = None, get_info: bool = False) -> tuple[Graph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        gradient = decision.to_numpy_graph()
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        jax_gradient = JaxGraph.from_numpy_graph(gradient)
        return jax_gradient, {}

    def get_metrics(
        self, decision: JaxGraph, cfg: DictConfig | None = None, get_info: bool = False
    ) -> tuple[list[float], dict]:
        """Returns the mean squared error of the decision :class:`Graph` w.r.t. the oracle :class:`Graph`."""
        gradient = decision.to_numpy_graph()
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        objective = np.nanmean(np.square(gradient.feature_flat_array), axis=1)
        return objective.tolist(), {}

    def get_metadata(self) -> ProblemMetadata:
        pass

    def save(self, *, path: str) -> None:
        pass


class TestProblem(Problem):
    __test__ = False

    def __init__(self, *, context: Graph, oracle: Graph):
        self.context = context
        self.oracle = oracle
        self.jax_context = JaxGraph.from_numpy_graph(context)
        self.jax_oracle = JaxGraph.from_numpy_graph(oracle)

    @property
    def decision_structure(self) -> GraphStructure:
        return TEST_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return TEST_CONTEXT_STRUCTURE

    def get_context(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.jax_context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)` computed by the AC Power Flow solver."""
        return deepcopy(self.jax_oracle), {}

    def get_gradient(self, decision: JaxGraph, cfg: DictConfig | None = None, get_info: bool = False) -> tuple[JaxGraph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        gradient = decision.to_numpy_graph()
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        jax_gradient = JaxGraph.from_numpy_graph(gradient)
        return jax_gradient, {}

    def get_metrics(self, decision: JaxGraph, cfg: DictConfig | None = None, get_info: bool = False) -> tuple[float, dict]:
        """Returns the mean squared error of the decision :class:`Graph` w.r.t. the oracle :class:`Graph`."""
        gradient = decision.to_numpy_graph()
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        objective = np.nanmean(np.square(gradient.feature_flat_array))
        return float(objective), {}

    def get_metadata(self) -> ProblemMetadata:
        pass

    def save(self, *, path: str) -> None:
        pass


def _generate_sparse_linear_system(n, m):
    """Generates sparse matrix A and vectors b and x such that Ax = b."""
    A_dense = np.random.randn(n, n)
    threshold = np.sort(np.reshape(np.abs(A_dense), -1))[-m]
    A = np.where(np.abs(A_dense) >= threshold, A_dense, 0)
    x = np.random.randn(n)
    b = A @ x
    return A, b, x


class TestProblemGenerator:
    __test__ = False
    """Generates random sparse linear systems."""

    def __init__(self, *, n_max: int = 32):
        self.n_max = n_max

    def generate_problem(self) -> TestProblem:
        n = np.random.randint(1, self.n_max + 1)
        m = np.random.randint(1, n**2 + 1)
        A, b, x = _generate_sparse_linear_system(n, m)

        # Context
        arrow_edge = Edge.from_dict(
            address_dict={"from": np.nonzero(A)[0], "to": np.nonzero(A)[1]}, feature_dict={"value": A[np.nonzero(A)]}
        )
        source_edge = Edge.from_dict(address_dict={"id": np.arange(n)}, feature_dict={"value": b})
        context = Graph.from_dict(edge_dict={"arrow": arrow_edge, "source": source_edge}, registry=np.arange(n))

        # Oracle
        source_edge = Edge.from_dict(address_dict=None, feature_dict={"value": x})
        oracle = Graph.from_dict(edge_dict={"source": source_edge}, registry=np.arange(n))

        return TestProblem(context=context, oracle=oracle)

    def generate_problem_batch(self, batch_size: int = 8) -> TestProblemBatch:

        context_list, oracle_list = [], []

        for _ in range(batch_size):
            problem = self.generate_problem()
            context = problem.context
            oracle = problem.oracle
            context_list.append(context)
            oracle_list.append(oracle)

        max_context_shape = GraphShape(
            edges={"arrow": np.array(self.n_max**2), "source": np.array(self.n_max)}, addresses=np.array(self.n_max)
        )
        max_oracle_shape = GraphShape(edges={"source": np.array(self.n_max)}, addresses=np.array(self.n_max))

        [context.pad(target_shape=max_context_shape) for context in context_list]
        [oracle.pad(target_shape=max_oracle_shape) for oracle in oracle_list]
        context_batch = collate_graphs(context_list)
        oracle_batch = collate_graphs(oracle_list)

        return TestProblemBatch(context=context_batch, oracle=oracle_batch)


class TestProblemLoader(ProblemLoader):
    __test__ = False

    def __init__(
        self,
        seed: int = 0,
        dataset_size: int = 32,
        batch_size: int = 8,
        n_max: int = 4,
        shuffle: bool = False,
    ):
        self.seed = seed
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.n_max = n_max
        self.shuffle = shuffle
        self.len = dataset_size
        self.current_step = 0

        self.generator = TestProblemGenerator(n_max=n_max)

    @property
    def decision_structure(self) -> GraphStructure:
        return TEST_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return TEST_CONTEXT_STRUCTURE

    def __iter__(self):
        self.current_step = 0
        np.random.seed(self.seed)
        return self

    def __next__(self) -> TestProblemBatch:
        if self.current_step >= self.len:
            raise StopIteration
        batch_start = self.current_step
        batch_end = min(self.current_step + self.batch_size, self.len)
        self.current_step = batch_end
        n_batch = batch_end - batch_start
        batch = self.generator.generate_problem_batch(batch_size=n_batch)
        return batch

    def __len__(self):
        return max(self.dataset_size // self.batch_size, 1)


def compare_single_graphs(a: JaxGraph, b: JaxGraph, rtol=1e-5, atol=1e-6):
    """
    Compare two single (non-batched) JaxGraph objects component-wise.
    """
    assert set(a.edges.keys()) == set(b.edges.keys()), f"Edge keys differ: {set(a.edges.keys())} vs {set(b.edges.keys())}"
    for k in a.edges:
        ae = a.edges[k]
        be = b.edges[k]
        # feature arrays
        if (ae.feature_array is None) != (be.feature_array is None):
            raise AssertionError(f"Feature presence mismatch for edge {k}")
        if ae.feature_array is not None:
            np.testing.assert_allclose(np.array(ae.feature_array), np.array(be.feature_array), rtol=rtol, atol=atol)
        # address_dict keys
        a_keys = set(ae.address_dict.keys()) if ae.address_dict is not None else set()
        b_keys = set(be.address_dict.keys()) if be.address_dict is not None else set()
        assert a_keys == b_keys
        for ak in a_keys:
            np.testing.assert_allclose(np.array(ae.address_dict[ak]), np.array(be.address_dict[ak]), rtol=rtol, atol=atol)
        # non_fictitious mask
        if ae.non_fictitious is None or be.non_fictitious is None:
            assert ae.non_fictitious is be.non_fictitious
        else:
            np.testing.assert_allclose(np.array(ae.non_fictitious), np.array(be.non_fictitious), rtol=rtol, atol=atol)


def compare_batched_graphs(*graphs, rtol=1e-6, atol=1e-6):
    """
    Compare a list of batched JaxGraph outputs.

    - Ensures same edge keys.
    - For each edge, ensures corresponding arrays have same shapes and are numerically close.
    - Also checks address_dict arrays and non_fictitious shapes.
    """

    if len(graphs) < 2:
        return

    # check keys
    keys0 = set(graphs[0].edges.keys())
    for g in graphs[1:]:
        if set(g.edges.keys()) != keys0:
            raise AssertionError(f"Edge keys differ: {keys0} vs {set(g.edges.keys())}")

    # for each edge key, compare feature_array, address_dict, non_fictitious
    for key in keys0:
        base = graphs[0].edges[key]
        # FEATURE ARRAYS (may be None)
        base_feat = base.feature_array
        base_np = None if base_feat is None else np.array(base_feat)
        for g in graphs[1:]:
            feat = g.edges[key].feature_array
            feat_np = None if feat is None else np.array(feat)
            # both None -> ok
            if base_np is None and feat_np is None:
                continue
            # shape must match
            if base_np is None or feat_np is None:
                raise AssertionError(
                    f"Feature-array presence mismatch for edge '{key}': {base_np is None} vs {feat_np is None}"
                )
            if base_np.shape != feat_np.shape:
                raise AssertionError(f"Feature-array shapes differ for edge '{key}': {base_np.shape} vs {feat_np.shape}")
            # numeric compare
            np.testing.assert_allclose(base_np, feat_np, rtol=rtol, atol=atol)

        # ADDRESS DICTS: keys must match, arrays comparable (possibly batched)
        base_addr_keys = set(base.address_dict.keys()) if base.address_dict is not None else set()
        for g in graphs[1:]:
            other_addr_keys = set(g.edges[key].address_dict.keys()) if g.edges[key].address_dict is not None else set()
            if base_addr_keys != other_addr_keys:
                raise AssertionError(f"Address dict keys differ for edge '{key}': {base_addr_keys} vs {other_addr_keys}")

        for ak in base_addr_keys:
            base_addr_np = np.array(base.address_dict[ak])
            for g in graphs[1:]:
                other_addr_np = np.array(g.edges[key].address_dict[ak])
                if base_addr_np.shape != other_addr_np.shape:
                    raise AssertionError(
                        f"Address array shapes differ for edge '{key}' addr '{ak}': {base_addr_np.shape} vs {other_addr_np.shape}"
                    )
                np.testing.assert_allclose(base_addr_np, other_addr_np, rtol=rtol, atol=atol)

        # non_fictitious masks
        base_nf = np.array(base.non_fictitious) if base.non_fictitious is not None else None
        for g in graphs[1:]:
            other_nf = np.array(g.edges[key].non_fictitious) if g.edges[key].non_fictitious is not None else None
            if (base_nf is None) != (other_nf is None):
                raise AssertionError(f"Non-fictitious presence mismatch for edge '{key}'")
            if base_nf is not None:
                if base_nf.shape != other_nf.shape:
                    raise AssertionError(f"Non-fictitious shapes differ for edge '{key}': {base_nf.shape} vs {other_nf.shape}")
                np.testing.assert_allclose(base_nf, other_nf, rtol=rtol, atol=atol)
