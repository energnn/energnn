# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import copy
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from energnn.graph import GraphStructure, HyperEdgeSetStructure
from energnn.graph import Graph, GraphShape, HyperEdgeSet, JaxBackend, NumpyBackend, collate_graphs
from ..batch import ProblemBatch
from ..loader import ProblemLoader
from ..problem import Problem

LINEAR_SYSTEM_CONTEXT_STRUCTURE = GraphStructure(
    hyper_edge_sets={
        "line": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=["susceptance"]),
        "bus": HyperEdgeSetStructure(port_list=["id"], feature_list=["active_power_injection"]),
    }
)
LINEAR_SYSTEM_DECISION_STRUCTURE = GraphStructure(
    hyper_edge_sets={"bus": HyperEdgeSetStructure(port_list=None, feature_list=["phase_angle"])}
)


class LinearSystemProblemBatch(ProblemBatch):
    __test__ = False

    def __init__(self, *, context: Graph, oracle: Graph):
        self.context = context
        self.oracle = oracle

        zero_decision = copy.deepcopy(oracle)
        # Vérifier opération
        zero_decision.feature_flat_array = 0.0 * zero_decision.feature_flat_array
        self.zero_decision = zero_decision

    @property
    def decision_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_CONTEXT_STRUCTURE

    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[Graph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)`."""
        return deepcopy(self.oracle), {}

    def get_zero_decision(self, get_info: bool = False) -> tuple[Graph, dict]:
        """Returns a decision filled with zeros."""
        return deepcopy(self.zero_decision), {}

    def get_gradient(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[Graph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        # jax_gradient = Graph.from_numpy_graph(gradient)
        return gradient, {}

    def get_score(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[list[float], dict]:
        """Returns the mean-squared error of the decision :class:`Graph` with regard to the oracle :class:`Graph`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        objective = jnp.nanmean(jnp.square(gradient.feature_flat_array), axis=1)
        return objective.tolist(), {}

    def save(self, *, path: str) -> None:
        pass


class LinearSystemProblem(Problem):
    __test__ = False

    def __init__(self, *, context: Graph, oracle: Graph):
        self.context = context
        self.oracle = oracle

        zero_decision = copy.deepcopy(oracle)
        zero_decision.feature_flat_array = 0.0 * zero_decision.feature_flat_array
        self.zero_decision = zero_decision

    @property
    def decision_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_CONTEXT_STRUCTURE

    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[Graph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)`."""
        return deepcopy(self.oracle), {}

    def get_zero_decision(self, get_info: bool = False) -> tuple[Graph, dict]:
        """Returns a decision filled with zeros."""
        return deepcopy(self.zero_decision), {}

    def get_gradient(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[Graph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        # jax_gradient = Graph.from_numpy_graph(gradient)
        return gradient, {}

    def get_score(
        self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[float, dict]:
        """Returns the mean-squared error of the decision :class:`Graph` with regard to the oracle :class:`Graph`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        objective = jnp.nanmean(jnp.square(gradient.feature_flat_array))
        return float(objective), {}

    def save(self, *, path: str) -> None:
        pass


def _generate_sparse_linear_system(n, m):
    """Generates sparse matrix B and vectors P and theta such that B theta = P for a DC network."""
    # Ensure connectivity by building a spanning tree first
    B = np.zeros((n, n))
    nodes = np.arange(n)
    np.random.shuffle(nodes)
    u, v = nodes[:-1], nodes[1:]
    weights = np.random.rand(n - 1) + 0.5
    B[u, v] = B[v, u] = -weights

    # Add remaining m - (n-1) edges among the still-free upper-triangular pairs
    iu, ju = np.triu_indices(n, k=1)
    free = np.flatnonzero(B[iu, ju] == 0)
    n_extra = min(m - (n - 1), free.size)
    if n_extra > 0:
        idx = np.random.choice(free, n_extra, replace=False)
        weights = np.random.rand(n_extra) + 0.5
        B[iu[idx], ju[idx]] = B[ju[idx], iu[idx]] = -weights

    # B is Laplacian matrix-like (off-diagonal < 0, diagonal = -sum(off-diagonal))
    # For a DC network: P = B * theta. B is the susceptance matrix.
    # To have a unique solution, we often fix one node's voltage (slack bus) or add some shunt conductance.
    # Here we'll add a small shunt to ensure invertibility if needed,
    # but the usual DC power flow has sum(P) = 0.
    # Let's make it more generic: B theta = P where B is the susceptance matrix.
    np.fill_diagonal(B, -B.sum(axis=1) + 0.1)  # 0.1 for shunt conductance to ground to ensure invertibility

    theta = np.random.randn(n)
    P = B @ theta
    return B, P, theta


class LinearSystemProblemGenerator:
    __test__ = False
    """Generates random sparse linear systems."""

    def __init__(self, *, seed: int = 0, n_max: int = 32):

        self.seed = seed
        self.n_max = n_max

        np.random.seed(seed)

    def generate_problem(self, backend: JaxBackend | NumpyBackend | None = None) -> LinearSystemProblem:
        # Graphs are built with a numpy backend: their shapes vary from one problem to the next,
        # and building them directly in jax would trigger one XLA compilation per new shape.
        if backend is None:
            backend = JaxBackend()
        n = np.random.randint(2, self.n_max + 1)
        m = np.random.randint(n - 1, n * (n - 1) // 2 + 1)
        B, P, theta = _generate_sparse_linear_system(n, m)

        # Context
        # Use line for off-diagonal terms
        rows, cols = np.nonzero(np.triu(B, k=1))
        numpy_backend = NumpyBackend()
        line = HyperEdgeSet.from_dict(
            backend=numpy_backend, port_dict={"from": rows, "to": cols}, feature_dict={"susceptance": -B[rows, cols]}
        )
        bus_context = HyperEdgeSet.from_dict(
            backend=numpy_backend, port_dict={"id": np.arange(n)}, feature_dict={"active_power_injection": P}
        )
        context = Graph.from_dict(
            backend=numpy_backend, hyper_edge_set_dict={"line": line, "bus": bus_context}, n_addresses=np.array(n)
        )

        # Oracle
        # Use bus for the solution (phase angles)
        bus_oracle = HyperEdgeSet.from_dict(backend=numpy_backend, port_dict=None, feature_dict={"phase_angle": theta})
        oracle = Graph.from_dict(backend=numpy_backend, hyper_edge_set_dict={"bus": bus_oracle}, n_addresses=np.array(n))

        if isinstance(backend, NumpyBackend):
            return LinearSystemProblem(context=context, oracle=oracle)
        return LinearSystemProblem(context=context.to_backend(backend), oracle=oracle.to_backend(backend))

    def generate_problem_batch(self, batch_size: int = 8) -> LinearSystemProblemBatch:

        context_list, oracle_list = [], []

        numpy_backend = NumpyBackend()
        for _ in range(batch_size):
            problem = self.generate_problem(backend=numpy_backend)
            context = problem.context
            oracle = problem.oracle
            context_list.append(context)
            oracle_list.append(oracle)

        max_context_shape = GraphShape(
            backend=numpy_backend,
            hyper_edge_sets={
                "line": np.array(self.n_max * (self.n_max - 1) // 2),
                "bus": np.array(self.n_max),
            },
            addresses=np.array(self.n_max),
        )
        max_oracle_shape = GraphShape(
            backend=numpy_backend, hyper_edge_sets={"bus": np.array(self.n_max)}, addresses=np.array(self.n_max)
        )

        # Padding and collating are done in numpy (variable shapes are free there); the padded
        # batch has a fixed shape, so the final conversion to jax compiles only once.
        [context.pad(target_shape=max_context_shape) for context in context_list]
        [oracle.pad(target_shape=max_oracle_shape) for oracle in oracle_list]
        context_batch = collate_graphs(context_list).to_backend(JaxBackend())
        oracle_batch = collate_graphs(oracle_list).to_backend(JaxBackend())

        return LinearSystemProblemBatch(context=context_batch, oracle=oracle_batch)


class LinearSystemProblemLoader(ProblemLoader):
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

        self.generator = LinearSystemProblemGenerator(seed=seed, n_max=n_max)
        # The loader resets its RNG at each epoch, so every epoch regenerates the exact same
        # batches: they are generated once and cached.
        self._batch_cache: list[LinearSystemProblemBatch] = []

    @property
    def decision_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_DECISION_STRUCTURE

    @property
    def context_structure(self) -> GraphStructure:
        return LINEAR_SYSTEM_CONTEXT_STRUCTURE

    def __iter__(self):
        self.current_step = 0
        np.random.seed(self.seed)
        return self

    def __next__(self) -> LinearSystemProblemBatch:
        if self.current_step >= self.len:
            raise StopIteration
        batch_start = self.current_step
        batch_end = min(self.current_step + self.batch_size, self.len)
        self.current_step = batch_end
        n_batch = batch_end - batch_start
        batch_index = batch_start // self.batch_size
        if batch_index < len(self._batch_cache):
            return self._batch_cache[batch_index]
        batch = self.generator.generate_problem_batch(batch_size=n_batch)
        self._batch_cache.append(batch)
        return batch

    def __len__(self):
        return max(self.dataset_size // self.batch_size, 1)
