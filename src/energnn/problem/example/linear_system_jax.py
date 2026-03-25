# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import copy
from copy import deepcopy

import numpy as np
import jax.numpy as jnp
from omegaconf import DictConfig

from energnn.graph import GraphStructure, HyperEdgeSetStructure
from energnn.graph.jax import JaxGraph, JaxGraphShape, JaxHyperEdgeSet, collate_graphs_jax
from ..batch import ProblemBatch
from ..loader import ProblemLoader
from ..problem import Problem

LINEAR_SYSTEM_CONTEXT_STRUCTURE = GraphStructure(
    hyper_edge_sets={
        "arrow": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=["value"]),
        "source": HyperEdgeSetStructure(port_list=["id"], feature_list=["value"]),
    }
)
LINEAR_SYSTEM_DECISION_STRUCTURE = GraphStructure(
    hyper_edge_sets={"source": HyperEdgeSetStructure(port_list=None, feature_list=["value"])}
)


class JaxLinearSystemProblemBatch(ProblemBatch):
    __test__ = False

    def __init__(self, *, context: JaxGraph, oracle: JaxGraph):
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

    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[JaxGraph, dict]:
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)`."""
        return deepcopy(self.oracle), {}

    def get_zero_decision(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """Returns a decision filled with zeros."""
        return deepcopy(self.zero_decision), {}

    def get_gradient(
        self, decision: JaxGraph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[JaxGraph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        # jax_gradient = JaxGraph.from_numpy_graph(gradient)
        return gradient, {}

    def get_score(
        self, decision: JaxGraph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[list[float], dict]:
        """Returns the mean-squared error of the decision :class:`Graph` with regard to the oracle :class:`Graph`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        objective = jnp.nanmean(jnp.square(gradient.feature_flat_array), axis=1)
        return objective.tolist(), {}

    def save(self, *, path: str) -> None:
        pass


class JaxLinearSystemProblem(Problem):
    __test__ = False

    def __init__(self, *, context: JaxGraph, oracle: JaxGraph):
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

    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[JaxGraph, dict]:
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)`."""
        return deepcopy(self.oracle), {}

    def get_zero_decision(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """Returns a decision filled with zeros."""
        return deepcopy(self.zero_decision), {}

    def get_gradient(
        self, decision: JaxGraph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
    ) -> tuple[JaxGraph, dict]:
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        # gradient = decision.to_numpy_graph()
        gradient = deepcopy(decision)
        gradient.feature_flat_array = gradient.feature_flat_array - self.oracle.feature_flat_array
        # jax_gradient = JaxGraph.from_numpy_graph(gradient)
        return gradient, {}

    def get_score(
        self, decision: JaxGraph, cfg: DictConfig | None = None, get_info: bool = False, step: int | None = None
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
    """Generates sparse matrix A and vectors b and x such that Ax = b."""
    A_dense = np.random.randn(n, n)
    threshold = np.sort(np.reshape(np.abs(A_dense), -1))[-m]
    A = np.where(np.abs(A_dense) >= threshold, A_dense, 0)
    x = np.random.randn(n)
    b = A @ x
    return A, b, x


class JaxLinearSystemProblemGenerator:
    __test__ = False
    """Generates random sparse linear systems."""

    def __init__(self, *, seed: int = 0, n_max: int = 32):

        self.seed = seed
        self.n_max = n_max

        np.random.seed(seed)

    def generate_problem(self) -> JaxLinearSystemProblem:
        n = np.random.randint(1, self.n_max + 1)
        m = np.random.randint(1, n**2 + 1)
        A, b, x = _generate_sparse_linear_system(n, m)

        # Context
        arrow_edge = JaxHyperEdgeSet.from_dict(
            port_dict={"from": np.nonzero(A)[0], "to": np.nonzero(A)[1]}, feature_dict={"value": A[np.nonzero(A)]}
        )
        source_edge = JaxHyperEdgeSet.from_dict(port_dict={"id": np.arange(n)}, feature_dict={"value": b})
        context = JaxGraph.from_dict(hyper_edge_set_dict={"arrow": arrow_edge, "source": source_edge}, n_addresses=n)

        # Oracle
        source_edge = JaxHyperEdgeSet.from_dict(port_dict=None, feature_dict={"value": x})
        oracle = JaxGraph.from_dict(hyper_edge_set_dict={"source": source_edge}, n_addresses=n)

        return JaxLinearSystemProblem(context=context, oracle=oracle)

    def generate_problem_batch(self, batch_size: int = 8) -> JaxLinearSystemProblemBatch:

        context_list, oracle_list = [], []

        for _ in range(batch_size):
            problem = self.generate_problem()
            context = problem.context
            oracle = problem.oracle
            context_list.append(context)
            oracle_list.append(oracle)

        max_context_shape = JaxGraphShape(
            hyper_edge_sets={"arrow": jnp.array(self.n_max**2), "source": jnp.array(self.n_max)}, addresses=jnp.array(self.n_max)
        )
        max_oracle_shape = JaxGraphShape(hyper_edge_sets={"source": jnp.array(self.n_max)}, addresses=jnp.array(self.n_max))

        [context.pad(target_shape=max_context_shape) for context in context_list]
        [oracle.pad(target_shape=max_oracle_shape) for oracle in oracle_list]
        context_batch = collate_graphs_jax(context_list)
        oracle_batch = collate_graphs_jax(oracle_list)

        return JaxLinearSystemProblemBatch(context=context_batch, oracle=oracle_batch)


class JaxLinearSystemProblemLoader(ProblemLoader):
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

        self.generator = JaxLinearSystemProblemGenerator(seed=seed, n_max=n_max)

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

    def __next__(self) -> JaxLinearSystemProblemBatch:
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
