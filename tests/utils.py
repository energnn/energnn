#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from copy import deepcopy

import numpy as np
from omegaconf import DictConfig

from energnn.graph import Edge, Graph, GraphShape, collate_graphs
from energnn.problem import Problem, ProblemBatch, ProblemLoader, ProblemMetadata


def sample_edge(*, n_obj: int, feature_list: list[str] = None, address_list: list[str] = None) -> Edge:
    """Samples a basic Edge from a uniform distribution."""
    if not feature_list:
        feature_dict = None
    else:
        feature_dict = {n: np.random.uniform(size=n_obj) for n in feature_list}
    if not address_list:
        address_dict = None
    else:
        address_dict = {n: np.random.permutation(np.arange(n_obj)) for n in address_list}
    return Edge.from_dict(address_dict=address_dict, feature_dict=feature_dict)


def sample_graph(*, edge_params: dict[str : tuple[int, list[str], list[str]]], n_addr: int) -> Graph:
    """Samples a basic Graph from a uniform distribution."""
    edge_dict = {k: sample_edge(**v) for k, v in edge_params.items()}
    return Graph.from_dict(edge_dict=edge_dict, registry=np.arange(n_addr))


def build_shape(*, edge_params: dict[str:int], n_addr: int) -> GraphShape:
    """Builds a basic GraphShape."""
    edges = {k: np.array(v) for k, v in edge_params.items()}
    addresses = np.array(n_addr)
    return GraphShape(edges=edges, addresses=addresses)


def build_coordinates_batch(*, n_batch: int, n_addr: int, d: int) -> np.ndarray:
    """Builds a basic Coordinates from a normal distribution."""
    return np.random.normal(size=(n_batch, n_addr, d))


class TestProblem(Problem):

    def __init__(self, *, context: Graph, oracle: Graph, zero_decision: Graph):
        self.context = context
        self.oracle = oracle
        self.zero_decision = zero_decision

    @classmethod
    def sample(
        cls,
        *,
        context_edge_params: dict[str : tuple[int, list[str], list[str]]],
        oracle_edge_params: dict[str : tuple[int, list[str], list[str]]],
        n_addr: int,
    ):
        context = sample_graph(edge_params=context_edge_params, n_addr=n_addr)
        oracle = sample_graph(edge_params=oracle_edge_params, n_addr=0)
        zero_decision = deepcopy(oracle)
        zero_decision.feature_flat_array *= 0.0
        return cls(context=context, oracle=oracle, zero_decision=zero_decision)

    def get_context(self, get_info: bool = False) -> (Graph, dict):
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.context), {}

    def get_zero_decision(self, get_info: bool = False) -> (Graph, dict):
        """Returns a decision :class:`Graph` :math:`y` filled with zeros."""
        return deepcopy(self.zero_decision), {}

    def get_oracle(self, get_info: bool = False) -> (Graph, dict):
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)` computed by the AC Power Flow solver."""
        return deepcopy(self.oracle), {}

    def get_gradient(self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False) -> (Graph, dict):
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        gradient = deepcopy(decision)
        gradient.feature_flat_array = decision.feature_flat_array - self.oracle.feature_flat_array
        return gradient, {}

    def get_metrics(self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False) -> (np.ndarray, dict):
        """Returns the mean squared error of the decision :class:`Graph` w.r.t. the oracle :class:`Graph`."""
        gradient = deepcopy(decision)
        gradient.feature_flat_array = decision.feature_flat_array - self.oracle.feature_flat_array
        objective = np.nanmean(np.square(gradient.feature_flat_array))
        return objective, {}

    def get_metadata(self) -> ProblemMetadata:
        pass

    def save(self, *, path: str) -> None:
        pass


class TestProblemBatch(ProblemBatch):

    def __init__(self, *, context: Graph, oracle: Graph, zero_decision: Graph):
        self.context = context
        self.oracle = oracle
        self.zero_decision = zero_decision

    @classmethod
    def sample(
        cls,
        *,
        context_edge_params: dict[str : tuple[int, list[str], list[str]]],
        oracle_edge_params: dict[str : tuple[int, list[str], list[str]]],
        n_addr: int,
        n_batch: int,
    ):
        context_list, oracle_list = [], []
        # context_shape_list, oracle_shape_list = [], []
        for _ in range(n_batch):
            current_context_edge_params = deepcopy(context_edge_params)
            current_oracle_edge_params = deepcopy(oracle_edge_params)
            current_n_addr = 0
            for k, d in context_edge_params.items():
                n_obj = np.random.randint(0, d["n_obj"])
                current_context_edge_params[k]["n_obj"] = n_obj
                if k in oracle_edge_params:
                    current_oracle_edge_params[k]["n_obj"] = n_obj
                if n_obj > current_n_addr:
                    current_n_addr = n_obj
            pb = TestProblem.sample(
                context_edge_params=current_context_edge_params,
                oracle_edge_params=current_oracle_edge_params,
                n_addr=current_n_addr,
            )
            context, _ = pb.get_context()
            oracle, _ = pb.get_oracle()
            context_list.append(context)
            oracle_list.append(oracle)
            # context_shape_list.append(context.true_shape)
            # oracle_shape_list.append(oracle.true_shape)

        max_context_shape = GraphShape(edges={k: np.array(n_addr) for k in context_edge_params}, addresses=np.array(n_addr))
        max_oracle_shape = GraphShape(edges={k: np.array(n_addr) for k in oracle_edge_params}, addresses=np.array(n_addr))

        # max_context_shape = max_shape(context_shape_list)
        # max_oracle_shape = max_shape(oracle_shape_list)
        [context.pad(target_shape=max_context_shape) for context in context_list]
        [oracle.pad(target_shape=max_oracle_shape) for oracle in oracle_list]
        context_batch = collate_graphs(context_list)
        oracle_batch = collate_graphs(oracle_list)
        zero_decision_batch = deepcopy(oracle_batch)
        zero_decision_batch.feature_flat_array *= 0.0
        return cls(context=context_batch, oracle=oracle_batch, zero_decision=zero_decision_batch)

    def get_context(self, get_info: bool = False) -> (Graph, dict):
        """Returns the context :class:`Graph` :math:`x`."""
        return deepcopy(self.context), {}

    def get_zero_decision(self, get_info: bool = False) -> (Graph, dict):
        """Returns a decision :class:`Graph` :math:`y` filled with zeros."""
        return deepcopy(self.zero_decision), {}

    def get_oracle(self, get_info: bool = False) -> (Graph, dict):
        r"""Returns the ground truth :class:`Graph` :math:`y^{\star}(x)` computed by the AC Power Flow solver."""
        return deepcopy(self.oracle), {}

    def get_gradient(self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False) -> (Graph, dict):
        r"""Returns the gradient :class:`Graph` :math:`\nabla_y f(y;x) = y - y^{\star}(x)`."""
        gradient = deepcopy(decision)
        gradient.feature_flat_array = decision.feature_flat_array - self.oracle.feature_flat_array
        return gradient, {}

    def get_metrics(self, decision: Graph, cfg: DictConfig | None = None, get_info: bool = False) -> (np.ndarray, dict):
        """Returns the mean squared error of the decision :class:`Graph` w.r.t. the oracle :class:`Graph`."""
        gradient = deepcopy(decision)
        gradient.feature_flat_array = decision.feature_flat_array - self.oracle.feature_flat_array
        objective = np.nanmean(np.square(gradient.feature_flat_array), axis=1)
        return objective, {}


class TestProblemLoader(ProblemLoader):

    def __init__(
        self,
        dataset_size: int,
        n_batch: int,
        context_edge_params: dict[str : tuple[int, list[str], list[str]]],
        oracle_edge_params: dict[str : tuple[int, list[str], list[str]]],
        n_addr: int,
        shuffle: bool = False,
    ):
        self.dataset_size = dataset_size
        self.n_batch = n_batch
        self.context_edge_params = context_edge_params
        self.oracle_edge_params = oracle_edge_params
        self.n_addr = n_addr
        self.shuffle = shuffle
        self.len = dataset_size
        self.current_step = 0

    def __iter__(self):
        self.current_step = 0
        return self

    def __next__(self) -> TestProblemBatch:
        if self.current_step >= self.len:
            raise StopIteration
        batch_start = self.current_step
        batch_end = min(self.current_step + self.n_batch, self.len)
        self.current_step = batch_end
        n_batch = batch_end - batch_start
        batch = TestProblemBatch.sample(
            context_edge_params=self.context_edge_params,
            oracle_edge_params=self.oracle_edge_params,
            n_addr=self.n_addr,
            n_batch=n_batch,
        )
        return batch

    def __len__(self):
        return max(self.dataset_size // self.n_batch, 1)
