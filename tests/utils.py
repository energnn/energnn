#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from energnn.graph import Edge, Graph, GraphShape, collate_graphs, separate_graphs
from energnn.graph.jax import JaxGraph
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


n_addr = 10
n_batch = 4
d = 2
dataset_size = 8


test_loader = TestProblemLoader(
    dataset_size=dataset_size,
    n_batch=n_batch,
    context_edge_params={
        "node": {"n_obj": n_addr, "feature_list": ["a", "b"], "address_list": ["0"]},
        "edge": {"n_obj": n_addr, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
    },
    oracle_edge_params={
        "node": {"n_obj": n_addr, "feature_list": ["e"]},
        "edge": {"n_obj": n_addr, "feature_list": ["f"]},
    },
    n_addr=n_addr,
    shuffle=True,
)

pb_batch = next(iter(test_loader))
np_context_batch, _ = pb_batch.get_context()
test_context_batch = JaxGraph.from_numpy_graph(np_context_batch)
np_context = separate_graphs(np_context_batch)[0]
test_context = JaxGraph.from_numpy_graph(np_context)

np_oracle_batch, _ = pb_batch.get_oracle()
test_oracle_batch = JaxGraph.from_numpy_graph(np_oracle_batch)
np_oracle = separate_graphs(np_oracle_batch)[0]
test_oracle = JaxGraph.from_numpy_graph(np_oracle)

np_test_coordinates_batch = build_coordinates_batch(n_batch=n_batch, n_addr=n_addr, d=d)
test_coordinates_batch = jnp.array(np_test_coordinates_batch)
test_coordinates = test_coordinates_batch[0]


test_out_structure = {"node": {"e": jnp.array([0])}, "edge": {"f": jnp.array([0])}}
