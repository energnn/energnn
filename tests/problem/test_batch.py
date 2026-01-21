#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import pytest

from energnn.problem.batch import ProblemBatch
from energnn.graph import separate_graphs
from tests.utils import TestProblemLoader


@pytest.fixture(scope="module")
def pb_loader():
    # Small deterministic loader used in other tests of the repo
    return TestProblemLoader(
        dataset_size=8,
        n_batch=4,
        context_edge_params={
            "node": {"n_obj": 10, "feature_list": ["a", "b"], "address_list": ["0"]},
            "edge": {"n_obj": 10, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
        },
        oracle_edge_params={
            "node": {"n_obj": 10, "feature_list": ["e"]},
            "edge": {"n_obj": 10, "feature_list": ["f"]},
        },
        n_addr=10,
        shuffle=False,
    )


@pytest.fixture(scope="module")
def pb_batch(pb_loader):
    # grab one batch instance from the loader for delegation in tests
    return next(iter(pb_loader))


class DelegateProblemBatch(ProblemBatch):
    """A tiny concrete ProblemBatch that delegates to a provided TestProblemLoader batch."""
    def __init__(self, pb_batch):
        # store a concrete batch that exposes get_context/get_zero_decision/get_gradient/get_metrics
        self._pb_batch = pb_batch

    def get_context(self, get_info: bool = False):
        # pb_batch.get_context returns (context_batch, info)
        return self._pb_batch.get_context(get_info=get_info)

    def get_zero_decision(self, get_info: bool = False):
        return self._pb_batch.get_zero_decision(get_info=get_info)

    def get_gradient(self, *, decision, get_info: bool = False, cfg=None):
        # delegate to pb_batch's get_gradient
        return self._pb_batch.get_gradient(decision=decision, get_info=get_info, cfg=cfg)

    def get_metrics(self, *, decision, get_info: bool = False, cfg=None):
        return self._pb_batch.get_metrics(decision=decision, get_info=get_info, cfg=cfg)


def test_problembatch_is_abstract():
    """ProblemBatch is an abstract base class and cannot be instantiated."""
    with pytest.raises(TypeError):
        ProblemBatch()  # should fail because abstract methods are not implemented


def test_get_context_returns_graph_and_info_flags(pb_batch):
    """get_context must return (Graph, dict) and obey get_info flag."""
    pb = DelegateProblemBatch(pb_batch)
    ctx0, info0 = pb.get_context(get_info=False)
    assert isinstance(info0, dict)
    assert info0 == {}

    ctx1, info1 = pb.get_context(get_info=True)
    assert isinstance(info1, dict)


def test_get_zero_decision_returns_graph_and_info_flags(pb_batch):
    """get_zero_decision must return a graph-like object and an info dict when requested."""
    pb = DelegateProblemBatch(pb_batch)
    zero0, info0 = pb.get_zero_decision(get_info=False)
    assert isinstance(info0, dict)
    assert info0 == {}

    zero1, info1 = pb.get_zero_decision(get_info=True)
    assert isinstance(info1, dict)


def test_get_decision_structure_returns_int_dimensions(pb_batch):
    """
    get_decision_structure uses get_zero_decision and separate_graphs internally.
    Ensure returned structure maps edge keys to dicts of ints.
    """
    pb = DelegateProblemBatch(pb_batch)
    structure = pb.get_decision_structure()

    assert isinstance(structure, dict)
    # Expect keys corresponding to edges (as in TestProblemLoader default)
    assert "node" in structure or "edge" in structure  # at least one expected edge key
    for edge_key, feat_map in structure.items():
        assert isinstance(feat_map, dict)
        for feat_name, v in feat_map.items():
            assert isinstance(v, int), "Decision structure values should be Python ints"


def test_get_gradient_shapes_match_decision(pb_batch):
    """get_gradient must return a gradient Graph with the same edge keys and shapes as the decision input."""
    pb = DelegateProblemBatch(pb_batch)
    zero_decision, _ = pb.get_zero_decision(get_info=False)

    grad, _ = pb.get_gradient(decision=zero_decision, get_info=False)

    # The gradient should have same edges and compatible shapes as the decision
    # Use separate_graphs to get an example separated (first element) to inspect shapes
    separated_decisions = separate_graphs(zero_decision)
    separated_gradients = separate_graphs(grad)
    assert len(separated_decisions) == len(separated_gradients)

    dec0 = separated_decisions[0]
    grad0 = separated_gradients[0]

    # Check edge keys equality
    assert set(dec0.edges.keys()) == set(grad0.edges.keys())

    # For each edge ensure feature_array shapes match
    for key in dec0.edges:
        dec_arr = dec0.edges[key].feature_array
        grad_arr = grad0.edges[key].feature_array
        assert dec_arr.shape == grad_arr.shape

def test_get_info_flag_propagates_to_gradient_and_metrics(pb_batch):
    """Ensure get_info flag works for gradient and metrics methods."""
    pb = DelegateProblemBatch(pb_batch)
    zero_decision, _ = pb.get_zero_decision(get_info=False)

    _, grad_info_false = pb.get_gradient(decision=zero_decision, get_info=False)
    _, grad_info_true = pb.get_gradient(decision=zero_decision, get_info=True)
    assert isinstance(grad_info_false, dict)
    assert isinstance(grad_info_true, dict)

    _, metrics_info_false = pb.get_metrics(decision=zero_decision, get_info=False)
    _, metrics_info_true = pb.get_metrics(decision=zero_decision, get_info=True)
    assert isinstance(metrics_info_false, dict)
    assert isinstance(metrics_info_true, dict)
