#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from energnn.graph import EdgeStructure, GraphStructure
from energnn.graph.edge import Edge
from energnn.graph.graph import Graph
from energnn.problem.problem import Problem


# -------------------------
# Helpers & Stub Classes
# -------------------------
def make_dummy_edge_mock(feature_names, feature_array=None):
    m = MagicMock(spec=Edge)
    m.feature_names = feature_names
    m.feature_array = feature_array
    return m


def make_dummy_graph_mock(edges: dict):
    m = MagicMock(spec=Graph)
    m.edges = edges
    return m


class StubProblem(Problem):
    """Base stub implementation for testing Problem interface."""

    def __init__(self):
        pass

    @property
    def context_structure(self) -> GraphStructure:
        return GraphStructure(edges={})

    @property
    def decision_structure(self) -> GraphStructure:
        return GraphStructure(edges={})

    def get_context(self, get_info=False):
        return make_dummy_graph_mock(edges={}), {}

    def get_zero_decision(self, get_info=False):
        return make_dummy_graph_mock(edges={}), {}

    def get_gradient(self, *, decision, get_info=False, cfg=None):
        return decision, {}

    def get_metrics(self, *, decision, get_info=False, cfg=None):
        return 0.0, {}

    def get_metadata(self):
        raise NotImplementedError

    def save(self, *, path: str) -> None:
        raise NotImplementedError

    def get_decision_structure(self) -> dict:
        """Standard implementation pattern for get_decision_structure."""
        zero_decision, _ = self.get_zero_decision(get_info=False)
        structure = {}
        for edge_key, edge in zero_decision.edges.items():
            if edge.feature_names is not None:
                structure[edge_key] = {name: int(idx) for name, idx in edge.feature_names.items()}
        return structure


# -------------------------
# Tests
# -------------------------
def test_problem_is_abstract():
    """Problem is abstract: instantiating it directly should raise TypeError."""
    with pytest.raises(TypeError):
        Problem()


@pytest.mark.parametrize(
    "feature_names, expected_values",
    [
        ({"a": 0, "b": 1}, {"a": 0, "b": 1}),
        ({"a": jnp.array(0), "b": np.int64(2)}, {"a": 0, "b": 2}),
    ],
)
def test_get_decision_structure_conversions(feature_names, expected_values):
    """get_decision_structure should correctly convert various int-like types to native ints."""

    class P(StubProblem):
        def get_zero_decision(self, get_info=False):
            edge = make_dummy_edge_mock(feature_names=feature_names)
            return make_dummy_graph_mock(edges={"node": edge}), {}

    p = P()
    ds = p.get_decision_structure()
    assert isinstance(ds, dict)
    assert ds["node"] == expected_values
    for val in ds["node"].values():
        assert isinstance(val, int)


def test_get_decision_structure_invalid_feature_value_raises():
    """If a feature name value cannot be converted to int, get_decision_structure should raise."""

    class P(StubProblem):
        def get_zero_decision(self, get_info=False):
            edge = make_dummy_edge_mock(feature_names={"bad": "not-an-int"})
            return make_dummy_graph_mock(edges={"node": edge}), {}

    p = P()
    with pytest.raises((TypeError, ValueError)):
        _ = p.get_decision_structure()


def test_get_methods_return_tuple_and_info():
    """Check each abstract method returns (Graph, dict) or (float, dict) and handles get_info flag."""

    class P(StubProblem):
        def get_context(self, get_info=False):
            g = make_dummy_graph_mock(edges={"c": make_dummy_edge_mock(feature_names={"x": 0})})
            info = {"cinfo": True} if get_info else {}
            return g, info

        def get_zero_decision(self, get_info=False):
            g = make_dummy_graph_mock(edges={"d": make_dummy_edge_mock(feature_names={"y": 0})})
            info = {"dinfo": 1} if get_info else {}
            return g, info

        def get_gradient(self, *, decision, get_info=False, cfg=None):
            keys = list(decision.edges.keys())
            g = make_dummy_graph_mock({k: make_dummy_edge_mock(feature_names=decision.edges[k].feature_names) for k in keys})
            info = {"ginfo": "ok"} if get_info else {}
            return g, info

        def get_metrics(self, *, decision, get_info=False, cfg=None):
            metric = 3.14
            info = {"minfo": "m"} if get_info else {}
            return metric, info

    p = P()
    ctx, info0 = p.get_context(get_info=False)
    assert isinstance(ctx, Graph)
    assert info0 == {}

    _, info1 = p.get_context(get_info=True)
    assert info1 == {"cinfo": True}

    zd, zd_info = p.get_zero_decision(get_info=False)
    assert isinstance(zd, Graph)
    assert zd_info == {}

    grad, g_info = p.get_gradient(decision=zd, get_info=True)
    assert isinstance(grad, Graph)
    assert g_info == {"ginfo": "ok"}

    metric, m_info = p.get_metrics(decision=zd, get_info=True)
    assert isinstance(metric, float)
    assert m_info == {"minfo": "m"}


def test_get_gradient_structure_matches_decision():
    """Check gradients returned have the same edge keys and shapes as the decision."""

    class P(StubProblem):
        def get_zero_decision(self, get_info=False):
            d_edge = make_dummy_edge_mock(feature_names={"a": 0, "b": 1}, feature_array=jnp.zeros((2, 3)))
            return make_dummy_graph_mock(edges={"node": d_edge}), {}

        def get_gradient(self, *, decision, get_info=False, cfg=None):
            ke = list(decision.edges.keys())[0]
            shape = decision.edges[ke].feature_array.shape
            g_edge = make_dummy_edge_mock(feature_names=decision.edges[ke].feature_names, feature_array=jnp.ones(shape))
            return make_dummy_graph_mock(edges={ke: g_edge}), {}

    p = P()
    decision, _ = p.get_zero_decision()
    gradient, _ = p.get_gradient(decision=decision)
    assert set(decision.edges.keys()) == set(gradient.edges.keys())
    for k in decision.edges:
        assert decision.edges[k].feature_array.shape == gradient.edges[k].feature_array.shape


def test_save_writes_file(tmp_path):
    """A concrete save implementation should create a file at the given path."""

    class P(StubProblem):
        def save(self, *, path: str) -> None:
            with open(path, "w") as f:
                f.write("saved")

    p = P()
    save_path = tmp_path / "save.txt"
    p.save(path=str(save_path))
    assert save_path.exists()
    assert save_path.read_text() == "saved"


def test_integration_minimal_pipeline():
    """Integration: context -> zero_decision -> gradient -> metrics with numeric checks."""

    class P(StubProblem):
        def get_context(self, get_info=False):
            edge = make_dummy_edge_mock(feature_names={"x": 0}, feature_array=jnp.array([[1.0, 2.0]]))
            return make_dummy_graph_mock(edges={"c": edge}), {}

        def get_zero_decision(self, get_info=False):
            d_edge = make_dummy_edge_mock(feature_names={"f0": 1}, feature_array=jnp.array([[1.0], [2.0]]))
            return make_dummy_graph_mock(edges={"node": d_edge}), {}

        def get_gradient(self, *, decision, get_info=False, cfg=None):
            g = {}
            for k, e in decision.edges.items():
                g[k] = make_dummy_edge_mock(feature_names=e.feature_names, feature_array=2.0 * e.feature_array)
            return make_dummy_graph_mock(edges=g), {}

        def get_metrics(self, *, decision, get_info=False, cfg=None):
            total = 0.0
            for e in decision.edges.values():
                total += float(jnp.sum(e.feature_array**2))
            return total, {}

    p = P()
    decision, _ = p.get_zero_decision()
    grad, _ = p.get_gradient(decision=decision)
    # gradient should be twice decision
    for k in grad.edges:
        np.testing.assert_allclose(np.array(grad.edges[k].feature_array), 2.0 * np.array(decision.edges[k].feature_array))
    metric, _ = p.get_metrics(decision=decision)
    # for decision [[1],[2]] metric = 1^2 + 2^2 = 5.0
    assert pytest.approx(metric, rel=1e-6) == 1.0**2 + 2.0**2
