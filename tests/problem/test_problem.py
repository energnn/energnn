#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from energnn.graph import GraphStructure
from energnn.graph import Graph, HyperEdgeSet
from energnn.problem.problem import Problem, SelfSupervisedProblem, SupervisedProblem


def make_dummy_edge_mock(feature_names, feature_array=None):
    m = MagicMock(spec=HyperEdgeSet)
    m.feature_names = feature_names
    m.feature_array = feature_array
    return m


def make_dummy_graph_mock(edges: dict):
    m = MagicMock(spec=Graph)
    m.hyper_edge_sets = edges
    return m


class StubProblem(SelfSupervisedProblem):
    """Base stub implementation for testing SelfSupervisedProblem interface."""

    def __init__(self):
        pass

    @property
    def context_structure(self) -> GraphStructure:
        return GraphStructure(hyper_edge_sets={})

    @property
    def decision_structure(self) -> GraphStructure:
        return GraphStructure(hyper_edge_sets={})

    def get_context(self, get_info=False):
        return make_dummy_graph_mock(edges={}), {}

    def get_zero_decision(self, get_info=False):
        return make_dummy_graph_mock(edges={}), {}

    def get_gradient(self, *, decision, get_info=False, step: int | None = None):
        return decision, {}

    def get_score(self, *, decision, get_info=False, step: int | None = None):
        return 0.0, {}

    def get_metadata(self):
        raise NotImplementedError

    def save(self, *, path: str) -> None:
        raise NotImplementedError

    def get_decision_structure(self) -> dict:
        """Standard implementation pattern for get_decision_structure."""
        zero_decision = make_dummy_graph_mock(edges={})
        structure = {}
        for edge_key, edge in zero_decision.hyper_edge_sets.items():
            if edge.feature_names is not None:
                structure[edge_key] = {name: int(idx) for name, idx in edge.feature_names.items()}
        return structure


def test_problem_is_abstract():
    """Problem is abstract: instantiating it directly should raise TypeError."""
    with pytest.raises(TypeError):
        Problem()
    with pytest.raises(TypeError):
        SelfSupervisedProblem()
    with pytest.raises(TypeError):
        SupervisedProblem()


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
        def get_decision_structure(self) -> dict:
            edge = make_dummy_edge_mock(feature_names=feature_names)
            zero_decision = make_dummy_graph_mock(edges={"node": edge})
            structure = {}
            for edge_key, edge in zero_decision.hyper_edge_sets.items():
                if edge.feature_names is not None:
                    structure[edge_key] = {name: int(idx) for name, idx in edge.feature_names.items()}
            return structure

    p = P()
    ds = p.get_decision_structure()
    assert isinstance(ds, dict)
    assert ds["node"] == expected_values
    for val in ds["node"].values():
        assert isinstance(val, int)


def test_get_decision_structure_invalid_feature_value_raises():
    """If a feature name value cannot be converted to int, get_decision_structure should raise."""

    class P(StubProblem):
        def get_decision_structure(self) -> dict:
            edge = make_dummy_edge_mock(feature_names={"bad": "not-an-int"})
            zero_decision = make_dummy_graph_mock(edges={"node": edge})
            structure = {}
            for edge_key, edge in zero_decision.hyper_edge_sets.items():
                if edge.feature_names is not None:
                    structure[edge_key] = {name: int(idx) for name, idx in edge.feature_names.items()}
            return structure

    p = P()
    with pytest.raises((TypeError, ValueError)):
        _ = p.get_decision_structure()


def test_get_methods_return_tuple_and_info():
    """Check each abstract method returns (Graph, dict) or (float, dict) and handles get_info flag."""

    class P(StubProblem):
        def get_context(self, get_info=False, step=None):
            g = make_dummy_graph_mock(edges={"c": make_dummy_edge_mock(feature_names={"x": 0})})
            info = {"cinfo": True} if get_info else {}
            return g, info

        def get_gradient(self, *, decision, get_info=False, step=None):
            keys = list(decision.hyper_edge_sets.keys())
            g = make_dummy_graph_mock(
                {k: make_dummy_edge_mock(feature_names=decision.hyper_edge_sets[k].feature_names) for k in keys}
            )
            info = {"ginfo": "ok"} if get_info else {}
            return g, info

        def get_score(self, *, decision, get_info=False, step=None):
            metric = 3.14
            info = {"minfo": "m"} if get_info else {}
            return metric, info

    p = P()
    ctx, info0 = p.get_context(get_info=False)
    assert isinstance(ctx, Graph)
    assert info0 == {}

    _, info1 = p.get_context(get_info=True)
    assert info1 == {"cinfo": True}

    grad, g_info = p.get_gradient(decision=ctx, get_info=True)
    assert isinstance(grad, Graph)
    assert g_info == {"ginfo": "ok"}

    metric, m_info = p.get_score(decision=ctx, get_info=True)
    assert isinstance(metric, float)
    assert m_info == {"minfo": "m"}


def test_get_gradient_structure_matches_decision():
    """Check gradients returned have the same edge keys and shapes as the decision."""

    class P(StubProblem):
        def get_zero_decision(self, get_info=False):
            d_edge = make_dummy_edge_mock(feature_names={"a": 0, "b": 1}, feature_array=jnp.zeros((2, 3)))
            return make_dummy_graph_mock(edges={"node": d_edge}), {}

        def get_gradient(self, *, decision, get_info=False, step=None):
            ke = list(decision.hyper_edge_sets.keys())[0]
            shape = decision.hyper_edge_sets[ke].feature_array.shape
            g_edge = make_dummy_edge_mock(
                feature_names=decision.hyper_edge_sets[ke].feature_names, feature_array=jnp.ones(shape)
            )
            return make_dummy_graph_mock(edges={ke: g_edge}), {}

    p = P()
    decision, _ = p.get_zero_decision()
    gradient, _ = p.get_gradient(decision=decision)
    assert set(decision.hyper_edge_sets.keys()) == set(gradient.hyper_edge_sets.keys())
    for k in decision.hyper_edge_sets:
        assert decision.hyper_edge_sets[k].feature_array.shape == gradient.hyper_edge_sets[k].feature_array.shape


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
    """Integration: context -> gradient -> score with numeric checks."""

    class P(StubProblem):
        def get_context(self, get_info=False, step=None):
            edge = make_dummy_edge_mock(feature_names={"x": 0}, feature_array=jnp.array([[1.0, 2.0]]))
            return make_dummy_graph_mock(edges={"c": edge}), {}

        def get_gradient(self, *, decision, get_info=False, step=None):
            g = {}
            for k, e in decision.hyper_edge_sets.items():
                g[k] = make_dummy_edge_mock(feature_names=e.feature_names, feature_array=2.0 * e.feature_array)
            return make_dummy_graph_mock(edges=g), {}

        def get_score(self, *, decision, get_info=False, step=None):
            total = 0.0
            for e in decision.hyper_edge_sets.values():
                total += float(jnp.sum(e.feature_array**2))
            return total, {}

    p = P()
    # Create a dummy decision
    edge = make_dummy_edge_mock(feature_names={"f0": 1}, feature_array=jnp.array([[1.0], [2.0]]))
    decision = make_dummy_graph_mock(edges={"node": edge})

    grad, _ = p.get_gradient(decision=decision)
    # gradient should be twice decision
    for k in grad.hyper_edge_sets:
        np.testing.assert_allclose(
            np.array(grad.hyper_edge_sets[k].feature_array), 2.0 * np.array(decision.hyper_edge_sets[k].feature_array)
        )
    metric, _ = p.get_score(decision=decision)
    # for decision [[1],[2]] metric = 1^2 + 2^2 = 5.0
    assert pytest.approx(metric, rel=1e-6) == 1.0**2 + 2.0**2


def test_supervised_problem_interface():
    """Test the SupervisedProblem interface."""

    class P(SupervisedProblem):
        def __init__(self, val=1.0):
            self.val = val

        @property
        def context_structure(self) -> GraphStructure:
            return GraphStructure(hyper_edge_sets={})

        @property
        def decision_structure(self) -> GraphStructure:
            return GraphStructure(hyper_edge_sets={})

        def get_context(self, get_info=False, step=None):
            return make_dummy_graph_mock(edges={}), {}

        def get_score(self, *, decision, get_info=False, step=None):
            return 0.0, {}

        def get_loss(self, *, decision, get_info=False, step=None):
            return self.val, {"info": "loss"}

        def save(self, *, path: str) -> None:
            pass

    p = P(val=42.0)
    loss, info = p.get_loss(decision=make_dummy_graph_mock(edges={}), get_info=True)
    assert loss == 42.0
    assert info == {"info": "loss"}


def test_problem_pytree_registration():
    """Test that Problem subclasses are correctly registered as JAX PyTrees."""

    class MyProblem(StubProblem):
        def __init__(self, a, b):
            self.a = a
            self.b = b

    p = MyProblem(a=1.0, b=jnp.array([2.0, 3.0]))
    leaves, treedef = jax.tree_util.tree_flatten(p)

    assert 1.0 in leaves

    p2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(p2, MyProblem)
    assert p2.a == p.a
    np.testing.assert_allclose(p2.b, p.b)
