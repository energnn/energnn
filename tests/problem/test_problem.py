#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import os
import tempfile

import numpy as np
import jax.numpy as jnp
import pytest

from energnn.problem.problem import Problem


# Minimal Graph/Edge
class DummyEdge:
    def __init__(self, feature_names, feature_array=None):
        """
        feature_names: a mapping-like object (dict) or None
        feature_array: arbitrary array (for gradient/shape tests)
        """
        self.feature_names = feature_names
        self.feature_array = feature_array


class DummyGraph:
    def __init__(self, edges: dict):
        # edges: mapping from edge_key -> DummyEdge
        self.edges = edges


def test_problem_is_abstract():
    # Problem is abstract: instantiating it should raise TypeError
    with pytest.raises(TypeError):
        Problem()  # cannot instantiate abstract base class


def test_get_decision_structure_returns_ints():
    """get_decision_structure should convert feature name values to native ints."""
    class P(Problem):
        def __init__(self):
            pass

        def get_context(self, get_info=False):
            return DummyGraph(edges={}), {}

        def get_zero_decision(self, get_info=False):
            # feature_names values are integers
            edge = DummyEdge(feature_names={"a": 0, "b": 1})
            return DummyGraph(edges={"node": edge}), {}

        def get_gradient(self, *, decision, get_info=False, cfg=None):
            # return a gradient graph (not used here)
            grad_edge = DummyEdge(feature_names={"a": 0, "b": 1}, feature_array=None)
            return DummyGraph(edges={"node": grad_edge}), {}

        def get_metrics(self, *, decision, get_info=False, cfg=None):
            return 0.0, {}

        def get_metadata(self):
            raise NotImplementedError

        def save(self, *, path: str) -> None:
            raise NotImplementedError

    p = P()
    ds = p.get_decision_structure()
    assert isinstance(ds, dict)
    assert "node" in ds
    assert ds["node"]["a"] == 0 and ds["node"]["b"] == 1
    assert isinstance(ds["node"]["a"], int) and isinstance(ds["node"]["b"], int)


def test_get_decision_structure_with_jax_numpy_values():
    """get_decision_structure should accept jnp/np integer-like types and return native ints."""
    class P(Problem):
        def __init__(self):
            pass

        def get_context(self, get_info=False):
            return DummyGraph(edges={}), {}

        def get_zero_decision(self, get_info=False):
            edge = DummyEdge(feature_names={"a": jnp.array(0), "b": np.int64(2)})
            return DummyGraph(edges={"node": edge}), {}

        def get_gradient(self, *, decision, get_info=False, cfg=None):
            return DummyGraph(edges={"node": DummyEdge(feature_names={"a": 0})}), {}

        def get_metrics(self, *, decision, get_info=False, cfg=None):
            return 0.0, {}

        def get_metadata(self):
            raise NotImplementedError

        def save(self, *, path: str) -> None:
            raise NotImplementedError

    p = P()
    ds = p.get_decision_structure()
    assert ds["node"]["a"] == 0 and ds["node"]["b"] == 2
    assert isinstance(ds["node"]["a"], int) and isinstance(ds["node"]["b"], int)


def test_get_decision_structure_invalid_feature_value_raises():
    """If a feature name value cannot be converted to int, get_decision_structure should raise."""
    class P(Problem):
        def __init__(self):
            pass

        def get_context(self, get_info=False):
            return DummyGraph(edges={}), {}

        def get_zero_decision(self, get_info=False):
            edge = DummyEdge(feature_names={"bad": "not-an-int"})
            return DummyGraph(edges={"node": edge}), {}

        def get_gradient(self, *, decision, get_info=False, cfg=None):
            return DummyGraph(edges={"node": DummyEdge(feature_names={"bad": 0})}), {}

        def get_metrics(self, *, decision, get_info=False, cfg=None):
            return 0.0, {}

        def get_metadata(self):
            raise NotImplementedError

        def save(self, *, path: str) -> None:
            raise NotImplementedError

    p = P()
    with pytest.raises((TypeError, ValueError)):
        _ = p.get_decision_structure()


def test_get_decision_structure_missing_feature_names_raises():
    """If an edge has feature_names=None the code should raise when attempting to iterate .items()."""
    class P(Problem):
        def __init__(self):
            pass

        def get_context(self, get_info=False):
            return DummyGraph(edges={}), {}

        def get_zero_decision(self, get_info=False):
            edge = DummyEdge(feature_names=None)
            return DummyGraph(edges={"node": edge}), {}

        def get_gradient(self, *, decision, get_info=False, cfg=None):
            return DummyGraph(edges={"node": DummyEdge(feature_names=None)}), {}

        def get_metrics(self, *, decision, get_info=False, cfg=None):
            return 0.0, {}

        def get_metadata(self):
            raise NotImplementedError

        def save(self, *, path: str) -> None:
            raise NotImplementedError

    p = P()
    with pytest.raises(AttributeError):
        _ = p.get_decision_structure()


def test_get_methods_return_tuple_and_info():
    """Check each abstract method returns (Graph, dict) or (float, dict) and handles get_info flag."""
    class P(Problem):
        def __init__(self):
            pass

        def get_context(self, get_info=False):
            g = DummyGraph(edges={"c": DummyEdge(feature_names={"x": 0})})
            info = {"cinfo": True} if get_info else {}
            return g, info

        def get_zero_decision(self, get_info=False):
            g = DummyGraph(edges={"d": DummyEdge(feature_names={"y": 0})})
            info = {"dinfo": 1} if get_info else {}
            return g, info

        def get_gradient(self, *, decision, get_info=False, cfg=None):
            # return same shape graph and info
            keys = list(decision.edges.keys())
            g = DummyGraph({k: DummyEdge(feature_names=decision.edges[k].feature_names) for k in keys})
            info = {"ginfo": "ok"} if get_info else {}
            return g, info

        def get_metrics(self, *, decision, get_info=False, cfg=None):
            # return float metric and optional info
            metric = 3.14
            info = {"minfo": "m"} if get_info else {}
            return metric, info

        def get_metadata(self):
            raise NotImplementedError

        def save(self, *, path: str) -> None:
            raise NotImplementedError

    p = P()
    ctx, info0 = p.get_context(get_info=False)
    assert isinstance(ctx, DummyGraph)
    assert info0 == {}

    ctx2, info1 = p.get_context(get_info=True)
    assert info1 == {"cinfo": True}

    zd, zd_info = p.get_zero_decision(get_info=False)
    assert isinstance(zd, DummyGraph)
    assert zd_info == {}

    grad, g_info = p.get_gradient(decision=zd, get_info=True)
    assert isinstance(grad, DummyGraph)
    assert g_info == {"ginfo": "ok"}

    metric, m_info = p.get_metrics(decision=zd, get_info=True)
    assert isinstance(metric, float)
    assert m_info == {"minfo": "m"}


def test_get_gradient_structure_matches_decision():
    """Check gradients returned have the same edge keys and shapes as the decision."""
    class P(Problem):
        def __init__(self):
            pass

        def get_context(self, get_info=False):
            return DummyGraph(edges={}), {}

        def get_zero_decision(self, get_info=False):
            # decision features shaped (2,3)
            d_edge = DummyEdge(feature_names={"a": 0, "b": 1}, feature_array=jnp.zeros((2, 3)))
            return DummyGraph(edges={"node": d_edge}), {}

        def get_gradient(self, *, decision, get_info=False, cfg=None):
            # return gradient with same keys and shape as decision.feature_array
            ke = list(decision.edges.keys())[0]
            shape = decision.edges[ke].feature_array.shape
            g_edge = DummyEdge(feature_names=decision.edges[ke].feature_names, feature_array=jnp.ones(shape))
            return DummyGraph(edges={ke: g_edge}), {}

        def get_metrics(self, *, decision, get_info=False, cfg=None):
            return 0.0, {}

        def get_metadata(self):
            raise NotImplementedError

        def save(self, *, path: str) -> None:
            raise NotImplementedError

    p = P()
    decision, _ = p.get_zero_decision()
    gradient, _ = p.get_gradient(decision=decision)
    assert set(decision.edges.keys()) == set(gradient.edges.keys())
    for k in decision.edges:
        assert decision.edges[k].feature_array.shape == gradient.edges[k].feature_array.shape


def test_save_writes_file_and_cleanup():
    """A concrete save implementation should create a file at the given path."""
    class P(Problem):
        def __init__(self):
            pass

        def get_context(self, get_info=False):
            return DummyGraph(edges={}), {}

        def get_zero_decision(self, get_info=False):
            return DummyGraph(edges={}), {}

        def get_gradient(self, *, decision, get_info=False, cfg=None):
            return DummyGraph(edges={}), {}

        def get_metrics(self, *, decision, get_info=False, cfg=None):
            return 0.0, {}

        def get_metadata(self):
            raise NotImplementedError

        def save(self, *, path: str) -> None:
            # simple save: write a small text file
            with open(path, "w") as f:
                f.write("saved")

    p = P()
    fd, path = tempfile.mkstemp(prefix="test_problem_save_", suffix=".txt")
    os.close(fd)
    # ensure file removed first
    if os.path.exists(path):
        os.remove(path)

    p.save(path=path)
    assert os.path.exists(path)
    # cleanup
    os.remove(path)


def test_integration_minimal_pipeline():
    """Integration: context -> zero_decision -> gradient -> metrics with numeric checks."""
    class P(Problem):
        def __init__(self):
            pass

        def get_context(self, get_info=False):
            # trivial context
            edge = DummyEdge(feature_names={"x": 0}, feature_array=jnp.array([[1.0, 2.0]]))
            return DummyGraph(edges={"c": edge}), {}

        def get_zero_decision(self, get_info=False):
            # decision with two objects and one feature column
            d_edge = DummyEdge(feature_names={"f0": 1}, feature_array=jnp.array([[1.0], [2.0]]))
            return DummyGraph(edges={"node": d_edge}), {}

        def get_gradient(self, *, decision, get_info=False, cfg=None):
            # gradient = 2 * decision.feature_array
            g = {}
            for k, e in decision.edges.items():
                g[k] = DummyEdge(feature_names=e.feature_names, feature_array=2.0 * e.feature_array)
            return DummyGraph(edges=g), {}

        def get_metrics(self, *, decision, get_info=False, cfg=None):
            # metric: sum of squares of all decision features -> numeric check
            total = 0.0
            for e in decision.edges.values():
                total += float(jnp.sum(e.feature_array ** 2))
            return total, {}

        def get_metadata(self):
            raise NotImplementedError

        def save(self, *, path: str) -> None:
            raise NotImplementedError

    p = P()
    ctx, _ = p.get_context()
    decision, _ = p.get_zero_decision()
    grad, _ = p.get_gradient(decision=decision)
    # gradient should be twice decision
    for k in grad.edges:
        np.testing.assert_allclose(np.array(grad.edges[k].feature_array), 2.0 * np.array(decision.edges[k].feature_array))
    metric, _ = p.get_metrics(decision=decision)
    # for decision [[1],[2]] metric = 1^2 + 2^2 = 5.0
    assert pytest.approx(metric, rel=1e-6) == 1.0 ** 2 + 2.0 ** 2
