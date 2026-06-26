# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from energnn.graph.backend import NumpyBackend
from energnn.graph.graph import Graph
from energnn.graph.hyper_edge_set import HyperEdgeSet
from energnn.graph.shape import GraphShape


def get_fixed_edge(backend=None):
    if backend is None:
        backend = NumpyBackend()
    return HyperEdgeSet.from_dict(
        port_dict={"dst": np.array([1, 2], dtype=np.float32), "src": np.array([0, 1], dtype=np.float32)},
        feature_dict={"b": np.array([0.1, 0.2], dtype=np.float32), "w": np.array([0.5, 1.0], dtype=np.float32)},
        backend=backend,
    )


def get_fixed_graphshape(backend=None):
    edge = get_fixed_edge(backend=backend)
    non_fictitious = np.ones((3,), dtype=np.float32)
    return GraphShape.from_dict(hyper_edge_set_dict={"etype": edge}, non_fictitious=non_fictitious, backend=backend)


def make_simple_edge(n_obj: int = 2, backend=None):
    if backend is None:
        backend = NumpyBackend()
    return HyperEdgeSet.from_dict(
        port_dict={"dst": np.arange(n_obj, dtype=np.float32), "src": np.arange(n_obj, dtype=np.float32)},
        feature_dict={f"f{i}": np.arange(n_obj, dtype=np.float32) + i for i in range(2)},
        backend=backend,
    )


def make_graph_with_n_addresses(n_addresses: int = 4, n_obj: int = 2, backend=None):
    if backend is None:
        backend = NumpyBackend()
    edge = make_simple_edge(n_obj=n_obj, backend=backend)
    return Graph.from_dict(hyper_edge_set_dict={"etype": edge}, n_addresses=n_addresses, backend=backend)


def assert_edges_equal(e1: HyperEdgeSet, e2: HyperEdgeSet):
    """Assert two HyperEdgeSet instances are numerically equal (backend-agnostic)."""
    if e1.port_dict is None:
        assert e2.port_dict is None
    else:
        assert set(e1.port_dict.keys()) == set(e2.port_dict.keys())
        for k in e1.port_dict:
            np.testing.assert_allclose(np.array(e1.port_dict[k]), np.array(e2.port_dict[k]))

    if e1.feature_array is None:
        assert e2.feature_array is None
    else:
        np.testing.assert_allclose(np.array(e1.feature_array), np.array(e2.feature_array))

    if e1.feature_names is None:
        assert e2.feature_names is None
    else:
        assert set(e1.feature_names.keys()) == set(e2.feature_names.keys())
        for k in e1.feature_names:
            np.testing.assert_allclose(np.array(e1.feature_names[k]), np.array(e2.feature_names[k]))

    if e1.non_fictitious is None:
        assert e2.non_fictitious is None
    else:
        np.testing.assert_allclose(np.array(e1.non_fictitious), np.array(e2.non_fictitious))


def assert_graphshape_equal(a: GraphShape, b: GraphShape):
    assert set(a.hyper_edge_sets.keys()) == set(b.hyper_edge_sets.keys())
    for k in a.hyper_edge_sets:
        np.testing.assert_allclose(np.array(a.hyper_edge_sets[k]), np.array(b.hyper_edge_sets[k]))
    np.testing.assert_allclose(np.array(a.addresses), np.array(b.addresses))


def assert_graphs_equal(g1: Graph, g2: Graph):
    assert set(g1.hyper_edge_sets.keys()) == set(g2.hyper_edge_sets.keys())
    for k in g1.hyper_edge_sets:
        assert_edges_equal(g1.hyper_edge_sets[k], g2.hyper_edge_sets[k])
    for k in g1.true_shape.hyper_edge_sets:
        np.testing.assert_allclose(np.array(g1.true_shape.hyper_edge_sets[k]), np.array(g2.true_shape.hyper_edge_sets[k]))
    np.testing.assert_allclose(np.array(g1.true_shape.addresses), np.array(g2.true_shape.addresses))
