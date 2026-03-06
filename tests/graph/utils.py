#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np

from energnn.graph.graph import Graph
from energnn.graph.hyper_edge_set import HyperEdgeSet
from energnn.graph.shape import GraphShape


def get_fixed_edge():
    address_dict = {"dst": np.array([1, 2], dtype=np.float32), "src": np.array([0, 1], dtype=np.float32)}
    feature_dict = {"b": np.array([0.1, 0.2], dtype=np.float32), "w": np.array([0.5, 1.0], dtype=np.float32)}
    edge = HyperEdgeSet.from_dict(address_dict=address_dict, feature_dict=feature_dict)
    return edge


def get_fixed_graphshape():
    """Build a simple GraphShape from two edges"""
    edge = get_fixed_edge()
    non_fictitious = np.ones((3,), dtype=np.float32)
    gs = GraphShape.from_dict(hyper_edge_set_dict={"etype": edge}, non_fictitious=non_fictitious)
    return gs


def make_simple_edge(n_obj: int = 2):
    address_dict = {"dst": np.arange(n_obj, dtype=np.float32), "src": np.arange(n_obj, dtype=np.float32)}
    feature_dict = {f"f{i}": np.arange(n_obj, dtype=np.float32) + i for i in range(2)}
    return HyperEdgeSet.from_dict(address_dict=address_dict, feature_dict=feature_dict)


def make_graph_with_registry(n_addresses: int = 4, n_obj: int = 2):
    edge = make_simple_edge(n_obj=n_obj)
    edges = {"etype": edge}
    registry = np.arange(n_addresses, dtype=np.float32)
    graph = Graph.from_dict(hyper_edge_set_dict=edges, registry=registry)
    return graph


def assert_edges_equal(e1: HyperEdgeSet, e2: HyperEdgeSet):
    """Assert two numpy Edges are equivalent (arrays allclose and same keys)."""
    # address_dict
    if e1.address_dict is None:
        assert e2.address_dict is None
    else:
        assert set(e1.address_dict.keys()) == set(e2.address_dict.keys())
        for k in e1.address_dict:
            np.testing.assert_allclose(e1.address_dict[k], e2.address_dict[k])

    # feature_array
    if e1.feature_array is None:
        assert e2.feature_array is None
    else:
        np.testing.assert_allclose(e1.feature_array, e2.feature_array)

    # feature_names (keys)
    if e1.feature_names is None:
        assert e2.feature_names is None
    else:
        assert set(e1.feature_names.keys()) == set(e2.feature_names.keys())
        # values may be arrays or scalars; compare as arrays
        for k in e1.feature_names:
            np.testing.assert_allclose(np.array(e1.feature_names[k]), np.array(e2.feature_names[k]))

    # non_fictitious
    if e1.non_fictitious is None:
        assert e2.non_fictitious is None
    else:
        np.testing.assert_allclose(e1.non_fictitious, e2.non_fictitious)


def assert_graphshape_equal(a: GraphShape, b: GraphShape):
    """Assert two GraphShape are equivalent (edges keys and values, addresses)."""
    assert set(a.hyper_edge_sets.keys()) == set(b.hyper_edge_sets.keys())
    for k in a.hyper_edge_sets:
        np.testing.assert_allclose(np.array(a.hyper_edge_sets[k]), np.array(b.hyper_edge_sets[k]))
    np.testing.assert_allclose(np.array(a.addresses), np.array(b.addresses))


def assert_graphs_equal(np_g: Graph, np_g2: Graph):
    """Simple comparator for Graph <-> Graph roundtrip checks (addresses lengths, edge arrays)."""
    assert set(np_g.hyper_edge_sets.keys()) == set(np_g2.hyper_edge_sets.keys())
    for k in np_g.hyper_edge_sets:
        e1 = np_g.hyper_edge_sets[k]
        e2 = np_g2.hyper_edge_sets[k]
        # compare feature arrays
        if e1.feature_array is None:
            assert e2.feature_array is None
        else:
            np.testing.assert_allclose(e1.feature_array, e2.feature_array)
        # compare address arrays
        if e1.address_dict is None:
            assert e2.address_dict is None
        else:
            for ak in e1.address_dict:
                np.testing.assert_allclose(e1.address_dict[ak], e2.address_dict[ak])
    # shapes
    for k in np_g.true_shape.hyper_edge_sets:
        np.testing.assert_allclose(np.array(np_g.true_shape.hyper_edge_sets[k]), np.array(np_g2.true_shape.hyper_edge_sets[k]))
    np.testing.assert_allclose(np.array(np_g.true_shape.addresses), np.array(np_g2.true_shape.addresses))
