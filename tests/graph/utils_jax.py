#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import chex
import jax.numpy as jnp

from energnn.graph.jax.graph import JaxGraph
from energnn.graph.jax.hyper_edge_set import JaxHyperEdgeSet
from energnn.graph.jax.shape import JaxGraphShape


def get_fixed_edge_jax():
    address_dict = {"dst": jnp.array([1, 2], dtype=jnp.float32), "src": jnp.array([0, 1], dtype=jnp.float32)}
    feature_dict = {"b": jnp.array([0.1, 0.2], dtype=jnp.float32), "w": jnp.array([0.5, 1.0], dtype=jnp.float32)}
    edge = JaxHyperEdgeSet.from_dict(port_dict=address_dict, feature_dict=feature_dict)
    return edge


def get_fixed_graphshape_jax():
    """Build a simple JaxGraphShape from two edges"""
    edge = get_fixed_edge_jax()
    non_fictitious = jnp.ones((3,), dtype=jnp.float32)
    gs = JaxGraphShape.from_dict(hyper_edge_set_dict={"etype": edge}, non_fictitious=non_fictitious)
    return gs


def make_simple_edge_jax(n_obj: int = 2):
    address_dict = {"dst": jnp.arange(n_obj, dtype=jnp.float32), "src": jnp.arange(n_obj, dtype=jnp.float32)}
    feature_dict = {f"f{i}": jnp.arange(n_obj, dtype=jnp.float32) + i for i in range(2)}
    return JaxHyperEdgeSet.from_dict(port_dict=address_dict, feature_dict=feature_dict)


def make_graph_with_registry_jax(n_addresses: int = 4, n_obj: int = 2):
    edge = make_simple_edge_jax(n_obj=n_obj)
    edges = {"etype": edge}
    graph = JaxGraph.from_dict(hyper_edge_set_dict=edges, n_addresses=n_addresses)
    return graph


def assert_edges_equal_jax(e1: JaxHyperEdgeSet, e2: JaxHyperEdgeSet):
    """Assert two jax Edges are equivalent (arrays allclose and same keys)."""
    # address_dict
    if e1.port_dict is None:
        assert e2.port_dict is None
    else:
        assert set(e1.port_dict.keys()) == set(e2.port_dict.keys())
        for k in e1.port_dict:
            chex.assert_trees_all_close(e1.port_dict[k], e2.port_dict[k])

    # feature_array
    if e1.feature_array is None:
        assert e2.feature_array is None
    else:
        chex.assert_trees_all_close(e1.feature_array, e2.feature_array)

    # feature_names (keys)
    if e1.feature_names is None:
        assert e2.feature_names is None
    else:
        assert set(e1.feature_names.keys()) == set(e2.feature_names.keys())
        # values may be arrays or scalars; compare as arrays
        for k in e1.feature_names:
            chex.assert_trees_all_close(jnp.array(e1.feature_names[k]), jnp.array(e2.feature_names[k]))

    # non_fictitious
    if e1.non_fictitious is None:
        assert e2.non_fictitious is None
    else:
        chex.assert_trees_all_close(e1.non_fictitious, e2.non_fictitious)


def assert_graphshape_equal_jax(a: JaxGraphShape, b: JaxGraphShape):
    """Assert two GraphShape are equivalent (edges keys and values, addresses)."""
    assert set(a.hyper_edge_sets.keys()) == set(b.hyper_edge_sets.keys())
    for k in a.hyper_edge_sets:
        chex.assert_trees_all_close(jnp.array(a.hyper_edge_sets[k]), jnp.array(b.hyper_edge_sets[k]))
    chex.assert_trees_all_close(jnp.array(a.addresses), jnp.array(b.addresses))


def assert_graphs_equal_jax(jax_g: JaxGraph, jax_g2: JaxGraph):
    """Simple comparator for Graph <-> Graph roundtrip checks (addresses lengths, edge arrays)."""
    assert set(jax_g.hyper_edge_sets.keys()) == set(jax_g2.hyper_edge_sets.keys())
    for k in jax_g.hyper_edge_sets:
        e1 = jax_g.hyper_edge_sets[k]
        e2 = jax_g2.hyper_edge_sets[k]
        # compare feature arrays
        if e1.feature_array is None:
            assert e2.feature_array is None
        else:
            chex.assert_trees_all_close(e1.feature_array, e2.feature_array)
        # compare address arrays
        if e1.port_dict is None:
            assert e2.port_dict is None
        else:
            for ak in e1.port_dict:
                chex.assert_trees_all_close(e1.port_dict[ak], e2.port_dict[ak])
    # shapes
    for k in jax_g.true_shape.hyper_edge_sets:
        chex.assert_trees_all_close(
            jnp.array(jax_g.true_shape.hyper_edge_sets[k]), jnp.array(jax_g2.true_shape.hyper_edge_sets[k])
        )
    chex.assert_trees_all_close(jnp.array(jax_g.true_shape.addresses), jnp.array(jax_g2.true_shape.addresses))
