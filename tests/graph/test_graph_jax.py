#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from energnn.graph.edge import Edge
from energnn.graph.graph import Graph
from energnn.graph.jax.graph import JaxGraph
from energnn.graph.jax.utils import np_to_jnp, jnp_to_np
from energnn.graph.jax.edge import JaxEdge
from energnn.graph.jax.shape import JaxGraphShape
from energnn.graph.shape import GraphShape
from tests.graph.utils import make_graph_with_registry


def assert_graphs_equal(np_g: Graph, np_g2: Graph):
    """Simple comparator for Graph <-> Graph roundtrip checks (addresses lengths, edge arrays)."""
    assert set(np_g.edges.keys()) == set(np_g2.edges.keys())
    for k in np_g.edges:
        e1 = np_g.edges[k]
        e2 = np_g2.edges[k]
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
    for k in np_g.true_shape.edges:
        np.testing.assert_allclose(np.array(np_g.true_shape.edges[k]), np.array(np_g2.true_shape.edges[k]))
    np.testing.assert_allclose(np.array(np_g.true_shape.addresses), np.array(np_g2.true_shape.addresses))


def test_from_numpy_and_to_numpy_roundtrip():
    np_graph = make_graph_with_registry(n_addresses=5, n_obj=4)
    jg = JaxGraph.from_numpy_graph(np_graph, dtype="float32")
    # internals are Jax objects
    assert isinstance(jg.edges["etype"], JaxEdge)
    assert isinstance(jg.true_shape, JaxGraphShape)
    assert isinstance(jg.current_shape, JaxGraphShape)
    assert isinstance(jg.non_fictitious_addresses, jax.Array)

    # convert back to numpy and compare
    np_round = jg.to_numpy_graph()
    assert isinstance(np_round, Graph)
    assert_graphs_equal(np_graph, np_round)


def test_pytree_flatten_and_unflatten_roundtrip():
    np_graph = make_graph_with_registry(n_addresses=4, n_obj=3)
    jg = JaxGraph.from_numpy_graph(np_graph, dtype="float32")
    children, aux = jax.tree_util.tree_flatten(jg)
    recon = jax.tree_util.tree_unflatten(aux, children)
    assert isinstance(recon, JaxGraph)
    # convert back and compare
    np_recon = recon.to_numpy_graph()
    assert_graphs_equal(np_graph, np_recon)


def test_feature_flat_array_concatenation_order_and_shape():
    # Build graph with two edge types to ensure concatenation order sorted by key
    # Create two edges with known feature lengths
    e1 = Edge.from_dict(address_dict={"a": np.array([0, 1])}, feature_dict={"f1": np.array([1.0, 2.0])})
    e2 = Edge.from_dict(address_dict={"b": np.array([0, 1])}, feature_dict={"f2": np.array([3.0, 4.0])})
    g = Graph.from_dict(edge_dict={"A": e1, "B": e2}, registry=np.arange(5, dtype=np.float32))
    jg = JaxGraph.from_numpy_graph(g, dtype="float32")
    # feature_flat_array should concatenate edge A then B (keys sorted)
    flat = jg.feature_flat_array
    # convert to numpy and compare to manual concatenation of each edge's feature_flat_array
    manual = np.concatenate([np.array(jnp.ravel(jnp.array(e.feature_flat_array))) for _, e in sorted(jg.edges.items())], axis=-1)
    np.testing.assert_allclose(np.array(flat), manual)


def test_quantiles_match_numpy_graph_quantiles():
    # Build numpy graph with deterministic features
    arr = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    edge = Edge.from_dict(address_dict={"a": np.arange(arr.size)}, feature_dict={"f0": arr})
    g = Graph.from_dict(edge_dict={"etype": edge}, registry=np.arange(10, dtype=np.float32))

    # NumPy quantiles
    q_np = g.quantiles(q_list=[0.0, 25.0, 50.0, 100.0])

    # JAX quantiles via JaxGraph and compare numerically (convert jax results to numpy)
    jg = JaxGraph.from_numpy_graph(g, dtype="float32")
    q_jax = jg.quantiles(q_list=[0.0, 25.0, 50.0, 100.0])
    # convert to numpy
    q_jax_np = {k: np.array(v) for k, v in q_jax.items()}

    # Compare expected numeric values
    for k in q_np:
        np.testing.assert_allclose(q_np[k], q_jax_np[k], rtol=1e-6, atol=1e-9)
