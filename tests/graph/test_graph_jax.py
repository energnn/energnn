#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import jax
import jax.numpy as jnp
import numpy as np

from energnn.graph.graph import Graph
from energnn.graph.hyper_edge_set import HyperEdgeSet
from energnn.graph.jax.graph import JaxGraph
from energnn.graph.jax.hyper_edge_set import JaxHyperEdgeSet
from energnn.graph.jax.shape import JaxGraphShape
from tests.graph.utils import assert_graphs_equal, make_graph_with_registry


def test_from_numpy_and_to_numpy_roundtrip():
    np_graph = make_graph_with_registry(n_addresses=5, n_obj=4)
    jg = JaxGraph.from_numpy_graph(np_graph, dtype="float32")
    # internals are Jax objects
    assert isinstance(jg.hyper_edge_sets["etype"], JaxHyperEdgeSet)
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
    e1 = HyperEdgeSet.from_dict(port_dict={"a": np.array([0, 1])}, feature_dict={"f1": np.array([1.0, 2.0])})
    e2 = HyperEdgeSet.from_dict(port_dict={"b": np.array([0, 1])}, feature_dict={"f2": np.array([3.0, 4.0])})
    g = Graph.from_dict(hyper_edge_set_dict={"A": e1, "B": e2}, n_addresses=np.array(5))
    jg = JaxGraph.from_numpy_graph(g, dtype="float32")
    # feature_flat_array should concatenate edge A then B (keys sorted)
    flat = jg.feature_flat_array
    # convert to numpy and compare to manual concatenation of each edge's feature_flat_array
    manual = np.concatenate(
        [np.array(jnp.ravel(jnp.array(e.feature_flat_array))) for _, e in sorted(jg.hyper_edge_sets.items())], axis=-1
    )
    np.testing.assert_allclose(np.array(flat), manual)


def test_quantiles_match_numpy_graph_quantiles():
    # Build numpy graph with deterministic features
    arr = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    edge = HyperEdgeSet.from_dict(port_dict={"a": np.arange(arr.size)}, feature_dict={"f0": arr})
    g = Graph.from_dict(hyper_edge_set_dict={"etype": edge}, n_addresses=np.array(10))

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
