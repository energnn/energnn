#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import copy

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from energnn.graph.graph import Graph
from energnn.graph.hyper_edge_set import HyperEdgeSet
from energnn.graph.jax.graph import (
    JaxGraph,
    collate_graphs_jax,
    separate_graphs_jax,
    concatenate_graphs_jax,
    get_statistics_jax,
    check_hyper_edge_set_dict_type_jax,
    check_valid_addresses_jax,
)
from energnn.graph.jax.hyper_edge_set import JaxHyperEdgeSet
from energnn.graph.jax.shape import JaxGraphShape
from tests.graph.utils import assert_graphs_equal, make_graph_with_registry
from tests.graph.utils_jax import (
    make_graph_with_registry_jax,
    make_simple_edge_jax,
)


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


def test_from_dict_and_basic_props():
    g = make_graph_with_registry_jax(n_addresses=5, n_obj=3)
    assert isinstance(g.true_shape, JaxGraphShape)
    assert isinstance(g.current_shape, JaxGraphShape)
    # As constructed, graph is single
    assert g.is_single is True
    assert g.is_batch is False
    # non_fictitious_addresses length equals registry length
    assert len(g.non_fictitious_addresses) == 5


def test_is_batch_detection_after_collation():
    g1 = make_graph_with_registry_jax(n_addresses=3, n_obj=2)
    g2 = make_graph_with_registry_jax(n_addresses=3, n_obj=2)
    batch = collate_graphs_jax([g1, g2])
    # batched graph should be detected as batch
    assert batch.is_batch is True
    assert batch.is_single is False
    # non_fictitious_addresses must be 2D
    assert batch.non_fictitious_addresses.ndim == 2


def test_feature_flat_array_getter_and_setter_and_shape_mismatch():
    # build graph with two edge types to test concatenation
    e1 = make_simple_edge_jax(n_obj=2)
    e2 = make_simple_edge_jax(n_obj=2)
    # rename features to ensure ordering across edges
    edges = {"a": e1, "b": e2}
    g = JaxGraph.from_dict(hyper_edge_set_dict=edges, n_addresses=jnp.array(4))
    flat = g.feature_flat_array
    # Should be 1D since single graph
    assert flat.ndim == 1
    # Create new flat with same shape and set
    new_flat = flat + 1.0
    g.feature_flat_array = new_flat
    chex.assert_trees_all_close(g.feature_flat_array, new_flat)
    # Wrong shape should raise
    with pytest.raises(ValueError):
        g.feature_flat_array = new_flat[:-1]


def test_pad_and_unpad_graph():
    g = make_graph_with_registry_jax(n_addresses=5, n_obj=2)
    # create a target shape with larger counts per edge
    target_edges = {k: jnp.array(int(v) + 3) for k, v in g.current_shape.hyper_edge_sets.items()}
    target_addresses = jnp.array(int(g.current_shape.addresses) + 4)
    target_shape = JaxGraphShape(hyper_edge_sets=target_edges, addresses=target_addresses)
    # pad
    g.pad(target_shape)
    # after pad, shapes should match target
    for k in target_edges:
        assert g.hyper_edge_sets[k].n_obj == int(target_edges[k])
    assert len(g.non_fictitious_addresses) == int(target_addresses)
    # unpad should restore true_shape
    g.unpad()
    for k, v in g.true_shape.hyper_edge_sets.items():
        assert g.hyper_edge_sets[k].n_obj == int(v)
    assert len(g.non_fictitious_addresses) == int(g.true_shape.addresses)


def test_count_connected_components_simple():
    # Build a graph with 3 addresses: 0 connected to 1 via one edge; 2 isolated
    # Edge with two objects: one connects 0 and 1, the other connects only 2 (self-loop)
    address_dict = {"u": jnp.array([0, 2], dtype=jnp.float32), "v": jnp.array([1, 2], dtype=jnp.float32)}
    feature_dict = {"val": jnp.array([0.1, 0.2], dtype=jnp.float32)}
    e = JaxHyperEdgeSet.from_dict(port_dict=address_dict, feature_dict=feature_dict)
    g = JaxGraph.from_dict(hyper_edge_set_dict={"e": e}, n_addresses=np.array(3))
    n_comp, labels = g.count_connected_components()
    # Expect two components: {0,1} and {2}
    assert n_comp == 2
    # labels length equals number of addresses
    assert labels.shape[0] == 3
    # ensure that labels for 0 and 1 are equal and different from 2
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_offset_addresses_affects_edges_but_not_registry():
    g1 = make_graph_with_registry_jax(n_addresses=4, n_obj=2)
    orig_a = copy.deepcopy(g1.hyper_edge_sets["etype"].port_dict)
    g1.offset_addresses(10)
    for k in orig_a:
        chex.assert_trees_all_close(g1.hyper_edge_sets["etype"].port_dict[k], orig_a[k] + 10)
    # registry mask unchanged by edge offset
    assert len(g1.non_fictitious_addresses) == 4


def test_quantiles_single_and_batch_behavior():
    # Single graph: feature array [0,1,2,3,4] -> known quantiles
    arr_single = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    e_single = JaxHyperEdgeSet.from_dict(port_dict={"a": jnp.arange(arr_single.size)}, feature_dict={"f0": arr_single})
    g_single = JaxGraph.from_dict(hyper_edge_set_dict={"E": e_single}, n_addresses=jnp.array(10))

    q_single = g_single.quantiles(q_list=[0.0, 50.0, 100.0])
    # 0th -> 0, 50th -> 2, 100th -> 4
    assert jnp.isclose(q_single["E/f0/0.0th-percentile"], 0.0)
    assert jnp.isclose(q_single["E/f0/50.0th-percentile"], 2.0)
    assert jnp.isclose(q_single["E/f0/100.0th-percentile"], 4.0)

    # Batch case: two graphs with features [0,1,2] and [3,4,5]
    arr_a = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    arr_b = jnp.array([3.0, 4.0, 5.0], dtype=jnp.float32)
    ea = JaxHyperEdgeSet.from_dict(port_dict={"a": jnp.arange(arr_a.size)}, feature_dict={"f0": arr_a})
    eb = JaxHyperEdgeSet.from_dict(port_dict={"a": jnp.arange(arr_b.size)}, feature_dict={"f0": arr_b})
    ga = JaxGraph.from_dict(hyper_edge_set_dict={"E": ea}, n_addresses=jnp.array(10))
    gb = JaxGraph.from_dict(hyper_edge_set_dict={"E": eb}, n_addresses=jnp.array(10))
    batch = collate_graphs_jax([ga, gb])

    q_batch = batch.quantiles(q_list=[50.0])
    # For first graph 50th percentile = 1.0, second = 4.0 -> result should be array([1.0, 4.0])
    key = "E/f0/50.0th-percentile"
    assert key in q_batch
    chex.assert_trees_all_close(q_batch[key], jnp.array([1.0, 4.0]), rtol=1e-6, atol=1e-9)


def test_collate_and_separate_graphs_roundtrip():
    g1 = make_graph_with_registry_jax(n_addresses=4, n_obj=2)
    g2 = make_graph_with_registry_jax(n_addresses=4, n_obj=2)
    batch = collate_graphs_jax([g1, g2])
    separated = separate_graphs_jax(batch)
    assert isinstance(separated, list)
    assert len(separated) == 2
    # compare some properties
    assert separated[0].true_shape.hyper_edge_sets.keys() == g1.true_shape.hyper_edge_sets.keys()
    # addresses recovered
    assert len(separated[0].non_fictitious_addresses) == len(g1.non_fictitious_addresses)


def test_concatenate_graphs_preserves_counts_and_addresses():
    g1 = make_graph_with_registry_jax(n_addresses=4, n_obj=2)
    g2 = make_graph_with_registry_jax(n_addresses=3, n_obj=3)
    cat = concatenate_graphs_jax([g1, g2])
    # addresses concatenated
    assert len(cat.non_fictitious_addresses) == len(g1.non_fictitious_addresses) + len(g2.non_fictitious_addresses)
    # true_shape addresses should be summed
    assert int(cat.true_shape.addresses) == int(g1.true_shape.addresses) + int(g2.true_shape.addresses)


def test_check_edge_dict_type_and_valid_addresses_errors():
    # not a dict
    with pytest.raises(TypeError):
        check_hyper_edge_set_dict_type_jax("not a dict")
    # value not an Edge
    with pytest.raises(TypeError):
        check_hyper_edge_set_dict_type_jax({"a": 123})
    # invalid addresses: create an edge with an address >= n_addresses
    e = make_simple_edge_jax(n_obj=2)
    e.port_dict["dst"] = np.array([0, 10], dtype=np.float32)  # 10 out of range for n_addresses=5
    with pytest.raises(AssertionError):
        check_valid_addresses_jax({"e": e}, jnp.array(5))


def test_get_statistics_basic_and_with_norm():
    # Build two edges with small known features
    e1 = JaxHyperEdgeSet.from_dict(port_dict={"a": jnp.array([0, 1])}, feature_dict={"x": jnp.array([1.0, 2.0])})
    e2 = JaxHyperEdgeSet.from_dict(port_dict={"a": jnp.array([0, 1])}, feature_dict={"x": jnp.array([2.0, 4.0])})
    g1 = JaxGraph.from_dict(hyper_edge_set_dict={"T": e1}, n_addresses=jnp.array(2))
    g2 = JaxGraph.from_dict(hyper_edge_set_dict={"T": e2}, n_addresses=jnp.array(2))

    stats = get_statistics_jax(g1, axis=None, norm_graph=g2)

    # Expected numerical values (analytically computed)
    arr = jnp.array([1.0, 2.0])
    rmse_expected = jnp.sqrt(np.mean(arr**2))  # sqrt((1^2 + 2^2)/2) = sqrt(2.5)
    mae_expected = jnp.mean(np.abs(arr))  # (1 + 2)/2 = 1.5
    mean_expected = jnp.mean(arr)  # 1.5
    std_expected = jnp.std(arr)  # population std = 0.5
    q90_expected = jnp.nanpercentile(arr, 90)  # 1.9
    q75_expected = jnp.nanpercentile(arr, 75)  # 1.75
    q50_expected = jnp.nanpercentile(arr, 50)  # 1.5
    q25_expected = jnp.nanpercentile(arr, 25)  # 1.25
    q10_expected = jnp.nanpercentile(arr, 10)  # 1.1
    qmin_expected = jnp.nanmin(arr)  # 1.0
    qmax_expected = jnp.nanmax(arr)  # 2.0

    # Normalization: norm array is [2,4] => demeaned variance = 1.0, mean absolute dev = 1.0
    # So nrmse = rmse / 1.0 == rmse, nmae = mae / 1.0 == mae
    rmse_key = "T/x/rmse"
    nrmse_key = "T/x/nrmse"
    mae_key = "T/x/mae"
    nmae_key = "T/x/nmae"
    mean_key = "T/x/mean"
    std_key = "T/x/std"
    max_key = "T/x/max"
    q90_key = "T/x/90th"
    q75_key = "T/x/75th"
    q50_key = "T/x/50th"
    q25_key = "T/x/25th"
    q10_key = "T/x/10th"
    min_key = "T/x/min"

    # Assert presence
    for k in [
        rmse_key,
        nrmse_key,
        mae_key,
        nmae_key,
        mean_key,
        std_key,
        max_key,
        q90_key,
        q75_key,
        q50_key,
        q25_key,
        q10_key,
        min_key,
    ]:
        assert k in stats

    # Numeric assertions
    chex.assert_trees_all_close(stats[rmse_key], rmse_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[nrmse_key], rmse_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[mae_key], mae_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[nmae_key], mae_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[mean_key], mean_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[std_key], std_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[max_key], qmax_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[q90_key], q90_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[q75_key], q75_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[q50_key], q50_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[q25_key], q25_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[q10_key], q10_expected, rtol=1e-6, atol=1e-9)
    chex.assert_trees_all_close(stats[min_key], qmin_expected, rtol=1e-6, atol=1e-9)
