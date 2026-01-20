#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import copy
import os
import pickle
import numpy as np
import pytest

from energnn.graph.edge import Edge
from energnn.graph.shape import GraphShape
from energnn.graph.graph import (
    Graph,
    collate_graphs,
    separate_graphs,
    concatenate_graphs,
    check_edge_dict_type,
    check_valid_addresses,
    get_statistics,
)
from tests.graph.utils import (
    make_simple_edge,
    make_graph_with_registry
)


def test_from_dict_and_basic_props():
    g = make_graph_with_registry(n_addresses=5, n_obj=3)
    assert isinstance(g.true_shape, GraphShape)
    assert isinstance(g.current_shape, GraphShape)
    # As constructed, graph is single
    assert g.is_single is True
    assert g.is_batch is False
    # non_fictitious_addresses length equals registry length
    assert len(g.non_fictitious_addresses) == 5


def test_is_batch_detection_after_collation():
    g1 = make_graph_with_registry(n_addresses=3, n_obj=2)
    g2 = make_graph_with_registry(n_addresses=3, n_obj=2)
    batch = collate_graphs([g1, g2])
    # batched graph should be detected as batch
    assert batch.is_batch is True
    assert batch.is_single is False
    # non_fictitious_addresses must be 2D
    assert batch.non_fictitious_addresses.ndim == 2


def test_feature_flat_array_getter_and_setter_and_shape_mismatch():
    # build graph with two edge types to test concatenation
    e1 = make_simple_edge(n_obj=2)
    e2 = make_simple_edge(n_obj=2)
    # rename features to ensure ordering across edges
    edges = {"a": e1, "b": e2}
    registry = np.arange(4, dtype=np.float32)
    g = Graph.from_dict(edge_dict=edges, registry=registry)
    flat = g.feature_flat_array
    # Should be 1D since single graph
    assert flat.ndim == 1
    # Create new flat with same shape and set
    new_flat = flat + 1.0
    g.feature_flat_array = new_flat
    np.testing.assert_allclose(g.feature_flat_array, new_flat)
    # Wrong shape should raise
    with pytest.raises(ValueError):
        g.feature_flat_array = new_flat[:-1]


def test_pad_and_unpad_graph():
    g = make_graph_with_registry(n_addresses=5, n_obj=2)
    # create a target shape with larger counts per edge
    target_edges = {k: np.array(int(v) + 3) for k, v in g.current_shape.edges.items()}
    target_addresses = np.array(int(g.current_shape.addresses) + 4)
    target_shape = GraphShape(edges=target_edges, addresses=target_addresses)
    # pad
    g.pad(target_shape)
    # after pad, shapes should match target
    for k in target_edges:
        assert g.edges[k].n_obj == int(target_edges[k])
    assert len(g.non_fictitious_addresses) == int(target_addresses)
    # unpad should restore true_shape
    g.unpad()
    for k, v in g.true_shape.edges.items():
        assert g.edges[k].n_obj == int(v)
    assert len(g.non_fictitious_addresses) == int(g.true_shape.addresses)


def test_count_connected_components_simple():
    # Build a graph with 3 addresses: 0 connected to 1 via one edge; 2 isolated
    # Edge with two objects: one connects 0 and 1, the other connects only 2 (self-loop)
    address_dict = {"u": np.array([0, 2], dtype=np.float32), "v": np.array([1, 2], dtype=np.float32)}
    feature_dict = {"val": np.array([0.1, 0.2], dtype=np.float32)}
    e = Edge.from_dict(address_dict=address_dict, feature_dict=feature_dict)
    registry = np.arange(3, dtype=np.float32)
    g = Graph.from_dict(edge_dict={"e": e}, registry=registry)
    n_comp, labels = g.count_connected_components()
    # Expect two components: {0,1} and {2}
    assert n_comp == 2
    # labels length equals number of addresses
    assert labels.shape[0] == 3
    # ensure that labels for 0 and 1 are equal and different from 2
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_offset_addresses_affects_edges_but_not_registry():
    g1 = make_graph_with_registry(n_addresses=4, n_obj=2)
    orig_a = copy.deepcopy(g1.edges["etype"].address_dict)
    g1.offset_addresses(10)
    for k in orig_a:
        np.testing.assert_allclose(g1.edges["etype"].address_dict[k], orig_a[k] + 10)
    # registry mask unchanged by edge offset
    assert len(g1.non_fictitious_addresses) == 4


def test_quantiles_single_and_batch_behavior():
    # Single graph: feature array [0,1,2,3,4] -> known quantiles
    arr_single = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    e_single = Edge.from_dict(address_dict={"a": np.arange(arr_single.size)}, feature_dict={"f0": arr_single})
    g_single = Graph.from_dict(edge_dict={"E": e_single}, registry=np.arange(10, dtype=np.float32))

    q_single = g_single.quantiles(q_list=[0.0, 50.0, 100.0])
    # 0th -> 0, 50th -> 2, 100th -> 4
    assert np.isclose(q_single["E/f0/0.0th-percentile"], 0.0)
    assert np.isclose(q_single["E/f0/50.0th-percentile"], 2.0)
    assert np.isclose(q_single["E/f0/100.0th-percentile"], 4.0)

    # Batch case: two graphs with features [0,1,2] and [3,4,5]
    arr_a = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    arr_b = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    ea = Edge.from_dict(address_dict={"a": np.arange(arr_a.size)}, feature_dict={"f0": arr_a})
    eb = Edge.from_dict(address_dict={"a": np.arange(arr_b.size)}, feature_dict={"f0": arr_b})
    ga = Graph.from_dict(edge_dict={"E": ea}, registry=np.arange(10, dtype=np.float32))
    gb = Graph.from_dict(edge_dict={"E": eb}, registry=np.arange(10, dtype=np.float32))
    batch = collate_graphs([ga, gb])

    q_batch = batch.quantiles(q_list=[50.0])
    # For first graph 50th percentile = 1.0, second = 4.0 -> result should be array([1.0, 4.0])
    key = "E/f0/50.0th-percentile"
    assert key in q_batch
    np.testing.assert_allclose(q_batch[key], np.array([1.0, 4.0]), rtol=1e-6, atol=1e-9)


def test_collate_and_separate_graphs_roundtrip():
    g1 = make_graph_with_registry(n_addresses=4, n_obj=2)
    g2 = make_graph_with_registry(n_addresses=4, n_obj=2)
    batch = collate_graphs([g1, g2])
    separated = separate_graphs(batch)
    assert isinstance(separated, list)
    assert len(separated) == 2
    # compare some properties
    assert separated[0].true_shape.edges.keys() == g1.true_shape.edges.keys()
    # addresses recovered
    assert len(separated[0].non_fictitious_addresses) == len(g1.non_fictitious_addresses)


def test_concatenate_graphs_preserves_counts_and_addresses():
    g1 = make_graph_with_registry(n_addresses=4, n_obj=2)
    g2 = make_graph_with_registry(n_addresses=3, n_obj=3)
    cat = concatenate_graphs([g1, g2])
    # addresses concatenated
    assert len(cat.non_fictitious_addresses) == len(g1.non_fictitious_addresses) + len(g2.non_fictitious_addresses)
    # true_shape addresses should be summed
    assert int(cat.true_shape.addresses) == int(g1.true_shape.addresses) + int(g2.true_shape.addresses)


def test_check_edge_dict_type_and_valid_addresses_errors():
    # not a dict
    with pytest.raises(TypeError):
        check_edge_dict_type("not a dict")
    # value not an Edge
    with pytest.raises(TypeError):
        check_edge_dict_type({"a": 123})
    # invalid addresses: create an edge with an address >= n_addresses
    e = make_simple_edge(n_obj=2)
    e.address_dict["dst"] = np.array([0, 10], dtype=np.float32)  # 10 out of range for n_addresses=5
    with pytest.raises(AssertionError):
        check_valid_addresses({"e": e}, np.array(5))


def test_get_statistics_basic_and_with_norm():
    # Build two edges with small known features
    e1 = Edge.from_dict(address_dict={"a": np.array([0, 1])}, feature_dict={"x": np.array([1.0, 2.0])})
    e2 = Edge.from_dict(address_dict={"a": np.array([0, 1])}, feature_dict={"x": np.array([2.0, 4.0])})
    g1 = Graph.from_dict(edge_dict={"T": e1}, registry=np.arange(2, dtype=np.float32))
    g2 = Graph.from_dict(edge_dict={"T": e2}, registry=np.arange(2, dtype=np.float32))

    stats = get_statistics(g1, axis=None, norm_graph=g2)

    # Expected numerical values (analytically computed)
    arr = np.array([1.0, 2.0])
    rmse_expected = np.sqrt(np.mean(arr**2))  # sqrt((1^2 + 2^2)/2) = sqrt(2.5)
    mae_expected = np.mean(np.abs(arr))      # (1 + 2)/2 = 1.5
    mean_expected = np.mean(arr)             # 1.5
    std_expected = np.std(arr)               # population std = 0.5
    q90_expected = np.nanpercentile(arr, 90) # 1.9
    q75_expected = np.nanpercentile(arr, 75) # 1.75
    q50_expected = np.nanpercentile(arr, 50) # 1.5
    q25_expected = np.nanpercentile(arr, 25) # 1.25
    q10_expected = np.nanpercentile(arr, 10) # 1.1
    qmin_expected = np.nanmin(arr)           # 1.0
    qmax_expected = np.nanmax(arr)           # 2.0

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
    for k in [rmse_key, nrmse_key, mae_key, nmae_key, mean_key, std_key, max_key, q90_key, q75_key, q50_key, q25_key, q10_key, min_key]:
        assert k in stats

    # Numeric assertions
    np.testing.assert_allclose(stats[rmse_key], rmse_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[nrmse_key], rmse_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[mae_key], mae_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[nmae_key], mae_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[mean_key], mean_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[std_key], std_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[max_key], qmax_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[q90_key], q90_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[q75_key], q75_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[q50_key], q50_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[q25_key], q25_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[q10_key], q10_expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats[min_key], qmin_expected, rtol=1e-6, atol=1e-9)
