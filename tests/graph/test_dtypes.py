# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Dtype guarantees: addresses and shape counts are integers, features are floats."""

import numpy as np
import pytest

from energnn.graph import Graph, GraphShape, HyperEdgeSet, collate_graphs, concatenate_graphs
from energnn.graph.backend import JaxBackend, NumpyBackend


def make_graph(backend=None, n_bus: int = 3) -> Graph:
    line = HyperEdgeSet.from_dict(
        backend=backend,
        port_dict={"from": np.arange(n_bus - 1), "to": np.arange(1, n_bus)},
        feature_dict={"susceptance": np.linspace(1.0, 2.0, n_bus - 1)},
    )
    bus = HyperEdgeSet.from_dict(
        backend=backend,
        port_dict={"id": np.arange(n_bus)},
        feature_dict={"p": np.linspace(-1.0, 1.0, n_bus)},
    )
    return Graph.from_dict(backend=backend, hyper_edge_set_dict={"line": line, "bus": bus}, n_addresses=np.array(n_bus))


def assert_int_dtype(arr):
    assert np.issubdtype(np.asarray(arr).dtype, np.integer), f"expected integer dtype, got {np.asarray(arr).dtype}"


def assert_float_dtype(arr, dtype=np.float32):
    assert np.asarray(arr).dtype == dtype, f"expected {dtype}, got {np.asarray(arr).dtype}"


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def test_from_dict_dtypes(backend):
    hes = HyperEdgeSet.from_dict(
        backend=backend,
        port_dict={"from": np.array([0, 1], dtype=np.int64)},
        feature_dict={"x": np.array([1, 2], dtype=np.int64)},
    )
    assert np.asarray(hes.port_dict["from"]).dtype == np.int32
    # Features are values, not indices: integer inputs are cast to float32.
    assert_float_dtype(hes.feature_array)


def test_from_dict_accepts_integer_valued_float_ports(backend):
    hes = HyperEdgeSet.from_dict(backend=backend, port_dict={"id": np.array([0.0, 1.0])}, feature_dict=None)
    assert np.asarray(hes.port_dict["id"]).dtype == np.int32


def test_from_dict_rejects_fractional_ports(backend):
    with pytest.raises(ValueError):
        HyperEdgeSet.from_dict(backend=backend, port_dict={"id": np.array([0.5, 1.0])}, feature_dict=None)


def test_from_dict_rejects_nan_float_ports(backend):
    with pytest.raises(ValueError):
        HyperEdgeSet.from_dict(backend=backend, port_dict={"id": np.array([np.nan, 1.0])}, feature_dict=None)


def test_graph_shape_dtypes(backend):
    graph = make_graph(backend)
    for count in graph.true_shape.hyper_edge_sets.values():
        assert_int_dtype(count)
        assert np.asarray(count).ndim == 0
    assert_int_dtype(graph.true_shape.addresses)
    assert np.asarray(graph.true_shape.addresses).ndim == 0


def test_large_addresses_are_exact():
    # 2**24 + 1 is not representable in float32: this is the regression the int pipeline fixes.
    big = 2**24 + 1
    hes = HyperEdgeSet.from_dict(port_dict={"id": np.array([0, big], dtype=np.int64)}, feature_dict=None)
    assert int(np.asarray(hes.port_dict["id"])[1]) == big


# ---------------------------------------------------------------------------
# Backend conversion
# ---------------------------------------------------------------------------


def test_to_backend_preserves_integer_ports_and_counts():
    graph = make_graph(NumpyBackend())
    for target in (JaxBackend(), NumpyBackend()):
        converted = graph.to_backend(target)
        assert_int_dtype(converted.hyper_edge_sets["line"].port_dict["from"])
        assert_int_dtype(converted.true_shape.addresses)
        for count in converted.true_shape.hyper_edge_sets.values():
            assert_int_dtype(count)
        assert_float_dtype(converted.hyper_edge_sets["line"].feature_array)


def test_bfloat16_backend_does_not_corrupt_ports():
    graph = make_graph(NumpyBackend(), n_bus=400)
    converted = graph.to_backend(JaxBackend(dtype="bfloat16"))

    # Features get the backend's floating dtype...
    assert str(converted.hyper_edge_sets["bus"].feature_array.dtype) == "bfloat16"
    # ...but ports stay integers: address 257+ is not representable in bfloat16.
    ports = np.asarray(converted.hyper_edge_sets["bus"].port_dict["id"])
    assert_int_dtype(ports)
    assert ports.tolist() == list(range(400))


def test_round_trip_keeps_dtypes():
    graph = make_graph(NumpyBackend())
    round_tripped = graph.to_backend(JaxBackend()).to_backend(NumpyBackend())
    assert np.asarray(round_tripped.hyper_edge_sets["line"].port_dict["from"]).dtype == np.int32
    assert_int_dtype(round_tripped.true_shape.addresses)
    assert_float_dtype(round_tripped.hyper_edge_sets["line"].feature_array)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def test_pad_and_collate_keep_integer_ports(backend):
    graphs = [make_graph(backend, n_bus=n) for n in (3, 5)]
    target = GraphShape(
        backend=backend,
        hyper_edge_sets={"line": np.array(4), "bus": np.array(5)},
        addresses=np.array(5),
    )
    for graph in graphs:
        graph.pad(target_shape=target)
    batch = collate_graphs(graphs)
    assert_int_dtype(batch.hyper_edge_sets["line"].port_dict["from"])


def test_concatenate_keeps_integer_ports(backend):
    graphs = [make_graph(backend, n_bus=n) for n in (3, 4)]
    merged = concatenate_graphs(graphs)
    ports = np.asarray(merged.hyper_edge_sets["bus"].port_dict["id"])
    assert_int_dtype(ports)
    # Offsets applied: second graph's buses are shifted by the first graph's 3 addresses.
    assert ports.tolist() == [0, 1, 2, 3, 4, 5, 6]


def test_shape_max_stays_integer(backend):
    shape_a = make_graph(backend, n_bus=3).true_shape
    shape_b = make_graph(backend, n_bus=5).true_shape
    merged = GraphShape.max(shape_a, shape_b)
    for count in merged.hyper_edge_sets.values():
        assert_int_dtype(count)
    assert int(merged.addresses) == 5


def test_shape_jsonable_round_trip_is_integer(backend):
    shape = make_graph(backend).true_shape
    restored = GraphShape.from_jsonable_dict(shape.to_jsonable_dict(), backend=backend)
    assert_int_dtype(restored.addresses)
    for count in restored.hyper_edge_sets.values():
        assert_int_dtype(count)
