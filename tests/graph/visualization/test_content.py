# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.graph import collate_graphs
from energnn.graph.visualization.content import feature_columns, per_address_array, read_graph


def test_read_graph_collects_real_objects(mixed_order_graph):
    topology = read_graph(mixed_order_graph)
    assert topology.n_addresses == 4
    assert topology.classes == ["gen", "line", "trafo3w"]
    assert topology.port_names["trafo3w"] == ["hv", "lv", "mv"]
    assert [h.ports for h in topology.of("line")] == [(0, 1), (1, 2), (2, 3)]
    assert [h.features for h in topology.of("gen")] == [{"p": 1.0}, {"p": 2.0}]
    assert [h.kind for h in topology.hyper_edges] == ["stub", "stub", "pair", "pair", "pair", "hub"]
    assert [h.key for h in topology.of("gen")] == [("gen", 0), ("gen", 1)]


def test_read_graph_skips_fictitious(mixed_order_graph, padded_shape):
    reference = read_graph(mixed_order_graph)
    mixed_order_graph.pad(padded_shape)
    padded = read_graph(mixed_order_graph)
    assert padded.n_addresses == reference.n_addresses and len(padded.address_mask) == 7
    assert padded.hyper_edges == reference.hyper_edges


def test_read_graph_rejects_batch(mixed_order_graph):
    with pytest.raises(ValueError, match="single"):
        read_graph(collate_graphs([mixed_order_graph, mixed_order_graph]))


def test_kinds_of_loops_hubs_and_portless_objects(multi_graph, portless_graph):
    assert [h.kind for h in read_graph(multi_graph).hyper_edges] == ["pair", "pair", "pair", "loop", "hub", "hub"]
    topology = read_graph(portless_graph)
    assert [h.kind for h in topology.of("bus")] == ["none"] * 3
    assert [h.features["phase_angle"] for h in topology.of("bus")] == pytest.approx([0.1, 0.2, 0.3])


def test_hyper_edge_positions_place_objects_with_finite_coordinates(located_graph):
    topology = read_graph(located_graph, hyper_edge_positions={"bus": ["x", "y"], "line": ["flow", "flow"]})
    buses = topology.of("bus")
    assert [h.kind for h in buses] == ["placed"] * 3
    np.testing.assert_allclose([h.position for h in buses], [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    assert [h.kind for h in topology.of("line")] == ["placed", "pair"]  # the NaN flow leaves line 1 unplaced
    assert [h.kind for h in topology.of("theta")] == ["none"] * 3


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"nope": ["x", "y"]}, "unknown hyper-edge class"),
        ({"bus": ["x"]}, "2 feature names"),
        ({"bus": ["x", "nope"]}, "has no feature"),
    ],
)
def test_hyper_edge_positions_errors(located_graph, spec, message):
    with pytest.raises(ValueError, match=message):
        read_graph(located_graph, hyper_edge_positions=spec)


def test_feature_columns_require_one_width(located_graph):
    topology = read_graph(located_graph)
    columns = feature_columns(topology, {"bus": ["load"], "line": ["flow"]}, "colors", (1, 2))
    np.testing.assert_allclose(columns["bus"], [[1.0], [2.0], [3.0]])
    assert np.isnan(columns["line"][1, 0])
    with pytest.raises(ValueError, match="same number of feature names"):
        feature_columns(topology, {"bus": ["load", "x"], "line": ["flow"]}, "colors", (1, 2))


def test_per_address_array_accepts_real_or_padded_length(mixed_order_graph, padded_shape):
    mixed_order_graph.pad(padded_shape)
    topology = read_graph(mixed_order_graph)
    assert per_address_array(np.arange(14, dtype=float).reshape(7, 2), "positions", topology, (2,)).shape == (4, 2)
    assert per_address_array(np.zeros((4, 1)), "colors", topology, (1, 2)).shape == (4, 1)
    for bad in (np.zeros((3, 2)), np.zeros((4, 3)), np.zeros(4), np.zeros((2, 4, 2))):
        with pytest.raises(ValueError, match="positions"):
            per_address_array(bad, "positions", topology, (2,))
