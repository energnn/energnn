# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.graph import collate_graphs
from energnn.graph.visualization.layout import extract_plot_data, object_geometries, spring_layout


def test_spring_layout_shape_and_scale():
    pos = spring_layout(5, np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))
    assert pos.shape == (5, 2)
    assert np.abs(pos).max() <= 1.0 + 1e-6


def test_spring_layout_no_edges():
    pos = spring_layout(3, np.zeros((0, 2), dtype=int))
    assert pos.shape == (3, 2)


def test_extract_plot_data_collects_real_objects(mixed_order_graph):
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0)
    assert data.n_addr == 4
    assert data.classes == ["gen", "line", "trafo3w"]
    assert data.ports["line"] == [[0, 1], [1, 2], [2, 3]]
    assert data.port_names["trafo3w"] == ["hv", "lv", "mv"]
    assert data.features["gen"] == [{"p": 1.0}, {"p": 2.0}]
    # one hub row for the single order-3 object, after the 4 addresses
    assert data.hub_ids == {("trafo3w", 0): 4}
    assert data.pos.shape == (5, 2)


def test_extract_plot_data_skips_fictitious(mixed_order_graph, padded_shape):
    reference = extract_plot_data(mixed_order_graph, iterations=10, seed=0)
    mixed_order_graph.pad(padded_shape)
    padded = extract_plot_data(mixed_order_graph, iterations=10, seed=0)
    assert padded.n_addr == reference.n_addr
    assert padded.ports == reference.ports
    assert padded.features == reference.features


def test_extract_plot_data_rejects_batch(mixed_order_graph):
    batch = collate_graphs([mixed_order_graph, mixed_order_graph])
    with pytest.raises(ValueError, match="single"):
        extract_plot_data(batch, iterations=10, seed=0)


def test_injected_positions_are_normalized(mixed_order_graph):
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, positions=positions)
    expected = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(data.pos[:4], expected, atol=1e-6)
    # the order-3 hub sits at the barycenter of its ports 0, 1, 2
    np.testing.assert_allclose(data.pos[4], expected[[0, 1, 2]].mean(axis=0), atol=1e-6)


def test_injected_positions_padded_length(mixed_order_graph, padded_shape):
    mixed_order_graph.pad(padded_shape)
    positions = np.arange(14, dtype=float).reshape(7, 2)  # padded length: extra rows dropped
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, positions=positions)
    assert data.n_addr == 4


def test_injected_positions_bad_shape(mixed_order_graph):
    with pytest.raises(ValueError, match="positions"):
        extract_plot_data(mixed_order_graph, iterations=10, seed=0, positions=np.zeros((3, 2)))


def test_object_geometries_one_per_object(mixed_order_graph):
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0)
    geoms = object_geometries(data)
    assert set(geoms) == {("line", 0), ("line", 1), ("line", 2), ("gen", 0), ("gen", 1), ("trafo3w", 0)}
    assert len(geoms[("trafo3w", 0)].lines) == 3  # one spoke per port
    assert len(geoms[("line", 0)].labels) == 2


def test_portless_objects_have_no_geometry(portless_graph):
    data = extract_plot_data(portless_graph, iterations=10, seed=0)
    assert data.ports["bus"] == [[], [], []]
    assert [f["phase_angle"] for f in data.features["bus"]] == pytest.approx([0.1, 0.2, 0.3])
    assert set(object_geometries(data)) == {("line", 0), ("line", 1)}


def test_parallel_edges_and_self_loops_are_separated(multi_graph):
    data = extract_plot_data(multi_graph, iterations=10, seed=0)
    geoms = object_geometries(data)
    markers = np.array([geoms[("line", i)].marker for i in range(4)])
    assert len(np.unique(np.round(markers, 6), axis=0)) == 4
    # the self-loop marker is off its address
    assert np.linalg.norm(data.pos[:3] - geoms[("line", 3)].marker, axis=1).min() > 0.05
    # the two parallel hubs are distinct
    assert np.linalg.norm(geoms[("trafo3w", 0)].marker - geoms[("trafo3w", 1)].marker) > 0.01
