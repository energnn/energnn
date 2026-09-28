# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.graph import collate_graphs
from energnn.graph.visualization.layout import (
    LOOP_RADIUS,
    address_radius,
    extract_plot_data,
    object_descriptors,
    object_geometries,
    spring_layout,
)

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
UNIT_SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


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
    assert data.ndim == 2
    assert data.n_frames == 1
    assert data.classes == ["gen", "line", "trafo3w"]
    assert data.ports["line"] == [[0, 1], [1, 2], [2, 3]]
    assert data.port_names["trafo3w"] == ["hv", "lv", "mv"]
    assert data.features["gen"] == [{"p": 1.0}, {"p": 2.0}]
    # one hub row for the single order-3 object, after the 4 addresses; z = 0 in 2D
    assert data.hub_ids == {("trafo3w", 0): 4}
    assert data.pos.shape == (1, 5, 3)
    assert np.all(data.pos[..., 2] == 0)
    assert data.colors is None


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


# ---------------------------------------------------------------------------
# Positions: 2D, 3D, frames
# ---------------------------------------------------------------------------


def test_injected_positions_are_normalized(mixed_order_graph):
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, positions=SQUARE)
    assert data.ndim == 2
    np.testing.assert_allclose(data.pos[0, :4, :2], UNIT_SQUARE, atol=1e-6)
    # the order-3 hub sits at the barycenter of its ports 0, 1, 2
    np.testing.assert_allclose(data.pos[0, 4, :2], UNIT_SQUARE[[0, 1, 2]].mean(axis=0), atol=1e-6)


def test_injected_positions_3d(mixed_order_graph):
    positions = np.concatenate([SQUARE, [[0.0], [5.0], [10.0], [5.0]]], axis=1)
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, positions=positions)
    assert data.ndim == 3
    assert data.pos.shape == (1, 5, 3)
    assert np.abs(data.pos).max() <= 1.0 + 1e-6
    assert not np.all(data.pos[..., 2] == 0)


def test_injected_positions_frames_share_one_normalization(mixed_order_graph):
    frames = np.stack([SQUARE, SQUARE + 10.0])  # the second frame is a translated copy
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, positions=frames)
    assert data.n_frames == 2
    shift = data.pos[1, :4] - data.pos[0, :4]
    assert np.allclose(shift, shift[0], atol=1e-6)  # translation preserved, not re-centered per frame
    assert np.linalg.norm(shift[0]) > 0.1


def test_injected_positions_padded_length(mixed_order_graph, padded_shape):
    mixed_order_graph.pad(padded_shape)
    positions = np.arange(14, dtype=float).reshape(7, 2)  # padded length: extra rows dropped
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, positions=positions)
    assert data.n_addr == 4


@pytest.mark.parametrize("bad", [np.zeros((3, 2)), np.zeros((4, 4)), np.zeros((4,)), np.zeros((2, 3, 2, 1))])
def test_injected_positions_bad_shape(mixed_order_graph, bad):
    with pytest.raises(ValueError, match="positions"):
        extract_plot_data(mixed_order_graph, iterations=10, seed=0, positions=bad)


# ---------------------------------------------------------------------------
# Address colors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_address_colors_are_normalized_per_channel(mixed_order_graph, n_channels):
    raw = np.arange(4 * n_channels, dtype=float).reshape(4, n_channels) * 10.0 - 5.0
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_colors=raw)
    assert data.colors is not None and data.color_range is not None
    assert data.colors.shape == (1, 4, n_channels)
    assert data.colors.min() == 0.0 and data.colors.max() == 1.0
    np.testing.assert_allclose(data.color_range[0], raw.min(axis=0))
    np.testing.assert_allclose(data.color_range[1], raw.max(axis=0))


def test_address_colors_constant_channel_and_nan(mixed_order_graph):
    raw = np.array([[1.0], [1.0], [np.nan], [1.0]])
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_colors=raw)
    assert np.all(data.colors == 0.5)


def test_address_colors_frames_broadcast_positions(mixed_order_graph):
    colors = np.random.default_rng(0).random((5, 4, 1))
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_colors=colors)
    assert data.n_frames == 5 and data.colors.shape == (5, 4, 1)
    assert np.all(data.pos[0] == data.pos[4])  # static layout repeated for every frame


def test_address_colors_frames_must_match_positions(mixed_order_graph):
    with pytest.raises(ValueError, match="frames"):
        extract_plot_data(
            mixed_order_graph, iterations=10, seed=0, positions=np.zeros((3, 4, 2)), address_colors=np.zeros((2, 4, 1))
        )


def test_address_colors_bad_channels(mixed_order_graph):
    with pytest.raises(ValueError, match="address_colors"):
        extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_colors=np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# Geometries and descriptors
# ---------------------------------------------------------------------------


def test_object_geometries_one_per_object(mixed_order_graph):
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0)
    geoms = object_geometries(data)
    assert set(geoms) == {("line", 0), ("line", 1), ("line", 2), ("gen", 0), ("gen", 1), ("trafo3w", 0)}
    assert len(geoms[("trafo3w", 0)].lines) == 3  # one spoke per port
    assert len(geoms[("line", 0)].labels) == 2
    assert all(line.shape[1] == 3 for g in geoms.values() for line in g.lines)


def test_object_geometries_follow_frames(mixed_order_graph):
    frames = np.stack([SQUARE, SQUARE[::-1]])
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, positions=frames)
    first, second = object_geometries(data, 0), object_geometries(data, 1)
    assert not np.allclose(first[("line", 0)].marker, second[("line", 0)].marker)


def test_portless_objects_have_no_geometry(portless_graph):
    data = extract_plot_data(portless_graph, iterations=10, seed=0)
    assert data.ports["bus"] == [[], [], []]
    assert [f["phase_angle"] for f in data.features["bus"]] == pytest.approx([0.1, 0.2, 0.3])
    assert set(object_geometries(data)) == {("line", 0), ("line", 1)}
    assert [d["kind"] for d in object_descriptors(data)["bus"]] == ["none"] * 3


def test_parallel_edges_and_self_loops_are_separated(multi_graph):
    data = extract_plot_data(multi_graph, iterations=10, seed=0)
    geoms = object_geometries(data)
    markers = np.array([geoms[("line", i)].marker for i in range(4)])
    assert len(np.unique(np.round(markers, 6), axis=0)) == 4
    # the self-loop marker is off its address
    assert np.linalg.norm(data.pos[0, :3] - geoms[("line", 3)].marker, axis=1).min() > 0.05
    # the two parallel hubs are distinct
    assert np.linalg.norm(geoms[("trafo3w", 0)].marker - geoms[("trafo3w", 1)].marker) > 0.01


def test_object_descriptors_kinds(multi_graph):
    data = extract_plot_data(multi_graph, iterations=10, seed=0)
    descriptors = object_descriptors(data)
    lines = descriptors["line"]
    assert [d["kind"] for d in lines] == ["pair", "pair", "pair", "loop"]
    assert sorted(d["fan"] for d in lines[:3]) == [-1.0, 0.0, 1.0]
    assert len(lines[3]["direction"]) == 2
    assert [d["kind"] for d in descriptors["trafo3w"]] == ["hub", "hub"]
    assert descriptors["trafo3w"][0]["hub"] == 3 and descriptors["trafo3w"][1]["hub"] == 4


def test_stubs_and_loops_keep_clear_of_the_address_circle(mixed_order_graph, multi_graph):
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0)
    r_addr = address_radius(data.n_addr)
    geoms = object_geometries(data)
    for i, (address,) in enumerate(data.ports["gen"]):
        distance = np.linalg.norm(geoms[("gen", i)].marker - data.pos[0, address])
        assert distance >= 2.0 * r_addr  # marker center beyond the circle plus a gap
    data = extract_plot_data(multi_graph, iterations=10, seed=0)
    loop = object_geometries(data)[("line", 3)]
    inner = np.linalg.norm(loop.lines[0] - data.pos[0, 2], axis=1).min()
    assert inner >= address_radius(data.n_addr)  # the loop circle starts outside the address circle
    radii = np.linalg.norm(loop.lines[0] - loop.lines[0].mean(axis=0), axis=1)
    assert np.ptp(radii) < 0.1 * LOOP_RADIUS  # a circle (the mean of the closed polyline is slightly biased)


def test_address_radius_shrinks_with_the_number_of_addresses():
    assert address_radius(1) == address_radius(4) > address_radius(400) > address_radius(10_000) > 0
