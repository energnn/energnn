# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.graph import collate_graphs
from energnn.graph.visualization.layout import (
    STUB_LENGTH,
    address_radius,
    layout_margin,
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
    assert data.classes == ["gen", "line", "trafo3w"]
    assert data.ports["line"] == [[0, 1], [1, 2], [2, 3]]
    assert data.port_names["trafo3w"] == ["hv", "lv", "mv"]
    assert data.features["gen"] == [{"p": 1.0}, {"p": 2.0}]
    # one hub row for the single order-3 object, after the 4 addresses; z = 0 in 2D
    assert data.hub_ids == {("trafo3w", 0): 4}
    assert data.pos.shape == (5, 3)
    assert np.all(data.pos[:, 2] == 0)
    assert data.colors is None and data.object_colors == {} and data.placed == frozenset()


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
# Positions: 2D, 3D, from hyper-edge features
# ---------------------------------------------------------------------------


def test_injected_positions_are_normalized(mixed_order_graph):
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_positions=SQUARE)
    assert data.ndim == 2
    np.testing.assert_allclose(data.pos[:4, :2], UNIT_SQUARE, atol=1e-6)
    # the order-3 hub sits at the barycenter of its ports 0, 1, 2
    np.testing.assert_allclose(data.pos[4, :2], UNIT_SQUARE[[0, 1, 2]].mean(axis=0), atol=1e-6)


def test_injected_positions_3d(mixed_order_graph):
    positions = np.concatenate([SQUARE, [[0.0], [5.0], [10.0], [5.0]]], axis=1)
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_positions=positions)
    assert data.ndim == 3
    assert data.pos.shape == (5, 3)
    assert np.abs(data.pos).max() <= 1.0 + 1e-6
    assert not np.all(data.pos[:, 2] == 0)


def test_injected_positions_padded_length(mixed_order_graph, padded_shape):
    mixed_order_graph.pad(padded_shape)
    positions = np.arange(14, dtype=float).reshape(7, 2)  # padded length: extra rows dropped
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_positions=positions)
    assert data.n_addr == 4


@pytest.mark.parametrize("bad", [np.zeros((3, 2)), np.zeros((4, 4)), np.zeros((4,)), np.zeros((2, 4, 2))])
def test_injected_positions_bad_shape(mixed_order_graph, bad):
    with pytest.raises(ValueError, match="address_positions"):
        extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_positions=bad)


# ---------------------------------------------------------------------------
# Address colors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_channels", [1, 2, 3])
def test_address_colors_are_normalized_per_channel(mixed_order_graph, n_channels):
    raw = np.arange(4 * n_channels, dtype=float).reshape(4, n_channels) * 10.0 - 5.0
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_colors=raw)
    assert data.colors is not None and data.color_range is not None
    assert data.colors.shape == (4, n_channels)
    assert data.colors.min() == 0.0 and data.colors.max() == 1.0
    np.testing.assert_allclose(data.color_range[0], raw.min(axis=0))
    np.testing.assert_allclose(data.color_range[1], raw.max(axis=0))


def test_address_colors_constant_channel_and_nan(mixed_order_graph):
    raw = np.array([[1.0], [1.0], [np.nan], [1.0]])
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_colors=raw)
    assert np.all(data.colors == 0.5)


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
    assert np.linalg.norm(data.pos[:3] - geoms[("line", 3)].marker, axis=1).min() > 0.05
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
    assert all("direction" not in d for d in descriptors["trafo3w"])  # regular hubs keep their barycenter


def test_degenerate_hub_descriptor_carries_its_direction(degenerate_hubs_graph):
    data = extract_plot_data(degenerate_hubs_graph, iterations=10, seed=0)
    descriptors = object_descriptors(data)
    assert len(descriptors["t5"][0]["direction"]) == 2 and len(descriptors["t3"][0]["direction"]) == 2
    assert "direction" not in descriptors["t3"][1] and "direction" not in descriptors["t4"][0]


def test_stubs_and_loops_keep_clear_of_the_address_circle(mixed_order_graph, multi_graph):
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0)
    r_addr = address_radius(data.n_addr)
    geoms = object_geometries(data)
    for i, (address,) in enumerate(data.ports["gen"]):
        distance = np.linalg.norm(geoms[("gen", i)].marker - data.pos[address])
        assert distance >= 2.0 * r_addr  # marker center beyond the circle plus a gap
    data = extract_plot_data(multi_graph, iterations=10, seed=0)
    loop = object_geometries(data)[("line", 3)]
    anchor = data.pos[2]
    assert np.linalg.norm(loop.marker - anchor) == pytest.approx(STUB_LENGTH * address_radius(data.n_addr))
    assert len(loop.lines) == 2 and all(len(line) == 17 for line in loop.lines)  # one fanned spoke per port
    for line in loop.lines:  # both spokes run from the address center to the marker
        np.testing.assert_allclose(line[0], anchor, atol=1e-9)
        np.testing.assert_allclose(line[-1], loop.marker, atol=1e-9)
    assert len(np.unique(np.round(loop.labels, 6), axis=0)) == 2


def test_address_radius_shrinks_with_the_number_of_addresses():
    assert address_radius(1) == address_radius(4) > address_radius(400) > address_radius(10_000) > 0


def test_layout_margin_covers_stubs_loops_and_fans(mixed_order_graph, multi_graph, portless_graph):
    with_stubs = extract_plot_data(mixed_order_graph, iterations=10, seed=0)
    r_addr = address_radius(with_stubs.n_addr)
    assert with_stubs.margin == pytest.approx((STUB_LENGTH + 0.62) * r_addr)
    geoms = object_geometries(with_stubs)
    reach = max(np.abs(g.marker).max() for g in geoms.values())
    assert reach <= 1.0 + with_stubs.margin
    with_loops = extract_plot_data(multi_graph, iterations=10, seed=0)
    assert with_loops.margin >= (STUB_LENGTH + 0.62) * address_radius(3)
    plain = extract_plot_data(portless_graph, iterations=10, seed=0)
    assert plain.margin == pytest.approx(0.62 * address_radius(3))
    assert layout_margin(3, {"line": [[0, 1], [0, 1]]}) >= 0.09  # fanned parallel edges


TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize("positions", [None, TRIANGLE], ids=["spring", "given"])
def test_hubs_with_all_ports_on_one_address_are_offset_like_stubs(degenerate_hubs_graph, positions):
    data = extract_plot_data(degenerate_hubs_graph, iterations=50, seed=0, address_positions=positions)
    r_addr = address_radius(data.n_addr)
    geoms = object_geometries(data)
    for key, address, order in ((("t3", 0), 2, 3), (("t5", 0), 1, 5)):
        hub = geoms[key].marker
        assert np.linalg.norm(hub - data.pos[address]) == pytest.approx(STUB_LENGTH * r_addr)
        assert len(geoms[key].lines) == order  # one (curved) spoke per port
        assert all(len(line) == 17 for line in geoms[key].lines)  # fanned Bezier curves, not segments
        assert len(np.unique(np.round(geoms[key].labels, 6), axis=0)) == order  # one label per port, all distinct
        assert len(np.unique(np.round([line[8] for line in geoms[key].lines], 6), axis=0)) == order


def test_hub_with_partially_repeated_ports(degenerate_hubs_graph):
    data = extract_plot_data(degenerate_hubs_graph, iterations=50, seed=0, address_positions=TRIANGLE)
    geom = object_geometries(data)[("t4", 0)]
    assert [len(line) for line in geom.lines] == [17, 17, 2, 2]  # a, b fanned to address 0; c, d straight
    assert len(np.unique(np.round(geom.labels, 6), axis=0)) == 4
    # the hub sits at the barycenter of the distinct addresses 0, 1, 2
    np.testing.assert_allclose(geom.marker, data.pos[[0, 1, 2]].mean(axis=0), atol=1e-6)


def test_hub_landing_on_one_of_its_addresses_is_pushed_away(mixed_order_graph):
    # addresses 0, 1, 2 collinear with 1 at the barycenter of the trafo3w ports (0, 1, 2)
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_positions=positions)
    hub = object_geometries(data)[("trafo3w", 0)].marker
    assert np.linalg.norm(hub - data.pos[1]) >= 1.5 * address_radius(data.n_addr)


def test_parallel_hubs_with_given_positions_are_spread(multi_graph):
    data = extract_plot_data(multi_graph, iterations=10, seed=0, address_positions=TRIANGLE)
    geoms = object_geometries(data)
    assert np.linalg.norm(geoms[("trafo3w", 0)].marker - geoms[("trafo3w", 1)].marker) >= 2 * address_radius(3)


def test_spring_layout_links_hubs_to_distinct_addresses_only(degenerate_hubs_graph):
    data = extract_plot_data(degenerate_hubs_graph, iterations=50, seed=0)
    assert data.margin >= (STUB_LENGTH + 0.62) * address_radius(3)  # degenerate hubs reach like stubs


# ---------------------------------------------------------------------------
# Missing coordinates and colors
# ---------------------------------------------------------------------------


def _chain(n: int):
    """A path graph 0-1-...-(n-1) over n addresses."""
    from energnn.graph.graph import Graph
    from energnn.graph.hyper_edge_set import HyperEdgeSet

    hes = {"line": HyperEdgeSet.from_dict(port_dict={"from": np.arange(n - 1), "to": np.arange(1, n)}, feature_dict=None)}
    return Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=n)


def test_missing_position_between_two_known_neighbors_is_their_midpoint():
    positions = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 4.0]])
    data = extract_plot_data(_chain(3), iterations=10, seed=0, address_positions=positions)
    np.testing.assert_allclose(data.pos[1], (data.pos[0] + data.pos[2]) / 2, atol=1e-9)
    assert data.inferred.tolist() == [False, True, False]


def test_missing_run_is_spread_evenly_along_the_chain():
    positions = np.full((6, 2), np.nan)
    positions[0], positions[5] = [0.0, 0.0], [10.0, 0.0]
    data = extract_plot_data(_chain(6), iterations=10, seed=0, address_positions=positions)
    xs = data.pos[:6, 0]
    np.testing.assert_allclose(np.diff(xs), np.diff(xs)[0], atol=1e-9)  # evenly spaced
    assert np.all(np.diff(xs) > 0)
    np.testing.assert_allclose(data.pos[:6, 1], 0.0, atol=1e-9)


def test_missing_positions_in_3d():
    positions = np.array([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [2.0, 2.0, 2.0]])
    data = extract_plot_data(_chain(3), iterations=10, seed=0, address_positions=positions)
    assert data.ndim == 3
    np.testing.assert_allclose(data.pos[1], (data.pos[0] + data.pos[2]) / 2, atol=1e-9)


def test_component_without_any_known_position_gets_a_layout_beside_the_known_cloud():
    from energnn.graph.graph import Graph
    from energnn.graph.hyper_edge_set import HyperEdgeSet

    # two components: 0-1 known, 2-3 entirely unknown
    hes = {"line": HyperEdgeSet.from_dict(port_dict={"from": np.array([0, 2]), "to": np.array([1, 3])}, feature_dict=None)}
    graph = Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=4)
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [np.nan, np.nan]])
    data = extract_plot_data(graph, iterations=10, seed=0, address_positions=positions)
    assert np.isfinite(data.pos).all()
    assert data.inferred.tolist() == [False, False, True, True]
    assert data.pos[2:4, 0].mean() > data.pos[0:2, 0].max()  # placed to the right of the known addresses
    assert np.linalg.norm(data.pos[2] - data.pos[3]) > 1e-3  # and not on top of each other


def test_all_positions_missing_is_an_error(mixed_order_graph):
    with pytest.raises(ValueError, match="no address position"):
        extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_positions=np.full((4, 2), np.nan))


def test_missing_color_is_flagged_not_averaged(mixed_order_graph):
    colors = np.array([[0.0], [np.nan], [2.0], [4.0]])
    data = extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_colors=colors)
    assert data.missing_colors is not None and data.missing_colors.tolist() == [False, True, False, False]
    np.testing.assert_allclose(data.color_range, [[0.0], [4.0]])  # the NaN does not stretch the range
    with pytest.raises(ValueError, match="all missing"):
        extract_plot_data(mixed_order_graph, iterations=10, seed=0, address_colors=np.full((4, 1), np.nan))


# ---------------------------------------------------------------------------
# Hyper-edge positions and colors read from features
# ---------------------------------------------------------------------------


def _located_graph():
    """Buses carrying their own (x, y) and a load, lines between them, one port-less decision class."""
    from energnn.graph.graph import Graph
    from energnn.graph.hyper_edge_set import HyperEdgeSet

    hes = {
        "bus": HyperEdgeSet.from_dict(
            port_dict={"id": np.array([0, 1, 2])},
            feature_dict={"x": np.array([0.0, 4.0, 0.0]), "y": np.array([0.0, 0.0, 3.0]), "load": np.array([1.0, 2.0, 3.0])},
        ),
        "line": HyperEdgeSet.from_dict(
            port_dict={"from": np.array([0, 1]), "to": np.array([1, 2])}, feature_dict={"flow": np.array([10.0, 5.0])}
        ),
        "theta": HyperEdgeSet.from_dict(port_dict=None, feature_dict={"value": np.array([0.5, 0.6, 0.7])}),
    }
    graph = Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=4)  # address 3 is not connected to anything
    graph.line.flow = np.array([10.0, np.nan])  # from_dict rejects NaN; the second flow is unknown
    return graph


def test_hyper_edge_positions_place_objects_and_derive_addresses():
    data = extract_plot_data(_located_graph(), iterations=10, seed=0, hyper_edge_positions={"bus": ["x", "y"]})
    assert data.ndim == 2 and data.placed == {("bus", 0), ("bus", 1), ("bus", 2)}
    assert all(("bus", i) in data.hub_ids for i in range(3))
    # the buses keep their given geometry (up to the fit in the [-1, 1] box) and each address sits on its bus
    bus_pos = data.pos[[data.hub_ids[("bus", i)] for i in range(3)], :2]
    scale = (bus_pos[1, 0] - bus_pos[0, 0]) / 4.0
    assert scale > 0
    np.testing.assert_allclose(bus_pos - bus_pos[0], scale * np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]), atol=1e-9)
    # each address is pushed off the bus that places it, like a stub, so both stay visible
    r_addr = address_radius(data.n_addr)
    for i in range(3):
        offset = np.linalg.norm(data.pos[i] - data.pos[data.hub_ids[("bus", i)]])
        assert offset == pytest.approx(STUB_LENGTH * r_addr)
    assert np.abs(data.pos).max() <= 1.0 + data.margin
    # address 3 has no position: reconstructed (beside the known cloud) and flagged
    assert data.inferred.tolist() == [False, False, False, True] and np.isfinite(data.pos).all()
    # a placed object is a hub whatever its order: one spoke to its port, marker at its position
    geoms = object_geometries(data)
    assert len(geoms[("bus", 0)].lines) == 1 and np.allclose(geoms[("bus", 0)].marker, data.pos[data.hub_ids[("bus", 0)]])
    assert [d["kind"] for d in object_descriptors(data)["bus"]] == ["hub"] * 3
    assert [d["kind"] for d in object_descriptors(data)["line"]] == ["pair", "pair"]


def test_hyper_edge_positions_average_over_the_objects_pointing_to_an_address():
    data = extract_plot_data(_located_graph(), iterations=10, seed=0, hyper_edge_positions={"line": ["flow", "flow"]})
    # line 1 has a NaN flow: it is drawn as a regular pair, and only line 0 (flow 10) places addresses 0 and 1
    assert data.placed == {("line", 0)}
    assert data.inferred.tolist() == [False, False, True, True]
    # both ends of line 0 are placed at its position, then pushed off it in distinct directions
    for a in (0, 1):
        assert np.linalg.norm(data.pos[a] - data.pos[data.hub_ids[("line", 0)]]) == pytest.approx(
            STUB_LENGTH * address_radius(data.n_addr)
        )
    assert np.linalg.norm(data.pos[0] - data.pos[1]) > 1e-3
    assert [d["kind"] for d in object_descriptors(data)["line"]] == ["hub", "pair"]


def test_hyper_edge_positions_with_address_positions_keep_the_given_addresses():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    data = extract_plot_data(
        _located_graph(), iterations=10, seed=0, address_positions=positions, hyper_edge_positions={"bus": ["x", "y"]}
    )
    # the addresses keep the given square (up to the fit of addresses and buses in the [-1, 1] box)
    square = data.pos[:4, :2]
    scale = square[1, 0] - square[0, 0]
    assert scale > 0
    np.testing.assert_allclose(square - square[0], scale * positions, atol=1e-9)
    assert data.inferred.tolist() == [False] * 4
    bus_pos = data.pos[[data.hub_ids[("bus", i)] for i in range(3)], :2]
    np.testing.assert_allclose(bus_pos - square[0], scale * np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]), atol=1e-9)
    assert bus_pos[1, 0] > square[:, 0].max()  # bus 1 is placed by its features, outside the address square


def test_hyper_edge_positions_place_portless_objects_as_lone_markers():
    spec = {"theta": ["value", "value"]}
    with pytest.raises(ValueError, match="no address position"):  # port-less objects place no address
        extract_plot_data(_located_graph(), iterations=10, seed=0, hyper_edge_positions=spec)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    data = extract_plot_data(_located_graph(), iterations=10, seed=0, address_positions=square, hyper_edge_positions=spec)
    geoms = object_geometries(data)
    assert all(geoms[("theta", i)].lines == [] for i in range(3))  # a marker, no spoke
    assert [d["kind"] for d in object_descriptors(data)["theta"]] == ["hub"] * 3
    assert data.inferred.tolist() == [False] * 4


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"nope": ["x", "y"]}, "unknown hyper-edge class"),
        ({"bus": ["x"]}, "2 or 3 feature names"),
        ({"bus": ["x", "nope"]}, "has no feature"),
        ({"bus": ["x", "y"], "line": ["flow", "flow", "flow"]}, "same number of feature names"),
    ],
)
def test_hyper_edge_positions_errors(spec, message):
    with pytest.raises(ValueError, match=message):
        extract_plot_data(_located_graph(), iterations=10, seed=0, hyper_edge_positions=spec)


def test_hyper_edge_positions_dimension_must_match_address_positions():
    with pytest.raises(ValueError, match="both be 2D or both be 3D"):
        extract_plot_data(
            _located_graph(),
            iterations=10,
            seed=0,
            address_positions=np.zeros((4, 3)),
            hyper_edge_positions={"bus": ["x", "y"]},
        )


def test_hyper_edge_colors_are_normalized_over_every_listed_class():
    data = extract_plot_data(_located_graph(), iterations=10, seed=0, hyper_edge_colors={"bus": ["load"], "line": ["flow"]})
    assert sorted(data.object_colors) == ["bus", "line"]
    np.testing.assert_allclose(data.object_color_range, [[1.0], [10.0]])  # one range over buses and lines, NaN ignored
    np.testing.assert_allclose(data.object_colors["bus"][:, 0], [0.0, 1 / 9, 2 / 9])
    np.testing.assert_allclose(data.object_colors["line"][:, 0], [1.0, 0.5])  # the NaN is flagged, not averaged
    assert data.missing_object_colors["line"].tolist() == [False, True]
    assert data.colors is None  # addresses keep their own colors
    with pytest.raises(ValueError, match="1 or 2 or 3 feature names"):
        extract_plot_data(_located_graph(), iterations=10, seed=0, hyper_edge_colors={"bus": ["x", "y", "load", "x"]})


def test_hyper_edge_colors_two_channels_and_address_colors_together():
    colors = np.array([[0.0], [1.0], [2.0], [3.0]])
    data = extract_plot_data(
        _located_graph(), iterations=10, seed=0, address_colors=colors, hyper_edge_colors={"bus": ["x", "load"]}
    )
    assert data.object_colors["bus"].shape == (3, 2) and data.colors.shape == (4, 1)
    np.testing.assert_allclose(data.object_color_range, [[0.0, 1.0], [4.0, 3.0]])
