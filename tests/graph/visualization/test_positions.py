# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.visualization.content import read_graph
from energnn.graph.visualization.positions import (
    FAN_HEIGHT,
    MARKER_RATIO,
    STUB_LENGTH,
    address_radius,
    lay_out,
    layout_margin,
    spring_layout,
)

from .conftest import SQUARE

UNIT_SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def test_spring_layout_shape_and_scale():
    pos = spring_layout(5, np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))
    assert pos.shape == (5, 2) and np.abs(pos).max() <= 1.0 + 1e-6
    assert spring_layout(3, np.zeros((0, 2), dtype=int)).shape == (3, 2)


def test_spring_layout_places_addresses_and_hubs(mixed_order_graph):
    layout = lay_out(read_graph(mixed_order_graph), iterations=10)
    assert layout.addresses.shape == (4, 2) and np.abs(layout.addresses).max() <= 1.0 + 1e-6
    assert list(layout.hubs) == [("trafo3w", 0)]  # one hub for the order-3 object, laid out with the addresses


def test_given_positions_are_fitted_to_the_box(mixed_order_graph):
    layout = lay_out(read_graph(mixed_order_graph), address_positions=SQUARE)
    np.testing.assert_allclose(layout.addresses, UNIT_SQUARE, atol=1e-6)
    np.testing.assert_allclose(layout.hubs[("trafo3w", 0)], UNIT_SQUARE[[0, 1, 2]].mean(axis=0), atol=1e-6)  # barycenter


def test_given_positions_padded_length_and_nan(mixed_order_graph, padded_shape):
    mixed_order_graph.pad(padded_shape)
    assert lay_out(read_graph(mixed_order_graph), address_positions=np.arange(14.0).reshape(7, 2)).addresses.shape == (4, 2)
    with pytest.raises(ValueError, match="NaN"):
        lay_out(read_graph(mixed_order_graph), address_positions=np.full((4, 2), np.nan))


def test_hub_with_all_ports_on_one_address_is_offset_like_a_stub():
    from energnn.graph.graph import Graph
    from energnn.graph.hyper_edge_set import HyperEdgeSet

    hes = {
        "line": HyperEdgeSet.from_dict(port_dict={"from": np.array([0]), "to": np.array([1])}, feature_dict=None),
        "t3": HyperEdgeSet.from_dict(port_dict={k: np.array([1]) for k in ("a", "b", "c")}, feature_dict=None),
    }
    topology = read_graph(Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=2))
    for layout in (lay_out(topology, iterations=10), lay_out(topology, address_positions=np.array([[0.0, 0.0], [1.0, 0.0]]))):
        offset = np.linalg.norm(layout.hubs[("t3", 0)] - layout.addresses[1])
        assert offset == pytest.approx(STUB_LENGTH * address_radius(2))


def test_placed_objects_derive_the_addresses_and_are_pushed_apart(located_graph):
    topology = read_graph(located_graph, hyper_edge_positions={"bus": ["x", "y"]})
    with pytest.raises(ValueError, match=r"addresses \[3\] are pointed to by no placed object"):
        lay_out(topology)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    layout = lay_out(topology, address_positions=square)  # given addresses win, the buses keep their features
    scale = layout.addresses[1, 0] - layout.addresses[0, 0]
    np.testing.assert_allclose(layout.addresses - layout.addresses[0], scale * square, atol=1e-9)
    buses = np.array([layout.hubs[("bus", i)] for i in range(3)])
    np.testing.assert_allclose(buses - layout.addresses[0], scale * np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]), atol=1e-9)


def test_placed_objects_alone_place_every_address():
    from energnn.graph.graph import Graph
    from energnn.graph.hyper_edge_set import HyperEdgeSet

    hes = {
        "bus": HyperEdgeSet.from_dict(
            port_dict={"id": np.array([0, 1, 2])},
            feature_dict={"x": np.array([0.0, 4.0, 0.0]), "y": np.array([0.0, 0.0, 3.0])},
        ),
        "line": HyperEdgeSet.from_dict(port_dict={"from": np.array([0, 1]), "to": np.array([1, 2])}, feature_dict=None),
    }
    topology = read_graph(Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=3), hyper_edge_positions={"bus": ["x", "y"]})
    layout = lay_out(topology)
    buses = np.array([layout.hubs[("bus", i)] for i in range(3)])
    scale = (buses[1, 0] - buses[0, 0]) / 4.0
    assert scale > 0
    np.testing.assert_allclose(buses - buses[0], scale * np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]), atol=1e-9)
    # each address is pushed off the bus that places it, like a stub, so both stay visible
    r_addr = address_radius(3)
    for i in range(3):
        assert np.linalg.norm(layout.addresses[i] - buses[i]) == pytest.approx(STUB_LENGTH * r_addr)
    assert np.abs(layout.addresses).max() <= 1.0 + layout.margin


def test_address_radius_shrinks_with_the_number_of_addresses():
    assert address_radius(1) == address_radius(4) > address_radius(400) > address_radius(10_000) > 0


def test_layout_margin_covers_stubs_loops_and_fans(mixed_order_graph, multi_graph, portless_graph):
    r_addr = address_radius(4)
    assert layout_margin(read_graph(mixed_order_graph)) == pytest.approx((STUB_LENGTH + MARKER_RATIO) * r_addr)
    assert layout_margin(read_graph(multi_graph)) >= max(FAN_HEIGHT, (STUB_LENGTH + MARKER_RATIO) * address_radius(3))
    assert layout_margin(read_graph(portless_graph)) == pytest.approx(MARKER_RATIO * address_radius(3))
