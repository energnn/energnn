# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.visualization.content import read_graph
from energnn.graph.visualization.geometry import geometries
from energnn.graph.visualization.positions import STUB_LENGTH, address_radius, lay_out

from .conftest import SQUARE

TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def _geoms(graph, **kwargs):
    topology = read_graph(graph, hyper_edge_positions=kwargs.pop("hyper_edge_positions", None))
    layout = lay_out(topology, iterations=10, **kwargs)
    return topology, layout, geometries(topology, layout)


def test_one_geometry_per_object_with_ports(mixed_order_graph, portless_graph):
    _, _, geoms = _geoms(mixed_order_graph)
    assert set(geoms) == {("line", 0), ("line", 1), ("line", 2), ("gen", 0), ("gen", 1), ("trafo3w", 0)}
    assert len(geoms[("trafo3w", 0)].lines) == 3 and len(geoms[("line", 0)].labels) == 2  # one spoke per port
    assert all(line.shape[1] == 2 for g in geoms.values() for line in g.lines)
    assert set(_geoms(portless_graph)[2]) == {("line", 0), ("line", 1)}  # port-less objects draw nothing


def test_stubs_keep_clear_of_the_address_circle(mixed_order_graph):
    topology, layout, geoms = _geoms(mixed_order_graph)
    r_addr = address_radius(4)
    for h in topology.of("gen"):
        assert np.linalg.norm(geoms[h.key].marker - layout.addresses[h.ports[0]]) == pytest.approx(STUB_LENGTH * r_addr)
    assert all(np.abs(g.marker).max() <= 1.0 + layout.margin for g in geoms.values())


def test_parallel_pairs_are_fanned_and_loops_drawn_as_two_spokes(multi_graph):
    _, layout, geoms = _geoms(multi_graph, address_positions=TRIANGLE)
    markers = np.array([geoms[("line", i)].marker for i in range(3)])
    assert len(np.unique(np.round(markers, 6), axis=0)) == 3  # three distinct curves between 0 and 1
    assert np.allclose(
        geoms[("line", 1)].marker, (layout.addresses[0] + layout.addresses[1]) / 2
    )  # the middle one is straight
    loop = geoms[("line", 3)]
    assert np.linalg.norm(loop.marker - layout.addresses[2]) == pytest.approx(STUB_LENGTH * address_radius(3))
    assert len(loop.lines) == 2 and all(len(line) == 17 for line in loop.lines)
    for line in loop.lines:  # both spokes run from the address center to the marker
        np.testing.assert_allclose(line[0], layout.addresses[2], atol=1e-9)
        np.testing.assert_allclose(line[-1], loop.marker, atol=1e-9)
    assert len(np.unique(np.round(loop.labels, 6), axis=0)) == 2


def test_hubs_with_repeated_ports_fan_their_spokes():
    from energnn.graph.graph import Graph
    from energnn.graph.hyper_edge_set import HyperEdgeSet

    hes = {
        "t4": HyperEdgeSet.from_dict(
            port_dict={"a": np.array([0]), "b": np.array([0]), "c": np.array([1]), "d": np.array([2])}, feature_dict=None
        )
    }
    _, layout, geoms = _geoms(Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=3), address_positions=TRIANGLE)
    geom = geoms[("t4", 0)]
    assert [len(line) for line in geom.lines] == [17, 17, 2, 2]  # a, b fanned to address 0; c, d straight
    assert len(np.unique(np.round(geom.labels, 6), axis=0)) == 4
    np.testing.assert_allclose(geom.marker, layout.addresses.mean(axis=0), atol=1e-6)


def test_placed_objects_are_hubs_at_their_position(located_graph):
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    topology, layout, geoms = _geoms(
        located_graph, address_positions=square, hyper_edge_positions={"bus": ["x", "y"], "theta": ["value", "value"]}
    )
    for h in topology.of("bus"):
        np.testing.assert_allclose(geoms[h.key].marker, layout.hubs[h.key])
        assert len(geoms[h.key].lines) == 1  # one spoke, to the bus's address
        np.testing.assert_allclose(geoms[h.key].lines[0][-1], layout.addresses[h.ports[0]])
    assert all(geoms[h.key].lines == [] for h in topology.of("theta"))  # a lone marker for a placed port-less object


def test_geometries_follow_given_positions(mixed_order_graph):
    _, _, first = _geoms(mixed_order_graph, address_positions=SQUARE)
    _, _, second = _geoms(mixed_order_graph, address_positions=SQUARE[::-1])
    assert not np.allclose(first[("line", 0)].marker, second[("line", 0)].marker)
