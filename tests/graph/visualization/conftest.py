# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.graph import Graph
from energnn.graph.hyper_edge_set import HyperEdgeSet
from energnn.graph.shape import GraphShape

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


@pytest.fixture
def mixed_order_graph() -> Graph:
    """Hyper-edges of order 2 (line), 1 (gen) and 3 (trafo3w) over 4 addresses."""
    hes = {
        "line": HyperEdgeSet.from_dict(
            port_dict={"from": np.array([0, 1, 2]), "to": np.array([1, 2, 3])},
            feature_dict={"x": np.array([0.1, 0.2, 0.3])},
        ),
        "gen": HyperEdgeSet.from_dict(port_dict={"bus": np.array([0, 3])}, feature_dict={"p": np.array([1.0, 2.0])}),
        "trafo3w": HyperEdgeSet.from_dict(
            port_dict={"hv": np.array([0]), "mv": np.array([1]), "lv": np.array([2])},
            feature_dict={"ratio": np.array([1.02])},
        ),
    }
    return Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=4)


@pytest.fixture
def portless_graph() -> Graph:
    """A decision-like graph: one class with features but no ports, plus one order-2 class."""
    hes = {
        "bus": HyperEdgeSet.from_dict(port_dict=None, feature_dict={"phase_angle": np.array([0.1, 0.2, 0.3])}),
        "line": HyperEdgeSet.from_dict(port_dict={"from": np.array([0, 1]), "to": np.array([1, 2])}, feature_dict=None),
    }
    return Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=3)


@pytest.fixture
def padded_shape() -> GraphShape:
    """A padding target larger than ``mixed_order_graph`` in every dimension."""
    return GraphShape(
        hyper_edge_sets={"line": np.array(6), "gen": np.array(5), "trafo3w": np.array(2)},
        addresses=np.array(7),
    )


@pytest.fixture
def multi_graph() -> Graph:
    """3 parallel lines 0-1, a self-loop on 2, and 2 parallel 3-port trafos."""
    hes = {
        "line": HyperEdgeSet.from_dict(
            port_dict={"from": np.array([0, 0, 0, 2]), "to": np.array([1, 1, 1, 2])},
            feature_dict={"x": np.array([0.1, 0.2, 0.3, 0.4])},
        ),
        "trafo3w": HyperEdgeSet.from_dict(
            port_dict={"hv": np.array([0, 0]), "mv": np.array([1, 1]), "lv": np.array([2, 2])},
            feature_dict={"ratio": np.array([1.0, 1.1])},
        ),
    }
    return Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=3)


@pytest.fixture
def located_graph() -> Graph:
    """Buses carrying their own (x, y) and a load, lines between them (the second line has a NaN flow, set after
    construction since from_dict rejects NaN), one port-less decision class, and address 3 pointed to by nothing."""
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
    graph = Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=4)
    graph.line.flow = np.array([10.0, np.nan])
    return graph
