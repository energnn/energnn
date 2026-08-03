# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from energnn.graph.graph import Graph, collate_graphs  # noqa: E402
from energnn.graph.hyper_edge_set import HyperEdgeSet  # noqa: E402
from energnn.graph.plot import plot_graph, spring_layout  # noqa: E402
from energnn.graph.shape import GraphShape  # noqa: E402


@pytest.fixture
def mixed_order_graph() -> Graph:
    hes = {
        "line": HyperEdgeSet.from_dict(
            port_dict={"from": np.array([0, 1, 2]), "to": np.array([1, 2, 3])},
            feature_dict={"x": np.array([0.1, 0.2, 0.3])},
        ),
        "gen": HyperEdgeSet.from_dict(
            port_dict={"bus": np.array([0, 3])},
            feature_dict={"p": np.array([1.0, 2.0])},
        ),
        "trafo3w": HyperEdgeSet.from_dict(
            port_dict={"hv": np.array([0]), "mv": np.array([1]), "lv": np.array([2])},
            feature_dict={"ratio": np.array([1.02])},
        ),
    }
    return Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=4)


def teardown_function() -> None:
    plt.close("all")


def test_spring_layout_shape_and_scale():
    pos = spring_layout(5, np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))
    assert pos.shape == (5, 2)
    assert np.abs(pos).max() <= 1.0 + 1e-6


def test_spring_layout_no_edges():
    pos = spring_layout(3, np.zeros((0, 2), dtype=int))
    assert pos.shape == (3, 2)


def test_plot_graph_returns_axes(mixed_order_graph):
    ax = plot_graph(mixed_order_graph)
    # one gray collection for addresses + one per hyper-edge class
    assert len(ax.collections) == 1 + len(mixed_order_graph.hyper_edge_sets)
    labels = [artist.get_label() for artist in ax.collections]
    assert labels == ["addresses", "gen", "line", "trafo3w"]


def test_plot_graph_into_existing_axes(mixed_order_graph):
    _, ax = plt.subplots()
    assert plot_graph(mixed_order_graph, ax=ax) is ax


def test_plot_graph_skips_fictitious(mixed_order_graph):
    ax_ref = plot_graph(mixed_order_graph)
    n_points_ref = [len(c.get_offsets()) for c in ax_ref.collections]

    target = GraphShape(
        hyper_edge_sets={"line": np.array(6), "gen": np.array(5), "trafo3w": np.array(2)},
        addresses=np.array(7),
    )
    mixed_order_graph.pad(target)
    ax_padded = plot_graph(mixed_order_graph)
    n_points_padded = [len(c.get_offsets()) for c in ax_padded.collections]
    assert n_points_padded == n_points_ref


def test_plot_graph_rejects_batch(mixed_order_graph):
    batch = collate_graphs([mixed_order_graph, mixed_order_graph])
    with pytest.raises(ValueError, match="single"):
        plot_graph(batch)


def test_plot_graph_dark_theme(mixed_order_graph):
    ax = plot_graph(mixed_order_graph, theme="dark")
    assert ax.get_facecolor() == matplotlib.colors.to_rgba("#1a1a19")
    assert ax.figure.get_facecolor() == matplotlib.colors.to_rgba("#1a1a19")


def test_plot_graph_auto_theme_follows_rcparams(mixed_order_graph):
    with matplotlib.rc_context({"figure.facecolor": "#2b2b2b"}):
        ax = plot_graph(mixed_order_graph)
    assert ax.get_facecolor() == matplotlib.colors.to_rgba("#1a1a19")


def test_plot_graph_invalid_theme(mixed_order_graph):
    with pytest.raises(ValueError, match="theme"):
        plot_graph(mixed_order_graph, theme="solarized")
