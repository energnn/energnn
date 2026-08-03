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
from energnn.graph.plot import InteractiveGraphPlot, plot_graph, plot_graph_interactive, spring_layout  # noqa: E402
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


def test_plot_graph_port_labels(mixed_order_graph):
    ax = plot_graph(mixed_order_graph, port_labels=True)
    texts = {t.get_text() for t in ax.texts}
    assert {"from", "to", "bus", "hv", "mv", "lv"} <= texts


def test_interactive_plot_content(mixed_order_graph):
    plot = plot_graph_interactive(mixed_order_graph)
    assert isinstance(plot, InteractiveGraphPlot)
    fragment = plot._repr_html_()
    # class names in the legend, address ids, port names, and feature values in tooltips
    for expected in ["line", "gen", "trafo3w", ">3</text>", "hv", "mv", "lv", "1.02", "address 0"]:
        assert expected in fragment
    assert fragment.count('class="addr"') == 4
    # one group per real hyper-edge: 3 lines + 2 gens + 1 trafo3w
    assert fragment.count('class="obj"') == 6


def test_interactive_plot_skips_fictitious(mixed_order_graph):
    target = GraphShape(
        hyper_edge_sets={"line": np.array(6), "gen": np.array(5), "trafo3w": np.array(2)},
        addresses=np.array(7),
    )
    mixed_order_graph.pad(target)
    fragment = plot_graph_interactive(mixed_order_graph)._repr_html_()
    assert fragment.count('class="addr"') == 4
    assert fragment.count('class="obj"') == 6


def test_interactive_plot_rejects_batch(mixed_order_graph):
    batch = collate_graphs([mixed_order_graph, mixed_order_graph])
    with pytest.raises(ValueError, match="single"):
        plot_graph_interactive(batch)


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


def test_parallel_edges_have_distinct_markers(multi_graph):
    ax = plot_graph(multi_graph)
    line_collection = next(c for c in ax.collections if c.get_label() == "line")
    offsets = np.asarray(line_collection.get_offsets())
    assert len(np.unique(np.round(offsets, 6), axis=0)) == 4


def test_self_loop_is_visible(multi_graph):
    ax = plot_graph(multi_graph)
    line_collection = next(c for c in ax.collections if c.get_label() == "line")
    loop_marker = np.asarray(line_collection.get_offsets())[3]
    address_positions = np.asarray(next(c for c in ax.collections if c.get_label() == "addresses").get_offsets())
    assert np.linalg.norm(address_positions - loop_marker, axis=1).min() > 0.05


def test_parallel_hubs_are_separated(multi_graph):
    ax = plot_graph(multi_graph)
    trafo_offsets = np.asarray(next(c for c in ax.collections if c.get_label() == "trafo3w").get_offsets())
    assert np.linalg.norm(trafo_offsets[0] - trafo_offsets[1]) > 0.01


def test_interactive_multi_graph(multi_graph):
    fragment = plot_graph_interactive(multi_graph)._repr_html_()
    assert fragment.count('class="obj"') == 6


def test_injected_positions(mixed_order_graph):
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    ax = plot_graph(mixed_order_graph, positions=positions)
    drawn = np.asarray(next(c for c in ax.collections if c.get_label() == "addresses").get_offsets())
    # normalized to [-1, 1] but the square must be preserved
    expected = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(drawn, expected, atol=1e-6)


def test_injected_positions_padded_length(mixed_order_graph):
    target = GraphShape(
        hyper_edge_sets={"line": np.array(6), "gen": np.array(5), "trafo3w": np.array(2)},
        addresses=np.array(7),
    )
    mixed_order_graph.pad(target)
    positions = np.arange(14, dtype=float).reshape(7, 2)  # padded length: extra rows dropped
    fragment = plot_graph_interactive(mixed_order_graph, positions=positions)._repr_html_()
    assert fragment.count('class="addr"') == 4


def test_injected_positions_bad_shape(mixed_order_graph):
    with pytest.raises(ValueError, match="positions"):
        plot_graph(mixed_order_graph, positions=np.zeros((3, 2)))


def test_interactive_plot_save(mixed_order_graph, tmp_path):
    path = str(tmp_path / "graph.html")
    plot_graph_interactive(mixed_order_graph).save(path)
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    assert content.startswith("<!DOCTYPE html>")
    assert "trafo3w" in content
