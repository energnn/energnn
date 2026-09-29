# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="plot_graph needs the 'viz' extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from energnn.graph.graph import collate_graphs  # noqa: E402
from energnn.graph.visualization import plot_graph  # noqa: E402
from energnn.graph.visualization.theme import THEMES  # noqa: E402

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def teardown_function() -> None:
    plt.close("all")


def _collection(ax, label):
    return next(c for c in ax.collections if c.get_label() == label)


def _rgba(color):
    return matplotlib.colors.to_rgba(color)


def test_plot_graph_returns_axes(mixed_order_graph):
    ax = plot_graph(mixed_order_graph)
    # one collection for addresses + one per hyper-edge class
    assert len(ax.collections) == 1 + len(mixed_order_graph.hyper_edge_sets)
    labels = [artist.get_label() for artist in ax.collections]
    assert labels == ["addresses", "gen", "line", "trafo3w"]


def test_plot_graph_into_existing_axes(mixed_order_graph):
    _, ax = plt.subplots()
    assert plot_graph(mixed_order_graph, ax=ax) is ax


def test_plot_graph_skips_fictitious(mixed_order_graph, padded_shape):
    n_points_ref = [len(c.get_offsets()) for c in plot_graph(mixed_order_graph).collections]
    mixed_order_graph.pad(padded_shape)
    n_points_padded = [len(c.get_offsets()) for c in plot_graph(mixed_order_graph).collections]
    assert n_points_padded == n_points_ref


def test_plot_graph_rejects_batch(mixed_order_graph):
    batch = collate_graphs([mixed_order_graph, mixed_order_graph])
    with pytest.raises(ValueError, match="single"):
        plot_graph(batch)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_addresses_are_hollow_circles_in_ink(mixed_order_graph, theme):
    ax = plot_graph(mixed_order_graph, theme=theme)
    addresses = _collection(ax, "addresses")
    assert ax.get_facecolor() == _rgba(THEMES[theme].surface)
    assert ax.figure.get_facecolor() == _rgba(THEMES[theme].surface)
    assert tuple(addresses.get_facecolor()[0]) == _rgba(THEMES[theme].surface)
    assert tuple(addresses.get_edgecolor()[0]) == _rgba(THEMES[theme].ink)
    numbers = [t for t in ax.texts if t.get_text().isdigit()]
    assert len(numbers) == 4 and all(_rgba(t.get_color()) == _rgba(THEMES[theme].ink) for t in numbers)


def test_plot_graph_auto_theme_follows_rcparams(mixed_order_graph):
    with matplotlib.rc_context({"figure.facecolor": "#2b2b2b"}):
        ax = plot_graph(mixed_order_graph)
    assert ax.get_facecolor() == _rgba(THEMES["dark"].surface)


def test_plot_graph_invalid_theme(mixed_order_graph):
    with pytest.raises(ValueError, match="theme"):
        plot_graph(mixed_order_graph, theme="solarized")


def test_plot_graph_port_labels(mixed_order_graph):
    ax = plot_graph(mixed_order_graph, port_labels=True)
    texts = {t.get_text() for t in ax.texts}
    assert {"from", "to", "bus", "hv", "mv", "lv"} <= texts


def test_edge_colors_off_uses_neutral(mixed_order_graph):
    ax = plot_graph(mixed_order_graph, edge_colors=False, theme="light")
    neutral = _rgba(THEMES["light"].neutral)
    assert all(tuple(_collection(ax, name).get_facecolor()[0]) == neutral for name in ("gen", "line", "trafo3w"))
    assert all(line.get_color() == THEMES["light"].neutral for line in ax.lines)
    colored = plot_graph(mixed_order_graph, theme="light")
    assert tuple(_collection(colored, "gen").get_facecolor()[0]) != neutral


def test_plot_graph_portless_class(portless_graph):
    labels = [c.get_label() for c in plot_graph(portless_graph).collections]
    assert labels == ["addresses", "line"]  # port-less buses draw nothing


def test_parallel_edges_have_distinct_markers(multi_graph):
    offsets = np.asarray(_collection(plot_graph(multi_graph), "line").get_offsets())
    assert len(np.unique(np.round(offsets, 6), axis=0)) == 4


def test_injected_positions(mixed_order_graph):
    drawn = np.asarray(_collection(plot_graph(mixed_order_graph, address_positions=SQUARE), "addresses").get_offsets())
    expected = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(drawn, expected, atol=1e-6)


def test_logo_artist(mixed_order_graph):
    from matplotlib.offsetbox import AnnotationBbox

    with_logo = plot_graph(mixed_order_graph)
    assert sum(isinstance(a, AnnotationBbox) for a in with_logo.artists) == 1
    without = plot_graph(mixed_order_graph, logo=False)
    assert not any(isinstance(a, AnnotationBbox) for a in without.artists)
    in_3d = plot_graph(mixed_order_graph, address_positions=np.concatenate([SQUARE, np.arange(4)[:, None]], axis=1))
    assert sum(isinstance(a, AnnotationBbox) for a in in_3d.artists) == 1
    in_3d.figure.canvas.draw()  # the artist must render on 3D axes too


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------


def test_plot_graph_3d_creates_3d_axes(mixed_order_graph):
    positions = np.concatenate([SQUARE, [[0.0], [5.0], [10.0], [5.0]]], axis=1)
    ax = plot_graph(mixed_order_graph, address_positions=positions, port_labels=True)
    assert hasattr(ax, "zaxis")
    assert [c.get_label() for c in ax.collections] == ["addresses", "gen", "line", "trafo3w"]
    assert len(ax.lines) == 3  # one line artist per class
    assert len(ax.texts) >= 4 + 6  # address numbers + port labels, as 3D texts


def test_plot_graph_3d_rejects_2d_axes(mixed_order_graph):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="projection='3d'"):
        plot_graph(mixed_order_graph, address_positions=np.zeros((4, 3)) + np.arange(4)[:, None], ax=ax)


# ---------------------------------------------------------------------------
# Address colors
# ---------------------------------------------------------------------------


def test_address_colors_one_channel_fills_and_adds_colorbar(mixed_order_graph):
    ax = plot_graph(mixed_order_graph, address_colors=np.array([[0.0], [1.0], [2.0], [3.0]]), theme="light")
    faces = _collection(ax, "addresses").get_facecolor()
    assert len(np.unique(faces, axis=0)) == 4
    assert tuple(faces[0]) == _rgba(THEMES["light"].sequential[0])
    assert tuple(faces[-1]) == _rgba(THEMES["light"].sequential[-1])
    assert len(ax.figure.axes) >= 2  # the colorbar axes was added


def test_address_colors_two_channels_adds_bivariate_legend(mixed_order_graph):
    ax = plot_graph(
        mixed_order_graph, address_colors=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), theme="light"
    )
    faces = [tuple(f) for f in _collection(ax, "addresses").get_facecolor()]
    assert faces == [_rgba(c) for c in THEMES["light"].bivariate]
    assert any(inset.images and inset.get_xlabel() == "channel 1" for inset in ax.child_axes)


def test_address_colors_three_channels_are_rgb(mixed_order_graph):
    ax = plot_graph(mixed_order_graph, address_colors=np.eye(4, 3), theme="light")
    faces = _collection(ax, "addresses").get_facecolor()
    np.testing.assert_allclose(faces[:3, :3], np.eye(3))


# ---------------------------------------------------------------------------
# Errors, inferred positions, feature-driven positions and colors
# ---------------------------------------------------------------------------


def test_plot_graph_import_error_mentions_extra(mixed_order_graph, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"energnn\[viz\]"):
        plot_graph(mixed_order_graph, theme="light")


def test_inferred_positions_and_missing_colors_are_drawn_distinctly(mixed_order_graph):
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [np.nan, np.nan], [0.0, 10.0]])
    colors = np.array([[0.0], [1.0], [2.0], [np.nan]])
    ax = plot_graph(mixed_order_graph, address_positions=positions, address_colors=colors, theme="light")
    addresses = _collection(ax, "addresses")
    styles = [ls for ls in addresses.get_linestyle()]
    assert styles[2] != styles[0]  # the inferred address has a dashed outline
    faces = addresses.get_facecolor()
    assert tuple(faces[3]) == _rgba(THEMES["light"].surface)  # missing color: hollow
    assert tuple(faces[0]) != _rgba(THEMES["light"].surface)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "address (position inferred)" in labels


def _located_graph():
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
    }
    graph = Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=3)
    graph.line.flow = np.array([10.0, np.nan])  # from_dict rejects NaN; the second flow is unknown
    return graph


def test_hyper_edge_positions_place_the_markers():
    ax = plot_graph(_located_graph(), hyper_edge_positions={"bus": ["x", "y"]}, theme="light")
    buses = np.asarray(_collection(ax, "bus").get_offsets())
    addresses = np.asarray(_collection(ax, "addresses").get_offsets())
    # the buses are drawn at their features (fitted to the box), each address a stub away from its bus
    np.testing.assert_allclose(buses, (np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]) - [4 / 3, 1.0]) / (8 / 3), atol=1e-6)
    assert np.all(np.linalg.norm(buses - addresses, axis=1) > 0.05)


def test_hyper_edge_colors_color_markers_and_lines_per_object():
    from matplotlib.collections import LineCollection

    ax = plot_graph(_located_graph(), hyper_edge_colors={"bus": ["load"], "line": ["flow"]}, theme="light")
    faces = _collection(ax, "bus").get_facecolor()
    assert len(np.unique(faces, axis=0)) == 3
    assert tuple(faces[0]) == _rgba(THEMES["light"].sequential[0])  # load 1 is the low end of the shared scale
    line_collections = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(line_collections) == 2  # colored classes use one collection each, with per-object colors
    lines = _collection(ax, "line")
    line_faces = lines.get_facecolor()
    assert tuple(line_faces[0]) == _rgba(THEMES["light"].sequential[-1])  # flow 10 is the high end
    assert tuple(line_faces[1]) == _rgba(THEMES["light"].neutral)  # NaN flow: the (now neutral) class color
    labels = [a.get_ylabel() for a in ax.figure.axes]
    assert "hyper-edges" in labels and "addresses" not in labels
    both = plot_graph(_located_graph(), address_colors=np.arange(3.0)[:, None], hyper_edge_colors={"bus": ["load"]})
    labels = [a.get_ylabel() for a in both.figure.axes]
    assert "hyper-edges" in labels and "addresses" in labels


def test_hyper_edge_colors_turn_the_other_classes_neutral():
    ax = plot_graph(_located_graph(), hyper_edge_colors={"bus": ["load"]}, theme="light")
    neutral = _rgba(THEMES["light"].neutral)
    assert tuple(_collection(ax, "line").get_facecolor()[0]) == neutral  # the uncolored class loses its class color
    assert all(line.get_color() == THEMES["light"].neutral for line in ax.lines)
    assert len(np.unique(_collection(ax, "bus").get_facecolor(), axis=0)) == 3  # the colored class keeps its colormap
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["addresses", "bus", "line"]
    assert all(_rgba(h.get_markerfacecolor()) == neutral for h in legend.legend_handles[1:])
