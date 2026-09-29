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
from matplotlib.collections import LineCollection  # noqa: E402

from energnn.graph.graph import collate_graphs  # noqa: E402
from energnn.graph.visualization import plot_graph  # noqa: E402
from energnn.graph.visualization.theme import THEMES  # noqa: E402

from .conftest import SQUARE  # noqa: E402


def teardown_function() -> None:
    plt.close("all")


def _collection(ax, label):
    return next(c for c in ax.collections if c.get_label() == label)


def _rgba(color):
    return matplotlib.colors.to_rgba(color)


def test_plot_graph_returns_axes_with_one_artist_per_class(mixed_order_graph):
    ax = plot_graph(mixed_order_graph)
    labels = [artist.get_label() for artist in ax.collections]
    assert labels[1:] == ["gen", "line", "trafo3w"]  # the addresses' scatter, then one scatter per class
    assert len(ax.lines) == 3  # one line artist per class
    _, other = plt.subplots()
    assert plot_graph(mixed_order_graph, ax=other) is other


def test_plot_graph_skips_fictitious(mixed_order_graph, padded_shape):
    n_points_ref = [len(c.get_offsets()) for c in plot_graph(mixed_order_graph).collections]
    mixed_order_graph.pad(padded_shape)
    assert [len(c.get_offsets()) for c in plot_graph(mixed_order_graph).collections] == n_points_ref


def test_plot_graph_rejects_batch(mixed_order_graph):
    with pytest.raises(ValueError, match="single"):
        plot_graph(collate_graphs([mixed_order_graph, mixed_order_graph]))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_addresses_are_hollow_circles_in_ink(mixed_order_graph, theme):
    ax = plot_graph(mixed_order_graph, theme=theme)
    addresses = ax.collections[0]
    assert ax.get_facecolor() == _rgba(THEMES[theme].surface) and ax.figure.get_facecolor() == _rgba(THEMES[theme].surface)
    assert tuple(addresses.get_facecolor()[0]) == _rgba(THEMES[theme].surface)
    assert tuple(addresses.get_edgecolor()[0]) == _rgba(THEMES[theme].ink)
    numbers = [t for t in ax.texts if t.get_text().isdigit()]
    assert len(numbers) == 4 and all(_rgba(t.get_color()) == _rgba(THEMES[theme].ink) for t in numbers)


def test_plot_graph_auto_theme_follows_rcparams(mixed_order_graph):
    with matplotlib.rc_context({"figure.facecolor": "#2b2b2b"}):
        ax = plot_graph(mixed_order_graph)
    assert ax.get_facecolor() == _rgba(THEMES["dark"].surface)
    with pytest.raises(ValueError, match="theme"):
        plot_graph(mixed_order_graph, theme="solarized")


def test_plot_graph_port_labels(mixed_order_graph):
    texts = {t.get_text() for t in plot_graph(mixed_order_graph, port_labels=True).texts}
    assert {"from", "to", "bus", "hv", "mv", "lv"} <= texts


def test_edge_colors_off_uses_neutral(mixed_order_graph):
    ax = plot_graph(mixed_order_graph, edge_colors=False, theme="light")
    neutral = _rgba(THEMES["light"].neutral)
    assert all(tuple(_collection(ax, name).get_facecolor()[0]) == neutral for name in ("gen", "line", "trafo3w"))
    assert all(line.get_color() == THEMES["light"].neutral for line in ax.lines)
    assert tuple(_collection(plot_graph(mixed_order_graph, theme="light"), "gen").get_facecolor()[0]) != neutral


def test_portless_class_draws_nothing(portless_graph):
    assert [c.get_label() for c in plot_graph(portless_graph).collections][1:] == ["line"]


def test_parallel_edges_have_distinct_markers(multi_graph):
    offsets = np.asarray(_collection(plot_graph(multi_graph), "line").get_offsets())
    assert len(np.unique(np.round(offsets, 6), axis=0)) == 4


def test_given_positions(mixed_order_graph):
    drawn = np.asarray(plot_graph(mixed_order_graph, address_positions=SQUARE).collections[0].get_offsets())
    np.testing.assert_allclose(drawn, [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], atol=1e-6)


def test_logo_artist(mixed_order_graph):
    from matplotlib.offsetbox import AnnotationBbox

    assert sum(isinstance(a, AnnotationBbox) for a in plot_graph(mixed_order_graph).artists) == 1
    assert not any(isinstance(a, AnnotationBbox) for a in plot_graph(mixed_order_graph, logo=False).artists)


def test_address_colors_fill_and_add_a_legend(mixed_order_graph):
    ax = plot_graph(mixed_order_graph, address_colors=np.array([[0.0], [1.0], [2.0], [np.nan]]), theme="light")
    faces = ax.collections[0].get_facecolor()
    assert tuple(faces[0]) == _rgba(THEMES["light"].sequential[0])
    assert tuple(faces[2]) == _rgba(THEMES["light"].sequential[-1])
    assert tuple(faces[3]) == _rgba(THEMES["light"].surface)  # missing color: hollow
    assert "addresses" in [a.get_ylabel() for a in ax.figure.axes]  # the colorbar
    two = plot_graph(
        mixed_order_graph, address_colors=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), theme="light"
    )
    assert [tuple(f) for f in two.collections[0].get_facecolor()] == [_rgba(c) for c in THEMES["light"].bivariate]
    assert any(inset.images and inset.get_title() == "addresses" for inset in two.child_axes)


def test_hyper_edge_positions_place_the_markers(located_graph):
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    ax = plot_graph(located_graph, address_positions=square, hyper_edge_positions={"bus": ["x", "y"]}, theme="light")
    buses = np.asarray(_collection(ax, "bus").get_offsets())
    addresses = np.asarray(ax.collections[0].get_offsets())
    scale = addresses[1, 0] - addresses[0, 0]
    np.testing.assert_allclose(buses - addresses[0], scale * np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]), atol=1e-6)


def test_hyper_edge_colors_color_per_object_and_turn_the_other_classes_neutral(located_graph):
    ax = plot_graph(located_graph, hyper_edge_colors={"bus": ["load"]}, theme="light")
    neutral = _rgba(THEMES["light"].neutral)
    assert len(np.unique(_collection(ax, "bus").get_facecolor(), axis=0)) == 3  # the colored class shows its colormap
    assert tuple(_collection(ax, "line").get_facecolor()[0]) == neutral  # the other class loses its class color
    assert sum(isinstance(c, LineCollection) for c in ax.collections) == 2  # bus and line: per-object colored lines
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["addresses", "bus", "line", "theta"]
    assert all(_rgba(h.get_markerfacecolor()) == neutral for h in legend.legend_handles[1:])
    labels = [a.get_ylabel() for a in ax.figure.axes]
    assert "hyper-edges" in labels and "addresses" not in labels
    both = plot_graph(located_graph, address_colors=np.arange(4.0)[:, None], hyper_edge_colors={"line": ["flow"]})
    assert {"hyper-edges", "addresses"} <= {a.get_ylabel() for a in both.figure.axes}


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
