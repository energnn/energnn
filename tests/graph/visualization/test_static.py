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
from matplotlib.animation import FuncAnimation  # noqa: E402

from energnn.graph.graph import collate_graphs  # noqa: E402
from energnn.graph.visualization import animate_graph, plot_graph  # noqa: E402
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
    drawn = np.asarray(_collection(plot_graph(mixed_order_graph, positions=SQUARE), "addresses").get_offsets())
    expected = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(drawn, expected, atol=1e-6)


def test_logo_artist(mixed_order_graph):
    from matplotlib.offsetbox import AnnotationBbox

    with_logo = plot_graph(mixed_order_graph)
    assert sum(isinstance(a, AnnotationBbox) for a in with_logo.artists) == 1
    without = plot_graph(mixed_order_graph, logo=False)
    assert not any(isinstance(a, AnnotationBbox) for a in without.artists)
    in_3d = plot_graph(mixed_order_graph, positions=np.concatenate([SQUARE, np.arange(4)[:, None]], axis=1))
    assert sum(isinstance(a, AnnotationBbox) for a in in_3d.artists) == 1
    in_3d.figure.canvas.draw()  # the artist must render on 3D axes too


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------


def test_plot_graph_3d_creates_3d_axes(mixed_order_graph):
    positions = np.concatenate([SQUARE, [[0.0], [5.0], [10.0], [5.0]]], axis=1)
    ax = plot_graph(mixed_order_graph, positions=positions, port_labels=True)
    assert hasattr(ax, "zaxis")
    assert [c.get_label() for c in ax.collections] == ["addresses", "gen", "line", "trafo3w"]
    assert len(ax.lines) == 3  # one line artist per class
    assert len(ax.texts) >= 4 + 6  # address numbers + port labels, as 3D texts


def test_plot_graph_3d_rejects_2d_axes(mixed_order_graph):
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="projection='3d'"):
        plot_graph(mixed_order_graph, positions=np.zeros((4, 3)) + np.arange(4)[:, None], ax=ax)


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
# Frames and animation
# ---------------------------------------------------------------------------


def test_frame_selects_time_step(mixed_order_graph):
    frames = np.stack([SQUARE, SQUARE[::-1]])
    first = _collection(plot_graph(mixed_order_graph, positions=frames, frame=0), "addresses").get_offsets()
    second = _collection(plot_graph(mixed_order_graph, positions=frames, frame=1), "addresses").get_offsets()
    np.testing.assert_allclose(np.asarray(first), np.asarray(second)[::-1], atol=1e-6)
    with pytest.raises(IndexError, match="frame"):
        plot_graph(mixed_order_graph, positions=frames, frame=2)


def test_animate_graph(mixed_order_graph):
    frames = np.stack([SQUARE, SQUARE[::-1], SQUARE])
    colors = np.random.default_rng(0).random((3, 4, 1))
    animation = animate_graph(mixed_order_graph, positions=frames, address_colors=colors, interval=10)
    assert isinstance(animation, FuncAnimation)
    ax = animation._fig.axes[0]
    n_before = len(ax.collections)
    animation._func(1)  # draw the second frame in place
    assert len(ax.collections) == n_before
    assert len(animation._fig.axes) >= 2  # colorbar added once, not per frame


def test_animate_graph_needs_frames(mixed_order_graph):
    with pytest.raises(ValueError, match="time axis"):
        animate_graph(mixed_order_graph, positions=SQUARE)


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
    ax = plot_graph(mixed_order_graph, positions=positions, address_colors=colors, theme="light")
    addresses = _collection(ax, "addresses")
    styles = [ls for ls in addresses.get_linestyle()]
    assert styles[2] != styles[0]  # the inferred address has a dashed outline
    faces = addresses.get_facecolor()
    assert tuple(faces[3]) == _rgba(THEMES["light"].surface)  # missing color: hollow
    assert tuple(faces[0]) != _rgba(THEMES["light"].surface)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "address (position inferred)" in labels
