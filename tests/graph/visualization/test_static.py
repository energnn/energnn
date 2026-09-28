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


def teardown_function() -> None:
    plt.close("all")


def _collection(ax, label):
    return next(c for c in ax.collections if c.get_label() == label)


def test_plot_graph_returns_axes(mixed_order_graph):
    ax = plot_graph(mixed_order_graph)
    # one gray collection for addresses + one per hyper-edge class
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


def test_plot_graph_portless_class(portless_graph):
    labels = [c.get_label() for c in plot_graph(portless_graph).collections]
    assert labels == ["addresses", "line"]  # port-less buses draw nothing


def test_parallel_edges_have_distinct_markers(multi_graph):
    offsets = np.asarray(_collection(plot_graph(multi_graph), "line").get_offsets())
    assert len(np.unique(np.round(offsets, 6), axis=0)) == 4


def test_injected_positions(mixed_order_graph):
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    drawn = np.asarray(_collection(plot_graph(mixed_order_graph, positions=positions), "addresses").get_offsets())
    # normalized to [-1, 1] but the square must be preserved
    expected = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(drawn, expected, atol=1e-6)


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
