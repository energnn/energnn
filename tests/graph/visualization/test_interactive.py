# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.graph import collate_graphs
from energnn.graph.visualization import InteractiveGraphPlot, plot_graph_interactive


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


def test_interactive_plot_skips_fictitious(mixed_order_graph, padded_shape):
    mixed_order_graph.pad(padded_shape)
    fragment = plot_graph_interactive(mixed_order_graph)._repr_html_()
    assert fragment.count('class="addr"') == 4
    assert fragment.count('class="obj"') == 6


def test_interactive_plot_rejects_batch(mixed_order_graph):
    batch = collate_graphs([mixed_order_graph, mixed_order_graph])
    with pytest.raises(ValueError, match="single"):
        plot_graph_interactive(batch)


def test_interactive_plot_themes(mixed_order_graph):
    auto = plot_graph_interactive(mixed_order_graph, theme="auto")._repr_html_()
    assert "prefers-color-scheme: dark" in auto
    dark = plot_graph_interactive(mixed_order_graph, theme="dark")._repr_html_()
    assert "prefers-color-scheme" not in dark and "--surface:#1a1a19" in dark
    with pytest.raises(ValueError, match="theme"):
        plot_graph_interactive(mixed_order_graph, theme="solarized")


def test_interactive_plot_unique_ids(mixed_order_graph):
    a = plot_graph_interactive(mixed_order_graph)._repr_html_()
    b = plot_graph_interactive(mixed_order_graph)._repr_html_()
    assert a.split('id="')[1].split('"')[0] != b.split('id="')[1].split('"')[0]


def test_interactive_portless_class(portless_graph):
    fragment = plot_graph_interactive(portless_graph)._repr_html_()
    assert fragment.count('class="obj"') == 2  # the two lines; port-less buses are listed in the legend only
    assert "bus" in fragment


def test_interactive_multi_graph(multi_graph):
    fragment = plot_graph_interactive(multi_graph)._repr_html_()
    assert fragment.count('class="obj"') == 6


def test_injected_positions_padded_length(mixed_order_graph, padded_shape):
    mixed_order_graph.pad(padded_shape)
    positions = np.arange(14, dtype=float).reshape(7, 2)  # padded length: extra rows dropped
    fragment = plot_graph_interactive(mixed_order_graph, positions=positions)._repr_html_()
    assert fragment.count('class="addr"') == 4


def test_interactive_plot_save(mixed_order_graph, tmp_path):
    path = str(tmp_path / "graph.html")
    plot_graph_interactive(mixed_order_graph).save(path)
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    assert content.startswith("<!DOCTYPE html>")
    assert "trafo3w" in content
