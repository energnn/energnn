# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import json

import numpy as np
import pytest

from energnn.graph.graph import collate_graphs
from energnn.graph.visualization import InteractiveGraphPlot, plot_graph_interactive

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def _payload(plot: InteractiveGraphPlot) -> dict:
    fragment = plot._repr_html_()
    start = fragment.index('<script type="application/json">') + len('<script type="application/json">')
    return json.loads(fragment[start : fragment.index("</script>", start)])


def test_interactive_plot_content(mixed_order_graph):
    plot = plot_graph_interactive(mixed_order_graph)
    assert isinstance(plot, InteractiveGraphPlot)
    fragment = plot._repr_html_()
    for expected in ["line", "gen", "trafo3w", "addresses", "energnn-plot-"]:
        assert expected in fragment
    payload = _payload(plot)
    assert payload["nAddr"] == 4 and payload["ndim"] == 2 and len(payload["frames"]) == 1
    assert [c["name"] for c in payload["classes"]] == ["gen", "line", "trafo3w"]
    # one descriptor per real hyper-edge, with tooltips holding ports and feature values
    assert sum(len(c["objects"]) for c in payload["classes"]) == 6
    trafo = payload["classes"][2]["objects"][0]
    assert trafo["kind"] == "hub" and "hv" in trafo["tip"] and "1.02" in trafo["tip"]
    assert payload["addrTips"][0].startswith("&lt;b&gt;address 0")
    assert payload["logo"].startswith("data:image/png;base64,")


def test_interactive_plot_skips_fictitious(mixed_order_graph, padded_shape):
    mixed_order_graph.pad(padded_shape)
    payload = _payload(plot_graph_interactive(mixed_order_graph))
    assert payload["nAddr"] == 4
    assert sum(len(c["objects"]) for c in payload["classes"]) == 6


def test_interactive_plot_rejects_batch(mixed_order_graph):
    batch = collate_graphs([mixed_order_graph, mixed_order_graph])
    with pytest.raises(ValueError, match="single"):
        plot_graph_interactive(batch)


def test_interactive_plot_themes(mixed_order_graph):
    auto = plot_graph_interactive(mixed_order_graph, theme="auto")._repr_html_()
    assert "prefers-color-scheme: dark" in auto
    dark = plot_graph_interactive(mixed_order_graph, theme="dark")._repr_html_()
    assert "prefers-color-scheme" not in dark and "--surface:#1a1a19" in dark and "--s0:" in dark and "--b11:" in dark
    with pytest.raises(ValueError, match="theme"):
        plot_graph_interactive(mixed_order_graph, theme="solarized")


def test_interactive_plot_unique_ids(mixed_order_graph):
    a = plot_graph_interactive(mixed_order_graph)._repr_html_()
    b = plot_graph_interactive(mixed_order_graph)._repr_html_()
    assert a.split('id="')[1].split('"')[0] != b.split('id="')[1].split('"')[0]


def test_interactive_edge_colors_off(mixed_order_graph):
    payload = _payload(plot_graph_interactive(mixed_order_graph, edge_colors=False))
    assert all(c["color"] is None for c in payload["classes"])
    colored = _payload(plot_graph_interactive(mixed_order_graph))
    assert [c["color"] for c in colored["classes"]] == ["var(--c0)", "var(--c1)", "var(--c2)"]


def test_interactive_logo_off(mixed_order_graph):
    assert _payload(plot_graph_interactive(mixed_order_graph, logo=False))["logo"] is None


def test_interactive_portless_class(portless_graph):
    payload = _payload(plot_graph_interactive(portless_graph))
    kinds = {c["name"]: [o["kind"] for o in c["objects"]] for c in payload["classes"]}
    assert kinds == {"bus": ["none"] * 3, "line": ["pair", "pair"]}


def test_interactive_multi_graph(multi_graph):
    payload = _payload(plot_graph_interactive(multi_graph))
    kinds = [o["kind"] for c in payload["classes"] for o in c["objects"]]
    assert sorted(kinds) == ["hub", "hub", "loop", "pair", "pair", "pair"]


def test_interactive_3d_and_frames(mixed_order_graph):
    frames = np.stack([np.concatenate([SQUARE, np.arange(4)[:, None]], axis=1)] * 3)
    plot = plot_graph_interactive(mixed_order_graph, positions=frames, interval=50)
    payload = _payload(plot)
    assert payload["ndim"] == 3 and len(payload["frames"]) == 3 and len(payload["frames"][0]) == 5
    assert payload["interval"] == 50
    fragment = plot._repr_html_()
    assert 'type="range" min="0" max="2"' in fragment and "drag to rotate" in fragment
    static = plot_graph_interactive(mixed_order_graph, positions=SQUARE)._repr_html_()
    assert 'type="range"' not in static and "drag to rotate" not in static


@pytest.mark.parametrize(
    "n_channels, scale", [(1, "0<!--scale-->3"), (2, "ch1 0&ndash;3<!--scale-->ch2 0&ndash;3"), (3, "RGB")]
)
def test_interactive_address_colors(mixed_order_graph, n_channels, scale):
    colors = np.arange(4, dtype=float)[:, None].repeat(n_channels, axis=1)
    plot = plot_graph_interactive(mixed_order_graph, address_colors=colors)
    payload = _payload(plot)
    assert np.asarray(payload["colors"]).shape == (1, 4, n_channels)
    assert payload["colors"][0][0] == [0.0] * n_channels and payload["colors"][0][3] == [1.0] * n_channels
    assert scale in plot._repr_html_()


def test_injected_positions_padded_length(mixed_order_graph, padded_shape):
    mixed_order_graph.pad(padded_shape)
    positions = np.arange(14, dtype=float).reshape(7, 2)  # padded length: extra rows dropped
    assert _payload(plot_graph_interactive(mixed_order_graph, positions=positions))["nAddr"] == 4


def test_interactive_plot_save(mixed_order_graph, tmp_path):
    path = str(tmp_path / "graph.html")
    plot_graph_interactive(mixed_order_graph).save(path)
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    assert content.startswith("<!DOCTYPE html>")
    assert "trafo3w" in content
