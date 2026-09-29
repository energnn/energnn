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
    assert payload["nAddr"] == 4 and payload["ndim"] == 2 and len(payload["pos"]) == 5
    assert [c["name"] for c in payload["classes"]] == ["gen", "line", "trafo3w"]
    # one descriptor per real hyper-edge, with tooltips holding ports and feature values
    assert sum(len(c["objects"]) for c in payload["classes"]) == 6
    trafo = payload["classes"][2]["objects"][0]
    assert trafo["kind"] == "hub" and "hv" in trafo["tip"] and "1.02" in trafo["tip"]
    assert payload["addrTips"][0] == "<b>address 0</b>"
    assert trafo["tip"].startswith("<b>trafo3w #0</b><br>hv &rarr; 0<br>")
    assert '<img class="logo" src="data:image/png;base64,' in fragment


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
    auto = plot_graph_interactive(mixed_order_graph, theme="auto")
    fragment = auto._repr_html_()
    assert ".dark{--surface:#1a1a19" in fragment and _payload(auto)["autoTheme"] is True
    dark = plot_graph_interactive(mixed_order_graph, theme="dark")
    fragment = dark._repr_html_()
    assert ".dark{" not in fragment and "--surface:#1a1a19" in fragment and "--s0:" in fragment and "--b11:" in fragment
    assert _payload(dark)["autoTheme"] is False
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
    assert 'class="logo"' not in plot_graph_interactive(mixed_order_graph, logo=False)._repr_html_()


def test_interactive_portless_class(portless_graph):
    payload = _payload(plot_graph_interactive(portless_graph))
    kinds = {c["name"]: [o["kind"] for o in c["objects"]] for c in payload["classes"]}
    assert kinds == {"bus": ["none"] * 3, "line": ["pair", "pair"]}


def test_interactive_multi_graph(multi_graph):
    payload = _payload(plot_graph_interactive(multi_graph))
    kinds = [o["kind"] for c in payload["classes"] for o in c["objects"]]
    assert sorted(kinds) == ["hub", "hub", "loop", "pair", "pair", "pair"]


def test_interactive_3d(mixed_order_graph):
    positions = np.concatenate([SQUARE, np.arange(4)[:, None]], axis=1)
    plot = plot_graph_interactive(mixed_order_graph, address_positions=positions)
    payload = _payload(plot)
    assert payload["ndim"] == 3 and len(payload["pos"]) == 5
    fragment = plot._repr_html_()
    assert "drag to rotate" in fragment
    assert 'data-mode="rotate"' in fragment and 'data-mode="pan"' in fragment and 'data-act="reset"' in fragment
    flat = plot_graph_interactive(mixed_order_graph, address_positions=SQUARE)._repr_html_()
    assert "drag to rotate" not in flat and 'data-mode="rotate"' not in flat and 'data-mode="pan"' in flat


@pytest.mark.parametrize(
    "n_channels, scale", [(1, "0<!--scale-->3"), (2, "ch1 0&ndash;3<!--scale-->ch2 0&ndash;3"), (3, "RGB")]
)
def test_interactive_address_colors(mixed_order_graph, n_channels, scale):
    colors = np.arange(4, dtype=float)[:, None].repeat(n_channels, axis=1)
    plot = plot_graph_interactive(mixed_order_graph, address_colors=colors)
    payload = _payload(plot)
    assert np.asarray(payload["colors"]).shape == (4, n_channels)
    assert payload["colors"][0] == [0.0] * n_channels and payload["colors"][3] == [1.0] * n_channels
    assert f'<span class="sc" data-ch="{n_channels}">addresses: {scale}</span>' in plot._repr_html_()
    assert all(o["color"] is None for c in payload["classes"] for o in c["objects"])


def test_injected_positions_padded_length(mixed_order_graph, padded_shape):
    mixed_order_graph.pad(padded_shape)
    positions = np.arange(14, dtype=float).reshape(7, 2)  # padded length: extra rows dropped
    assert _payload(plot_graph_interactive(mixed_order_graph, address_positions=positions))["nAddr"] == 4


def test_interactive_plot_save(mixed_order_graph, tmp_path):
    path = str(tmp_path / "graph.html")
    plot_graph_interactive(mixed_order_graph).save(path)
    with open(path, encoding="utf-8") as handle:
        content = handle.read()
    assert content.startswith("<!DOCTYPE html>")
    assert "trafo3w" in content


def test_interactive_degenerate_hubs(degenerate_hubs_graph):
    payload = _payload(plot_graph_interactive(degenerate_hubs_graph))
    kinds = {c["name"]: [o["kind"] for o in c["objects"]] for c in payload["classes"]}
    assert kinds["t3"] == ["hub", "hub"] and kinds["t4"] == ["hub"] and kinds["t5"] == ["hub"]
    hubs = {c["name"]: [o["hub"] for o in c["objects"]] for c in payload["classes"] if c["name"] != "line"}
    assert sorted(h for hs in hubs.values() for h in hs) == [3, 4, 5, 6]
    assert len(payload["pos"]) == 7  # 3 addresses + 4 hubs, all positioned by Python


def test_interactive_inferred_positions_and_missing_colors(mixed_order_graph):
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [np.nan, np.nan], [0.0, 10.0]])
    colors = np.array([[0.0], [1.0], [2.0], [np.nan]])
    plot = plot_graph_interactive(mixed_order_graph, address_positions=positions, address_colors=colors)
    payload = _payload(plot)
    assert payload["inferred"] == [0, 0, 1, 0]
    assert payload["colors"][3] is None and payload["colors"][0] == [0.0]
    assert all(np.isfinite(np.asarray(payload["pos"])).ravel())
    assert "position inferred" in plot._repr_html_()


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


def test_interactive_hyper_edge_positions_and_colors():
    plot = plot_graph_interactive(
        _located_graph(), hyper_edge_positions={"bus": ["x", "y"]}, hyper_edge_colors={"line": ["flow"]}
    )
    payload = _payload(plot)
    classes = {c["name"]: c for c in payload["classes"]}
    assert [o["kind"] for o in classes["bus"]["objects"]] == ["hub"] * 3  # placed buses are hubs with one spoke
    assert len(payload["pos"]) == 6  # 3 addresses + 3 placed buses
    assert [o["hub"] for o in classes["bus"]["objects"]] == [3, 4, 5]
    assert [o["color"] for o in classes["line"]["objects"]] == [[0.5], None]  # NaN flow: class color; lone value: mid
    assert all(o["color"] is None for o in classes["bus"]["objects"])
    assert '<span class="sc" data-ch="1">hyper-edges: 10<!--scale-->10</span>' in plot._repr_html_()
    assert "addresses:" not in plot._repr_html_()
