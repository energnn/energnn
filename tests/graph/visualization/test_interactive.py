# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

from energnn.graph.graph import collate_graphs
from energnn.graph.visualization import InteractiveGraphPlot, plot_graph_interactive
from energnn.graph.visualization.theme import THEMES, rgb_to_hex, sequential_rgb


def _count(fragment: str, pattern: str) -> int:
    return len(re.findall(pattern, fragment))


def test_interactive_plot_content(mixed_order_graph):
    plot = plot_graph_interactive(mixed_order_graph)
    assert isinstance(plot, InteractiveGraphPlot)
    fragment = plot._repr_html_()
    assert 'id="energnn-plot-' in fragment and "<style>" in fragment and "<script>" in fragment
    assert re.search(r"__[A-Z_]+__", fragment) is None  # every placeholder of the template, css and script is filled
    assert _count(fragment, '<g class="addr"') == 4 and _count(fragment, '<g class="obj"') == 6
    assert _count(fragment, "<polyline ") == 3 + 2 + 3  # lines, stubs, and the three spokes of the hub
    # tooltips hold the ports and the feature values, escaped for the attribute
    assert "&lt;b&gt;trafo3w #0&lt;/b&gt;&lt;br&gt;hv &amp;rarr; 0&lt;br&gt;" in fragment and "ratio = 1.02" in fragment
    assert 'data-tip="&lt;b&gt;address 0&lt;/b&gt;"' in fragment
    for expected in (">gen<", ">line<", ">trafo3w<", "addresses", 'class="logo" src="data:image/png;base64,'):
        assert expected in fragment
    assert 'data-ch="' not in fragment  # nothing colored by a value


def test_interactive_plot_skips_fictitious_and_rejects_batches(mixed_order_graph, padded_shape):
    with pytest.raises(ValueError, match="single"):
        plot_graph_interactive(collate_graphs([mixed_order_graph, mixed_order_graph]))
    mixed_order_graph.pad(padded_shape)
    fragment = plot_graph_interactive(mixed_order_graph, address_positions=np.arange(14.0).reshape(7, 2))._repr_html_()
    assert _count(fragment, '<g class="addr"') == 4 and _count(fragment, '<g class="obj"') == 6


def test_interactive_plot_themes_and_ids(mixed_order_graph):
    auto = plot_graph_interactive(mixed_order_graph, theme="auto")._repr_html_()
    assert ".dark{--surface:#1a1a19" in auto and 'data-auto-theme="1"' in auto
    dark = plot_graph_interactive(mixed_order_graph, theme="dark")._repr_html_()
    assert ".dark{" not in dark and "--surface:#1a1a19" in dark and "--s0:" in dark and "--b11:" in dark
    assert 'data-auto-theme="0"' in dark
    with pytest.raises(ValueError, match="theme"):
        plot_graph_interactive(mixed_order_graph, theme="solarized")
    ids = [re.search(r'id="(energnn-plot-\d+)"', plot_graph_interactive(mixed_order_graph)._repr_html_()) for _ in range(2)]
    assert ids[0].group(1) != ids[1].group(1)


def test_interactive_edge_colors_and_logo_off(mixed_order_graph):
    plain = plot_graph_interactive(mixed_order_graph, edge_colors=False, logo=False)._repr_html_()
    assert "var(--c0)" not in plain and 'stroke="var(--neutral)"' in plain and 'class="logo"' not in plain
    colored = plot_graph_interactive(mixed_order_graph)._repr_html_()
    assert all(f"var(--c{i})" in colored for i in range(3))


def test_interactive_portless_class_and_multi_graph(portless_graph, multi_graph):
    assert _count(plot_graph_interactive(portless_graph)._repr_html_(), '<g class="obj"') == 2  # port-less buses draw nothing
    fragment = plot_graph_interactive(multi_graph)._repr_html_()
    assert _count(fragment, '<g class="obj"') == 6 and _count(fragment, "<polyline ") == 3 + 2 + 3 + 3


@pytest.mark.parametrize("n_channels, scale", [(1, "0<!--scale-->3"), (2, "ch1 0&ndash;3<!--scale-->ch2 0&ndash;3")])
def test_interactive_address_colors(mixed_order_graph, n_channels, scale):
    colors = np.arange(4, dtype=float)[:, None].repeat(n_channels, axis=1)
    colors[3] = np.nan  # the range is then 0 to 2
    fragment = plot_graph_interactive(mixed_order_graph, address_colors=colors)._repr_html_()
    channels = re.findall(r'<g class="addr" data-tip="[^"]*"( data-ch="[^"]*")?>', fragment)
    assert channels == [
        ' data-ch="' + ",".join(["0.0000"] * n_channels) + '"',
        ' data-ch="' + ",".join(["0.5000"] * n_channels) + '"',
        ' data-ch="' + ",".join(["1.0000"] * n_channels) + '"',
        "",
    ]
    assert f'<span class="sc" data-ch="{n_channels}">addresses: {scale.replace("3", "2")}</span>' in fragment  # NaN ignored
    assert "no color given" in fragment


def test_interactive_hyper_edge_positions_and_colors(located_graph):
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    plot = plot_graph_interactive(
        located_graph, address_positions=square, hyper_edge_positions={"bus": ["x", "y"]}, hyper_edge_colors={"line": ["flow"]}
    )
    fragment = plot._repr_html_()
    objects = re.findall(r'<g class="obj" data-tip="&lt;b&gt;(\w+) #(\d)&lt;/b&gt;[^"]*"( data-ch="[^"]*")?>', fragment)
    assert [(cls, ch) for cls, _, ch in objects] == [
        ("bus", ""),
        ("bus", ""),
        ("bus", ""),
        ("line", ' data-ch="0.5000"'),
        ("line", ""),
    ]
    assert "var(--c0)" not in fragment  # every class drawn in neutral once one is colored by a value
    assert '<span class="sc" data-ch="1">hyper-edges: 10<!--scale-->10</span>' in fragment and "addresses:" not in fragment


def test_interactive_plot_save(mixed_order_graph, tmp_path):
    path = str(tmp_path / "graph.html")
    plot_graph_interactive(mixed_order_graph).save(path)
    content = Path(path).read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>") and "trafo3w" in content


# ---------------------------------------------------------------------------
# The script itself, run by node against a DOM rebuilt from the fragment
# ---------------------------------------------------------------------------


class _Tree(HTMLParser):
    """Turns the fragment into a JSON tree of elements, comments and text for the node shim."""

    VOID = {"img", "br"}

    def __init__(self) -> None:
        super().__init__()
        self.root: dict = {"tag": "root", "attrs": {}, "children": []}
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = {"tag": tag, "attrs": dict(attrs), "children": []}
        self.stack[-1]["children"].append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1]["children"].append({"tag": tag, "attrs": dict(attrs), "children": []})

    def handle_endtag(self, tag):
        if tag not in self.VOID:
            self.stack.pop()

    def handle_comment(self, data):
        self.stack[-1]["children"].append({"comment": data})

    def handle_data(self, data):
        if data.strip():
            self.stack[-1]["children"].append({"text": data})


@pytest.mark.skipif(shutil.which("node") is None, reason="node is needed to run plot.js")
def test_plot_js_paints_colors_scales_zoom_and_tooltips(located_graph, tmp_path):
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    fragment = plot_graph_interactive(
        located_graph,
        address_positions=square,
        address_colors=np.array([[0.0], [1.0], [2.0], [np.nan]]),
        hyper_edge_colors={"bus": ["load", "x"]},
        theme="light",
    )._repr_html_()
    tree = _Tree()
    tree.feed(fragment)
    (tmp_path / "tree.json").write_text(json.dumps(tree.root))
    shim = Path(__file__).with_name("plot_js_shim.js")
    result = subprocess.run(["node", str(shim), str(tmp_path / "tree.json")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    expected = rgb_to_hex(sequential_rgb(np.array([0.0, 0.5, 1.0]), THEMES["light"]))  # the script mirrors the Python colormap
    assert state["addressFills"][:3] == expected and state["addressFills"][3] == "var(--surface)"
    assert len(set(state["busFills"])) == 3 and all(f.startswith("#") for f in state["busFills"])
    assert state["lineStrokes"] == ["var(--neutral)"] * 2  # the class not colored by a value stays neutral
    assert state["scaleSvgs"] == 2  # one color scale per family in the legend
    assert (
        state["transformAfterZoom"] == "translate(-80.0 -80.0) scale(1.250)"
        and state["transformAfterReset"] == "translate(0.0 0.0) scale(1.000)"
    )
    assert state["tooltipOnHover"].startswith("<b>address 0</b>")
