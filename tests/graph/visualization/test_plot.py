# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import json

import numpy as np
import pytest

pytest.importorskip("plotly", reason="plot_graph needs the 'viz' extra")
import plotly.graph_objects as go  # noqa: E402

from energnn.graph.graph import Graph, collate_graphs  # noqa: E402
from energnn.graph.hyper_edge_set import HyperEdgeSet  # noqa: E402
from energnn.graph.visualization import GraphFigure, plot_graph  # noqa: E402
from energnn.graph.visualization.plot import (  # noqa: E402
    ADDRESS_RADIUS,
    MARKER_RATIO,
    N_SHADES,
    STUB_LENGTH,
    THEMES,
    _markers,
    _positions,
    _swatches,
)

from .conftest import SQUARE  # noqa: E402

UNIT_SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
RADIUS = 0.04


def _trace(figure, name, mode="markers"):
    """The marker trace of a class (or of the addresses), or with ``mode="lines"`` the traces of its spokes."""
    if mode == "lines":
        return [t for t in figure.data if t.legendgroup == name and t.mode == "lines"]
    return next(t for t in figure.data if t.name == name)


def _xy(trace):
    return np.stack([trace.x, trace.y], axis=-1)


def _placed(graph, address_positions=None, hyper_edge_positions=None):
    """The addresses and the markers of a graph, as ``plot_graph`` places them."""
    classes = {name: hes for name, hes in sorted(graph.hyper_edge_sets.items()) if hes.port_dict}
    addresses, placed = _positions(classes, graph.n_addresses, address_positions, hyper_edge_positions or {}, 10, 0)
    return addresses, _markers(classes, addresses, placed, RADIUS)


def test_figure_content(mixed_order_graph):
    figure = plot_graph(mixed_order_graph)
    assert isinstance(figure, GraphFigure) and isinstance(figure, go.Figure)
    assert [t.name for t in figure.data if t.name] == ["gen", "line", "trafo3w", "addresses"]
    assert [len(_trace(figure, name).x) for name in ("gen", "line", "trafo3w", "addresses")] == [2, 3, 1, 4]
    # one spoke per port, each made of the marker, the address and a NaN that ends the line
    assert [len(_trace(figure, name, "lines")[0].x) for name in ("gen", "line", "trafo3w")] == [2 * 3, 3 * 2 * 3, 3 * 3]
    spokes = _xy(_trace(figure, "trafo3w", "lines")[0]).reshape(3, 3, 2)
    addresses = _xy(_trace(figure, "addresses"))
    np.testing.assert_allclose(spokes[:, 0], np.tile(_xy(_trace(figure, "trafo3w")), (3, 1)))
    np.testing.assert_allclose(spokes[:, 1], addresses[[0, 2, 1]])  # ports in sorted order: hv, lv, mv
    assert np.isnan(spokes[:, 2]).all()
    # tooltips: a template per class, filled from the index, the ports and the features of each object
    trafo = _trace(figure, "trafo3w")
    assert trafo.hovertemplate == (
        "<b>trafo3w #%{customdata[0]}</b><br>hv → %{customdata[1]}<br>lv → %{customdata[2]}<br>mv → %{customdata[3]}"
        "<br>ratio = %{customdata[4]:.5~g}<extra></extra>"
    )
    np.testing.assert_allclose(trafo.customdata, [[0, 0, 2, 1, 1.02]])
    np.testing.assert_allclose(_trace(figure, "gen").customdata, [[0, 0, 1.0], [1, 3, 2.0]])
    assert list(_trace(figure, "addresses").text) == ["0", "1", "2", "3"]
    assert _trace(figure, "addresses").hovertemplate == "<b>address %{text}</b><extra></extra>"
    assert len({t.marker.symbol for t in figure.data if t.name}) == 4 and figure.layout.width == figure.layout.height == 640


def test_skips_fictitious_and_rejects_batches(mixed_order_graph, padded_shape):
    with pytest.raises(ValueError, match="single"):
        plot_graph(collate_graphs([mixed_order_graph, mixed_order_graph]))
    reference = plot_graph(mixed_order_graph, address_positions=SQUARE)
    mixed_order_graph.pad(padded_shape)
    padded = plot_graph(mixed_order_graph, address_positions=SQUARE)
    assert mixed_order_graph.n_addresses == 7  # the graph itself is left padded
    assert len(padded.data) == len(reference.data)
    for a, b in zip(padded.data, reference.data):
        np.testing.assert_allclose(_xy(a), _xy(b))


def test_classes_without_port_or_object_are_not_drawn(portless_graph):
    assert [t.name for t in plot_graph(portless_graph).data if t.name] == ["line", "addresses"]
    only_portless = Graph.from_dict(hyper_edge_set_dict={"bus": portless_graph.hyper_edge_sets["bus"]}, n_addresses=3)
    assert [t.name for t in plot_graph(only_portless).data] == ["addresses"]


def test_themes_and_class_colors(mixed_order_graph):
    for name, theme in THEMES.items():
        figure = plot_graph(mixed_order_graph, theme=name)
        assert figure.layout.paper_bgcolor == figure.layout.plot_bgcolor == theme.surface
        assert [_trace(figure, cls).marker.color for cls in ("gen", "line", "trafo3w")] == list(theme.palette[:3])
        assert _trace(figure, "line", "lines")[0].line.color == theme.palette[1]
        assert _trace(figure, "addresses").marker.color == theme.surface
    assert plot_graph(mixed_order_graph).layout.paper_bgcolor == THEMES["light"].surface  # "auto" is built light
    plain = plot_graph(mixed_order_graph, hyper_edge_colors=False)
    assert {_trace(plain, cls).marker.color for cls in ("gen", "line", "trafo3w")} == {THEMES["light"].neutral}
    with pytest.raises(ValueError, match="theme"):
        plot_graph(mixed_order_graph, theme="solarized")


def test_spring_layout_is_used_without_positions(mixed_order_graph):
    first, second = (_xy(_trace(plot_graph(mixed_order_graph, iterations=10), "addresses")) for _ in range(2))
    np.testing.assert_allclose(first, second)  # same seed, same layout
    assert np.abs(first).max() <= 1.0 + 1e-6
    other = _xy(_trace(plot_graph(mixed_order_graph, iterations=10, seed=1), "addresses"))
    assert not np.allclose(first, other)


def test_given_positions_are_fitted_to_the_box(mixed_order_graph):
    addresses, markers = _placed(mixed_order_graph, address_positions=SQUARE)
    np.testing.assert_allclose(addresses, UNIT_SQUARE, atol=1e-9)
    np.testing.assert_allclose(markers["line"], (UNIT_SQUARE[[0, 1, 2]] + UNIT_SQUARE[[1, 2, 3]]) / 2.0, atol=1e-9)
    np.testing.assert_allclose(markers["trafo3w"], UNIT_SQUARE[[0, 1, 2]].mean(axis=0, keepdims=True), atol=1e-9)
    # an object with a single address is a short distance away from it
    np.testing.assert_allclose(np.linalg.norm(markers["gen"] - UNIT_SQUARE[[0, 3]], axis=1), STUB_LENGTH * RADIUS)


@pytest.mark.parametrize("bad", [np.zeros((3, 2)), np.zeros((4, 3)), np.zeros(4), np.full((4, 2), np.nan)])
def test_address_positions_errors(mixed_order_graph, bad):
    with pytest.raises(ValueError, match="address_positions"):
        plot_graph(mixed_order_graph, address_positions=bad)


def test_objects_on_the_same_addresses_are_moved_apart(multi_graph):
    addresses, markers = _placed(multi_graph, address_positions=TRIANGLE)
    # 3 parallel lines between addresses 0 and 1: the middle one on the barycenter, the others on either side of it
    lines, middle = markers["line"][:3], (addresses[0] + addresses[1]) / 2.0
    np.testing.assert_allclose(lines[1], middle, atol=1e-9)
    np.testing.assert_allclose(lines[0] + lines[2], 2.0 * middle, atol=1e-9)
    assert abs((lines[2] - lines[0]) @ (addresses[1] - addresses[0])) < 1e-9 and np.linalg.norm(lines[2] - lines[0]) > 0.1
    # the self-loop on address 2 is drawn like an object with a single address
    assert np.linalg.norm(markers["line"][3] - addresses[2]) == pytest.approx(STUB_LENGTH * RADIUS)
    # 2 parallel transformers: on either side of their common barycenter
    np.testing.assert_allclose(markers["trafo3w"].mean(axis=0), addresses.mean(axis=0), atol=1e-9)
    assert np.linalg.norm(markers["trafo3w"][1] - markers["trafo3w"][0]) > 0.05


def test_objects_on_a_single_address_are_spread_around_it():
    hes = {
        "gen": HyperEdgeSet.from_dict(port_dict={"bus": np.array([0, 0, 1])}, feature_dict=None),
        "shunt": HyperEdgeSet.from_dict(port_dict={"a": np.array([0]), "b": np.array([0])}, feature_dict=None),
    }
    graph = Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=2)
    addresses, markers = _placed(graph, address_positions=np.array([[0.0, 0.0], [1.0, 0.0]]))
    around = np.concatenate([markers["gen"][:2], markers["shunt"]]) - addresses[0]  # 3 objects of 2 classes on address 0
    np.testing.assert_allclose(np.linalg.norm(around, axis=1), STUB_LENGTH * RADIUS)
    np.testing.assert_allclose(around.sum(axis=0), 0.0, atol=1e-9)  # evenly spread: a third of a turn apart
    assert np.linalg.norm(markers["gen"][2] - addresses[1]) == pytest.approx(STUB_LENGTH * RADIUS)


@pytest.mark.parametrize("n, spacing_radius", [(4, ADDRESS_RADIUS), (2500, 150.0 / 50.0)])
def test_symbols_keep_their_size_and_distances_follow_the_spacing(n, spacing_radius):
    bus = HyperEdgeSet.from_dict(port_dict={"id": np.arange(n)}, feature_dict={"v": np.ones(n)})
    graph = Graph.from_dict(hyper_edge_set_dict={"bus": bus}, n_addresses=n)
    figure = plot_graph(graph, address_positions=np.random.default_rng(0).uniform(size=(n, 2)), size=320)
    # in pixels, whatever the number of addresses: half the sizes of the default 640-pixel figure
    assert _trace(figure, "addresses").marker.size == pytest.approx(ADDRESS_RADIUS)
    assert _trace(figure, "bus").marker.size == pytest.approx(MARKER_RATIO * ADDRESS_RADIUS)
    # in layout units, shrinking beyond 133 addresses
    stubs = np.linalg.norm(_xy(_trace(figure, "bus")) - _xy(_trace(figure, "addresses")), axis=1)
    np.testing.assert_allclose(stubs, STUB_LENGTH * spacing_radius / 290.0, rtol=1e-3)


@pytest.mark.parametrize("n, font_size", [(1000, 12.4), (1001, 9.8), (10001, 8.1)])
def test_address_numbers_shrink_to_fit_in_their_circle(n, font_size):
    graph = Graph.from_dict(hyper_edge_set_dict={}, n_addresses=n)
    addresses = _trace(plot_graph(graph, address_positions=np.zeros((n, 2))), "addresses")
    assert addresses.marker.size == 2 * ADDRESS_RADIUS
    assert addresses.textfont.size == pytest.approx(font_size, abs=0.05)


def test_hyper_edge_positions_place_the_markers(located_graph):
    square = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [4.0, 3.0]])
    addresses, markers = _placed(located_graph, address_positions=square, hyper_edge_positions={"line": ["flow", "flow"]})
    scale = addresses[1, 0] - addresses[0, 0]
    # the first line goes where its features say, in the frame of the addresses; the second one has NaN features
    np.testing.assert_allclose(markers["line"][0] - addresses[0], scale / 4.0 * np.array([10.0, 10.0]), atol=1e-9)
    np.testing.assert_allclose(markers["line"][1], (addresses[1] + addresses[2]) / 2.0, atol=1e-9)
    assert np.abs(np.concatenate([addresses, markers["line"]])).max() == pytest.approx(1.0)


def test_hyper_edge_positions_place_the_addresses(located_graph):
    with pytest.raises(ValueError, match=r"addresses \[3\] are pointed to by no placed hyper-edge"):
        plot_graph(located_graph, hyper_edge_positions={"bus": ["x", "y"]})
    hes = {name: located_graph.hyper_edge_sets[name] for name in ("bus", "line")}
    addresses, markers = _placed(
        Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=3), hyper_edge_positions={"bus": ["x", "y"]}
    )
    scale = (addresses[1, 0] - addresses[0, 0]) / 4.0
    assert scale > 0
    np.testing.assert_allclose(addresses - addresses[0], scale * np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]), atol=1e-9)
    # a bus sits on the address it places, which would hide it: it is moved away like any object with a single address
    np.testing.assert_allclose(np.linalg.norm(markers["bus"] - addresses, axis=1), STUB_LENGTH * RADIUS)


@pytest.mark.parametrize(
    "spec, error",
    [
        ({"nope": ["x", "y"]}, KeyError),
        ({"theta": ["value", "value"]}, KeyError),  # a class without port is not drawn
        ({"bus": ["x"]}, ValueError),
        ({"bus": ["x", "nope"]}, KeyError),
    ],
)
def test_hyper_edge_positions_errors(located_graph, spec, error):
    with pytest.raises(error):
        plot_graph(located_graph, hyper_edge_positions=spec)


def test_address_colors(mixed_order_graph):
    figure = plot_graph(mixed_order_graph, address_colors=np.array([0.0, 1.0, 2.0, np.nan]))
    addresses = _trace(figure, "addresses")
    # the shade of each address, the last one being the color of the missing values: the surface, as without colors
    assert list(addresses.marker.color) == [0, round((N_SHADES - 1) / 2), N_SHADES - 1, N_SHADES]
    assert (addresses.marker.cmin, addresses.marker.cmax) == (0, N_SHADES)
    assert (
        addresses.marker.colorscale[-1] == (1.0, THEMES["light"].surface) and len(addresses.marker.colorscale) == N_SHADES + 1
    )
    np.testing.assert_allclose(addresses.customdata, [0.0, 1.0, 2.0, np.nan])
    assert "value = %{customdata:.5~g}" in addresses.hovertemplate
    colorbar = next(t for t in figure.data if t.marker.showscale)
    assert (colorbar.marker.cmin, colorbar.marker.cmax) == (0.0, 2.0) and colorbar.marker.colorbar.title.text == "addresses"
    constant = plot_graph(mixed_order_graph, address_colors=np.ones(4))
    assert set(_trace(constant, "addresses").marker.color) == {round((N_SHADES - 1) / 2)}  # the middle of the colormap
    with pytest.raises(ValueError, match="all missing"):
        plot_graph(mixed_order_graph, address_colors=np.full(4, np.nan))
    with pytest.raises(ValueError, match="address_colors"):
        plot_graph(mixed_order_graph, address_colors=np.zeros((4, 2)))


def test_hyper_edge_colors_share_one_scale_over_the_listed_classes(located_graph):
    figure = plot_graph(located_graph, hyper_edge_colors={"bus": "load", "line": "flow"})
    neutral = THEMES["light"].neutral
    colorbar = next(t for t in figure.data if t.marker.showscale)
    assert (colorbar.marker.cmin, colorbar.marker.cmax) == (1.0, 10.0)  # one range over buses and lines, NaN ignored
    assert colorbar.marker.colorbar.title.text == "bus.load, line.flow"
    buses, lines = _trace(figure, "bus"), _trace(figure, "line")
    assert list(buses.marker.color) == [0, round(1 / 9 * (N_SHADES - 1)), round(2 / 9 * (N_SHADES - 1))]
    assert list(lines.marker.color) == [N_SHADES - 1, N_SHADES]  # the NaN flow keeps the neutral color
    assert lines.marker.colorscale[-1] == (1.0, neutral)
    # the spokes follow: one trace per shade, in the color of that shade
    assert [t.line.color for t in _trace(figure, "bus", "lines")] == [
        buses.marker.colorscale[i][1] for i in buses.marker.color
    ]
    assert [t.line.color for t in _trace(figure, "line", "lines")] == [lines.marker.colorscale[N_SHADES - 1][1], neutral]
    assert [len(t.x) for t in _trace(figure, "line", "lines")] == [2 * 3, 2 * 3]


def test_classes_not_colored_by_a_value_turn_neutral(located_graph):
    figure = plot_graph(located_graph, hyper_edge_colors={"line": "flow"}, address_colors=np.arange(4.0))
    assert _trace(figure, "bus").marker.color == THEMES["light"].neutral
    assert [t.marker.colorbar.title.text for t in figure.data if t.marker.showscale] == ["line.flow", "addresses"]
    assert len({t.marker.colorbar.x for t in figure.data if t.marker.showscale}) == 2  # side by side
    for spec in ({"nope": "x"}, {"bus": "nope"}, {"theta": "value"}):
        with pytest.raises(KeyError):
            plot_graph(located_graph, hyper_edge_colors=spec)
    located_graph.line.flow = np.array([np.nan, np.nan])
    with pytest.raises(ValueError, match="all missing"):
        plot_graph(located_graph, hyper_edge_colors={"line": "flow"})


def test_figure_is_displayed_and_saved_with_the_wheel_zoom(mixed_order_graph, tmp_path, monkeypatch):
    figure = plot_graph(mixed_order_graph)
    shown = []
    monkeypatch.setattr("plotly.io.show", lambda fig, *args, **kwargs: shown.append(kwargs))
    figure.show()
    figure.show(config={"staticPlot": True})  # the caller's options win
    assert [kwargs["config"].get("scrollZoom") for kwargs in shown] == [True, None]
    path = tmp_path / "graph.html"
    figure.write_html(path)
    content = path.read_text(encoding="utf-8")
    assert "<html>" in content and '"scrollZoom": true' in content and "trafo3w" in content


def _notebook_html(figure):
    """What a notebook gets to display the figure, and the script constants named in ``names`` read from it."""
    formatter = pytest.importorskip("IPython.core.formatters").DisplayFormatter()
    data, _ = formatter.format(figure)
    assert set(data) == {"text/html", "text/plain"} and data["text/plain"] == "<GraphFigure>"
    return data["text/html"]


def test_notebooks_get_the_figure_as_html_with_a_script(mixed_order_graph, monkeypatch):
    monkeypatch.setattr("plotly.io.show", lambda *args, **kwargs: pytest.fail("plotly's display is not used"))
    html = _notebook_html(plot_graph(mixed_order_graph, size=500))
    assert html.startswith('<div style="width:500px;height:500px">') and 'import("https://cdn.plot.ly/plotly-' in html
    assert '"scrollZoom": true' in html and "trafo3w" in html
    assert "__" not in html.replace("__bdata", "")  # every field of the template was filled


@pytest.mark.parametrize(
    "theme, built, auto", [("auto", "light", "true"), ("light", "light", "false"), ("dark", "dark", "false")]
)
def test_only_the_auto_theme_follows_the_notebook(mixed_order_graph, theme, built, auto):
    html = _notebook_html(plot_graph(mixed_order_graph, theme=theme))
    assert f'const built = "{built}", auto = {auto}, pairs = [["#fcfcfb", "#1a1a19"], ["#0b0b0b", "#ffffff"]' in html


def test_every_color_of_a_figure_has_its_counterpart_in_the_other_theme(located_graph):
    figures = {
        name: plot_graph(located_graph, theme=name, hyper_edge_colors={"bus": "x"}, address_colors=np.arange(4.0))
        for name in THEMES
    }
    table = dict(zip(_swatches(THEMES["light"]), _swatches(THEMES["dark"])))
    assert len(table) == len(_swatches(THEMES["light"]))  # no light color with two counterparts

    def swap(value):  # as the script of the notebook does
        if isinstance(value, str):
            return table.get(value, value)
        if isinstance(value, (list, tuple)):
            return [swap(item) for item in value]
        return {key: swap(item) for key, item in value.items()} if isinstance(value, dict) else value

    light, dark = (json.loads(figures[name].to_json()) for name in ("light", "dark"))
    assert swap(light) == dark


def test_the_script_cannot_be_ended_by_a_text_of_the_figure(mixed_order_graph):
    figure = plot_graph(mixed_order_graph)
    figure.update_layout(title="</script><b>")
    html = figure._repr_html_()
    assert html.count("</script>") == 1 and html.endswith("</script>")


def test_big_graphs_are_drawn_with_webgl(mixed_order_graph, monkeypatch):
    assert {t.type for t in plot_graph(mixed_order_graph).data} == {"scatter"}
    monkeypatch.setattr("energnn.graph.visualization.plot.SVG_UP_TO", 9)  # the graph has 4 addresses and 6 objects
    assert {t.type for t in plot_graph(mixed_order_graph).data} == {"scattergl"}
    colored = plot_graph(mixed_order_graph, address_colors=np.arange(4.0))
    assert [t.type for t in colored.data if t.marker.showscale] == ["scatter"]  # the colorbar is no drawing
