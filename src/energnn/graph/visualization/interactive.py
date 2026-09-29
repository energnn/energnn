# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Interactive, dependency-free HTML/SVG rendering of a Graph.

Python runs the whole pipeline and writes the SVG (polylines, markers, address circles, labels), the
legend and the tooltips. The embedded script (``assets/plot.js``) only adds what needs the browser:
zoom and pan, tooltips, the notebook's theme, and the colors that depend on it. The elements colored
by a value carry their normalized channels in a ``data-ch`` attribute, the legend's color scales carry
their channel count, and the markers, address circles and labels that must keep their size under zoom
sit at the origin of a ``.fx`` group translated to their position; that is the whole contract between
the two sides.
"""

from __future__ import annotations

import html
import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

from energnn.graph.visualization.assets import script_js, stylesheet_css, template_html
from energnn.graph.visualization.colors import ColorScale, Colors, resolve_colors
from energnn.graph.visualization.content import FeatureSpec, Topology, read_graph
from energnn.graph.visualization.geometry import Geometry, geometries
from energnn.graph.visualization.positions import MARKER_RATIO, Layout, address_radius, lay_out
from energnn.graph.visualization.theme import SVG_MARKERS, THEMES, Theme

if TYPE_CHECKING:
    from energnn.graph.graph import Graph

_plot_ids = itertools.count()
_PAD = 30  # canvas margin, in pixels


class InteractiveGraphPlot:
    """Self-contained HTML/SVG rendering of a Graph, displayed inline by notebooks."""

    def __init__(self, html_fragment: str) -> None:
        self._html = html_fragment

    def _repr_html_(self) -> str:
        return self._html

    def save(self, file_path: str) -> None:
        """Write the plot as a standalone HTML page."""
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(
                "<!DOCTYPE html>\n<html><head><meta charset='utf-8'></head><body>\n" + self._html + "\n</body></html>"
            )


# ---------------------------------------------------------------------------
# SVG pieces
# ---------------------------------------------------------------------------


def _marker_points(shape: str, r: float) -> list[tuple[float, float]]:
    """Vertices of a marker polygon of nominal radius ``r`` centered on the origin."""
    if shape == "square":
        return [(-r, -r), (r, -r), (r, r), (-r, r)]
    if shape == "triangle-up":
        return [(0, -1.2 * r), (1.1 * r, 0.85 * r), (-1.1 * r, 0.85 * r)]
    if shape == "triangle-down":
        return [(0, 1.2 * r), (1.1 * r, -0.85 * r), (-1.1 * r, -0.85 * r)]
    if shape == "diamond":
        return [(0, -1.3 * r), (1.3 * r, 0), (0, 1.3 * r), (-1.3 * r, 0)]
    if shape == "plus":
        t = 0.45 * r
        return [(-t, -r), (t, -r), (t, -t), (r, -t), (r, t), (t, t), (t, r), (-t, r), (-t, t), (-r, t), (-r, -t), (-t, -t)]
    if shape == "cross":
        c = np.sqrt(2.0) / 2.0
        return [(c * (x - y), c * (x + y)) for x, y in _marker_points("plus", r)]
    if shape == "pentagon":
        return [(1.2 * r * np.sin(2 * np.pi * k / 5), -1.2 * r * np.cos(2 * np.pi * k / 5)) for k in range(5)]
    if shape == "star":
        return [
            (radius * np.sin(np.pi * k / 5), -radius * np.cos(np.pi * k / 5))
            for k, radius in ((k, 1.5 * r if k % 2 == 0 else 0.65 * r) for k in range(10))
        ]
    raise ValueError(f"Unknown marker shape '{shape}'.")


def _svg_marker(shape: str, r: float, fill: str) -> str:
    """A marker polygon centered on the origin."""
    points = " ".join(f"{dx:.1f},{dy:.1f}" for dx, dy in _marker_points(shape, r))
    return f'<polygon class="mk" points="{points}" fill="{fill}"/>'


def _fixed(x: float, y: float, inner: str) -> str:
    """A group translated to ``(x, y)`` whose content keeps its size under zoom (the script counter-scales it)."""
    return f'<g class="fx" data-at="{x:.1f},{y:.1f}" transform="translate({x:.1f} {y:.1f})">{inner}</g>'


def _tip(title: str, ports: list[tuple[str, int]], features: dict[str, float]) -> str:
    """Tooltip HTML for one object (names and values escaped), ready to sit in a ``data-tip`` attribute."""
    parts = [f"<b>{html.escape(title)}</b>"]
    parts += [f"{html.escape(name)} &rarr; {address}" for name, address in ports]
    parts += [f"{html.escape(name)} = {value:.5g}" for name, value in features.items()]
    return html.escape("<br>".join(parts), quote=True)


def _channels_attr(scale: ColorScale | None, row: int) -> str:
    """``data-ch="u,v"`` for an element colored by a value, empty when it keeps its default color."""
    if scale is None or scale.missing[row]:
        return ""
    return ' data-ch="' + ",".join(f"{c:.4f}" for c in scale.channels[row]) + '"'


class _Canvas:
    """Maps layout units (the ``[-1, 1]`` box plus its margin) to canvas pixels."""

    def __init__(self, size: int, margin: float, n_addresses: int) -> None:
        self.size = size
        self.scale = (size - 2 * _PAD) / (2.0 + 2.0 * margin)
        self.margin = margin
        self.r_addr = address_radius(n_addresses) * self.scale
        self.r_mark = MARKER_RATIO * self.r_addr
        self.font_size = max(round(0.95 * self.r_addr), 7)

    def px(self, point: np.ndarray) -> tuple[float, float]:
        return _PAD + (point[0] + 1 + self.margin) * self.scale, _PAD + (1 + self.margin - point[1]) * self.scale

    def points(self, line: np.ndarray) -> str:
        return " ".join("{:.1f},{:.1f}".format(*self.px(p)) for p in line)


def _svg_body(topology: Topology, layout: Layout, colors: Colors, geoms: dict, canvas: _Canvas, class_colors) -> str:
    parts = []
    for row, h in enumerate(topology.hyper_edges):
        geom: Geometry | None = geoms.get(h.key)
        if geom is None:
            continue
        color = class_colors[h.cls]
        tip = _tip(f"{h.cls} #{h.index}", list(zip(topology.port_names[h.cls], h.ports)), h.features)
        parts.append(f'<g class="obj" data-tip="{tip}"{_channels_attr(colors.hyper_edges, row)}>')
        parts += [f'<polyline points="{canvas.points(line)}" stroke="{color}"/>' for line in geom.lines]
        for name, at in zip(topology.port_names[h.cls], geom.labels):
            parts.append(_fixed(*canvas.px(at), f'<text class="pl" y="-3" text-anchor="middle">{html.escape(name)}</text>'))
        shape = SVG_MARKERS[topology.classes.index(h.cls) % len(SVG_MARKERS)]
        parts.append(_fixed(*canvas.px(geom.marker), _svg_marker(shape, canvas.r_mark, color)))
        parts.append("</g>")
    for i, at in enumerate(layout.addresses):
        x, y = canvas.px(at)
        tip = _tip(f"address {i}", [], {})
        missing = colors.addresses is not None and colors.addresses.missing[i]
        if missing:
            tip += html.escape("<br><i>no color given</i>", quote=True)
        parts.append(f'<g class="addr" data-tip="{tip}"{_channels_attr(colors.addresses, i)}>')
        circle = f'<circle r="{canvas.r_addr:.1f}" fill="var(--surface)"/><text font-size="{canvas.font_size}">{i}</text>'
        parts.append(_fixed(x, y, circle) + "</g>")
    return "".join(parts)


def _legend_html(topology: Topology, colors: Colors, class_colors: dict[str, str]) -> str:
    items = [
        '<span><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="var(--surface)" stroke="var(--ink)"'
        ' stroke-width="1.2"/></svg>addresses</span>'
    ]
    for class_index, cls in enumerate(topology.classes):
        shape = SVG_MARKERS[class_index % len(SVG_MARKERS)]
        marker = f'<g transform="translate(7 7)">{_svg_marker(shape, 4.5, class_colors[cls])}</g>'
        items.append(f'<span><svg width="14" height="14">{marker}</svg>{html.escape(cls)}</span>')
    for label, scale in (("addresses", colors.addresses), ("hyper-edges", colors.hyper_edges)):
        if scale is not None:
            items.append(_scale_html(label, scale))
    return "".join(items)


def _scale_html(label: str, scale: ColorScale) -> str:
    """A color scale of the legend; the script draws the colormap at the ``<!--scale-->`` anchor."""
    lo, hi = scale.low, scale.high
    if scale.n_channels == 1:
        text = f"{lo[0]:.3g}<!--scale-->{hi[0]:.3g}"
    else:
        text = f"ch1 {lo[0]:.3g}&ndash;{hi[0]:.3g}<!--scale-->ch2 {lo[1]:.3g}&ndash;{hi[1]:.3g}"
    return f'<span class="sc" data-ch="{scale.n_channels}">{label}: {text}</span>'


def _theme_css(uid: str, theme: str) -> str:
    """CSS custom properties of the palette; with ``auto`` the script adds the ``dark`` class when the notebook is dark."""

    def variables(t: Theme) -> str:
        slots = "".join(f"--c{i}:{c};" for i, c in enumerate(t.palette))
        seq = "".join(f"--s{i}:{c};" for i, c in enumerate(t.sequential))
        biv = "".join(f"--b{k}:{c};" for k, c in zip(("00", "10", "01", "11"), t.bivariate))
        return f"--surface:{t.surface};--ink:{t.ink};--neutral:{t.neutral};{slots}{seq}{biv}"

    if theme == "auto":
        return f"#{uid}{{{variables(THEMES['light'])}}}#{uid}.dark{{{variables(THEMES['dark'])}}}"
    return f"#{uid}{{{variables(THEMES[theme])}}}"


def plot_graph_interactive(
    graph: Graph,
    *,
    address_positions: Any = None,
    hyper_edge_positions: FeatureSpec | None = None,
    address_colors: Any = None,
    hyper_edge_colors: FeatureSpec | None = None,
    edge_colors: bool = True,
    iterations: int = 150,
    seed: int = 0,
    size: int = 640,
    theme: str = "auto",
) -> InteractiveGraphPlot:
    """
    Render a single Graph as a self-contained interactive HTML/SVG figure.

    Address indices are always visible; hovering any object (address, hyper-edge marker or line) shows a
    tooltip with its port addresses and feature values, and reveals the port names along its connections.
    The mouse wheel zooms, dragging pans, double-click resets the view, and the toolbar offers the same.
    The result displays inline in Jupyter/IDE notebooks (via ``_repr_html_``) and can be written to a
    standalone HTML file with :meth:`InteractiveGraphPlot.save`. No dependency is required.

    :param graph: A single Graph; batched graphs must first go through :func:`energnn.graph.separate_graphs`.
    :param address_positions: Optional address coordinates of shape ``(n_addresses, 2)``; replaces the
        force-directed layout. Padded graphs may pass the padded length, fictitious rows are dropped.
    :param hyper_edge_positions: Optional ``{class: [x_feature, y_feature]}``: the objects of that class are
        drawn at the coordinates held by those features, as a marker with one spoke per port. Without
        ``address_positions``, each address sits at the mean position of the placed objects pointing to it,
        and every address must be pointed to by one.
    :param address_colors: Optional per-address values of shape ``(n_addresses, C)`` with ``C`` in {1, 2}:
        sequential colormap or bivariate colormap, normalized per channel. A NaN leaves the address uncolored.
    :param hyper_edge_colors: Optional ``{class: [feature, ...]}`` with 1 or 2 features: the markers and
        lines of those objects are colored from these features like the addresses above, with their own color
        scale shared by every listed class. Every other class is then drawn in neutral gray, so that only the
        feature colors carry a meaning. A NaN keeps that neutral color.
    :param edge_colors: If False, hyper-edges are drawn in the neutral gray instead of one color per class
        (marker shapes still tell classes apart).
    :param iterations: Number of layout relaxation steps (unused when positions are given).
    :param seed: Seed for the layout's random initial positions.
    :param size: Width and height of the drawing, in pixels.
    :param theme: ``"light"``, ``"dark"``, or ``"auto"`` to follow the notebook's theme (the background color
        of the output cell, or the OS preference when it cannot be read).
    :return: An :class:`InteractiveGraphPlot`.
    :raises ValueError: If the graph is not single, if ``theme`` is invalid, if an array has a wrong shape,
        or if a feature spec names an unknown class or feature.
    """
    if theme not in ("light", "dark", "auto"):
        raise ValueError("theme must be 'light', 'dark' or 'auto'.")
    topology = read_graph(graph, hyper_edge_positions=hyper_edge_positions)
    layout = lay_out(topology, address_positions=address_positions, iterations=iterations, seed=seed)
    colors = resolve_colors(topology, address_colors=address_colors, hyper_edge_colors=hyper_edge_colors)
    geoms = geometries(topology, layout)
    # once a class is colored by its features, the others are drawn in neutral so the colormap stands alone
    n_colors = len(THEMES["light"].palette)
    class_colors = {
        cls: f"var(--c{i % n_colors})" if edge_colors and colors.hyper_edges is None else "var(--neutral)"
        for i, cls in enumerate(topology.classes)
    }
    canvas = _Canvas(size, layout.margin, topology.n_addresses)
    uid = f"energnn-plot-{next(_plot_ids)}"
    # the pieces holding their own __UID__ (stylesheet, script) go in first, the id is substituted last
    pieces = {
        "__CSS__": _theme_css(uid, theme) + stylesheet_css(),
        "__SCRIPT__": script_js(),
        "__AUTO_THEME__": "1" if theme == "auto" else "0",
        "__LEGEND__": _legend_html(topology, colors, class_colors),
        "__SIZE__": str(size),
        "__BODY__": _svg_body(topology, layout, colors, geoms, canvas, class_colors),
        "__UID__": uid,
    }
    fragment = template_html().strip()
    for placeholder, value in pieces.items():
        fragment = fragment.replace(placeholder, value)
    return InteractiveGraphPlot(fragment)
