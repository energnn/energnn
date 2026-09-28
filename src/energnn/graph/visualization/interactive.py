# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Interactive, dependency-free HTML/SVG rendering of a Graph."""

from __future__ import annotations

import html
import itertools
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from energnn.graph.visualization.layout import ObjGeom, PlotData, extract_plot_data, object_geometries
from energnn.graph.visualization.theme import ADDRESS_COLOR, SVG_MARKERS, THEMES, Theme

if TYPE_CHECKING:
    from energnn.graph.graph import Graph

_plot_ids = itertools.count()


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
# SVG building blocks
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


def _svg_marker(shape: str, x: float, y: float, r: float, color: str, extra: str = "") -> str:
    points = " ".join(f"{x + dx:.1f},{y + dy:.1f}" for dx, dy in _marker_points(shape, r))
    return f'<polygon class="mk" points="{points}" fill="{color}" {extra}/>'


def _tip(title: str, port_lines: list[tuple[str, int]], feature_lines: dict[str, float]) -> str:
    """Build the tooltip HTML for one object and escape it for use in an attribute."""
    parts = [f"<b>{html.escape(title)}</b>"]
    parts += [f"{html.escape(pn)} &rarr; {addr}" for pn, addr in port_lines]
    parts += [f"{html.escape(fn)} = {value:.5g}" for fn, value in feature_lines.items()]
    return html.escape("<br>".join(parts), quote=True)


def _theme_css(uid: str, theme: str) -> str:
    """CSS custom properties for the palette; ``auto`` follows the viewer's color-scheme preference."""

    def _vars(t: Theme) -> str:
        slots = "".join(f"--c{i}:{c};" for i, c in enumerate(t.palette))
        return f"--surface:{t.surface};--ink:{t.ink};{slots}"

    if theme == "auto":
        light, dark = THEMES["light"], THEMES["dark"]
        return f"#{uid}{{{_vars(light)}}}" f"@media (prefers-color-scheme: dark){{#{uid}{{{_vars(dark)}}}}}"
    return f"#{uid}{{{_vars(THEMES[theme])}}}"


def _css(uid: str, theme: str, stroke: float) -> str:
    return (
        f"{_theme_css(uid, theme)}"
        f"#{uid}{{position:relative;display:inline-block;font-family:system-ui,sans-serif;"
        f"background:var(--surface);border-radius:6px}}"
        f"#{uid} .lg{{display:flex;flex-wrap:wrap;gap:4px 14px;padding:8px 12px 0;color:var(--ink);font-size:12px}}"
        f"#{uid} .lg span{{display:inline-flex;align-items:center;gap:5px}}"
        f"#{uid} .obj .pl{{opacity:0;fill:var(--ink);font-size:9px;pointer-events:none}}"
        f"#{uid} .obj:hover .pl{{opacity:1}}"
        f"#{uid} .obj:hover polyline{{stroke-width:{2.2 * stroke:.1f}px}}"
        f"#{uid} .obj:hover .mk,#{uid} .addr:hover circle{{stroke:var(--ink);stroke-width:1.5px}}"
        f"#{uid} .tip{{display:none;position:absolute;pointer-events:none;background:var(--surface);color:var(--ink);"
        f"border:1px solid {ADDRESS_COLOR};border-radius:4px;padding:5px 8px;font-size:11px;line-height:1.5;"
        f"white-space:nowrap;z-index:10}}"
        f"#{uid} svg.cv{{cursor:grab}}"
        f"#{uid} svg.cv:active{{cursor:grabbing}}"
    )


def _script(uid: str, size: int) -> str:
    """Tooltips on hover; wheel to zoom on the cursor, drag to pan, double-click to reset."""
    return (
        f"(function(){{var root=document.getElementById('{uid}');var tip=root.querySelector('.tip');"
        f"root.querySelectorAll('[data-tip]').forEach(function(el){{"
        f"el.addEventListener('mousemove',function(e){{tip.innerHTML=el.getAttribute('data-tip');"
        f"tip.style.display='block';var r=root.getBoundingClientRect();"
        f"tip.style.left=(e.clientX-r.left+14)+'px';tip.style.top=(e.clientY-r.top+14)+'px';}});"
        f"el.addEventListener('mouseleave',function(){{tip.style.display='none';}});}});"
        f"var svg=root.querySelector('svg.cv');var vb=[0,0,{size},{size}];var drag=null;"
        f"function apply(){{svg.setAttribute('viewBox',vb.join(' '));}}"
        f"svg.addEventListener('wheel',function(e){{e.preventDefault();"
        f"var k=e.deltaY<0?0.8:1.25;var r=svg.getBoundingClientRect();"
        f"var mx=vb[0]+(e.clientX-r.left)/r.width*vb[2];var my=vb[1]+(e.clientY-r.top)/r.height*vb[3];"
        f"vb=[mx-(mx-vb[0])*k,my-(my-vb[1])*k,vb[2]*k,vb[3]*k];apply();}},{{passive:false}});"
        f"svg.addEventListener('mousedown',function(e){{e.preventDefault();drag=[e.clientX,e.clientY,vb[0],vb[1]];}});"
        f"window.addEventListener('mousemove',function(e){{if(drag){{var r=svg.getBoundingClientRect();"
        f"vb[0]=drag[2]-(e.clientX-drag[0])/r.width*vb[2];vb[1]=drag[3]-(e.clientY-drag[1])/r.height*vb[3];apply();}}}});"
        f"window.addEventListener('mouseup',function(){{drag=null;}});"
        f"svg.addEventListener('dblclick',function(){{vb=[0,0,{size},{size}];apply();}});}})();"
    )


def _object_svg(
    name: str, i: int, data: PlotData, geom: ObjGeom, shape: str, color_var: str, stroke: float, r_mark: float, to_px: Callable
) -> str:
    """One hoverable ``<g>`` per hyper-edge object: its polylines, port labels and marker."""
    port_names = data.port_names[name]
    tip = _tip(f"{name} #{i}", list(zip(port_names, data.ports[name][i])), data.features[name][i])
    body: list[str] = []
    for line in geom.lines:
        points_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in to_px(line))
        body.append(
            f'<polyline points="{points_attr}" fill="none"'
            f' stroke="{color_var}" stroke-width="{stroke:.1f}" stroke-opacity="0.85"/>'
        )
    for port_name, at in zip(port_names, geom.labels):
        p = to_px(at)
        body.append(f'<text class="pl" x="{p[0]:.1f}" y="{p[1] - 3:.1f}" text-anchor="middle">{html.escape(port_name)}</text>')
    point = to_px(geom.marker)
    body.append(_svg_marker(shape, point[0], point[1], r_mark, color_var, 'stroke="var(--surface)" stroke-width="1"'))
    return f'<g class="obj" data-tip="{tip}">{"".join(body)}</g>'


def _address_svg(i: int, xy: np.ndarray, r_addr: float) -> str:
    label = (
        f'<text x="{xy[0]:.1f}" y="{xy[1]:.1f}" text-anchor="middle" dominant-baseline="central"'
        f' fill="#ffffff" font-size="{max(round(0.95 * r_addr), 7)}" pointer-events="none">{i}</text>'
    )
    return (
        f'<g class="addr" data-tip="{_tip(f"address {i}", [], {})}">'
        f'<circle cx="{xy[0]:.1f}" cy="{xy[1]:.1f}" r="{r_addr:.1f}" fill="{ADDRESS_COLOR}"/>{label}</g>'
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_graph_interactive(
    graph: Graph,
    *,
    positions: Any = None,
    iterations: int = 150,
    seed: int = 0,
    size: int = 640,
    theme: str = "auto",
) -> InteractiveGraphPlot:
    """
    Render a single Graph as a self-contained interactive HTML/SVG figure.

    Address indices are always visible; hovering any object (address, hyper-edge
    marker or line) shows a tooltip with its port addresses and feature values, and
    reveals the port names along its connections. The mouse wheel zooms, dragging
    pans, and double-click resets the view. The result displays inline in
    Jupyter/IDE notebooks (via ``_repr_html_``) and can be written to a standalone
    HTML file with :meth:`InteractiveGraphPlot.save`. No dependency is required.

    :param graph: A single Graph; batched graphs must first go through
        :func:`energnn.graph.separate_graphs`.
    :param positions: Optional address coordinates of shape ``(n_addresses, 2)`` (e.g.
        latent coordinates from a coupler); replaces the force-directed layout. Padded
        graphs may pass the padded length, fictitious rows are dropped.
    :param iterations: Number of layout relaxation steps (unused when ``positions`` is given).
    :param seed: Seed for the layout's random initial positions.
    :param size: Width and height of the drawing, in pixels.
    :param theme: ``"light"``, ``"dark"``, or ``"auto"`` to follow the viewer's
        color-scheme preference via CSS.
    :return: An :class:`InteractiveGraphPlot`.
    :raises ValueError: If the graph is not single, or if ``theme`` is invalid.
    """
    if theme not in ("light", "dark", "auto"):
        raise ValueError("theme must be 'light', 'dark' or 'auto'.")

    data = extract_plot_data(graph, iterations=iterations, seed=seed, positions=positions)
    geoms = object_geometries(data)

    pad = 30.0

    def to_px(points: np.ndarray) -> np.ndarray:
        return (points + 1.0) / 2.0 * (size - 2 * pad) + pad

    r_addr = float(np.clip(150.0 / np.sqrt(max(data.n_addr, 1)), 5.0, 13.0))
    r_mark = 0.62 * r_addr
    stroke = float(np.clip(r_addr / 6.0, 1.0, 2.0))
    uid = f"energnn-plot-{next(_plot_ids)}"
    n_colors = len(THEMES["light"].palette)

    legend = [f'<span><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="{ADDRESS_COLOR}"/></svg>addresses</span>']
    svg: list[str] = []
    for class_index, name in enumerate(data.classes):
        color_var = f"var(--c{class_index % n_colors})"
        shape = SVG_MARKERS[class_index % len(SVG_MARKERS)]
        legend.append(
            f'<span><svg width="14" height="14">{_svg_marker(shape, 7, 7, 4.5, color_var)}</svg>{html.escape(name)}</span>'
        )
        for i in range(len(data.ports[name])):
            if (name, i) in geoms:  # objects without ports have nothing to draw
                svg.append(_object_svg(name, i, data, geoms[(name, i)], shape, color_var, stroke, r_mark, to_px))
    address_px = to_px(data.pos[: data.n_addr])
    svg += [_address_svg(i, address_px[i], r_addr) for i in range(data.n_addr)]

    fragment = (
        f"<style>{_css(uid, theme, stroke)}</style>"
        f'<div id="{uid}"><div class="lg">{"".join(legend)}</div>'
        f'<svg class="cv" width="{size}" height="{size}" viewBox="0 0 {size} {size}">{"".join(svg)}</svg>'
        f'<div class="tip"></div><script>{_script(uid, size)}</script></div>'
    )
    return InteractiveGraphPlot(fragment)
