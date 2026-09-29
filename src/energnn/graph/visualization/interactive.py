# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Interactive, dependency-free HTML/SVG rendering of a Graph.

Python extracts the topology (which addresses each object connects, how parallel edges
are fanned out) and the per-frame address positions and colors; the embedded script
(``assets/plot.js``) computes the geometry, so the view can be rotated (3D), zoomed,
panned and stepped through time without a server.
"""

from __future__ import annotations

import html
import itertools
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from energnn.graph.visualization.assets import logo_data_uri, script_js
from energnn.graph.visualization.layout import (
    FAN_HEIGHT,
    STUB_LENGTH,
    PlotData,
    address_radius,
    extract_plot_data,
    object_descriptors,
)
from energnn.graph.visualization.theme import SVG_MARKERS, THEMES, Theme

if TYPE_CHECKING:
    from energnn.graph.graph import Graph

_plot_ids = itertools.count()
_PAD = 30  # canvas margin in pixels, as in plot.js


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
# Static HTML pieces
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
    """Tooltip HTML for one object (names and values escaped; travels through the JSON payload)."""
    parts = [f"<b>{html.escape(title)}</b>"]
    parts += [f"{html.escape(pn)} &rarr; {addr}" for pn, addr in port_lines]
    parts += [f"{html.escape(fn)} = {value:.5g}" for fn, value in feature_lines.items()]
    return "<br>".join(parts)


def _theme_vars(t: Theme) -> str:
    slots = "".join(f"--c{i}:{c};" for i, c in enumerate(t.palette))
    seq = "".join(f"--s{i}:{c};" for i, c in enumerate(t.sequential))
    biv = "".join(f"--b{k}:{c};" for k, c in zip(("00", "10", "01", "11"), t.bivariate))
    return f"--surface:{t.surface};--ink:{t.ink};--neutral:{t.neutral};{slots}{seq}{biv}"


def _theme_css(uid: str, theme: str) -> str:
    """CSS custom properties for the palette; with ``auto`` the script adds the ``dark`` class when the notebook is dark."""
    if theme == "auto":
        return f"#{uid}{{{_theme_vars(THEMES['light'])}}}#{uid}.dark{{{_theme_vars(THEMES['dark'])}}}"
    return f"#{uid}{{{_theme_vars(THEMES[theme])}}}"


def _css(uid: str, theme: str, stroke: float, logo_width: int) -> str:
    return (
        f"{_theme_css(uid, theme)}"
        f"#{uid}{{position:relative;display:inline-block;font-family:system-ui,sans-serif;"
        f"background:var(--surface);border-radius:6px;color:var(--ink)}}"
        f"#{uid} .lg{{display:flex;flex-wrap:wrap;align-items:center;gap:4px 14px;padding:8px 12px 0;font-size:12px}}"
        f"#{uid} .lg span{{display:inline-flex;align-items:center;gap:5px}}"
        f"#{uid} .lg .sc{{font-size:9px;gap:3px}}"
        f"#{uid} .obj .pl{{opacity:0;fill:var(--ink);font-size:9px;pointer-events:none}}"
        f"#{uid} .obj:hover .pl{{opacity:1}}"
        f"#{uid} .obj:hover polyline{{stroke-width:{2.2 * stroke:.1f}px}}"
        f"#{uid} .obj:hover .mk{{stroke:var(--ink);stroke-width:1.5px}}"
        f"#{uid} .addr:hover circle{{stroke-width:2.5px}}"
        f"#{uid} .tip{{display:none;position:absolute;pointer-events:none;background:var(--surface);"
        f"border:1px solid var(--neutral);border-radius:4px;padding:5px 8px;font-size:11px;line-height:1.5;"
        f"white-space:nowrap;z-index:10}}"
        f"#{uid} .cw{{position:relative}}"
        f"#{uid} svg.cv{{cursor:grab;display:block}}"
        f"#{uid} svg.cv:active{{cursor:grabbing}}"
        f"#{uid} .logo{{position:absolute;right:10px;bottom:8px;width:{logo_width}px;opacity:0.9;"
        f"pointer-events:none}}"
        f"#{uid} .tb{{position:absolute;top:8px;right:10px;display:flex;gap:4px}}"
        f"#{uid} .tb button{{font:inherit;font-size:13px;width:26px;height:26px;padding:0;border:1px solid var(--neutral);"
        f"border-radius:4px;background:var(--surface);color:var(--ink);cursor:pointer;opacity:0.85}}"
        f"#{uid} .tb button.on{{background:var(--ink);color:var(--surface)}}"
        f"#{uid} .tl{{display:flex;align-items:center;gap:8px;padding:4px 12px 8px;font-size:11px}}"
        f"#{uid} .tl input{{flex:1}}"
        f"#{uid} .tl button{{font:inherit;padding:1px 8px;border:1px solid var(--neutral);border-radius:4px;"
        f"background:var(--surface);color:var(--ink);cursor:pointer}}"
    )


def _payload(data: PlotData, size: int, edge_colors: bool, interval: int, loop_pause: int, theme: str) -> dict[str, Any]:
    """Everything the script needs, JSON-serializable."""
    r_units = address_radius(data.n_addr)
    r_addr = r_units * (size - 2 * _PAD) / (2.0 + 2.0 * data.margin)  # in pixels, on the padded canvas
    r_mark = 0.62 * r_addr
    descriptors = object_descriptors(data)
    n_colors = len(THEMES["light"].palette)
    classes = []
    for class_index, name in enumerate(data.classes):
        objects = []
        for i, item in enumerate(descriptors[name]):
            tip = _tip(f"{name} #{i}", list(zip(data.port_names[name], item["ports"])), data.features[name][i])
            objects.append(item | {"tip": tip})
        classes.append(
            {
                "name": name,
                "shape": SVG_MARKERS[class_index % len(SVG_MARKERS)],
                "color": f"var(--c{class_index % n_colors})" if edge_colors else None,
                "portNames": data.port_names[name],
                "objects": objects,
            }
        )
    return {
        "size": size,
        "ndim": data.ndim,
        "nAddr": data.n_addr,
        "rAddr": r_addr,
        "addrR": r_units,
        "stub": STUB_LENGTH,
        "fanH": FAN_HEIGHT,
        "margin": data.margin,
        "stroke": float(np.clip(r_addr / 6.0, 1.0, 2.0)),
        "fontSize": max(round(0.95 * r_addr), 7),
        "markers": {shape: [[round(x, 2), round(y, 2)] for x, y in _marker_points(shape, r_mark)] for shape in SVG_MARKERS},
        "frames": np.round(data.pos, 4).tolist(),
        "colors": None if data.colors is None else np.round(data.colors, 4).tolist(),
        "classes": classes,
        "addrTips": [_tip(f"address {i}", [], {}) for i in range(data.n_addr)],
        "interval": interval,
        "pause": loop_pause,
        "autoTheme": theme == "auto",
    }


def _legend_html(data: PlotData, edge_colors: bool) -> str:
    n_colors = len(THEMES["light"].palette)
    items = [
        '<span><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="var(--surface)" stroke="var(--ink)"'
        ' stroke-width="1.2"/></svg>addresses</span>'
    ]
    for class_index, name in enumerate(data.classes):
        color = f"var(--c{class_index % n_colors})" if edge_colors else "var(--neutral)"
        shape = SVG_MARKERS[class_index % len(SVG_MARKERS)]
        items.append(
            f'<span><svg width="14" height="14">{_svg_marker(shape, 7, 7, 4.5, color)}</svg>{html.escape(name)}</span>'
        )
    if data.colors is not None and data.color_range is not None:
        lo, hi = data.color_range
        n_channels = data.colors.shape[-1]
        if n_channels == 1:
            items.append(f'<span class="sc">{lo[0]:.3g}<!--scale-->{hi[0]:.3g}</span>')
        elif n_channels == 2:
            items.append(
                f'<span class="sc">ch1 {lo[0]:.3g}&ndash;{hi[0]:.3g}<!--scale-->ch2 {lo[1]:.3g}&ndash;{hi[1]:.3g}</span>'
            )
        else:
            items.append('<span class="sc">RGB</span>')
    return "".join(items)


def _toolbar_html(ndim: int) -> str:
    """Mode buttons (drag rotates in 3D, or pans) and zoom in / zoom out / reset actions."""
    buttons = []
    if ndim == 3:
        buttons.append('<button type="button" data-mode="rotate" title="drag rotates the view">&#x21bb;</button>')
    buttons += [
        '<button type="button" data-mode="pan" title="drag pans the view">&#x2725;</button>',
        '<button type="button" data-act="zin" title="zoom in">+</button>',
        '<button type="button" data-act="zout" title="zoom out">&minus;</button>',
        '<button type="button" data-act="reset" title="reset the view">&#x2302;</button>',
    ]
    return f'<div class="tb">{"".join(buttons)}</div>'


def plot_graph_interactive(
    graph: Graph,
    *,
    positions: Any = None,
    address_colors: Any = None,
    edge_colors: bool = True,
    iterations: int = 150,
    seed: int = 0,
    size: int = 640,
    theme: str = "auto",
    logo: bool = True,
    interval: int = 100,
    loop_pause: int = 1000,
) -> InteractiveGraphPlot:
    """
    Render a single Graph as a self-contained interactive HTML/SVG figure.

    Address indices are always visible; hovering any object (address, hyper-edge
    marker or line) shows a tooltip with its port addresses and feature values, and
    reveals the port names along its connections. The mouse wheel zooms (markers,
    lines and labels keep their size), dragging pans (or rotates the view for 3D
    positions, shift-drag then pans), double-click resets the view, and the toolbar
    offers the same. When ``positions`` or ``address_colors`` carry a
    time axis, a slider and a play button step through the frames; playback
    interpolates positions and colors between frames and pauses at the end of
    the series before looping. The result
    displays inline in Jupyter/IDE notebooks (via ``_repr_html_``) and can be
    written to a standalone HTML file with :meth:`InteractiveGraphPlot.save`.
    No dependency is required.

    :param graph: A single Graph; batched graphs must first go through
        :func:`energnn.graph.separate_graphs`.
    :param positions: Optional address coordinates of shape ``(n_addresses, 2)`` or
        ``(n_addresses, 3)``; replaces the force-directed layout. A leading axis gives a
        series of frames. Padded graphs may pass the padded length, fictitious rows are dropped.
    :param address_colors: Optional per-address values of shape ``(n_addresses, C)`` with
        ``C`` in {1, 2, 3}: sequential colormap, bivariate colormap or RGB; normalized per
        channel over all frames. A leading axis gives a series of frames.
    :param edge_colors: If False, hyper-edges are drawn in the neutral gray instead of one
        color per class (marker shapes still tell classes apart).
    :param iterations: Number of layout relaxation steps (unused when ``positions`` is given).
    :param seed: Seed for the layout's random initial positions.
    :param size: Width and height of the drawing, in pixels.
    :param theme: ``"light"``, ``"dark"``, or ``"auto"`` to follow the notebook's theme (the
        background color of the output cell, or the OS preference when it cannot be read).
    :param logo: If True, draw the EnerGNN mark in the bottom-right corner.
    :param interval: Duration of one frame when playing a time series, in milliseconds.
    :param loop_pause: Pause at the end of the series before looping, in milliseconds.
    :return: An :class:`InteractiveGraphPlot`.
    :raises ValueError: If the graph is not single, if ``theme`` is invalid, or if the
        positions/colors arrays have a wrong shape.
    """
    if theme not in ("light", "dark", "auto"):
        raise ValueError("theme must be 'light', 'dark' or 'auto'.")

    data = extract_plot_data(graph, iterations=iterations, seed=seed, positions=positions, address_colors=address_colors)
    payload = _payload(data, size, edge_colors, interval, loop_pause, theme)
    uid = f"energnn-plot-{next(_plot_ids)}"

    timeline = ""
    if data.n_frames > 1:
        timeline = (
            f'<div class="tl"><button type="button">&#x25B6;</button>'
            f'<input type="range" min="0" max="{data.n_frames - 1}" value="0" step="1"/><span class="fr"></span></div>'
        )
    hint = ", drag to rotate or pan (toolbar), shift-drag to pan" if data.ndim == 3 else ", drag to pan"
    # the logo and the toolbar sit over the canvas, outside the SVG, so zoom and pan leave them in place
    logo_html = f'<img class="logo" src="{logo_data_uri()}" alt="EnerGNN"/>' if logo else ""
    toolbar = _toolbar_html(data.ndim)
    fragment = (
        f"<style>{_css(uid, theme, payload['stroke'], max(round(0.15 * size), 60))}</style>"
        f'<div id="{uid}" title="scroll to zoom{hint}, double-click to reset">'
        f'<div class="lg">{_legend_html(data, edge_colors)}</div>'
        f'<div class="cw"><svg class="cv" width="{size}" height="{size}" viewBox="0 0 {size} {size}"></svg>'
        f"{toolbar}{logo_html}</div>"
        f'{timeline}<div class="tip"></div>'
        f'<script type="application/json">{json.dumps(payload, separators=(",", ":"))}</script>'
        f"<script>{script_js().replace('__UID__', uid)}</script></div>"
    )
    return InteractiveGraphPlot(fragment)
