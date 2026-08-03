# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import html
import itertools
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from energnn.graph.graph import Graph


class _Theme(NamedTuple):
    palette: tuple[str, ...]
    surface: str
    ink: str


# Colorblind-validated categorical palettes; slot order is part of the validation, and the
# dark palette is the same hues re-stepped for a dark surface, not an automatic inversion.
_THEMES = {
    "light": _Theme(
        palette=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
        surface="#fcfcfb",
        ink="#0b0b0b",
    ),
    "dark": _Theme(
        palette=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
        surface="#1a1a19",
        ink="#ffffff",
    ),
}
# Marker shapes double the color encoding so classes stay separable without color.
_MARKERS = ["s", "^", "D", "v", "P", "X", "p", "*"]
_SVG_MARKERS = ["square", "triangle-up", "diamond", "triangle-down", "plus", "cross", "pentagon", "star"]
_ADDRESS_COLOR = "#898781"

_plot_ids = itertools.count()


def _resolve_theme(theme: str) -> str:
    """Return "light" or "dark"; "auto" follows the luminance of matplotlib's figure facecolor."""
    if theme in _THEMES:
        return theme
    if theme != "auto":
        raise ValueError("theme must be 'light', 'dark' or 'auto'.")
    import matplotlib

    r, g, b = matplotlib.colors.to_rgb(matplotlib.rcParams["figure.facecolor"])
    return "dark" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "light"


def spring_layout(n_nodes: int, edges: np.ndarray, *, iterations: int = 150, seed: int = 0) -> np.ndarray:
    """
    Compute a Fruchterman-Reingold force-directed layout.

    :param n_nodes: Number of nodes to place.
    :param edges: Integer array of shape ``(n_edges, 2)`` listing node pairs.
    :param iterations: Number of relaxation steps.
    :param seed: Seed for the random initial positions.
    :return: Positions array of shape ``(n_nodes, 2)`` scaled to ``[-1, 1]``.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n_nodes, 2))
    if n_nodes <= 1:
        return pos
    k = 1.0 / np.sqrt(n_nodes)
    temperature = 0.1
    cooling = temperature / (iterations + 1)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
    if len(edges):
        adjacency[edges[:, 0], edges[:, 1]] = True
        adjacency[edges[:, 1], edges[:, 0]] = True
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dist, 1.0)
        dist = np.maximum(dist, 0.01)
        force = k * k / dist**2 - adjacency * dist / k
        displacement = (delta * force[..., None]).sum(axis=1)
        length = np.maximum(np.linalg.norm(displacement, axis=-1, keepdims=True), 1e-9)
        pos += displacement / length * np.minimum(length, temperature)
        temperature -= cooling
    pos -= pos.mean(axis=0)
    scale = np.abs(pos).max()
    return pos / scale if scale > 0 else pos


class _PlotData(NamedTuple):
    n_addr: int
    classes: list[str]
    ports: dict[str, list[list[int]]]  # class -> per real object, port addresses (sorted port order)
    port_names: dict[str, list[str]]  # class -> sorted port names
    features: dict[str, list[dict[str, float]]]  # class -> per real object, feature name -> value
    pos: np.ndarray  # star-expansion layout: addresses then hubs
    hub_ids: dict[tuple[str, int], int]  # (class, object index) -> hub row in pos


def _extract_plot_data(graph: Graph, *, iterations: int, seed: int) -> _PlotData:
    """Collect real (non-fictitious) objects of a single Graph and lay them out."""
    if not graph.is_single:
        raise ValueError("plot_graph only handles single graphs; use separate_graphs() on a batch first.")

    g = graph.to_numpy_backend()
    n_addr = int(np.asarray(g.non_fictitious_addresses).sum())

    classes = sorted(g.hyper_edge_sets)
    ports: dict[str, list[list[int]]] = {}
    port_names: dict[str, list[str]] = {}
    features: dict[str, list[dict[str, float]]] = {}
    for name in classes:
        hes = g.hyper_edge_sets[name]
        mask = np.asarray(hes.non_fictitious) > 0
        if hes.port_dict is not None:
            port_names[name] = sorted(hes.port_dict)
            stacked = np.stack([np.asarray(hes.port_dict[k]) for k in port_names[name]], axis=-1)
            ports[name] = [list(map(int, row)) for row in stacked[mask]]
        else:
            port_names[name] = []
            ports[name] = [[] for _ in range(int(mask.sum()))]
        if hes.feature_names is not None:
            feature_array = np.asarray(hes.feature_array)[mask]
            features[name] = [
                {fn: float(feature_array[j, int(idx)]) for fn, idx in sorted(hes.feature_names.items())}
                for j in range(len(feature_array))
            ]
        else:
            features[name] = [{} for _ in range(len(ports[name]))]

    layout_edges: list[tuple[int, int]] = []
    hub_ids: dict[tuple[str, int], int] = {}
    next_id = n_addr
    for name in classes:
        for i, edge_ports in enumerate(ports[name]):
            if len(edge_ports) == 2:
                layout_edges.append((edge_ports[0], edge_ports[1]))
            elif len(edge_ports) >= 3:
                hub_ids[(name, i)] = next_id
                layout_edges.extend((next_id, p) for p in edge_ports)
                next_id += 1
    pos = spring_layout(next_id, np.array(layout_edges, dtype=int).reshape(-1, 2), iterations=iterations, seed=seed)

    return _PlotData(n_addr, classes, ports, port_names, features, pos, hub_ids)


def plot_graph(
    graph: Graph,
    *,
    ax: Axes | None = None,
    address_labels: bool = True,
    port_labels: bool = False,
    iterations: int = 150,
    seed: int = 0,
    node_size: float | None = None,
    theme: str = "auto",
) -> Axes:
    """
    Plot a single (non-batched) Graph with one color and marker per hyper-edge class.

    Addresses are drawn as gray circles. Hyper-edges of order 1 are drawn as a small
    marker attached to their address, hyper-edges of order 2 as a line between their
    two addresses with a marker at midpoint, and hyper-edges of order 3 or more as a
    hub marker connected to all their ports. Fictitious (padded) objects and
    addresses are skipped.

    For an interactive version with feature tooltips, see :func:`plot_graph_interactive`.

    Requires ``matplotlib``.

    :param graph: A single Graph; batched graphs must first go through
        :func:`energnn.graph.separate_graphs`.
    :param ax: Axes to draw into; a new figure is created when None.
    :param address_labels: If True, write the address index on each address node.
    :param port_labels: If True, write the port name along each port connection.
    :param iterations: Number of layout relaxation steps.
    :param seed: Seed for the layout's random initial positions.
    :param node_size: Address marker area; inferred from the number of addresses when None.
    :param theme: ``"light"``, ``"dark"``, or ``"auto"`` to follow matplotlib's current
        figure facecolor (e.g. dark notebook themes).
    :return: The matplotlib Axes containing the plot.
    :raises ImportError: If matplotlib is not installed.
    :raises ValueError: If the graph is not single, or if ``theme`` is invalid.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_graph requires matplotlib; install it with 'pip install matplotlib'.") from exc

    palette, surface, ink = _THEMES[_resolve_theme(theme)]
    data = _extract_plot_data(graph, iterations=iterations, seed=seed)
    n_addr, pos = data.n_addr, data.pos

    if node_size is None:
        node_size = float(np.clip(4000.0 / max(n_addr, 1), 12.0, 130.0))
    line_width = float(np.clip(1.4 * np.sqrt(node_size / 130.0), 0.7, 1.4))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.set_aspect("equal")
    ax.axis("off")

    def _port_label(text: str, at: np.ndarray) -> None:
        ax.annotate(text, (float(at[0]), float(at[1])), ha="center", va="center", fontsize=6, color=_ADDRESS_COLOR, zorder=4)

    for class_index, name in enumerate(data.classes):
        color = palette[class_index % len(palette)]
        segments_x: list[Any] = []
        segments_y: list[Any] = []
        for i, edge_ports in enumerate(data.ports[name]):
            if len(edge_ports) == 2:
                a, b = pos[edge_ports[0]], pos[edge_ports[1]]
                segments_x += [a[0], b[0], None]
                segments_y += [a[1], b[1], None]
                if port_labels:
                    _port_label(data.port_names[name][0], a + 0.22 * (b - a))
                    _port_label(data.port_names[name][1], b + 0.22 * (a - b))
            elif len(edge_ports) >= 3:
                hub = pos[data.hub_ids[(name, i)]]
                for port_index, p in enumerate(edge_ports):
                    segments_x += [hub[0], pos[p][0], None]
                    segments_y += [hub[1], pos[p][1], None]
                    if port_labels:
                        _port_label(data.port_names[name][port_index], (hub + pos[p]) / 2.0)
        if segments_x:
            ax.plot(segments_x, segments_y, color=color, linewidth=line_width, alpha=0.85, zorder=1)

    ax.scatter(
        pos[:n_addr, 0],
        pos[:n_addr, 1],
        s=node_size,
        c=_ADDRESS_COLOR,
        edgecolors=surface,
        linewidths=1.5,
        zorder=3,
        label="addresses",
    )
    if address_labels:
        for i in range(n_addr):
            ax.annotate(str(i), pos[i], ha="center", va="center", fontsize=7, color="#ffffff", zorder=4)

    for class_index, name in enumerate(data.classes):
        color = palette[class_index % len(palette)]
        marker = _MARKERS[class_index % len(_MARKERS)]
        xs, ys = [], []
        for i, edge_ports in enumerate(data.ports[name]):
            if len(edge_ports) == 1:
                # Deterministic radial offset so several order-1 edges on one address stay visible.
                angle = 2.0 * np.pi * ((class_index * 0.37 + i * 0.61) % 1.0)
                point = pos[edge_ports[0]] + 0.045 * np.array([np.cos(angle), np.sin(angle)])
                if port_labels:
                    _port_label(data.port_names[name][0], point + np.array([0.0, -0.03]))
            elif len(edge_ports) == 2:
                point = (pos[edge_ports[0]] + pos[edge_ports[1]]) / 2.0
            elif len(edge_ports) >= 3:
                point = pos[data.hub_ids[(name, i)]]
            else:
                continue
            xs.append(point[0])
            ys.append(point[1])
        if xs:
            ax.scatter(
                xs, ys, s=0.45 * node_size, c=color, marker=marker, edgecolors=surface, linewidths=0.8, zorder=3.5, label=name
            )

    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, labelcolor=ink, fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# Interactive (HTML/SVG) plot
# ---------------------------------------------------------------------------


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


def plot_graph_interactive(
    graph: Graph,
    *,
    iterations: int = 150,
    seed: int = 0,
    size: int = 640,
    theme: str = "auto",
) -> InteractiveGraphPlot:
    """
    Render a single Graph as a self-contained interactive HTML/SVG figure.

    Address indices are always visible; hovering any object (address, hyper-edge
    marker or line) shows a tooltip with its port addresses and feature values, and
    reveals the port names along its connections. The result displays inline in
    Jupyter/IDE notebooks (via ``_repr_html_``) and can be written to a standalone
    HTML file with :meth:`InteractiveGraphPlot.save`. No dependency is required.

    :param graph: A single Graph; batched graphs must first go through
        :func:`energnn.graph.separate_graphs`.
    :param iterations: Number of layout relaxation steps.
    :param seed: Seed for the layout's random initial positions.
    :param size: Width and height of the drawing, in pixels.
    :param theme: ``"light"``, ``"dark"``, or ``"auto"`` to follow the viewer's
        color-scheme preference via CSS.
    :return: An :class:`InteractiveGraphPlot`.
    :raises ValueError: If the graph is not single, or if ``theme`` is invalid.
    """
    if theme not in ("light", "dark", "auto"):
        raise ValueError("theme must be 'light', 'dark' or 'auto'.")

    data = _extract_plot_data(graph, iterations=iterations, seed=seed)
    n_addr, pos = data.n_addr, data.pos

    pad = 30.0
    xy = (pos + 1.0) / 2.0 * (size - 2 * pad) + pad
    r_addr = float(np.clip(150.0 / np.sqrt(max(n_addr, 1)), 5.0, 13.0))
    r_mark = 0.62 * r_addr
    stroke = float(np.clip(r_addr / 6.0, 1.0, 2.0))
    uid = f"energnn-plot-{next(_plot_ids)}"

    light, dark = _THEMES["light"], _THEMES["dark"]

    def _vars(t: _Theme) -> str:
        slots = "".join(f"--c{i}:{c};" for i, c in enumerate(t.palette))
        return f"--surface:{t.surface};--ink:{t.ink};{slots}"

    if theme == "auto":
        theme_css = f"#{uid}{{{_vars(light)}}}" f"@media (prefers-color-scheme: dark){{#{uid}{{{_vars(dark)}}}}}"
    else:
        theme_css = f"#{uid}{{{_vars(_THEMES[theme])}}}"

    css = (
        f"{theme_css}"
        f"#{uid}{{position:relative;display:inline-block;font-family:system-ui,sans-serif;"
        f"background:var(--surface);border-radius:6px}}"
        f"#{uid} .lg{{display:flex;flex-wrap:wrap;gap:4px 14px;padding:8px 12px 0;color:var(--ink);font-size:12px}}"
        f"#{uid} .lg span{{display:inline-flex;align-items:center;gap:5px}}"
        f"#{uid} .obj .pl{{opacity:0;fill:var(--ink);font-size:9px;pointer-events:none}}"
        f"#{uid} .obj:hover .pl{{opacity:1}}"
        f"#{uid} .obj:hover line{{stroke-width:{2.2 * stroke:.1f}px}}"
        f"#{uid} .obj:hover .mk,#{uid} .addr:hover circle{{stroke:var(--ink);stroke-width:1.5px}}"
        f"#{uid} .tip{{display:none;position:absolute;pointer-events:none;background:var(--surface);color:var(--ink);"
        f"border:1px solid {_ADDRESS_COLOR};border-radius:4px;padding:5px 8px;font-size:11px;line-height:1.5;"
        f"white-space:nowrap;z-index:10}}"
    )

    svg: list[str] = []
    legend: list[str] = [
        f'<span><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="{_ADDRESS_COLOR}"/></svg>addresses</span>'
    ]

    for class_index, name in enumerate(data.classes):
        color_var = f"var(--c{class_index % len(light.palette)})"
        shape = _SVG_MARKERS[class_index % len(_SVG_MARKERS)]
        legend.append(
            f'<span><svg width="14" height="14">{_svg_marker(shape, 7, 7, 4.5, color_var)}</svg>{html.escape(name)}</span>'
        )
        port_names = data.port_names[name]
        for i, edge_ports in enumerate(data.ports[name]):
            tip = _tip(f"{name} #{i}", list(zip(port_names, edge_ports)), data.features[name][i])
            body: list[str] = []

            def _spoke(a: np.ndarray, b: np.ndarray, label: str, frac: float = 0.5) -> None:
                body.append(
                    f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}"'
                    f' stroke="{color_var}" stroke-width="{stroke:.1f}" stroke-opacity="0.85"/>'
                )
                at = a + frac * (b - a)
                body.append(
                    f'<text class="pl" x="{at[0]:.1f}" y="{at[1] - 3:.1f}" text-anchor="middle">{html.escape(label)}</text>'
                )

            if len(edge_ports) == 1:
                angle = 2.0 * np.pi * ((class_index * 0.37 + i * 0.61) % 1.0)
                point = xy[edge_ports[0]] + 2.2 * r_addr * np.array([np.cos(angle), np.sin(angle)])
                _spoke(xy[edge_ports[0]], point, port_names[0], frac=0.55)
            elif len(edge_ports) == 2:
                a, b = xy[edge_ports[0]], xy[edge_ports[1]]
                _spoke(a, b, port_names[0], frac=0.25)
                body.append(
                    f'<text class="pl" x="{(b + 0.25 * (a - b))[0]:.1f}" y="{(b + 0.25 * (a - b))[1] - 3:.1f}"'
                    f' text-anchor="middle">{html.escape(port_names[1])}</text>'
                )
                point = (a + b) / 2.0
            elif len(edge_ports) >= 3:
                point = xy[data.hub_ids[(name, i)]]
                for port_index, p in enumerate(edge_ports):
                    _spoke(point, xy[p], port_names[port_index])
            else:
                continue
            body.append(_svg_marker(shape, point[0], point[1], r_mark, color_var, 'stroke="var(--surface)" stroke-width="1"'))
            svg.append(f'<g class="obj" data-tip="{tip}">{"".join(body)}</g>')

    for i in range(n_addr):
        label = (
            f'<text x="{xy[i][0]:.1f}" y="{xy[i][1]:.1f}" text-anchor="middle" dominant-baseline="central"'
            f' fill="#ffffff" font-size="{max(round(0.95 * r_addr), 7)}" pointer-events="none">{i}</text>'
        )
        svg.append(
            f'<g class="addr" data-tip="{_tip(f"address {i}", [], {})}">'
            f'<circle cx="{xy[i][0]:.1f}" cy="{xy[i][1]:.1f}" r="{r_addr:.1f}" fill="{_ADDRESS_COLOR}"/>{label}</g>'
        )

    script = (
        f"(function(){{var root=document.getElementById('{uid}');var tip=root.querySelector('.tip');"
        f"root.querySelectorAll('[data-tip]').forEach(function(el){{"
        f"el.addEventListener('mousemove',function(e){{tip.innerHTML=el.getAttribute('data-tip');"
        f"tip.style.display='block';var r=root.getBoundingClientRect();"
        f"tip.style.left=(e.clientX-r.left+14)+'px';tip.style.top=(e.clientY-r.top+14)+'px';}});"
        f"el.addEventListener('mouseleave',function(){{tip.style.display='none';}});}});}})();"
    )

    fragment = (
        f"<style>{css}</style>"
        f'<div id="{uid}"><div class="lg">{"".join(legend)}</div>'
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">{"".join(svg)}</svg>'
        f'<div class="tip"></div><script>{script}</script></div>'
    )
    return InteractiveGraphPlot(fragment)
