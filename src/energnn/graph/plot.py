# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

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
_ADDRESS_COLOR = "#898781"


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


def plot_graph(
    graph: Graph,
    *,
    ax: Axes | None = None,
    address_labels: bool = False,
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

    Requires ``matplotlib``.

    :param graph: A single Graph; batched graphs must first go through
        :func:`energnn.graph.separate_graphs`.
    :param ax: Axes to draw into; a new figure is created when None.
    :param address_labels: If True, write the address index on each address node.
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

    if not graph.is_single:
        raise ValueError("plot_graph only handles single graphs; use separate_graphs() on a batch first.")

    palette, surface, ink = _THEMES[_resolve_theme(theme)]

    g = graph.to_numpy_backend()
    n_addr = int(np.asarray(g.non_fictitious_addresses).sum())
    if node_size is None:
        node_size = float(np.clip(4000.0 / max(n_addr, 1), 12.0, 130.0))
    line_width = float(np.clip(1.4 * np.sqrt(node_size / 130.0), 0.7, 1.4))

    # Collect the port lists of real (non-fictitious) hyper-edges, per class.
    classes = sorted(g.hyper_edge_sets)
    edges_by_class: dict[str, list[list[int]]] = {}
    for name in classes:
        hes = g.hyper_edge_sets[name]
        mask = np.asarray(hes.non_fictitious) > 0
        if hes.port_dict is None:
            edges_by_class[name] = []
            continue
        ports = np.stack([np.asarray(hes.port_dict[k]) for k in sorted(hes.port_dict)], axis=-1)
        edges_by_class[name] = [list(map(int, row)) for row in ports[mask]]

    # Star expansion for the layout: addresses in [0, n_addr), then one hub per order>=3 edge.
    layout_edges: list[tuple[int, int]] = []
    hub_ids: dict[tuple[str, int], int] = {}
    next_id = n_addr
    for name in classes:
        for i, edge_ports in enumerate(edges_by_class[name]):
            if len(edge_ports) == 2:
                layout_edges.append((edge_ports[0], edge_ports[1]))
            elif len(edge_ports) >= 3:
                hub_ids[(name, i)] = next_id
                layout_edges.extend((next_id, p) for p in edge_ports)
                next_id += 1
    pos = spring_layout(next_id, np.array(layout_edges, dtype=int).reshape(-1, 2), iterations=iterations, seed=seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.set_aspect("equal")
    ax.axis("off")

    for class_index, name in enumerate(classes):
        color = palette[class_index % len(palette)]
        segments_x: list[Any] = []
        segments_y: list[Any] = []
        for i, edge_ports in enumerate(edges_by_class[name]):
            if len(edge_ports) == 2:
                a, b = pos[edge_ports[0]], pos[edge_ports[1]]
                segments_x += [a[0], b[0], None]
                segments_y += [a[1], b[1], None]
            elif len(edge_ports) >= 3:
                hub = pos[hub_ids[(name, i)]]
                for p in edge_ports:
                    segments_x += [hub[0], pos[p][0], None]
                    segments_y += [hub[1], pos[p][1], None]
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

    for class_index, name in enumerate(classes):
        color = palette[class_index % len(palette)]
        marker = _MARKERS[class_index % len(_MARKERS)]
        xs, ys = [], []
        for i, edge_ports in enumerate(edges_by_class[name]):
            if len(edge_ports) == 1:
                # Deterministic radial offset so several order-1 edges on one address stay visible.
                angle = 2.0 * np.pi * ((class_index * 0.37 + i * 0.61) % 1.0)
                point = pos[edge_ports[0]] + 0.045 * np.array([np.cos(angle), np.sin(angle)])
            elif len(edge_ports) == 2:
                point = (pos[edge_ports[0]] + pos[edge_ports[1]]) / 2.0
            elif len(edge_ports) >= 3:
                point = pos[hub_ids[(name, i)]]
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
