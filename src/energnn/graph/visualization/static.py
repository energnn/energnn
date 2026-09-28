# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Static matplotlib rendering of a Graph. Requires the ``viz`` extra (``pip install energnn[viz]``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from energnn.graph.visualization.layout import ObjGeom, ObjKey, PlotData, extract_plot_data, object_geometries
from energnn.graph.visualization.theme import ADDRESS_COLOR, MARKERS, THEMES, resolve_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from energnn.graph.graph import Graph


def _draw_connections(
    ax: Axes, data: PlotData, geoms: dict[ObjKey, ObjGeom], palette: tuple[str, ...], line_width: float, port_labels: bool
) -> None:
    """One line artist per class holding all its polylines (``None`` breaks between them)."""
    for class_index, name in enumerate(data.classes):
        color = palette[class_index % len(palette)]
        segments_x: list[Any] = []
        segments_y: list[Any] = []
        for i in range(len(data.ports[name])):
            geom = geoms.get((name, i))
            if geom is None:  # objects without ports have nothing to draw
                continue
            for line in geom.lines:
                segments_x += list(line[:, 0]) + [None]
                segments_y += list(line[:, 1]) + [None]
            if port_labels:
                for port_name, at in zip(data.port_names[name], geom.labels):
                    ax.annotate(
                        port_name,
                        (float(at[0]), float(at[1])),
                        ha="center",
                        va="center",
                        fontsize=6,
                        color=ADDRESS_COLOR,
                        zorder=4,
                    )
        if segments_x:
            ax.plot(segments_x, segments_y, color=color, linewidth=line_width, alpha=0.85, zorder=1)


def _draw_addresses(ax: Axes, data: PlotData, node_size: float, surface: str, address_labels: bool) -> None:
    pos = data.pos[: data.n_addr]
    ax.scatter(
        pos[:, 0], pos[:, 1], s=node_size, c=ADDRESS_COLOR, edgecolors=surface, linewidths=1.5, zorder=3, label="addresses"
    )
    if address_labels:
        for i in range(data.n_addr):
            ax.annotate(str(i), pos[i], ha="center", va="center", fontsize=7, color="#ffffff", zorder=4)


def _draw_markers(
    ax: Axes, data: PlotData, geoms: dict[ObjKey, ObjGeom], palette: tuple[str, ...], node_size: float, surface: str
) -> None:
    """One scatter artist per class, labelled with the class name for the legend."""
    for class_index, name in enumerate(data.classes):
        markers = np.array([geoms[(name, i)].marker for i in range(len(data.ports[name])) if (name, i) in geoms])
        markers = markers.reshape(-1, 2)
        if len(markers):
            ax.scatter(
                markers[:, 0],
                markers[:, 1],
                s=0.45 * node_size,
                c=palette[class_index % len(palette)],
                marker=MARKERS[class_index % len(MARKERS)],
                edgecolors=surface,
                linewidths=0.8,
                zorder=3.5,
                label=name,
            )


def plot_graph(
    graph: Graph,
    *,
    ax: Axes | None = None,
    address_labels: bool = True,
    port_labels: bool = False,
    positions: Any = None,
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

    Requires ``matplotlib``, installed by the ``viz`` extra: ``pip install energnn[viz]``.

    :param graph: A single Graph; batched graphs must first go through
        :func:`energnn.graph.separate_graphs`.
    :param ax: Axes to draw into; a new figure is created when None.
    :param address_labels: If True, write the address index on each address node.
    :param port_labels: If True, write the port name along each port connection.
    :param positions: Optional address coordinates of shape ``(n_addresses, 2)`` (e.g.
        latent coordinates from a coupler); replaces the force-directed layout. Padded
        graphs may pass the padded length, fictitious rows are dropped.
    :param iterations: Number of layout relaxation steps (unused when ``positions`` is given).
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
        raise ImportError("plot_graph requires matplotlib; install it with 'pip install energnn[viz]'.") from exc

    palette, surface, ink = THEMES[resolve_theme(theme)]
    data = extract_plot_data(graph, iterations=iterations, seed=seed, positions=positions)
    geoms = object_geometries(data)

    if node_size is None:
        node_size = float(np.clip(4000.0 / max(data.n_addr, 1), 12.0, 130.0))
    line_width = float(np.clip(1.4 * np.sqrt(node_size / 130.0), 0.7, 1.4))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.set_aspect("equal")
    ax.axis("off")

    _draw_connections(ax, data, geoms, palette, line_width, port_labels)
    _draw_addresses(ax, data, node_size, surface, address_labels)
    _draw_markers(ax, data, geoms, palette, node_size, surface)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, labelcolor=ink, fontsize=9)
    return ax
