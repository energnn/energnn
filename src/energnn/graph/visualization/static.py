# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Static matplotlib rendering of a Graph. Requires the ``viz`` extra (``pip install energnn[viz]``)."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

from energnn.graph.visualization.assets import logo_png
from energnn.graph.visualization.colors import ColorScale, Colors, resolve_colors
from energnn.graph.visualization.content import FeatureSpec, Topology, read_graph
from energnn.graph.visualization.geometry import Geometry, geometries
from energnn.graph.visualization.positions import MARKER_RATIO, Layout, lay_out
from energnn.graph.visualization.theme import MARKERS, THEMES, Theme, bivariate_rgb, channels_to_rgb, resolve_theme, rgb_to_hex

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from energnn.graph.graph import Graph

_IMPORT_HINT = "requires matplotlib; install it with 'pip install energnn[viz]'."
_ADDRESS_AREA = (4000.0, 12.0, 130.0)  # address marker area: 4000 / n clipped to [12, 130] points²


class _Scene(NamedTuple):
    topology: Topology
    layout: Layout
    colors: Colors
    geoms: dict[tuple[str, int], Geometry]
    theme: Theme
    node_size: float
    line_width: float
    class_colors: dict[str, str]  # class -> its color (palette, or neutral)


def _class_colors(topology: Topology, theme: Theme, edge_colors: bool) -> dict[str, str]:
    palette = theme.palette
    return {cls: palette[i % len(palette)] if edge_colors else theme.neutral for i, cls in enumerate(topology.classes)}


def _fills(scale: ColorScale | None, fallback: list[str], theme: Theme) -> list[str]:
    """One color per row of ``scale``: from the channels, or the fallback where the value is missing."""
    if scale is None:
        return fallback
    colors = rgb_to_hex(channels_to_rgb(scale.channels, theme))
    return [back if missing else color for color, back, missing in zip(colors, fallback, scale.missing)]


def _xy(point: np.ndarray) -> tuple[float, float]:
    return float(point[0]), float(point[1])


def _draw_connections(ax: Axes, scene: _Scene, port_labels: bool) -> None:
    """One line artist per class (NaN breaks between its polylines), or one collection when its objects carry
    their own colors."""
    from matplotlib.collections import LineCollection

    colored = scene.colors.hyper_edges
    fills = _fills(colored, [scene.class_colors[h.cls] for h in scene.topology.hyper_edges], scene.theme)
    for cls in scene.topology.classes:
        lines: list[np.ndarray] = []
        colors: list[str] = []
        for row, h in enumerate(scene.topology.hyper_edges):
            geom = scene.geoms.get(h.key)
            if h.cls != cls or geom is None:
                continue
            lines += geom.lines
            colors += [fills[row]] * len(geom.lines)
            if port_labels:
                for name, at in zip(scene.topology.port_names[cls], geom.labels):
                    ax.annotate(name, _xy(at), ha="center", va="center", fontsize=6, color=scene.theme.neutral, zorder=4)
        if not lines:
            continue
        if colored is None:
            chunks = [chunk for line in lines for chunk in (line, np.full((1, 2), np.nan))]
            ax.plot(*np.concatenate(chunks).T, color=colors[0], linewidth=scene.line_width, alpha=0.85, zorder=1)
        else:
            ax.add_collection(LineCollection(lines, colors=colors, linewidths=scene.line_width, alpha=0.85, zorder=1))


def _draw_addresses(ax: Axes, scene: _Scene, address_labels: bool) -> None:
    """Hollow circles outlined in the theme's ink, filled with the address colors when given."""
    pos = scene.layout.addresses
    face = _fills(scene.colors.addresses, [scene.theme.surface] * len(pos), scene.theme)
    ax.scatter(pos[:, 0], pos[:, 1], s=scene.node_size, c=face, edgecolors=scene.theme.ink, linewidths=1.2, zorder=3)
    if address_labels:
        for i, at in enumerate(pos):
            ax.annotate(str(i), _xy(at), ha="center", va="center", fontsize=7, color=scene.theme.ink, zorder=4)


def _draw_markers(ax: Axes, scene: _Scene) -> None:
    """One scatter artist per class, labelled with the class name."""
    fills = _fills(scene.colors.hyper_edges, [scene.class_colors[h.cls] for h in scene.topology.hyper_edges], scene.theme)
    for class_index, cls in enumerate(scene.topology.classes):
        rows = [row for row, h in enumerate(scene.topology.hyper_edges) if h.cls == cls and h.key in scene.geoms]
        if not rows:
            continue
        markers = np.array([scene.geoms[scene.topology.hyper_edges[row].key].marker for row in rows])
        ax.scatter(
            markers[:, 0], markers[:, 1], s=MARKER_RATIO**2 * scene.node_size * 1.2, c=[fills[row] for row in rows],
            marker=MARKERS[class_index % len(MARKERS)], edgecolors=scene.theme.surface, linewidths=0.8, zorder=3.5, label=cls,
        )  # fmt: skip


def _legend(ax: Axes, scene: _Scene) -> None:
    """A hollow circle for the addresses whatever their fill, one marker per class in its class color."""
    from matplotlib.lines import Line2D

    def marker(shape: str, face: str, edge: str, label: str) -> Line2D:
        return Line2D(
            [], [], linestyle="none", marker=shape, markersize=7, markerfacecolor=face, markeredgecolor=edge, label=label
        )

    handles = [marker("o", scene.theme.surface, scene.theme.ink, "addresses")]
    for class_index, cls in enumerate(scene.topology.classes):
        handles.append(marker(MARKERS[class_index % len(MARKERS)], scene.class_colors[cls], scene.theme.surface, cls))
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, labelcolor=scene.theme.ink, fontsize=9
    )


def _color_legend(ax: Axes, scale: ColorScale | None, theme: Theme, label: str, slot: int) -> None:
    """A colorbar for 1 channel, a bivariate square for 2; ``slot`` stacks the squares of the addresses and the
    hyper-edges."""
    import matplotlib.colors
    from matplotlib.cm import ScalarMappable

    if scale is None:
        return
    if scale.n_channels == 1:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("energnn", theme.sequential)
        mappable = ScalarMappable(norm=matplotlib.colors.Normalize(float(scale.low[0]), float(scale.high[0])), cmap=cmap)
        colorbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.3, pad=0.02, anchor=(0.0, 0.0), label=label)
        cast(Any, colorbar).outline.set_visible(False)
        colorbar.ax.tick_params(labelsize=7, colors=theme.ink)
        colorbar.ax.yaxis.label.set_color(theme.ink)
        colorbar.ax.yaxis.label.set_fontsize(7)
    else:
        grid = np.linspace(0.0, 1.0, 32)
        u, v = np.meshgrid(grid, grid)
        inset = ax.inset_axes((1.02, 0.02 + 0.34 * slot, 0.14, 0.14))  # bottom right, clear of the class legend
        extent = (scale.low[0], scale.high[0], scale.low[1], scale.high[1])
        inset.imshow(bivariate_rgb(u, v, theme), origin="lower", extent=extent, aspect="auto")
        inset.set_title(label, fontsize=6, color=theme.ink, pad=2)
        inset.set_xlabel("channel 1", fontsize=6, color=theme.ink, labelpad=1)
        inset.set_ylabel("channel 2", fontsize=6, color=theme.ink, labelpad=1)
        inset.set_xticks([scale.low[0], scale.high[0]])
        inset.set_yticks([scale.low[1], scale.high[1]])
        inset.tick_params(labelsize=5, colors=theme.ink, length=2)
        for side in ("left", "right", "top", "bottom"):
            inset.spines[side].set_visible(False)


def _add_logo(ax: Axes) -> None:
    """The EnerGNN mark in the bottom-right corner of the axes, at a fixed pixel size."""
    import matplotlib.image
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    image = OffsetImage(matplotlib.image.imread(io.BytesIO(logo_png()), format="png"), zoom=0.3, alpha=0.9)  # ~96 px wide
    box = AnnotationBbox(image, (1.0, 0.0), xycoords="axes fraction", box_alignment=(1.0, 0.0), frameon=False, pad=0.0)
    box.set_zorder(5)
    ax.add_artist(box)


def plot_graph(
    graph: Graph,
    *,
    ax: Axes | None = None,
    address_labels: bool = True,
    port_labels: bool = False,
    address_positions: Any = None,
    hyper_edge_positions: FeatureSpec | None = None,
    address_colors: Any = None,
    hyper_edge_colors: FeatureSpec | None = None,
    edge_colors: bool = True,
    iterations: int = 150,
    seed: int = 0,
    node_size: float | None = None,
    theme: str = "auto",
    logo: bool = True,
) -> Axes:
    """
    Plot a single (non-batched) Graph with one color and marker per hyper-edge class.

    Addresses are drawn as hollow circles (filled with ``address_colors`` when given). Hyper-edges of
    order 1 are a small marker attached to their address, hyper-edges of order 2 a line between their two
    addresses with a marker at midpoint, hyper-edges of order 3 or more a hub marker connected to all
    their ports. Fictitious (padded) objects and addresses are skipped. For an interactive version with
    feature tooltips, see :func:`plot_graph_interactive`.

    Requires ``matplotlib``, installed by the ``viz`` extra: ``pip install energnn[viz]``.

    :param graph: A single Graph; batched graphs must first go through :func:`energnn.graph.separate_graphs`.
    :param ax: Axes to draw into; a new figure is created when None.
    :param address_labels: If True, write the address index on each address node.
    :param port_labels: If True, write the port name along each port connection.
    :param address_positions: Optional address coordinates of shape ``(n_addresses, 2)`` (e.g. latent
        coordinates from a coupler); replaces the force-directed layout. Padded graphs may pass the padded
        length, fictitious rows are dropped.
    :param hyper_edge_positions: Optional ``{class: [x_feature, y_feature]}``: the objects of that class are
        drawn at the coordinates held by those features, as a marker with one spoke per port. Without
        ``address_positions``, each address sits at the mean position of the placed objects pointing to it,
        and every address must be pointed to by one.
    :param address_colors: Optional per-address values of shape ``(n_addresses, C)`` with ``C`` in {1, 2}:
        1 channel is mapped through the sequential colormap (with a colorbar), 2 channels through the
        bivariate colormap (with its legend). Values are normalized per channel. A NaN leaves the address
        uncolored.
    :param hyper_edge_colors: Optional ``{class: [feature, ...]}`` with 1 or 2 features: the markers and
        lines of those objects are colored from these features like the addresses above, with their own color
        scale shared by every listed class. Every other class is then drawn in neutral gray, so that only the
        feature colors carry a meaning. A NaN keeps that neutral color.
    :param edge_colors: If False, hyper-edges are drawn in the neutral gray instead of one color per class
        (marker shapes still tell classes apart).
    :param iterations: Number of layout relaxation steps (unused when positions are given).
    :param seed: Seed for the layout's random initial positions.
    :param node_size: Address marker area; inferred from the number of addresses when None.
    :param theme: ``"light"``, ``"dark"``, or ``"auto"`` to follow matplotlib's current figure facecolor
        (e.g. dark notebook themes).
    :param logo: If True, draw the EnerGNN mark in the bottom-right corner.
    :return: The matplotlib Axes containing the plot.
    :raises ImportError: If matplotlib is not installed.
    :raises ValueError: If the graph is not single, if ``theme`` is invalid, if an array has a wrong shape,
        or if a feature spec names an unknown class or feature.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_graph " + _IMPORT_HINT) from exc

    resolved = THEMES[resolve_theme(theme)]
    topology = read_graph(graph, hyper_edge_positions=hyper_edge_positions)
    layout = lay_out(topology, address_positions=address_positions, iterations=iterations, seed=seed)
    colors = resolve_colors(topology, address_colors=address_colors, hyper_edge_colors=hyper_edge_colors)
    if node_size is None:
        area, low, high = _ADDRESS_AREA
        node_size = float(np.clip(area / max(topology.n_addresses, 1), low, high))
    line_width = float(np.clip(1.4 * np.sqrt(node_size / _ADDRESS_AREA[2]), 0.7, 1.4))
    # once a class is colored by its features, the others are drawn in neutral so the colormap stands alone
    class_colors = _class_colors(topology, resolved, edge_colors and colors.hyper_edges is None)
    scene = _Scene(topology, layout, colors, geometries(topology, layout), resolved, node_size, line_width, class_colors)

    if ax is None:
        # constrained layout keeps the legend and color scales, placed outside the axes, inside the figure
        fig = plt.figure(figsize=(7, 7), layout="constrained")
        ax = fig.add_subplot()
        fig.set_facecolor(resolved.surface)
    ax.set_facecolor(resolved.surface)
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_connections(ax, scene, port_labels)
    _draw_addresses(ax, scene, address_labels)
    _draw_markers(ax, scene)
    _legend(ax, scene)
    _color_legend(ax, colors.addresses, resolved, "addresses", 0)
    _color_legend(ax, colors.hyper_edges, resolved, "hyper-edges", 1 if colors.addresses is not None else 0)
    if logo:
        _add_logo(ax)
    return ax
