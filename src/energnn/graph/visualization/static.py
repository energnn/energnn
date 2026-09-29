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
from energnn.graph.visualization.layout import FeatureSpec, ObjGeom, ObjKey, PlotData, extract_plot_data, object_geometries
from energnn.graph.visualization.theme import MARKERS, THEMES, Theme, bivariate_rgb, channels_to_rgb, resolve_theme, rgb_to_hex

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from energnn.graph.graph import Graph

_IMPORT_HINT = "requires matplotlib; install it with 'pip install energnn[viz]'."


class _Style(NamedTuple):
    theme: Theme
    node_size: float
    line_width: float
    address_labels: bool
    port_labels: bool
    edge_colors: bool


# matplotlib's 3D axes share the 2D API with an extra coordinate; the helpers below hide the
# difference (and the ``Any`` casts hide it from the type checker, which only knows 2D axes).


def _is_3d(ax: Axes) -> bool:
    return hasattr(ax, "zaxis")


def _plot(ax: Axes, points: np.ndarray, **kwargs: Any) -> None:
    if _is_3d(ax):
        cast(Any, ax).plot(points[:, 0], points[:, 1], points[:, 2], **kwargs)
    else:
        ax.plot(points[:, 0], points[:, 1], **kwargs)


def _plot_collection(ax: Axes, lines: list[np.ndarray], colors: list[str], **kwargs: Any) -> None:
    """One collection holding several polylines, each with its own color."""
    if _is_3d(ax):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection  # type: ignore[import-untyped]

        cast(Any, ax).add_collection3d(Line3DCollection(lines, colors=colors, **kwargs))
    else:
        from matplotlib.collections import LineCollection

        ax.add_collection(LineCollection([line[:, :2] for line in lines], colors=colors, **kwargs))


def _scatter(ax: Axes, points: np.ndarray, **kwargs: Any) -> None:
    if _is_3d(ax):
        cast(Any, ax).scatter(points[:, 0], points[:, 1], points[:, 2], **kwargs)
    else:
        ax.scatter(points[:, 0], points[:, 1], **kwargs)


def _text(ax: Axes, at: np.ndarray, text: str, **kwargs: Any) -> None:
    if _is_3d(ax):
        cast(Any, ax).text(float(at[0]), float(at[1]), float(at[2]), text, **kwargs)
    else:
        ax.annotate(text, (float(at[0]), float(at[1])), **kwargs)


def _style_axes(ax: Axes, surface: str, margin: float) -> None:
    ax.set_facecolor(surface)
    if _is_3d(ax):
        ax3d = cast(Any, ax)
        ax3d.set_axis_off()
        limit = 1.05 + margin
        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_box_aspect((1, 1, 1))
    else:
        ax.set_aspect("equal")
        ax.axis("off")


def _class_color(class_index: int, style: _Style) -> str:
    return style.theme.palette[class_index % len(style.theme.palette)] if style.edge_colors else style.theme.neutral


def _object_colors(data: PlotData, name: str, class_index: int, style: _Style) -> list[str] | None:
    """Per-object colors of a class colored by ``hyper_edge_colors`` (the class color where the value is missing)."""
    channels = data.object_colors.get(name)
    if channels is None:
        return None
    colors = rgb_to_hex(channels_to_rgb(channels, style.theme))
    fallback = _class_color(class_index, style)
    return [fallback if missing else color for color, missing in zip(colors, data.missing_object_colors[name])]


def _draw_connections(ax: Axes, data: PlotData, geoms: dict[ObjKey, ObjGeom], style: _Style) -> None:
    """Per class, one line artist holding all its polylines (NaN breaks between them), or one collection when
    its objects carry their own colors."""
    for class_index, name in enumerate(data.classes):
        colors = _object_colors(data, name, class_index, style)
        lines: list[np.ndarray] = []
        line_colors: list[str] = []
        for i in range(len(data.ports[name])):
            geom = geoms.get((name, i))
            if geom is None:  # objects without ports have nothing to draw
                continue
            lines += geom.lines
            line_colors += [colors[i] if colors else _class_color(class_index, style)] * len(geom.lines)
            if style.port_labels:
                for port_name, at in zip(data.port_names[name], geom.labels):
                    _text(ax, at, port_name, ha="center", va="center", fontsize=6, color=style.theme.neutral, zorder=4)
        if not lines:
            continue
        if colors is None:
            chunks = [chunk for line in lines for chunk in (line, np.full((1, 3), np.nan))]
            _plot(ax, np.concatenate(chunks), color=line_colors[0], linewidth=style.line_width, alpha=0.85, zorder=1)
        else:
            _plot_collection(ax, lines, line_colors, linewidths=style.line_width, alpha=0.85, zorder=1)


def _draw_addresses(ax: Axes, data: PlotData, style: _Style) -> None:
    """Hollow circles outlined in the theme's ink (filled with the address colors when given)."""
    pos = data.pos[: data.n_addr]
    face: Any = style.theme.surface
    if data.colors is not None:
        face = rgb_to_hex(channels_to_rgb(data.colors, style.theme))
        if data.missing_colors is not None:  # a missing color leaves the address hollow
            face = [style.theme.surface if missing else color for color, missing in zip(face, data.missing_colors)]
    # inferred positions (not given) get a dashed outline
    linestyles = ["dashed" if inferred else "solid" for inferred in data.inferred]
    _scatter(
        ax, pos, s=style.node_size, c=face, edgecolors=style.theme.ink, linewidths=1.2, linestyles=linestyles, zorder=3,
        label="addresses",
    )  # fmt: skip
    if style.address_labels:
        for i in range(data.n_addr):
            _text(ax, pos[i], str(i), ha="center", va="center", fontsize=7, color=style.theme.ink, zorder=4)


def _draw_markers(ax: Axes, data: PlotData, geoms: dict[ObjKey, ObjGeom], style: _Style) -> None:
    """One scatter artist per class, labelled with the class name."""
    for class_index, name in enumerate(data.classes):
        drawn = [i for i in range(len(data.ports[name])) if (name, i) in geoms]
        if not drawn:
            continue
        colors = _object_colors(data, name, class_index, style)
        _scatter(
            ax,
            np.array([geoms[(name, i)].marker for i in drawn]).reshape(-1, 3),
            s=0.45 * style.node_size,
            c=[colors[i] for i in drawn] if colors else _class_color(class_index, style),
            marker=MARKERS[class_index % len(MARKERS)],
            edgecolors=style.theme.surface,
            linewidths=0.8,
            zorder=3.5,
            label=name,
        )


def _legend(ax: Axes, data: PlotData, style: _Style) -> None:
    """A hollow circle for the addresses whatever their fill, one marker per class in its class color."""
    from matplotlib.lines import Line2D

    def marker(shape: str, face: str, **kwargs: Any) -> Line2D:
        return Line2D([], [], linestyle="none", marker=shape, markersize=7, markerfacecolor=face, **kwargs)

    handles: list[Any] = [
        marker("o", style.theme.surface, markeredgecolor=style.theme.ink, markeredgewidth=1.2, label="addresses")
    ]
    if data.inferred.any():
        from matplotlib.patches import Patch

        handles.append(
            Patch(
                facecolor=style.theme.surface, edgecolor=style.theme.ink, linestyle="--", label="address (position inferred)"
            )
        )
    for class_index, name in enumerate(data.classes):
        handles.append(
            marker(
                MARKERS[class_index % len(MARKERS)], _class_color(class_index, style), markeredgecolor=style.theme.surface,
                markeredgewidth=0.8, label=name,
            )  # fmt: skip
        )
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, labelcolor=style.theme.ink, fontsize=9
    )


def _add_color_legend(
    ax: Axes, colors: np.ndarray | None, color_range: np.ndarray | None, theme: Theme, label: str, slot: int
):
    """Colorbar for 1-channel colors, bivariate square for 2 channels, nothing for RGB; ``slot`` stacks several."""
    import matplotlib.colors
    from matplotlib.cm import ScalarMappable

    if colors is None or color_range is None:
        return
    lo, hi = color_range
    if colors.shape[-1] == 1:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("energnn", theme.sequential)
        mappable = ScalarMappable(norm=matplotlib.colors.Normalize(float(lo[0]), float(hi[0])), cmap=cmap)
        colorbar = _figure(ax).colorbar(mappable, ax=ax, shrink=0.3, pad=0.02, anchor=(0.0, 0.0), label=label)
        cast(Any, colorbar).outline.set_visible(False)
        colorbar.ax.tick_params(labelsize=7, colors=theme.ink)
        colorbar.ax.yaxis.label.set_color(theme.ink)
        colorbar.ax.yaxis.label.set_fontsize(7)
    elif colors.shape[-1] == 2:
        grid = np.linspace(0.0, 1.0, 32)
        u, v = np.meshgrid(grid, grid)
        inset = ax.inset_axes((1.02, 0.02 + 0.34 * slot, 0.14, 0.14))  # bottom right, clear of the class legend
        inset.imshow(bivariate_rgb(u, v, theme), origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]), aspect="auto")
        inset.set_title(label, fontsize=6, color=theme.ink, pad=2)
        inset.set_xlabel("channel 1", fontsize=6, color=theme.ink, labelpad=1)
        inset.set_ylabel("channel 2", fontsize=6, color=theme.ink, labelpad=1)
        inset.set_xticks([lo[0], hi[0]])
        inset.set_yticks([lo[1], hi[1]])
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


def _figure(ax: Axes) -> Figure:
    return cast("Figure", ax.get_figure(root=True))


def _axes_for(ax: Axes | None, data: PlotData, surface: str) -> Axes:
    import matplotlib.pyplot as plt

    if ax is None:
        # constrained layout keeps the legend and color scales, placed outside the axes, inside the figure
        fig = plt.figure(figsize=(7, 7), layout="constrained")
        ax = fig.add_subplot(projection="3d" if data.ndim == 3 else None)
        fig.set_facecolor(surface)
    elif data.ndim == 3 and not _is_3d(ax):
        raise ValueError("3D positions need an Axes created with projection='3d'.")
    return ax


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

    Addresses are drawn as hollow circles (filled with ``address_colors`` when given).
    Hyper-edges of order 1 are drawn as a small marker attached to their address,
    hyper-edges of order 2 as a line between their two addresses with a marker at
    midpoint, and hyper-edges of order 3 or more as a hub marker connected to all
    their ports. Fictitious (padded) objects and addresses are skipped.

    For an interactive version with feature tooltips, see :func:`plot_graph_interactive`.

    Requires ``matplotlib``, installed by the ``viz`` extra: ``pip install energnn[viz]``.

    :param graph: A single Graph; batched graphs must first go through
        :func:`energnn.graph.separate_graphs`.
    :param ax: Axes to draw into; a new figure is created when None. 3D positions need
        an Axes created with ``projection="3d"``.
    :param address_labels: If True, write the address index on each address node.
    :param port_labels: If True, write the port name along each port connection.
    :param address_positions: Optional address coordinates of shape ``(n_addresses, 2)`` or
        ``(n_addresses, 3)`` (e.g. latent coordinates from a coupler); replaces the
        force-directed layout. Padded graphs may pass the padded length, fictitious rows are
        dropped. NaN rows are reconstructed from the graph (each missing address at the mean of
        its neighbors) and drawn with a dashed outline.
    :param hyper_edge_positions: Optional ``{class: [x_feature, y_feature]}`` (or three features
        for 3D): the objects of that class are drawn at the coordinates held by those features,
        as a marker with one spoke per port. Addresses without ``address_positions`` sit at the
        mean position of the placed objects pointing to them, the others being reconstructed as
        above.
    :param address_colors: Optional per-address values of shape ``(n_addresses, C)`` with
        ``C`` in {1, 2, 3}: 1 channel is mapped through the sequential colormap (with a
        colorbar), 2 channels through the bivariate colormap (with its legend), 3 channels
        are RGB. Values are normalized per channel. A NaN leaves the address uncolored.
    :param hyper_edge_colors: Optional ``{class: [feature, ...]}`` with 1, 2 or 3 features: the
        markers and lines of those objects are colored from these features like the addresses
        above, with their own color scale shared by every listed class. A NaN keeps the class
        color.
    :param edge_colors: If False, hyper-edges are drawn in the neutral gray instead of one
        color per class (marker shapes still tell classes apart). This is also what happens as
        soon as ``hyper_edge_colors`` is given, so that only the feature colors carry a meaning.
    :param iterations: Number of layout relaxation steps (unused when positions are given).
    :param seed: Seed for the layout's random initial positions.
    :param node_size: Address marker area; inferred from the number of addresses when None.
    :param theme: ``"light"``, ``"dark"``, or ``"auto"`` to follow matplotlib's current
        figure facecolor (e.g. dark notebook themes).
    :param logo: If True, draw the EnerGNN mark in the bottom-right corner.
    :return: The matplotlib Axes containing the plot.
    :raises ImportError: If matplotlib is not installed.
    :raises ValueError: If the graph is not single, if ``theme`` is invalid, if the
        positions/colors arrays have a wrong shape, or if a feature spec names an unknown class
        or feature.
    """
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise ImportError("plot_graph " + _IMPORT_HINT) from exc

    resolved = THEMES[resolve_theme(theme)]
    data = extract_plot_data(
        graph,
        iterations=iterations,
        seed=seed,
        address_positions=address_positions,
        hyper_edge_positions=hyper_edge_positions,
        address_colors=address_colors,
        hyper_edge_colors=hyper_edge_colors,
    )
    if node_size is None:
        node_size = float(np.clip(4000.0 / max(data.n_addr, 1), 12.0, 130.0))
    line_width = float(np.clip(1.4 * np.sqrt(node_size / 130.0), 0.7, 1.4))
    # once a class is colored by its features, the others are drawn in neutral so the colormap stands alone
    style = _Style(resolved, node_size, line_width, address_labels, port_labels, edge_colors and not data.object_colors)

    ax = _axes_for(ax, data, style.theme.surface)
    geoms = object_geometries(data)
    _style_axes(ax, style.theme.surface, data.margin)
    _draw_connections(ax, data, geoms, style)
    _draw_addresses(ax, data, style)
    _draw_markers(ax, data, geoms, style)
    _legend(ax, data, style)
    _add_color_legend(ax, data.colors, data.color_range, style.theme, "addresses", 0)
    if data.object_colors:
        stacked = np.concatenate(list(data.object_colors.values()))
        _add_color_legend(
            ax, stacked, data.object_color_range, style.theme, "hyper-edges", 1 if data.colors is not None else 0
        )
    if logo:
        _add_logo(ax)
    return ax
