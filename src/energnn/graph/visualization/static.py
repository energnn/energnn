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
from energnn.graph.visualization.layout import ObjGeom, ObjKey, PlotData, extract_plot_data, object_geometries
from energnn.graph.visualization.theme import MARKERS, THEMES, Theme, bivariate_rgb, channels_to_rgb, resolve_theme, rgb_to_hex

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
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


def _style_axes(ax: Axes, surface: str) -> None:
    ax.set_facecolor(surface)
    if _is_3d(ax):
        ax3d = cast(Any, ax)
        ax3d.set_axis_off()
        ax3d.set_xlim(-1.1, 1.1)
        ax3d.set_ylim(-1.1, 1.1)
        ax3d.set_zlim(-1.1, 1.1)
        ax3d.set_box_aspect((1, 1, 1))
    else:
        ax.set_aspect("equal")
        ax.axis("off")


def _draw_connections(ax: Axes, data: PlotData, geoms: dict[ObjKey, ObjGeom], style: _Style) -> None:
    """One line artist per class holding all its polylines (NaN breaks between them)."""
    for class_index, name in enumerate(data.classes):
        color = style.theme.palette[class_index % len(style.theme.palette)] if style.edge_colors else style.theme.neutral
        chunks: list[np.ndarray] = []
        for i in range(len(data.ports[name])):
            geom = geoms.get((name, i))
            if geom is None:  # objects without ports have nothing to draw
                continue
            for line in geom.lines:
                chunks += [line, np.full((1, 3), np.nan)]
            if style.port_labels:
                for port_name, at in zip(data.port_names[name], geom.labels):
                    _text(ax, at, port_name, ha="center", va="center", fontsize=6, color=style.theme.neutral, zorder=4)
        if chunks:
            _plot(ax, np.concatenate(chunks), color=color, linewidth=style.line_width, alpha=0.85, zorder=1)


def _draw_addresses(ax: Axes, data: PlotData, frame: int, style: _Style) -> None:
    """Hollow circles outlined in the theme's ink (filled with the address colors when given)."""
    pos = data.pos[frame, : data.n_addr]
    face: Any = style.theme.surface
    if data.colors is not None:
        face = rgb_to_hex(channels_to_rgb(data.colors[frame], style.theme))
    _scatter(ax, pos, s=style.node_size, c=face, edgecolors=style.theme.ink, linewidths=1.2, zorder=3, label="addresses")
    if style.address_labels:
        for i in range(data.n_addr):
            _text(ax, pos[i], str(i), ha="center", va="center", fontsize=7, color=style.theme.ink, zorder=4)


def _draw_markers(ax: Axes, data: PlotData, geoms: dict[ObjKey, ObjGeom], style: _Style) -> None:
    """One scatter artist per class, labelled with the class name for the legend."""
    for class_index, name in enumerate(data.classes):
        markers = np.array([geoms[(name, i)].marker for i in range(len(data.ports[name])) if (name, i) in geoms])
        if len(markers):
            _scatter(
                ax,
                markers.reshape(-1, 3),
                s=0.45 * style.node_size,
                c=style.theme.palette[class_index % len(style.theme.palette)] if style.edge_colors else style.theme.neutral,
                marker=MARKERS[class_index % len(MARKERS)],
                edgecolors=style.theme.surface,
                linewidths=0.8,
                zorder=3.5,
                label=name,
            )


def _render_frame(ax: Axes, data: PlotData, frame: int, style: _Style) -> None:
    """Draw one frame into ``ax`` (which must hold no artists yet)."""
    geoms = object_geometries(data, frame)
    _style_axes(ax, style.theme.surface)
    _draw_connections(ax, data, geoms, style)
    _draw_addresses(ax, data, frame, style)
    _draw_markers(ax, data, geoms, style)
    _legend(ax, style)


def _legend(ax: Axes, style: _Style) -> None:
    """Class markers from the scatter artists; a hollow circle for the addresses whatever their fill."""
    from matplotlib.lines import Line2D

    addresses = Line2D(
        [], [], linestyle="none", marker="o", markersize=7, markerfacecolor=style.theme.surface,
        markeredgecolor=style.theme.ink, markeredgewidth=1.2, label="addresses",
    )  # fmt: skip
    classes = [artist for artist in ax.collections if artist.get_label() != "addresses"]
    ax.legend(
        handles=[addresses, *classes],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        labelcolor=style.theme.ink,
        fontsize=9,
    )


def _add_color_legend(ax: Axes, data: PlotData, theme: Theme) -> None:
    """Colorbar for 1-channel address colors, bivariate square for 2 channels, nothing for RGB."""
    import matplotlib.colors
    from matplotlib.cm import ScalarMappable

    if data.colors is None or data.color_range is None:
        return
    lo, hi = data.color_range
    if data.colors.shape[-1] == 1:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("energnn", theme.sequential)
        mappable = ScalarMappable(norm=matplotlib.colors.Normalize(float(lo[0]), float(hi[0])), cmap=cmap)
        colorbar = _figure(ax).colorbar(mappable, ax=ax, shrink=0.3, pad=0.02, anchor=(0.0, 0.0))
        cast(Any, colorbar).outline.set_visible(False)
        colorbar.ax.tick_params(labelsize=7, colors=theme.ink)
    elif data.colors.shape[-1] == 2:
        grid = np.linspace(0.0, 1.0, 32)
        u, v = np.meshgrid(grid, grid)
        inset = ax.inset_axes((1.02, 0.5, 0.14, 0.14))
        inset.imshow(bivariate_rgb(u, v, theme), origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]), aspect="auto")
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

    image = OffsetImage(matplotlib.image.imread(io.BytesIO(logo_png()), format="png"), zoom=0.3, alpha=0.9)
    box = AnnotationBbox(image, (1.0, 0.0), xycoords="axes fraction", box_alignment=(1.0, 0.0), frameon=False, pad=0.0)
    box.set_zorder(5)
    ax.add_artist(box)


def _figure(ax: Axes) -> Figure:
    return cast("Figure", ax.get_figure(root=True))


def _prepare(
    graph: Graph, positions: Any, address_colors: Any, iterations: int, seed: int, theme: str, node_size: float | None, **flags
) -> tuple[PlotData, _Style]:
    resolved = THEMES[resolve_theme(theme)]
    data = extract_plot_data(graph, iterations=iterations, seed=seed, positions=positions, address_colors=address_colors)
    if node_size is None:
        node_size = float(np.clip(4000.0 / max(data.n_addr, 1), 12.0, 130.0))
    line_width = float(np.clip(1.4 * np.sqrt(node_size / 130.0), 0.7, 1.4))
    return data, _Style(resolved, node_size, line_width, **flags)


def _axes_for(ax: Axes | None, data: PlotData, surface: str) -> Axes:
    import matplotlib.pyplot as plt

    if ax is None:
        # constrained layout keeps the legend and color scale, placed outside the axes, inside the figure
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
    positions: Any = None,
    address_colors: Any = None,
    edge_colors: bool = True,
    frame: int = 0,
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

    For an interactive version with feature tooltips, see :func:`plot_graph_interactive`;
    for a series of frames, see :func:`animate_graph`.

    Requires ``matplotlib``, installed by the ``viz`` extra: ``pip install energnn[viz]``.

    :param graph: A single Graph; batched graphs must first go through
        :func:`energnn.graph.separate_graphs`.
    :param ax: Axes to draw into; a new figure is created when None. 3D positions need
        an Axes created with ``projection="3d"``.
    :param address_labels: If True, write the address index on each address node.
    :param port_labels: If True, write the port name along each port connection.
    :param positions: Optional address coordinates of shape ``(n_addresses, 2)`` or
        ``(n_addresses, 3)`` (e.g. latent coordinates from a coupler); replaces the
        force-directed layout. A leading axis gives a series of frames, see ``frame``.
        Padded graphs may pass the padded length, fictitious rows are dropped.
    :param address_colors: Optional per-address values of shape ``(n_addresses, C)`` with
        ``C`` in {1, 2, 3}: 1 channel is mapped through the sequential colormap (with a
        colorbar), 2 channels through the bivariate colormap (with its legend), 3 channels
        are RGB. Values are normalized per channel over all frames. A leading axis gives a
        series of frames.
    :param edge_colors: If False, hyper-edges are drawn in the neutral gray instead of one
        color per class (marker shapes still tell classes apart).
    :param frame: Index of the frame to draw when ``positions`` or ``address_colors`` have
        a time axis.
    :param iterations: Number of layout relaxation steps (unused when ``positions`` is given).
    :param seed: Seed for the layout's random initial positions.
    :param node_size: Address marker area; inferred from the number of addresses when None.
    :param theme: ``"light"``, ``"dark"``, or ``"auto"`` to follow matplotlib's current
        figure facecolor (e.g. dark notebook themes).
    :param logo: If True, draw the EnerGNN mark in the bottom-right corner.
    :return: The matplotlib Axes containing the plot.
    :raises ImportError: If matplotlib is not installed.
    :raises ValueError: If the graph is not single, if ``theme`` is invalid, or if the
        positions/colors arrays have a wrong shape.
    """
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise ImportError("plot_graph " + _IMPORT_HINT) from exc

    flags = {"address_labels": address_labels, "port_labels": port_labels, "edge_colors": edge_colors}
    data, style = _prepare(graph, positions, address_colors, iterations, seed, theme, node_size, **flags)
    if not 0 <= frame < data.n_frames:
        raise IndexError(f"frame {frame} out of range for {data.n_frames} frame(s).")
    ax = _axes_for(ax, data, style.theme.surface)
    _render_frame(ax, data, frame, style)
    _add_color_legend(ax, data, style.theme)
    if logo:
        _add_logo(ax)
    return ax


def animate_graph(
    graph: Graph,
    *,
    positions: Any = None,
    address_colors: Any = None,
    interval: int = 100,
    ax: Axes | None = None,
    address_labels: bool = True,
    port_labels: bool = False,
    edge_colors: bool = True,
    iterations: int = 150,
    seed: int = 0,
    node_size: float | None = None,
    theme: str = "auto",
    logo: bool = True,
) -> FuncAnimation:
    """
    Animate a Graph over the frames of ``positions`` and/or ``address_colors``.

    Same drawing as :func:`plot_graph`, one frame per time step. Display the result in a
    notebook with ``IPython.display.HTML(animation.to_jshtml())`` and save it with
    ``animation.save("graph.gif")`` (or ``.mp4`` with ffmpeg installed).

    :param graph: A single Graph.
    :param positions: Address coordinates with a leading time axis, ``(n_frames, n_addresses, 2 or 3)``.
    :param address_colors: Per-address channels with a leading time axis, ``(n_frames, n_addresses, C)``.
    :param interval: Delay between frames, in milliseconds.
    :param ax: Axes to draw into; a new figure is created when None.
    :return: A :class:`matplotlib.animation.FuncAnimation`; keep a reference to it while it plays.
    :raises ValueError: If neither ``positions`` nor ``address_colors`` has a time axis.
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError as exc:
        raise ImportError("animate_graph " + _IMPORT_HINT) from exc

    flags = {"address_labels": address_labels, "port_labels": port_labels, "edge_colors": edge_colors}
    data, style = _prepare(graph, positions, address_colors, iterations, seed, theme, node_size, **flags)
    if data.n_frames < 2:
        raise ValueError("animate_graph needs positions or address_colors with a leading time axis of length >= 2.")
    ax = _axes_for(ax, data, style.theme.surface)
    _render_frame(ax, data, 0, style)
    _add_color_legend(ax, data, style.theme)
    if logo:
        _add_logo(ax)

    def update(frame: int) -> list:
        for artist in [*ax.lines, *ax.collections, *ax.texts]:
            artist.remove()
        _render_frame(ax, data, frame, style)
        return []

    return FuncAnimation(_figure(ax), update, frames=data.n_frames, interval=interval, blit=False)
