# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Interactive rendering of a Graph as a plotly figure. Requires the ``viz`` extra (``pip install energnn[viz]``).

**How a graph is drawn.** The addresses are numbered circles. Every hyper-edge is a marker, whose shape and
color tell its class, with one straight line (a *spoke*) to the address of each of its ports. The marker sits
at the barycenter of these addresses: a line between two buses is thus a segment with its marker in the middle,
a three-winding transformer a star. Two cases would hide a marker, and are handled by :func:`_markers`:

- hyper-edges pointing to a single address (a bus, a generator) are spread around it, a short distance away;
- hyper-edges pointing to the same addresses (parallel lines) are fanned out on either side of their barycenter.

**How it works.** :func:`plot_graph` works on a copy of the Graph without its padding (:meth:`Graph.unpad`),
in which the classes that draw nothing (without port or without hyper-edge) are ignored. Then :func:`_positions`
places the addresses in the ``[-1, 1]`` box, from the user's coordinates or from a force-directed layout
(:mod:`.layout`), :func:`_markers` places the hyper-edges, :func:`_shades` turns the values to display into
colors, and the figure is assembled from plotly *traces*, a trace being a set of points or lines sharing a
style: for each class, one trace for its markers and one for its spokes, then one trace for the addresses.

Big graphs use traces of the ``scattergl`` type, drawn by the graphics card (WebGL), which keeps zoom and pan
fluid with hundreds of thousands of hyper-edges, provided that the browser has access to a graphics card. Small
graphs use the ``scatter`` type, drawn as SVG like the rest of a web page: a page cannot hold more than a few
WebGL figures at once, and a notebook is a single page. Zoom, pan, tooltips, the legend and the export to HTML
are plotly's.

**How it is displayed.** The figure is a :class:`GraphFigure`, a plotly figure. A notebook displays it as a
piece of HTML (:data:`_NOTEBOOK_HTML`) holding the figure and a short script: the script has plotly.js draw the
figure, and gives it the light or the dark colors (:data:`THEMES`) according to the notebook around it. Only a
script can do that, since the Python side cannot see the theme of the notebook.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from energnn.graph.visualization.layout import spring_layout

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.colors import sample_colorscale
    from plotly.offline import get_plotlyjs_version
except ImportError as exc:  # pragma: no cover
    raise ImportError("energnn.graph.visualization requires plotly; install it with 'pip install energnn[viz]'.") from exc

if TYPE_CHECKING:
    from energnn.graph.graph import Graph
    from energnn.graph.hyper_edge_set import HyperEdgeSet


class Theme(NamedTuple):
    """The named colors of one theme, all as ``"#rrggbb"`` strings."""

    surface: str  #: background
    ink: str  #: text and address outlines
    neutral: str  #: gray of the things without a meaningful color
    palette: tuple[str, ...]  #: one color per hyper-edge class, in class order (cycled if needed)
    sequential: tuple[str, ...]  #: stops of the colormap of the values, low -> high


# Built around the EnerGNN brand gradient (teal, green, lime) and the LF Energy blues; the palette order was
# validated for color-vision deficiencies.
THEMES = {
    "light": Theme(
        surface="#fcfcfb",
        ink="#0b0b0b",
        neutral="#898781",
        palette=("#00b58a", "#1f8ff0", "#8fcc0d", "#1b3f78", "#f0883a", "#d6499a", "#7b5be6", "#d9a800"),
        sequential=("#003070", "#0090f0", "#00d0a0", "#b0f010"),
    ),
    "dark": Theme(
        surface="#1a1a19",
        ink="#ffffff",
        neutral="#9c9a93",
        palette=("#19e0ad", "#4aa6ff", "#b7f21a", "#8fb0e6", "#ff9d4d", "#ec6fb3", "#a08cf5", "#f2c230"),
        sequential=("#2a5db0", "#0090f0", "#00d0a0", "#b0f010"),
    ),
}
#: Marker shapes per class, in class order (cycled if needed): classes stay separable without the colors.
#: They are the filled shapes of plotly that differ enough at marker size and do not look like an address.
SYMBOLS = (
    *("square", "triangle-up", "diamond", "triangle-down", "cross", "x", "pentagon", "star"),
    *("hourglass", "bowtie", "triangle-left", "triangle-right", "diamond-wide", "diamond-tall", "star-diamond", "square-x"),
    *("hexagram", "triangle-ne", "triangle-sw", "square-cross", "triangle-se", "triangle-nw", "star-square", "diamond-cross"),
)

# --- sizes: lengths are in layout units (the addresses fill the [-1, 1] box), marker sizes in pixels ---------
#: Distance from an address to the markers of the hyper-edges pointing only to it, in address radii.
STUB_LENGTH = 2.6
#: Largest distance between two neighbors among hyper-edges fanned out, in layout units.
FAN_HEIGHT = 0.09
#: Radius of an address, in pixels on a 640-pixel figure, whatever the size of the graph and the zoom.
ADDRESS_RADIUS = 13.0
#: Half the width of a digit of an address number, and the margin around the number, in font sizes: the
#: numbers are written as large as the longest of them fits in an address circle, 3 digits at full size.
DIGIT_HALF_WIDTH, LABEL_MARGIN = 0.28, 0.21
#: Radius of a class marker, as a fraction of the address radius.
MARKER_RATIO = 0.62
#: Number of shades of the colormap that the values are colored with.
N_SHADES = 16
#: Up to this many things to draw (addresses and hyper-edges), the figure is drawn as SVG; beyond, with WebGL.
SVG_UP_TO = 1000
#: Display options, that plotly takes apart from the figure: the mouse wheel zooms, no plotly logo in the toolbar.
_CONFIG = {"scrollZoom": True, "displaylogo": False}

#: What a notebook displays (see :meth:`GraphFigure._repr_html_`, which fills the ``__FIELDS__``): an element,
#: and the script that draws the figure in it. The element first holds a message, which stays if the script
#: is not run (a notebook only runs the scripts of the outputs it trusts). The script goes through these steps:
#:
#: 1. it loads plotly.js, the library that draws plotly figures in a web page, from plotly's CDN, unless the
#:    page already has it (the reader therefore needs an internet access);
#: 2. ``draw`` draws the figure with ``Plotly.react``, and is called again every second to follow the notebook;
#: 3. with the "auto" theme, ``draw`` first reads the theme of the notebook (``notebookTheme``: is the first
#:    background found around the figure dark? and failing any background, does the system prefer dark?). If
#:    it is not the theme the figure was built in, ``swap`` replaces every color of the figure by its
#:    counterpart in the other theme, the two lists of colors being given side by side by :func:`_swatches`;
#: 4. ``draw`` also undoes a CSS zoom set around the figure, which plotly.js does not support: its tooltips
#:    would show up for another hyper-edge than the one under the mouse. PyCharm sets such a zoom when the IDE
#:    is zoomed.
#:
#: A notebook may insert the same HTML several times (PyCharm does it twice for every output, and JupyterLab
#: for every view of an output), which runs the script as many times: each run must only deal with its own
#: element. This is why the script takes the element just before itself, rather than looking for an id. And
#: an element may be out of the page for a while (JupyterLab takes the cells away from the page while they
#: are scrolled out of view): ``draw`` waits for it, and gives up after 5 minutes, the output being then
#: taken as deleted.
_NOTEBOOK_HTML = """<div style="width:__WIDTH__px;height:__HEIGHT__px">\
Interactive figure, drawn by a script: trust this notebook to display it.</div>
<script>
(() => {
  const gd = document.currentScript.previousElementSibling, figure = __FIGURE__, config = __CONFIG__;
  const built = "__THEME__", auto = __AUTO__, pairs = __PAIRS__;  // pairs: the [light, dark] colors of the themes
  const swap = (value, table) =>  // a copy of value (a figure or a part of it) with its colors replaced
    typeof value === "string" ? table[value] ?? value
    : Array.isArray(value) ? value.map((item) => swap(item, table))
    : value && typeof value === "object"
      ? Object.fromEntries(Object.entries(value).map(([key, item]) => [key, swap(item, table)]))
    : value;
  const notebookTheme = () => {
    for (let element = gd.parentElement; element; element = element.parentElement) {
      const color = getComputedStyle(element).backgroundColor.match(/[\\d.]+/g);  // red, green, blue[, opacity]
      if (color && (color.length < 4 || +color[3] > 0))  // an element that has a background
        return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2] < 127.5 ? "dark" : "light";
    }
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  };
  let timer, shown, secondsAway = 0;  // shown: the theme the figure is drawn in
  const draw = () => {
    if (!gd.isConnected) return ++secondsAway > 300 && clearInterval(timer);  // out of the page
    secondsAway = 0;
    const zoomAround = (gd.currentCSSZoom || 1) / (parseFloat(gd.style.zoom) || 1);  // step 4
    gd.style.zoom = Math.abs(zoomAround - 1) > 0.001 ? 1 / zoomAround : "";
    const theme = auto ? notebookTheme() : built;  // step 3
    if (theme === shown) return;
    shown = theme;
    const table = Object.fromEntries(pairs.map(([light, dark]) => (theme === "dark" ? [light, dark] : [dark, light])));
    const themed = theme === built ? figure : swap(figure, table);
    Plotly.react(gd, themed.data, { ...themed.layout, uirevision: 1 }, config);  // uirevision keeps zoom and pan
  };
  (window.Plotly ? Promise.resolve() : import("__PLOTLYJS__"))  // step 1
    .then(() => {
      gd.textContent = "";
      timer = setInterval(draw, 1000);  // step 2
      draw();
    })
    .catch((error) => {
      gd.textContent = "The figure could not be drawn (plotly.js is loaded from __PLOTLYJS__): " + error;
    });
})();
</script>"""


def _swatches(theme: Theme) -> list[str]:
    """Every color that a figure may hold in this theme, written as in the figure, in the same order for all themes."""
    shades = sample_colorscale(list(theme.sequential), np.linspace(0.0, 1.0, N_SHADES))  # as _shades does
    return [theme.surface, theme.ink, theme.neutral, *theme.palette, *theme.sequential, *shades]


class GraphFigure(go.Figure):
    """The plotly figure of a graph: a :class:`plotly.graph_objects.Figure`, with its own display.

    It differs from a plain figure in two ways. The mouse wheel zooms, wherever the figure is displayed
    (:data:`_CONFIG`). And a notebook displays it as HTML with a script (:data:`_NOTEBOOK_HTML`) that makes it
    follow the theme of the notebook, instead of plotly's display, which sends the figure for the notebook to
    draw as it is.
    """

    #: Whether the figure takes the theme of the notebook that displays it: set by ``plot_graph(theme="auto")``.
    _follows_notebook = False

    # plotly takes the display options as an argument of the methods that display or export a figure
    def show(self, *args: Any, **kwargs: Any) -> Any:
        return super().show(*args, **{"config": _CONFIG, **kwargs})

    def to_html(self, *args: Any, **kwargs: Any) -> str:
        return super().to_html(*args, **{"config": _CONFIG, **kwargs})

    def write_html(self, *args: Any, **kwargs: Any) -> Any:
        return super().write_html(*args, **{"config": _CONFIG, **kwargs})

    # A notebook asks an object how to display it through the three members below, in this order. The first
    # one is plotly's display: it is turned off, so that the notebook goes on to the next.
    _ipython_display_ = None

    def _repr_mimebundle_(self, *args: Any, **kwargs: Any) -> dict:
        # the plain text is what is shown where HTML cannot be; without it, it would be the content of the figure
        return {"text/html": self._repr_html_(), "text/plain": f"<{type(self).__name__}>"}

    def _repr_html_(self) -> str:
        fields = {
            "__WIDTH__": str(self.layout.width or 640),
            "__HEIGHT__": str(self.layout.height or 640),
            "__CONFIG__": json.dumps(_CONFIG),
            "__THEME__": "dark" if self.layout.paper_bgcolor == THEMES["dark"].surface else "light",
            "__AUTO__": json.dumps(self._follows_notebook),
            "__PAIRS__": json.dumps(list(zip(_swatches(THEMES["light"]), _swatches(THEMES["dark"])))),
            "__PLOTLYJS__": f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js",
            # last, so that no field is looked for in the texts of the user that the figure holds (plotly writes
            # them without "<", so that none can end the script)
            "__FIGURE__": pio.to_json(self, validate=False),
        }
        html = _NOTEBOOK_HTML
        for field, value in fields.items():
            html = html.replace(field, value)
        return html


def _positions(
    classes: dict[str, HyperEdgeSet],
    n: int,
    address_positions: Any,
    hyper_edge_positions: dict[str, list[str]],
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Place the ``n`` addresses in the ``[-1, 1]`` box.

    The coordinates come, in order of precedence, from ``address_positions``; from the hyper-edges placed by
    ``hyper_edge_positions``, each address going to the mean position of the placed hyper-edges that point to
    it; otherwise from a force-directed layout. User coordinates are fitted into the box with one translation
    and one scale, so that the drawing keeps their geometry.

    :param classes: The hyper-edge sets drawn, by class name.
    :return: The address positions, shape ``(n, 2)``, and for each class of ``hyper_edge_positions`` the
        positions of its hyper-edges, shape ``(n_obj, 2)``, with NaN rows for the hyper-edges whose features
        are NaN.
    :raises ValueError: On a wrong shape or NaN in ``address_positions``, or when the placed hyper-edges leave
        an address without position.
    """
    placed = {
        cls: np.stack([(classes[cls].feature_dict or {})[name] for name in (x, y)], axis=-1).astype(float)
        for cls, (x, y) in hyper_edge_positions.items()
    }
    if address_positions is not None:
        addresses = np.asarray(address_positions, dtype=float)
        if addresses.shape != (n, 2) or np.isnan(addresses).any():
            raise ValueError(f"address_positions must have shape {(n, 2)} and no NaN; got shape {addresses.shape}.")
    elif placed:
        total, hits = np.zeros((n, 2)), np.zeros(n)
        for cls, xy in placed.items():
            known = np.isfinite(xy).all(axis=1)
            ports = classes[cls].port_array[known]
            np.add.at(total, ports, xy[known, None])  # each hyper-edge adds its position to its addresses
            np.add.at(hits, ports, 1)
        if not hits.all():
            raise ValueError(
                f"addresses {np.flatnonzero(hits == 0).tolist()} are pointed to by no placed hyper-edge, so they have "
                "no position; give address_positions, or place a class whose ports cover every address."
            )
        addresses = total / hits[:, None]
    else:
        return _spring(classes, n, iterations, seed), {}
    points = np.concatenate([addresses, *[xy[np.isfinite(xy).all(axis=1)] for xy in placed.values()]])
    center = points.mean(axis=0)
    scale = float(np.abs(points - center).max()) or 1.0
    return (addresses - center) / scale, {cls: (xy - center) / scale for cls, xy in placed.items()}


def _spring(classes: dict[str, HyperEdgeSet], n: int, iterations: int, seed: int) -> np.ndarray:
    """Place the ``n`` addresses with :func:`.layout.spring_layout`.

    The simulated graph has one node per address. A hyper-edge with two ports links its two addresses. A
    hyper-edge with more ports gets a node of its own, linked to each of its addresses, which pulls them
    together; this node is only there for the simulation, the marker is placed afterwards like any other.
    """
    edges, n_nodes = [np.zeros((0, 2), dtype=int)], n
    for hes in classes.values():
        ports = hes.port_array
        if ports.shape[1] == 2:
            edges.append(ports)
        elif ports.shape[1] > 2:
            nodes = n_nodes + np.arange(len(ports))
            n_nodes += len(nodes)
            edges.append(np.stack([np.repeat(nodes, ports.shape[1]), ports.ravel()], axis=-1))
    return spring_layout(n_nodes, np.concatenate(edges), iterations=iterations, seed=seed)[:n]


def _markers(
    classes: dict[str, HyperEdgeSet], addresses: np.ndarray, placed: dict[str, np.ndarray], radius: float
) -> dict[str, np.ndarray]:
    """Place the marker of every hyper-edge: at its given position if any, otherwise at the barycenter of its addresses.

    Hyper-edges pointing to the same addresses (whatever their classes) would have the same barycenter. They
    are ranked, and moved apart according to their rank:

    - around the address, ``STUB_LENGTH`` radii away from it, when they point to a single address;
    - otherwise along the perpendicular to the line joining their two extreme addresses, by at most
      ``FAN_HEIGHT`` between two neighbors, and less when these addresses are close to each other.

    A hyper-edge placed by its features on its only address (a bus that gives its position to its address)
    would be hidden by it, and is moved away like the others.

    :param radius: The address radius that distances are counted in, in layout units.
    :return: For each class, the marker positions, shape ``(n_obj, 2)``.
    """
    ports = [hes.port_array for hes in classes.values()]
    # the addresses of each hyper-edge, sorted and padded with -1 to the same width for all classes: its group key
    width = max(p.shape[1] for p in ports)
    keys = np.concatenate([np.pad(np.sort(p), ((0, 0), (0, width - p.shape[1])), constant_values=-1) for p in ports])
    single = keys[:, 0] == keys.max(axis=1)
    keys[single, 1:] = -1  # a single address, however many ports point to it
    _, group, count = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    group = group.reshape(-1)
    by_group = np.argsort(group, kind="stable")
    rank = np.empty(len(keys))
    rank[by_group] = np.arange(len(keys)) - (np.cumsum(count) - count)[group[by_group]]  # 0, 1, ... within each group
    count = count[group]

    first, last = addresses[keys[:, 0]], addresses[keys.max(axis=1)]
    angle = 2.0 * np.pi * rank / count + 0.6
    around = first + STUB_LENGTH * radius * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    chord = last - first
    length = np.maximum(np.linalg.norm(chord, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-chord[:, 1], chord[:, 0]], axis=-1) / length
    fan = (rank - (count - 1) / 2.0)[:, None] * np.minimum(0.3 * length, FAN_HEIGHT) * normal
    barycenter = np.concatenate([addresses[p].mean(axis=1) for p in ports])
    markers = np.where(single[:, None], around, barycenter + fan)

    sizes = np.cumsum([len(p) for p in ports])[:-1]
    out = dict(zip(classes, np.split(markers, sizes)))
    for (cls, xy), alone, p in zip(out.items(), np.split(single, sizes), ports):
        if cls in placed:
            hidden = alone & (np.linalg.norm(placed[cls] - addresses[p[:, 0]], axis=1) < radius)
            given = np.isfinite(placed[cls]).all(axis=1) & ~hidden
            xy[given] = placed[cls][given]
    return out


def _shades(values: np.ndarray, low: float, high: float, theme: Theme, default: str) -> tuple[np.ndarray, list[str], dict]:
    """Color ``values`` through the theme's colormap, ``low`` and ``high`` being its ends.

    :return: The shade of each value, as an index into the list of colors returned next: ``N_SHADES`` colors
        from low to high, then ``default``, the color of the NaN values. Last, the marker properties that make
        plotly draw these colors: sending it the shades rather than one color per point keeps big figures light.
    """
    colors = sample_colorscale(list(theme.sequential), np.linspace(0.0, 1.0, N_SHADES)) + [default]
    unit = (values - low) / (high - low) if high > low else np.full(len(values), 0.5)  # 0 at low, 1 at high
    shade = np.where(np.isnan(values), N_SHADES, np.rint(np.nan_to_num(unit) * (N_SHADES - 1))).astype(int)
    colorscale = [[i / N_SHADES, color] for i, color in enumerate(colors)]
    return shade, colors, dict(color=shade, cmin=0, cmax=N_SHADES, colorscale=colorscale)


def _range(values: np.ndarray, what: str) -> tuple[float, float]:
    """The lowest and highest of ``values``, NaN ignored."""
    if np.isnan(values).all():
        raise ValueError(f"{what} are all missing (NaN).")
    return float(np.nanmin(values)), float(np.nanmax(values))


def _colorbar(title: str, low: float, high: float, theme: Theme, column: int) -> dict:
    """The legend of a colormap: an empty trace, there for its colorbar only (plotly ties a colorbar to a trace)."""
    if high == low:  # a constant value is in the middle of the colormap
        low, high = low - 0.5, high + 0.5
    colorbar = dict(title=dict(text=title, side="right"), thickness=12, len=0.6, x=1.02 + 0.14 * column)
    colorbar.update(outlinewidth=0, ticks="")  # no frame and no tick marks, which would not follow the theme
    marker = dict(color=[low, high], cmin=low, cmax=high, colorscale=list(theme.sequential), colorbar=colorbar, showscale=True)
    return dict(type="scatter", x=[None], y=[None], mode="markers", marker=marker, showlegend=False, hoverinfo="skip")


def plot_graph(
    graph: Graph,
    *,
    address_positions: Any = None,
    hyper_edge_positions: dict[str, list[str]] | None = None,
    address_colors: Any = None,
    hyper_edge_colors: dict[str, str] | bool | None = None,
    iterations: int = 150,
    seed: int = 0,
    size: int = 640,
    theme: str = "auto",
) -> GraphFigure:
    """
    Draw a single Graph as an interactive plotly figure.

    Addresses are numbered circles; each hyper-edge is a marker joined to its addresses, with one color and
    one marker shape per class. Hovering an address or a marker shows its ports and feature values. The mouse
    wheel zooms, dragging pans, double-click resets the view; clicking a class in the legend hides it.

    A notebook displays the figure when it is the last line of a cell. It is a
    :class:`plotly.graph_objects.Figure`, so everything plotly offers applies to it, such as ``write_html``
    to save it as a standalone page.

    :param graph: A single Graph; batched graphs must first go through :func:`energnn.graph.separate_graphs`.
        Its padding (fictitious hyper-edges and addresses) is not drawn.
    :param address_positions: Optional address coordinates of shape ``(n_addresses, 2)``, ``n_addresses``
        being the number of real addresses; replaces the force-directed layout.
    :param hyper_edge_positions: Optional ``{class: [x_feature, y_feature]}``: the markers of that class are
        drawn at the coordinates held by those features. Without ``address_positions``, each address sits at
        the mean position of the placed hyper-edges pointing to it, and every address must be pointed to by one.
    :param address_colors: Optional values of shape ``(n_addresses,)``, shown as the color of the addresses.
        A NaN leaves the address uncolored.
    :param hyper_edge_colors: By default, one color per class. ``False`` draws every class in the neutral gray
        (marker shapes still tell classes apart). A dict ``{class: feature}`` colors the markers and spokes of
        those classes by that feature, on a color scale shared by every listed class; every other class is
        then drawn in the neutral gray, so that only these colors carry a meaning. A NaN keeps that neutral
        color.
    :param iterations: Number of layout relaxation steps (unused when positions are given).
    :param seed: Seed for the layout's random initial positions.
    :param size: Width and height of the figure, in pixels.
    :param theme: ``"light"``, ``"dark"``, or ``"auto"``: the figure is light, but a notebook displays it in
        its own theme, light or dark, and follows its changes.
    :return: The figure.
    :raises ValueError: If the graph is not single, if ``theme`` is invalid, if an array has a wrong shape,
        or if a value to display is NaN everywhere.
    :raises KeyError: If ``hyper_edge_positions`` or ``hyper_edge_colors`` names a class that is not drawn
        or a feature that it does not have.
    """
    if theme not in (*THEMES, "auto"):
        raise ValueError(f"theme must be one of {[*THEMES, 'auto']}; got '{theme}'.")
    if not graph.is_single:
        raise ValueError("Only single graphs can be drawn; use separate_graphs() on a batch first.")
    style = THEMES["light" if theme == "auto" else theme]
    g = graph.to_numpy_backend()
    g.unpad()
    n = g.n_addresses
    classes = {name: hes for name, hes in sorted(g.hyper_edge_sets.items()) if hes.port_dict and hes.n_obj}
    # Symbols have the same size in pixels whatever the graph, and plotly keeps it under zoom: a big graph is
    # an overview at first, and reads like a small one once zoomed in. The distances between symbols follow
    # the spacing of the addresses instead: they are those of an address radius that shrinks beyond 133
    # addresses. In layout units, 290 pixels of a 640-pixel figure go from the center of the box to its side.
    radius_px = ADDRESS_RADIUS * size / 640.0
    addresses, placed = _positions(classes, n, address_positions, hyper_edge_positions or {}, iterations, seed)
    spacing_radius = min(ADDRESS_RADIUS, 150.0 / np.sqrt(max(n, 1))) / 290.0
    markers = _markers(classes, addresses, placed, spacing_radius) if classes else {}
    # single precision is plenty for a drawing, and halves what is sent to the browser
    addresses, markers = addresses.astype(np.float32), {cls: xy.astype(np.float32) for cls, xy in markers.items()}

    kind = "scatter" if n + sum(hes.n_obj for hes in classes.values()) <= SVG_UP_TO else "scattergl"
    traces = []
    by_feature = hyper_edge_colors if isinstance(hyper_edge_colors, dict) else {}
    values = {cls: (classes[cls].feature_dict or {})[feature].astype(float) for cls, feature in by_feature.items()}
    if values:
        low, high = _range(np.concatenate(list(values.values())), "hyper_edge_colors")
        title = ", ".join(f"{cls}.{feature}" for cls, feature in by_feature.items())
        traces.append(_colorbar(title, low, high, style, column=0))
    for k, (cls, hes) in enumerate(classes.items()):
        ports, features = hes.port_array, hes.feature_dict or {}
        n_ports = ports.shape[1]
        own = style.neutral if values or hyper_edge_colors is False else style.palette[k % len(style.palette)]
        shade, shades, fill = np.zeros(hes.n_obj, dtype=int), [own], dict(color=own)
        if cls in values:
            shade, shades, fill = _shades(values[cls], low, high, style, own)
        # one spoke per port: the marker, the address, and a NaN that ends the line -> (n_obj, n_ports, 3, 2)
        spokes = np.full((hes.n_obj, n_ports, 3, 2), np.nan, dtype=np.float32)
        spokes[:, :, 0], spokes[:, :, 1] = markers[cls][:, None], addresses[ports]
        for i in np.unique(shade):  # plotly gives one color to all the lines of a trace: one trace per shade
            x, y = spokes[shade == i].reshape(-1, 2).T
            line = dict(color=shades[i], width=1.2)
            traces.append(
                dict(type=kind, x=x, y=y, mode="lines", line=line, legendgroup=cls, showlegend=False, hoverinfo="skip")
            )
        # the tooltip is a template that plotly fills with the row of the hovered hyper-edge in customdata
        tip = [f"<b>{cls} #%{{customdata[0]}}</b>"]
        tip += [f"{name} → %{{customdata[{1 + i}]}}" for i, name in enumerate(hes.port_names or {})]
        tip += [f"{name} = %{{customdata[{1 + n_ports + i}]:.5~g}}" for i, name in enumerate(features)]
        symbol = SYMBOLS[k % len(SYMBOLS)]
        x, y = markers[cls].T
        traces.append(
            dict(
                type=kind,
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    symbol=symbol, size=2.0 * MARKER_RATIO * radius_px, line=dict(color=style.surface, width=1), **fill
                ),
                customdata=np.column_stack([np.arange(hes.n_obj), ports, *features.values()]).astype(np.float32),
                hovertemplate="<br>".join(tip) + "<extra></extra>",
                name=cls,
                legendgroup=cls,
            )
        )

    address_values, fill, value_tip = None, dict(color=style.surface), ""
    if address_colors is not None:
        address_values = np.asarray(address_colors, dtype=float)
        if address_values.shape != (n,):
            raise ValueError(f"address_colors must have shape {(n,)}; got {address_values.shape}.")
        low, high = _range(address_values, "address_colors")
        fill, value_tip = _shades(address_values, low, high, style, style.surface)[2], "<br>value = %{customdata:.5~g}"
        traces.append(_colorbar("addresses", low, high, style, column=int(bool(values))))
    x, y = addresses.T
    font_size = radius_px / max(LABEL_MARGIN + DIGIT_HALF_WIDTH * len(str(n - 1)), 1.05)
    traces.append(
        dict(
            type=kind,
            x=x,
            y=y,
            mode="markers+text",
            marker=dict(size=2.0 * radius_px, line=dict(color=style.ink, width=1.2), **fill),
            text=np.arange(n).astype(str),  # strings: plotly fails to draw numbers here
            textfont=dict(size=max(font_size, 7.0), color=style.ink),
            customdata=address_values,
            hovertemplate="<b>address %{text}</b>" + value_tip + "<extra></extra>",
            name="addresses",
        )
    )
    layout = dict(
        template="none",  # plotly's default styles, which are the user's settings: the figure sets its own
        width=size,
        height=size,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=style.surface,
        plot_bgcolor=style.surface,
        font=dict(color=style.ink),
        legend=dict(orientation="h", yanchor="top", y=0.0, x=0.0, itemsizing="constant"),  # below: the toolbar is above
        dragmode="pan",
        hovermode="closest",
        hoverdistance=2,  # in pixels around the markers: the tooltip is the one of the hyper-edge under the mouse
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),  # same scale on both axes
    )
    figure = GraphFigure(dict(data=traces, layout=layout))
    figure._follows_notebook = theme == "auto"
    return figure
