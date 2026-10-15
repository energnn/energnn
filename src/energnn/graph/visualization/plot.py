# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Interactive rendering of a Graph as a plotly figure. Requires the ``viz`` extra (``pip install energnn[viz]``).

**How a graph is drawn.** The addresses are circles, holding their number or a label given by the user. Every
hyper-edge is a marker, whose shape and color tell its class, with one straight line (a *spoke*) to the address
of each of its ports: a line between two buses is thus a segment with its marker in the middle, a three-winding
transformer a star, a generator a short stub out of its bus.

**How it works.** :func:`plot_graph` works on a copy of the Graph without its padding (:meth:`Graph.unpad`),
in which the classes that draw nothing (without port or without hyper-edge) are ignored. Then :func:`_layout`
places the addresses and the hyper-edges in the ``[-1, 1]`` box with one force-directed layout (:mod:`.layout`),
in which the coordinates given by the user, if any, hold their addresses or hyper-edges in place. Then
:func:`_shades` turns the values to display into colors, and the figure is assembled from plotly *traces*, a
trace being a set of points or lines sharing a style: for each class, one trace for its markers and one for its
spokes, then one trace for the addresses.

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
from collections.abc import Sequence
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

# --- sizes, in pixels: the symbols keep their size whatever the graph and the zoom ------------------------------
#: Radius of an address, in pixels on a 640-pixel figure, whatever the size of the graph and the zoom.
ADDRESS_RADIUS = 13.0
#: Half the width of a character of an address label (a digit, or about that for a letter), and the margin
#: around the label, in font sizes: the labels are written as large as the longest of them fits in an address
#: circle, 3 characters at full size.
CHAR_HALF_WIDTH, LABEL_MARGIN = 0.28, 0.21
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


def _layout(
    classes: dict[str, HyperEdgeSet],
    n: int,
    address_positions: Any,
    hyper_edge_positions: dict[str, list[str]],
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Place the ``n`` addresses and every hyper-edge in the ``[-1, 1]`` box, with one force-directed layout.

    The simulated graph has one node per address and one per hyper-edge, joined by one edge per port
    (:func:`.layout.spring_layout`). The forces alone put every marker where the drawing needs it: a line
    settles between its two addresses, the hyper-edges pointing to a single address are pushed around it a
    short distance away, and parallel lines push each other to either side of their shared addresses.

    The coordinates given by the user (``address_positions``, and the features named by ``hyper_edge_positions``)
    are fitted into the box with one translation and one scale, so that the drawing keeps their geometry, and
    hold their nodes in place: the layout places the other nodes (a NaN row, the hyper-edges of the other
    classes...) among them, which may take them slightly out of the box. Without any coordinate, the layout
    places every node.

    :param classes: The hyper-edge sets drawn, by class name.
    :return: The address positions, shape ``(n, 2)``, and for each class the positions of its markers, shape
        ``(n_obj, 2)``.
    :raises ValueError: On a wrong shape of ``address_positions``, or when the given coordinates are NaN everywhere.
    """
    sizes = [hes.n_obj for hes in classes.values()]
    start = dict(zip(classes, n + np.cumsum([0, *sizes])))  # the node of the first hyper-edge of each class
    fixed = np.full((n + sum(sizes), 2), np.nan)
    if address_positions is not None:
        addresses = np.asarray(address_positions, dtype=float)
        if addresses.shape != (n, 2):
            raise ValueError(f"address_positions must have shape {(n, 2)}; got shape {addresses.shape}.")
        fixed[:n] = addresses
    for cls, (x, y) in hyper_edge_positions.items():
        features = classes[cls].feature_dict or {}
        fixed[start[cls] : start[cls] + classes[cls].n_obj] = np.stack([features[x], features[y]], axis=-1)
    known = np.isfinite(fixed).all(axis=1)
    if known.any():  # into the box, with one translation and one scale: the layout's forces are balanced there
        center = fixed[known].mean(axis=0)
        fixed = (fixed - center) / (float(np.abs(fixed[known] - center).max()) or 1.0)
    elif address_positions is not None or hyper_edge_positions:
        raise ValueError("address_positions and hyper_edge_positions place nothing: every coordinate is NaN.")
    if known.all():
        pos = fixed
    else:
        edges = [np.zeros((0, 2), dtype=int)] + [
            np.stack([np.repeat(start[cls] + np.arange(hes.n_obj), hes.port_array.shape[1]), hes.port_array.ravel()], -1)
            for cls, hes in classes.items()
        ]
        pos = spring_layout(len(fixed), np.concatenate(edges), iterations=iterations, seed=seed, fixed=fixed)
    return pos[:n], {cls: pos[start[cls] : start[cls] + hes.n_obj] for cls, hes in classes.items()}


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


def _port_lines(port_names: Sequence[str], ports: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The port lines of the tooltip of every hyper-edge, ``"name → label"`` joined by line breaks: shape ``(n_obj,)``."""
    lines = [np.char.add(f"{name} → ", labels[ports[:, i]]) for i, name in enumerate(port_names)]
    text = lines[0]
    for line in lines[1:]:
        text = np.char.add(np.char.add(text, "<br>"), line)
    return text


def plot_graph(
    graph: Graph,
    *,
    address_positions: Any = None,
    hyper_edge_positions: dict[str, list[str]] | None = None,
    address_labels: Sequence[str] | None = None,
    address_colors: Any = None,
    hyper_edge_colors: dict[str, str] | bool | None = None,
    iterations: int = 150,
    seed: int = 0,
    size: int = 640,
    theme: str = "auto",
) -> GraphFigure:
    """
    Draw a single Graph as an interactive plotly figure.

    Addresses are numbered circles (or labeled ones, with ``address_labels``); each hyper-edge is a marker
    joined to its addresses, with one color and one marker shape per class. Hovering an address or a marker
    shows its ports and feature values. The mouse wheel zooms, dragging pans, double-click resets the view;
    clicking a class in the legend hides it.

    A notebook displays the figure when it is the last line of a cell. It is a
    :class:`plotly.graph_objects.Figure`, so everything plotly offers applies to it, such as ``write_html``
    to save it as a standalone page.

    :param graph: A single Graph; batched graphs must first go through :func:`energnn.graph.separate_graphs`.
        Its padding (fictitious hyper-edges and addresses) is not drawn.
    :param address_positions: Optional address coordinates of shape ``(n_addresses, 2)``, ``n_addresses``
        being the number of real addresses: the addresses are drawn there, and the hyper-edges are placed
        among them by the force-directed layout. A row of NaN leaves the address to the layout as well.
    :param hyper_edge_positions: Optional ``{class: [x_feature, y_feature]}``: the markers of that class are
        drawn at the coordinates held by those features (NaN leaves the marker to the layout), and the layout
        places the addresses and the other classes among them. With ``address_positions`` as well, the
        drawing keeps the geometry of both: a hyper-edge drawn on its only address hides it.
    :param address_labels: Optional ``n_addresses`` strings (bus names, for instance), written in the address
        circles instead of the address numbers, and naming the addresses in the tooltips, where the number of
        an address stays readable. The labels are written as large as the longest of them fits in a circle:
        keep them short, a long label is written small and overflows its circle.
    :param address_colors: Optional values of shape ``(n_addresses,)``, shown as the color of the addresses.
        A NaN leaves the address uncolored.
    :param hyper_edge_colors: By default, one color per class. ``False`` draws every class in the neutral gray
        (marker shapes still tell classes apart). A dict ``{class: feature}`` colors the markers and spokes of
        those classes by that feature, on a color scale shared by every listed class; every other class is
        then drawn in the neutral gray, so that only these colors carry a meaning. A NaN keeps that neutral
        color.
    :param iterations: Number of layout relaxation steps (unused when everything has a position).
    :param seed: Seed for the layout's random initial positions.
    :param size: Width and height of the figure, in pixels.
    :param theme: ``"light"``, ``"dark"``, or ``"auto"``: the figure is light, but a notebook displays it in
        its own theme, light or dark, and follows its changes.
    :return: The figure.
    :raises ValueError: If the graph is not single, if ``theme`` is invalid, if an array has a wrong shape,
        if there is not one label per address, or if a value to display or the given coordinates are NaN everywhere.
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
    labels = None if address_labels is None else np.array([str(label) for label in address_labels])
    if labels is not None and labels.shape != (n,):
        raise ValueError(f"address_labels must hold {n} strings, one per address; got {labels.shape[0]}.")
    classes = {name: hes for name, hes in sorted(g.hyper_edge_sets.items()) if hes.port_dict and hes.n_obj}
    # Symbols have the same size in pixels whatever the graph, and plotly keeps it under zoom: a big graph is
    # an overview at first, and reads like a small one once zoomed in.
    radius_px = ADDRESS_RADIUS * size / 640.0
    addresses, markers = _layout(classes, n, address_positions, hyper_edge_positions or {}, iterations, seed)
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
        # the tooltip is a template that plotly fills with the row of the hovered hyper-edge in customdata; the
        # labels of its addresses, being strings, go through the text of the hyper-edge instead (not drawn)
        port_names: list[str] = list(hes.port_names or [])  # in column order of port_array
        tip = [f"<b>{cls} #%{{customdata[0]}}</b>"]
        if labels is None:
            tip += [f"{name} → %{{customdata[{1 + i}]}}" for i, name in enumerate(port_names)]
        else:
            tip += ["%{text}"]
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
                text=None if labels is None else _port_lines(port_names, ports, labels),
                hovertemplate="<br>".join(tip) + "<extra></extra>",
                name=cls,
                legendgroup=cls,
            )
        )

    # the tooltip of an address: its label and number, then its value if any, the two last from customdata
    text, tip, columns, fill = np.arange(n).astype(str), ["<b>address %{text}</b>"], [], dict(color=style.surface)
    if labels is not None:
        text, tip, columns = labels, ["<b>%{text}</b>", "address %{customdata[0]}"], [np.arange(n)]
    if address_colors is not None:
        address_values = np.asarray(address_colors, dtype=float)
        if address_values.shape != (n,):
            raise ValueError(f"address_colors must have shape {(n,)}; got {address_values.shape}.")
        low, high = _range(address_values, "address_colors")
        fill = _shades(address_values, low, high, style, style.surface)[2]
        tip.append(f"value = %{{customdata[{len(columns)}]:.5~g}}")
        columns.append(address_values)
        traces.append(_colorbar("addresses", low, high, style, column=int(bool(values))))
    x, y = addresses.T
    font_size = radius_px / max(LABEL_MARGIN + CHAR_HALF_WIDTH * max((len(label) for label in text), default=1), 1.05)
    traces.append(
        dict(
            type=kind,
            x=x,
            y=y,
            mode="markers+text",
            marker=dict(size=2.0 * radius_px, line=dict(color=style.ink, width=1.2), **fill),
            text=text,  # strings: plotly fails to draw numbers here
            textfont=dict(size=max(font_size, 7.0), color=style.ink),
            customdata=np.column_stack(columns) if columns else None,
            hovertemplate="<br>".join(tip) + "<extra></extra>",
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
