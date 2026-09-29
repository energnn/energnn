# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Visualization of :class:`energnn.graph.Graph` objects.

- :func:`plot_graph` draws a static matplotlib figure. It needs the ``viz`` extra: ``pip install energnn[viz]``.
- :func:`plot_graph_interactive` builds a self-contained HTML/SVG figure with tooltips, zoom and pan,
  displayed inline by notebooks. It has no extra dependency.

Both accept a single graph only (batches must first go through :func:`energnn.graph.separate_graphs`),
and the same four optional inputs: ``address_positions`` (an array, one row per address) or
``hyper_edge_positions`` (``{class: [x_feature, y_feature]}``) to place things, ``address_colors`` (an
array) or ``hyper_edge_colors`` (``{class: [feature, ...]}``) to color them, with 1 or 2 channels.

Design
------

Rendering is a pipeline of four numpy-only steps, one module each, followed by a renderer:

1. :mod:`.content` reads the Graph once into a :class:`~.content.Topology`: the real addresses, the classes
   with their port names, and one :class:`~.content.HyperEdge` record per real object. A record knows its
   ports, its features, and its ``kind``, i.e. how it is drawn: ``stub`` (order 1), ``pair`` (order 2),
   ``loop`` (order 2 on one address), ``hub`` (order 3+), ``placed`` (any order, positioned by its features)
   or ``none`` (no port). Every later rule dispatches on that kind.
2. :mod:`.positions` places the addresses and the hubs in the ``[-1, 1]`` box, from the user's coordinates
   or a spring layout, and computes the margin the drawings need around the box. The address radius (in
   layout units, shrinking with the number of addresses) is the unit of every other size.
3. :mod:`.colors` normalizes the color channels of the addresses and of the hyper-edges to ``[0, 1]``.
4. :mod:`.geometry` turns each record into polylines, a marker position and label anchors.

:mod:`.static` draws that with matplotlib; :mod:`.interactive` writes it as SVG and hands the browser only
what needs it (zoom, pan, tooltips, theme-dependent colors), see ``assets/plot.js``.
"""

from .interactive import InteractiveGraphPlot, plot_graph_interactive
from .positions import spring_layout
from .static import plot_graph

__all__ = [
    "InteractiveGraphPlot",
    "plot_graph",
    "plot_graph_interactive",
    "spring_layout",
]
