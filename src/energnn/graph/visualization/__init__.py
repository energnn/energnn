# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Visualization of :class:`energnn.graph.Graph` objects.

- :func:`plot_graph` draws a static matplotlib figure. It needs the ``viz`` extra:
  ``pip install energnn[viz]``.
- :func:`plot_graph_interactive` builds a self-contained HTML/SVG figure with tooltips,
  zoom, pan and 3D rotation, displayed inline by notebooks. It has no extra dependency.

Both accept a single graph only (batches must first go through
:func:`energnn.graph.separate_graphs`). Addresses can be placed from an array
(``address_positions``) and hyper-edges from their features (``hyper_edge_positions``);
addresses can be colored from an array (``address_colors``) and hyper-edges from their
features (``hyper_edge_colors``), with 1, 2 or 3 channels.
"""

from .interactive import InteractiveGraphPlot, plot_graph_interactive
from .layout import spring_layout
from .static import plot_graph

__all__ = [
    "InteractiveGraphPlot",
    "plot_graph",
    "plot_graph_interactive",
    "spring_layout",
]
