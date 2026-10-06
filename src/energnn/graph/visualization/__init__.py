# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Visualization of :class:`energnn.graph.Graph` objects.

:func:`plot_graph` draws a single graph (batches must first go through :func:`energnn.graph.separate_graphs`)
as an interactive `plotly <https://plotly.com/python/>`_ figure, with tooltips, zoom and pan, displayed inline
by notebooks. It needs the ``viz`` extra: ``pip install energnn[viz]``.

Things are placed by ``address_positions`` (an array, one row per address) or ``hyper_edge_positions``
(``{class: [x_feature, y_feature]}``), and colored by ``address_colors`` (an array, one value per address) or
``hyper_edge_colors`` (``{class: feature}``, or ``False`` for no color at all).

The module is made of :mod:`.plot`, which builds the figure, and :mod:`.layout`, the force-directed layout
that places the addresses when no coordinate is given.
"""

from .layout import spring_layout
from .plot import GraphFigure, plot_graph

__all__ = [
    "GraphFigure",
    "plot_graph",
    "spring_layout",
]
