# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Step 3 of the drawing pipeline: turn user values into color channels.

**Channels, not colors.** The pipeline never handles RGB colors itself: it normalizes the user's values to
*channels* in ``[0, 1]``, one channel per value, and the theme (:mod:`.theme`) turns the channels into
colors at rendering time. This keeps the light and dark themes, matplotlib and the browser consistent:
they all read the same channels.

- one channel (a scalar per thing, e.g. an error) goes through the *sequential* colormap, a single ramp
  from low to high, and gets a colorbar;
- two channels (e.g. a prediction and a target) go through the *bivariate* colormap, a square blending two
  hues, and get that square as a legend.

Two families of things can be colored, each with its own scale: the addresses (from an array) and the
hyper-edges (from features of the listed classes). The result is a :class:`Colors`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from energnn.graph.visualization.content import FeatureSpec, Topology, feature_columns, per_address_array

#: The allowed numbers of channels.
CHANNELS = (1, 2)


@dataclass(frozen=True)
class ColorScale:
    """The normalized channels of one family of things (the addresses, or the hyper-edges).

    :param channels: Shape ``(n, C)``, each channel scaled to ``[0, 1]`` over the values given (0.5 where the
        value is missing, but see ``missing``).
    :param missing: Shape ``(n,)``, True where no value was given (a NaN, or, for the hyper-edges, an object
        of a class that is not colored); the renderers then keep the default color of that thing.
    :param low: Shape ``(C,)``, the raw minimum of each channel, for the legend.
    :param high: Shape ``(C,)``, the raw maximum of each channel, for the legend.
    """

    channels: np.ndarray
    missing: np.ndarray
    low: np.ndarray
    high: np.ndarray

    @property
    def n_channels(self) -> int:
        """1 (sequential colormap) or 2 (bivariate colormap)."""
        return self.channels.shape[1]


@dataclass(frozen=True)
class Colors:
    """The color scales of the drawing.

    :param addresses: One row per real address, or None when ``address_colors`` is not given.
    :param hyper_edges: One row per hyper-edge of the topology, in the same order, or None when
        ``hyper_edge_colors`` is not given. Objects of the classes that are not listed are ``missing``.
    """

    addresses: ColorScale | None
    hyper_edges: ColorScale | None


def resolve_colors(topology: Topology, *, address_colors: Any = None, hyper_edge_colors: FeatureSpec | None = None) -> Colors:
    """Normalize the user's color values.

    :param topology: Where the hyper-edges and their features come from.
    :param address_colors: Optional array of shape ``(n_addresses, C)`` with ``C`` in {1, 2}.
    :param hyper_edge_colors: Optional ``{class: [feature, ...]}`` with 1 or 2 features per class; the scale
        is shared by every listed class, i.e. normalized over all their objects together.
    :return: The color scales.
    :raises ValueError: On a wrong shape or channel count, an unknown class or feature, or values all missing.
    """
    addresses = None
    if address_colors is not None:
        addresses = _scale(per_address_array(address_colors, "address_colors", topology, CHANNELS), "address_colors")
    hyper_edges = None
    if hyper_edge_colors:
        columns = feature_columns(topology, hyper_edge_colors, "hyper_edge_colors", CHANNELS)
        width = next(iter(columns.values())).shape[1]
        # one row per hyper-edge of the topology, NaN (hence "missing") for the classes that are not listed
        rows = [columns[h.cls][h.index] if h.cls in columns else np.full(width, np.nan) for h in topology.hyper_edges]
        hyper_edges = _scale(np.array(rows, dtype=float).reshape(-1, width), "hyper_edge_colors")
    return Colors(addresses, hyper_edges)


def _scale(raw: np.ndarray, what: str) -> ColorScale:
    """Normalize raw values of shape ``(n, C)`` to channels in ``[0, 1]``, per channel, ignoring NaN.

    A constant channel (maximum equal to minimum) maps to 0.5, the middle of the colormap.

    :raises ValueError: If every row holds a NaN, i.e. there is nothing to color.
    """
    missing = np.isnan(raw).any(axis=1)
    if missing.all():
        raise ValueError(f"{what} are all missing (NaN).")
    low, high = np.nanmin(raw, axis=0), np.nanmax(raw, axis=0)
    span = np.where(high > low, high - low, 1.0)
    channels = np.where(high > low, (raw - low) / span, 0.5)
    return ColorScale(np.nan_to_num(channels, nan=0.5), missing, low, high)
