# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Step 3 of the pipeline: color channels for the addresses and the hyper-edges.

Both accept 1 channel (sequential colormap) or 2 channels (bivariate colormap), normalized per channel
to ``[0, 1]`` over the values given; the theme turns the channels into RGB at rendering time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from energnn.graph.visualization.content import FeatureSpec, Topology, feature_columns, per_address_array

CHANNELS = (1, 2)


@dataclass(frozen=True)
class ColorScale:
    """Normalized channels of one family of drawn things (addresses or hyper-edges)."""

    channels: np.ndarray  # (n, C) in [0, 1] (0.5 where the value is missing)
    missing: np.ndarray  # (n,) bool: no value given (NaN, or the class is not colored)
    low: np.ndarray  # (C,) raw minimum per channel
    high: np.ndarray  # (C,) raw maximum per channel

    @property
    def n_channels(self) -> int:
        return self.channels.shape[1]


@dataclass(frozen=True)
class Colors:
    addresses: ColorScale | None  # one row per real address
    hyper_edges: ColorScale | None  # one row per hyper-edge of the topology; ``missing`` for the classes not listed


def resolve_colors(topology: Topology, *, address_colors: Any = None, hyper_edge_colors: FeatureSpec | None = None) -> Colors:
    """Normalize the address colors (an array) and the hyper-edge colors (features of the listed classes).

    :raises ValueError: On a wrong shape or channel count, an unknown class or feature, or values all missing.
    """
    addresses = None
    if address_colors is not None:
        addresses = _scale(per_address_array(address_colors, "address_colors", topology, CHANNELS), "address_colors")
    hyper_edges = None
    if hyper_edge_colors:
        columns = feature_columns(topology, hyper_edge_colors, "hyper_edge_colors", CHANNELS)
        width = next(iter(columns.values())).shape[1]
        rows = [columns[h.cls][h.index] if h.cls in columns else np.full(width, np.nan) for h in topology.hyper_edges]
        hyper_edges = _scale(np.array(rows, dtype=float).reshape(-1, width), "hyper_edge_colors")
    return Colors(addresses, hyper_edges)


def _scale(raw: np.ndarray, what: str) -> ColorScale:
    missing = np.isnan(raw).any(axis=1)
    if missing.all():
        raise ValueError(f"{what} are all missing (NaN).")
    low, high = np.nanmin(raw, axis=0), np.nanmax(raw, axis=0)
    span = np.where(high > low, high - low, 1.0)
    channels = np.where(high > low, (raw - low) / span, 0.5)
    return ColorScale(np.nan_to_num(channels, nan=0.5), missing, low, high)
