# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Color themes, colormaps and marker shapes shared by the static and interactive renderers.

The palettes are built around the EnerGNN brand gradient (teal ``#00d0a0`` -> green
``#70e040`` -> lime ``#b0f010``) and the LF Energy blues (``#0090f0``, ``#003070``):

- the categorical palette (one color per hyper-edge class) starts with those hues and
  continues with harmonized accents, in an order validated for color-vision deficiencies;
- the sequential colormap (1-channel address colors) runs along the brand gradient,
  dark blue -> blue -> teal -> lime;
- the bivariate colormap (2-channel address colors) blends blue (channel 1) and lime
  (channel 2) into teal, from a neutral corner.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Theme(NamedTuple):
    palette: tuple[str, ...]  # categorical, one slot per hyper-edge class
    surface: str  # background
    ink: str  # text
    neutral: str  # addresses and uncolored hyper-edges
    sequential: tuple[str, ...]  # 1-channel colormap stops, low -> high
    bivariate: tuple[str, str, str, str]  # 2-channel corners: (0,0), (1,0), (0,1), (1,1)


THEMES = {
    "light": Theme(
        palette=("#00b58a", "#1f8ff0", "#8fcc0d", "#1b3f78", "#f0883a", "#d6499a", "#7b5be6", "#d9a800"),
        surface="#fcfcfb",
        ink="#0b0b0b",
        neutral="#898781",
        sequential=("#003070", "#0090f0", "#00d0a0", "#b0f010"),
        bivariate=("#e4e4df", "#0090f0", "#b0f010", "#00d0a0"),
    ),
    "dark": Theme(
        palette=("#19e0ad", "#4aa6ff", "#b7f21a", "#8fb0e6", "#ff9d4d", "#ec6fb3", "#a08cf5", "#f2c230"),
        surface="#1a1a19",
        ink="#ffffff",
        neutral="#9c9a93",
        sequential=("#2a5db0", "#0090f0", "#00d0a0", "#b0f010"),
        bivariate=("#3b3b39", "#0090f0", "#b0f010", "#00d0a0"),
    ),
}
# Marker shapes double the color encoding so classes stay separable without color.
MARKERS = ["s", "^", "D", "v", "P", "X", "p", "*"]
SVG_MARKERS = ["square", "triangle-up", "diamond", "triangle-down", "plus", "cross", "pentagon", "star"]


def resolve_theme(theme: str) -> str:
    """Return ``"light"`` or ``"dark"``; ``"auto"`` follows the luminance of matplotlib's figure facecolor."""
    if theme in THEMES:
        return theme
    if theme != "auto":
        raise ValueError("theme must be 'light', 'dark' or 'auto'.")
    import matplotlib

    r, g, b = matplotlib.colors.to_rgb(matplotlib.rcParams["figure.facecolor"])
    return "dark" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "light"


# ---------------------------------------------------------------------------
# Color mapping (numpy only, shared by both renderers)
# ---------------------------------------------------------------------------


def hex_to_rgb(color: str) -> np.ndarray:
    """``"#rrggbb"`` -> float array in ``[0, 1]``."""
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def rgb_to_hex(rgb: np.ndarray) -> list[str]:
    """Float array of shape ``(n, 3)`` in ``[0, 1]`` -> list of ``"#rrggbb"``."""
    ints = np.clip(np.rint(np.asarray(rgb, dtype=float) * 255.0), 0, 255).astype(int)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in ints]


def sequential_rgb(values: np.ndarray, theme: Theme) -> np.ndarray:
    """Map values in ``[0, 1]`` (any shape) through the theme's sequential colormap -> ``(..., 3)``."""
    stops = np.stack([hex_to_rgb(c) for c in theme.sequential])
    x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0) * (len(stops) - 1)
    lo = np.floor(x).astype(int).clip(0, len(stops) - 2)
    t = (x - lo)[..., None]
    return stops[lo] * (1 - t) + stops[lo + 1] * t


def bivariate_rgb(u: np.ndarray, v: np.ndarray, theme: Theme) -> np.ndarray:
    """Bilinear blend of the theme's four corner colors for ``u`` (channel 1) and ``v`` (channel 2) in ``[0, 1]``."""
    c00, c10, c01, c11 = (hex_to_rgb(c) for c in theme.bivariate)
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)[..., None]
    v = np.clip(np.asarray(v, dtype=float), 0.0, 1.0)[..., None]
    return (1 - u) * (1 - v) * c00 + u * (1 - v) * c10 + (1 - u) * v * c01 + u * v * c11


def channels_to_rgb(values: np.ndarray, theme: Theme) -> np.ndarray:
    """Normalized channels of shape ``(..., C)`` with ``C`` in {1, 2} -> RGB of shape ``(..., 3)``."""
    if values.shape[-1] == 1:
        return sequential_rgb(values[..., 0], theme)
    if values.shape[-1] == 2:
        return bivariate_rgb(values[..., 0], values[..., 1], theme)
    raise ValueError(f"colors must have 1 or 2 channels; got {values.shape[-1]}.")
