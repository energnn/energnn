# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Color themes, colormaps and marker shapes shared by the static and interactive renderers.

**Two themes**, light and dark, each a :class:`Theme` of named colors: the *surface* (background), the *ink*
(text and address outlines), the *neutral* gray (things without a meaningful color), a *palette* of
categorical colors (one per hyper-edge class, in class order) and the two colormaps used for values:

- the *sequential* colormap (one channel) is a ramp through four color stops, low to high;
- the *bivariate* colormap (two channels) is a square whose corners are four colors: the color of a point is
  the bilinear blend of the corners, channel 1 running left to right and channel 2 bottom to top.

The palettes are built around the EnerGNN brand gradient (teal ``#00d0a0`` -> green ``#70e040`` -> lime
``#b0f010``) and the LF Energy blues (``#0090f0``, ``#003070``): the categorical palette starts with those
hues and continues with harmonized accents, in an order validated for color-vision deficiencies; the
sequential colormap runs dark blue -> blue -> teal -> lime; the bivariate colormap blends blue (channel 1) and
lime (channel 2) into teal from a neutral corner. The dark variants are re-stepped for a dark surface.

**Marker shapes** double the color encoding: each class gets a shape as well as a color, so classes stay
separable for color-blind readers, on a black-and-white print, or when every class is drawn in neutral.

The color mapping functions at the end are numpy-only and used by both renderers; ``assets/plot.js`` mirrors
them for the browser (see :mod:`.interactive`).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Theme(NamedTuple):
    """The named colors of one theme, all as ``"#rrggbb"`` strings."""

    palette: tuple[str, ...]  #: categorical colors, one per hyper-edge class in class order (cycled if needed)
    surface: str  #: background
    ink: str  #: text and address outlines (black on light, white on dark)
    neutral: str  #: gray of the things without a meaningful color
    sequential: tuple[str, ...]  #: 1-channel colormap stops, low -> high
    bivariate: tuple[str, str, str, str]  #: 2-channel colormap corners: (0,0), (1,0), (0,1), (1,1)


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
#: Marker shapes per class, in class order, as matplotlib marker codes (cycled if there are more classes).
MARKERS = ["s", "^", "D", "v", "P", "X", "p", "*"]
#: The same shapes, by name, for the SVG polygons of the interactive renderer (see ``_marker_points`` there).
SVG_MARKERS = ["square", "triangle-up", "diamond", "triangle-down", "plus", "cross", "pentagon", "star"]


def resolve_theme(theme: str) -> str:
    """Return ``"light"`` or ``"dark"`` for the static renderer.

    ``"auto"`` follows matplotlib's current figure background: a dark notebook theme usually sets
    ``figure.facecolor`` to a dark color, which is detected by its luminance.

    :raises ValueError: If ``theme`` is none of ``"light"``, ``"dark"`` and ``"auto"``.
    """
    if theme in THEMES:
        return theme
    if theme != "auto":
        raise ValueError("theme must be 'light', 'dark' or 'auto'.")
    import matplotlib

    r, g, b = matplotlib.colors.to_rgb(matplotlib.rcParams["figure.facecolor"])
    return "dark" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "light"  # relative luminance, as in sRGB


# ---------------------------------------------------------------------------
# Color mapping (numpy only, shared by both renderers)
# ---------------------------------------------------------------------------


def hex_to_rgb(color: str) -> np.ndarray:
    """``"#rrggbb"`` -> float array of shape ``(3,)`` in ``[0, 1]``."""
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def rgb_to_hex(rgb: np.ndarray) -> list[str]:
    """Float array of shape ``(n, 3)`` in ``[0, 1]`` -> list of ``"#rrggbb"``."""
    ints = np.clip(np.rint(np.asarray(rgb, dtype=float) * 255.0), 0, 255).astype(int)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in ints]


def sequential_rgb(values: np.ndarray, theme: Theme) -> np.ndarray:
    """Map channel values in ``[0, 1]`` through the theme's sequential colormap.

    The colormap is piecewise linear between its stops: with four stops, a value ``x`` falls in one of three
    segments and is linearly interpolated between the two stops of that segment.

    :param values: Any shape.
    :return: RGB in ``[0, 1]``, shape ``values.shape + (3,)``.
    """
    stops = np.stack([hex_to_rgb(c) for c in theme.sequential])
    x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0) * (len(stops) - 1)  # position along the stops
    lo = np.floor(x).astype(int).clip(0, len(stops) - 2)  # index of the stop just below
    t = (x - lo)[..., None]  # fraction of the way to the next stop
    return stops[lo] * (1 - t) + stops[lo + 1] * t


def bivariate_rgb(u: np.ndarray, v: np.ndarray, theme: Theme) -> np.ndarray:
    """Map two channels in ``[0, 1]`` through the theme's bivariate colormap.

    The color is the bilinear blend of the four corner colors: ``(0, 0)`` at ``u = v = 0``, ``(1, 0)`` at
    ``u = 1``, ``(0, 1)`` at ``v = 1``, ``(1, 1)`` at both.

    :param u: Channel 1, any shape.
    :param v: Channel 2, same shape.
    :return: RGB in ``[0, 1]``, shape ``u.shape + (3,)``.
    """
    c00, c10, c01, c11 = (hex_to_rgb(c) for c in theme.bivariate)
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)[..., None]
    v = np.clip(np.asarray(v, dtype=float), 0.0, 1.0)[..., None]
    return (1 - u) * (1 - v) * c00 + u * (1 - v) * c10 + (1 - u) * v * c01 + u * v * c11


def channels_to_rgb(values: np.ndarray, theme: Theme) -> np.ndarray:
    """Normalized channels of shape ``(..., C)`` -> RGB of shape ``(..., 3)``, through the colormap matching ``C``.

    :raises ValueError: If ``C`` is not 1 or 2.
    """
    if values.shape[-1] == 1:
        return sequential_rgb(values[..., 0], theme)
    if values.shape[-1] == 2:
        return bivariate_rgb(values[..., 0], values[..., 1], theme)
    raise ValueError(f"colors must have 1 or 2 channels; got {values.shape[-1]}.")
