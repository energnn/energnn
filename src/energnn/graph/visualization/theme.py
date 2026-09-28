# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Color themes and marker shapes shared by the static and interactive renderers."""

from __future__ import annotations

from typing import NamedTuple


class Theme(NamedTuple):
    palette: tuple[str, ...]
    surface: str
    ink: str


# Colorblind-validated categorical palettes; slot order is part of the validation, and the
# dark palette is the same hues re-stepped for a dark surface, not an automatic inversion.
THEMES = {
    "light": Theme(
        palette=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
        surface="#fcfcfb",
        ink="#0b0b0b",
    ),
    "dark": Theme(
        palette=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
        surface="#1a1a19",
        ink="#ffffff",
    ),
}
# Marker shapes double the color encoding so classes stay separable without color.
MARKERS = ["s", "^", "D", "v", "P", "X", "p", "*"]
SVG_MARKERS = ["square", "triangle-up", "diamond", "triangle-down", "plus", "cross", "pentagon", "star"]
ADDRESS_COLOR = "#898781"


def resolve_theme(theme: str) -> str:
    """Return ``"light"`` or ``"dark"``; ``"auto"`` follows the luminance of matplotlib's figure facecolor."""
    if theme in THEMES:
        return theme
    if theme != "auto":
        raise ValueError("theme must be 'light', 'dark' or 'auto'.")
    import matplotlib

    r, g, b = matplotlib.colors.to_rgb(matplotlib.rcParams["figure.facecolor"])
    return "dark" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "light"
