# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Static assets shipped with the visualization module (the EnerGNN logo)."""

from __future__ import annotations

import base64
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def logo_png() -> bytes:
    """The EnerGNN mark as PNG bytes (192 px, transparent background)."""
    return resources.files(__name__).joinpath("energnn_logo.png").read_bytes()


@lru_cache(maxsize=1)
def logo_data_uri() -> str:
    """The logo as a ``data:`` URI, for embedding in HTML/SVG."""
    return "data:image/png;base64," + base64.b64encode(logo_png()).decode("ascii")


@lru_cache(maxsize=1)
def script_js() -> str:
    """The interactive renderer's script; ``__UID__`` is to be replaced by the plot's element id."""
    return resources.files(__name__).joinpath("plot.js").read_text(encoding="utf-8")
