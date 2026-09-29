# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Static assets shipped with the visualization module: the interactive plot's HTML skeleton, stylesheet and script."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources


def _read(name: str) -> str:
    return resources.files(__name__).joinpath(name).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def template_html() -> str:
    """The interactive plot's HTML skeleton, with ``__NAME__`` placeholders."""
    return _read("plot.html")


@lru_cache(maxsize=1)
def stylesheet_css() -> str:
    """The interactive plot's stylesheet, with a ``__UID__`` placeholder."""
    return _read("plot.css")


@lru_cache(maxsize=1)
def script_js() -> str:
    """The interactive plot's script, with a ``__UID__`` placeholder."""
    return _read("plot.js")
