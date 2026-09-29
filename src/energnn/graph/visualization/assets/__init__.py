# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Static assets of the interactive renderer, read from this package at run time.

- ``plot.html``: the HTML skeleton of a plot, with ``__NAME__`` placeholders filled by
  :func:`~energnn.graph.visualization.interactive.plot_graph_interactive`;
- ``plot.css``: its stylesheet, scoped to the plot's element id (``__UID__``);
- ``plot.js``: its script (zoom, pan, tooltips, theme), also scoped by ``__UID__``.

The files are plain text so that they can be edited with syntax highlighting; each function below caches its
file, which is read once per process.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources


def _read(name: str) -> str:
    """Read an asset; a leading ``<!-- ... -->`` or ``/* ... */`` comment documents the file and is not shipped."""
    text = resources.files(__name__).joinpath(name).read_text(encoding="utf-8")
    return re.sub(r"^\s*(<!--.*?-->|/\*.*?\*/)\s*", "", text, count=1, flags=re.S)


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
