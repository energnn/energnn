# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Lightweight text-table rendering used by ``Graph``/``HyperEdgeSet`` printing."""

from __future__ import annotations

MAX_ROWS = 30
ELLIPSIS = "⋯"

Column = tuple[str, list[str]]
Group = tuple[str, list[Column]]


def format_int_column(values) -> list[str]:
    return [str(int(v)) for v in values]


def format_float_column(values) -> list[str]:
    return [f"{float(v):.6g}" for v in values]


def select_rows(n_rows: int, max_rows: int = MAX_ROWS) -> tuple[list[int], int | None]:
    """Pick the row indices to display.

    Returns the indices and, when truncated, the position at which an ellipsis
    row must be inserted (``None`` otherwise).
    """
    if n_rows <= max_rows:
        return list(range(n_rows)), None
    head = (max_rows + 1) // 2
    tail = max_rows - head
    return list(range(head)) + list(range(n_rows - tail, n_rows)), head


def render_grouped_table(
    index_columns: list[Column],
    groups: list[Group],
    ellipsis_at: int | None = None,
) -> list[str]:
    """Render right-aligned columns grouped in blocks separated by ``│``.

    ``index_columns`` forms an unnamed leading block; each entry of ``groups``
    is a ``(group_name, columns)`` block whose name is centered above it.
    """
    blocks = [("", index_columns)] + [g for g in groups if g[1]]
    blocks = [b for b in blocks if b[1]]
    if not blocks:
        return []

    specs = []
    for group_name, columns in blocks:
        widths = [max(len(name), max((len(c) for c in cells), default=0)) for name, cells in columns]
        content = sum(widths) + 2 * (len(columns) - 1)
        if len(group_name) > content:
            widths[0] += len(group_name) - content
            content = len(group_name)
        specs.append((group_name, columns, widths, content))

    lines = []
    if any(name for name, *_ in specs):
        lines.append(" │ ".join(name.center(content) for name, _, _, content in specs).rstrip())
    lines.append(
        " │ ".join(
            "  ".join(name.rjust(w) for (name, _), w in zip(columns, widths)) for _, columns, widths, _ in specs
        ).rstrip()
    )
    lines.append("─┼─".join("─" * content for *_, content in specs))

    n_rows = len(specs[0][1][0][1])
    for r in range(n_rows):
        if r == ellipsis_at:
            lines.append(" │ ".join(ELLIPSIS.center(content) for *_, content in specs).rstrip())
        lines.append(
            " │ ".join(
                "  ".join(cells[r].rjust(w) for (_, cells), w in zip(columns, widths)) for _, columns, widths, _ in specs
            ).rstrip()
        )
    return lines
