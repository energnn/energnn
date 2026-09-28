# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Text rendering of ``Graph``, ``HyperEdgeSet`` and ``GraphStructure``.

The classes only delegate their ``__str__`` to :func:`format_graph`,
:func:`format_hyper_edge_set` and :func:`format_graph_structure`; all the
layout logic lives here so that the data classes stay free of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from energnn.graph.graph import Graph
    from energnn.graph.hyper_edge_set import HyperEdgeSet
    from energnn.graph.structure import GraphStructure

MAX_ROWS = 30
ELLIPSIS = "⋯"
FICTITIOUS_MARKER = "×"

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


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


# ---------------------------------------------------------------------------
# HyperEdgeSet
# ---------------------------------------------------------------------------


def hyper_edge_set_summary(hyper_edge_set: HyperEdgeSet) -> str:
    """One-line description: object, port and feature counts."""
    parts = []
    if hyper_edge_set.is_batch:
        parts.append(f"batch of {hyper_edge_set.n_batch}")
    obj_part = _plural(hyper_edge_set.n_obj, "object")
    if hyper_edge_set.non_fictitious is not None and not hyper_edge_set.is_batch:
        mask = np.asarray(hyper_edge_set.non_fictitious)
        n_real, n_total = int(mask.sum()), int(mask.size)
        if n_real != n_total:
            obj_part += f" ({n_real} real)"
    parts.append(obj_part)
    n_ports = len(hyper_edge_set.port_dict) if hyper_edge_set.port_dict is not None else 0
    parts.append(_plural(n_ports, "port"))
    n_feat = len(hyper_edge_set.feature_names) if hyper_edge_set.feature_names is not None else 0
    parts.append(_plural(n_feat, "feature"))
    return " · ".join(parts)


def hyper_edge_set_table_lines(hyper_edge_set: HyperEdgeSet) -> list[str]:
    """Render ports and features as an aligned text table."""
    if hyper_edge_set.is_single:
        n_rows = hyper_edge_set.n_obj
        index_values = [("", [str(i) for i in range(n_rows)])]
    elif hyper_edge_set.is_batch:
        n_batch, n_obj = hyper_edge_set.n_batch, hyper_edge_set.n_obj
        n_rows = n_batch * n_obj
        index_values = [
            ("batch", [str(b) for b in range(n_batch) for _ in range(n_obj)]),
            ("obj", [str(i) for _ in range(n_batch) for i in range(n_obj)]),
        ]
    else:
        raise ValueError("HyperEdgeSet is neither single nor batched.")

    has_fictitious = False
    if hyper_edge_set.non_fictitious is not None:
        mask = np.asarray(hyper_edge_set.non_fictitious).reshape(-1)
        if mask.size == n_rows and not np.all(mask):
            has_fictitious = True
            index_values.append(("", ["" if m else FICTITIOUS_MARKER for m in mask]))

    rows, ellipsis_at = select_rows(n_rows)

    def take(cells: list[str]) -> list[str]:
        return [cells[r] for r in rows]

    index_columns = [(name, take(cells)) for name, cells in index_values]

    groups = []
    if hyper_edge_set.port_dict is not None:
        columns = [
            (k, take(format_int_column(np.asarray(v).reshape(-1)))) for k, v in sorted(hyper_edge_set.port_dict.items())
        ]
        groups.append(("ports", columns))
    if hyper_edge_set.feature_dict is not None:
        columns = [
            (k, take(format_float_column(np.asarray(v).reshape(-1))))
            for k, v in sorted(hyper_edge_set.feature_dict.items())
        ]
        groups.append(("features", columns))

    lines = render_grouped_table(index_columns, groups, ellipsis_at=ellipsis_at)
    if has_fictitious:
        lines.append(f"{FICTITIOUS_MARKER} fictitious object")
    return lines


def format_hyper_edge_set(hyper_edge_set: HyperEdgeSet) -> str:
    """Full text representation of a ``HyperEdgeSet``."""
    return "\n".join([f"HyperEdgeSet · {hyper_edge_set_summary(hyper_edge_set)}", *hyper_edge_set_table_lines(hyper_edge_set)])


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def format_graph(graph: Graph) -> str:
    """Full text representation of a ``Graph``: a header line and one tree branch per hyper-edge set."""
    sets = sorted(graph.hyper_edge_sets.items()) if graph.hyper_edge_sets is not None else []

    header_parts = [_plural(len(sets), "hyper-edge set")]
    if sets and graph.is_batch:
        header_parts.insert(0, f"batch of {sets[0][1].n_batch}")
    if graph.true_shape is not None:
        addresses = np.unique(np.asarray(graph.true_shape.addresses).reshape(-1))
        if addresses.size == 1:
            header_parts.append(f"{int(addresses[0])} addresses")
        elif addresses.size > 1:
            header_parts.append(f"{int(addresses.min())}–{int(addresses.max())} addresses")
    lines = ["Graph · " + " · ".join(header_parts)]

    for i, (name, hyper_edge_set) in enumerate(sets):
        last = i == len(sets) - 1
        branch, continuation = ("└─ ", "   ") if last else ("├─ ", "│  ")
        lines.append("│")
        lines.append(f"{branch}{name} · {hyper_edge_set_summary(hyper_edge_set)}")
        lines.extend(continuation + line for line in hyper_edge_set_table_lines(hyper_edge_set))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GraphStructure
# ---------------------------------------------------------------------------


def format_graph_structure(structure: GraphStructure) -> str:
    """Full text representation of a ``GraphStructure``: one line per hyper-edge set."""
    items = list(structure.hyper_edge_sets.items())
    lines = [f"GraphStructure · {_plural(len(items), 'hyper-edge set')}"]
    if not items:
        return lines[0]

    name_width = max(len(name) for name, _ in items)
    port_cells = ["ports: " + (", ".join(s.port_list) if s.port_list else "—") for _, s in items]
    port_width = max(len(cell) for cell in port_cells)
    for i, ((name, edge_structure), ports) in enumerate(zip(items, port_cells)):
        branch = "└─ " if i == len(items) - 1 else "├─ "
        features = "features: " + (", ".join(edge_structure.feature_list) if edge_structure.feature_list else "—")
        lines.append(f"{branch}{name.ljust(name_width)}   {ports.ljust(port_width)}   {features}")
    return "\n".join(lines)
