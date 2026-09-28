# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Text rendering of ``Graph``, ``HyperEdgeSet`` and ``GraphStructure``.

The classes only delegate their ``__str__`` to :func:`format_graph`,
:func:`format_hyper_edge_set` and :func:`format_graph_structure`; all the
layout logic lives here so that the data classes stay free of it.

The output deliberately uses Unicode box-drawing glyphs (``│ ─ ┼``), ``·``,
``×`` and ``⋯``. They render in notebooks and in every UTF-8 terminal, which is
where these reprs are meant to be read. There is no ASCII fallback: a
``__str__`` cannot know where it will be written, and switching glyphs based on
``sys.stdout`` would make the same object print differently depending on the
call site. Writing these strings to a file opened with a non-UTF-8 encoding
(e.g. the Windows default) raises ``UnicodeEncodeError``; pass
``encoding="utf-8"`` to ``open``/``logging.FileHandler`` in that case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import shutil

import numpy as np

if TYPE_CHECKING:
    from energnn.graph.graph import Graph
    from energnn.graph.hyper_edge_set import HyperEdgeSet
    from energnn.graph.structure import GraphStructure

MAX_ROWS = 30
DEFAULT_WIDTH = 100
ELLIPSIS = "⋯"
FICTITIOUS_MARKER = "×"

Column = tuple[str, list[str]]
Group = tuple[str, list[Column]]


SIGNIFICANT_DIGITS = 6


def format_int_column(values: Sequence[int] | np.ndarray) -> list[str]:
    """Format integer cells."""
    return [str(int(v)) for v in values]


def format_float_column(values: Sequence[float] | np.ndarray) -> list[str]:
    """Format float cells with one common notation and precision for the whole column.

    Finite values are first written with ``SIGNIFICANT_DIGITS`` significant digits. If any of
    them needs scientific notation, the whole column is written in scientific notation with a
    shared mantissa precision; otherwise it is written in fixed-point with a shared number of
    decimals. Non-finite values are written as ``nan``/``inf``.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    reprs = [f"{v:.{SIGNIFICANT_DIGITS}g}" for v in arr[finite]]

    if any("e" in r for r in reprs):
        precision = max(_mantissa_digits(r) for r in reprs) - 1
        fmt = f"{{:.{precision}e}}"
    else:
        decimals = max((len(r.split(".")[1]) if "." in r else 0 for r in reprs), default=0)
        fmt = f"{{:.{decimals}f}}"

    return [fmt.format(v) if ok else str(v) for v, ok in zip(arr, finite)]


def _mantissa_digits(g_repr: str) -> int:
    """Number of significant digits in a ``g``-formatted number (``"1.23e+06"`` -> 3)."""
    mantissa = g_repr.split("e")[0].lstrip("-")
    if "." not in mantissa:
        mantissa = mantissa.rstrip("0")
    return max(len(mantissa.replace(".", "").lstrip("0")), 1)


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


def available_width() -> int:
    """Width budget for tables: the terminal width when known, ``DEFAULT_WIDTH`` otherwise (e.g. notebooks)."""
    return shutil.get_terminal_size(fallback=(DEFAULT_WIDTH, 24)).columns


def _block_specs(blocks: list[Group]) -> list[tuple[str, list[Column], list[int], int]]:
    """Compute column widths and total content width of each block."""
    specs = []
    for group_name, columns in blocks:
        widths = [max(len(name), max((len(c) for c in cells), default=0)) for name, cells in columns]
        content = sum(widths) + 2 * (len(columns) - 1)
        if len(group_name) > content:
            widths[0] += len(group_name) - content
            content = len(group_name)
        specs.append((group_name, columns, widths, content))
    return specs


def _table_width(blocks: list[Group]) -> int:
    return sum(content for *_, content in _block_specs(blocks)) + 3 * (len(blocks) - 1)


def select_columns(index_columns: list[Column], groups: list[Group], max_width: int) -> list[Group]:
    """Drop columns from the middle of ``groups`` until the table fits in ``max_width``.

    Index columns are always kept. Data columns are kept alternately from the head and the tail
    of the flattened column list, and an ``ELLIPSIS`` column marks the cut. At least one data
    column is kept even if the table still overflows.
    """
    flat = [(g, c) for g, (_, columns) in enumerate(groups) for c in range(len(columns))]
    n_rows = len(index_columns[0][1]) if index_columns else (len(groups[0][1][0][1]) if flat else 0)

    def build(kept: set[tuple[int, int]], cut: tuple[int, int] | None) -> list[Group]:
        blocks: list[Group] = [("", index_columns)] if index_columns else []
        for g, (name, columns) in enumerate(groups):
            new_columns = [col for c, col in enumerate(columns) if (g, c) in kept]
            if cut is not None and cut[0] == g:
                new_columns.insert(sum((g, c) in kept for c in range(cut[1])), (ELLIPSIS, [ELLIPSIS] * n_rows))
            if new_columns:
                blocks.append((name, new_columns))
        return blocks

    if _table_width(build(set(flat), None)) <= max_width:
        return build(set(flat), None)

    kept: set[tuple[int, int]] = set()
    head, tail = 0, len(flat) - 1
    while head <= tail:
        candidate = flat[head] if len(kept) % 2 == 0 else flat[tail]
        trial = kept | {candidate}
        cut = flat[head + 1 if candidate == flat[head] else head]
        if kept and _table_width(build(trial, cut)) > max_width:
            break
        kept = trial
        if candidate == flat[head]:
            head += 1
        else:
            tail -= 1
    return build(kept, flat[head])


def render_grouped_table(
    index_columns: list[Column],
    groups: list[Group],
    ellipsis_at: int | None = None,
    max_width: int | None = None,
) -> list[str]:
    """Render right-aligned columns grouped in blocks separated by ``│``.

    ``index_columns`` forms an unnamed leading block; each entry of ``groups``
    is a ``(group_name, columns)`` block whose name is centered above it.
    ``ellipsis_at`` inserts an ellipsis row before that row index; ``max_width``
    (``available_width()`` by default) elides middle columns so the lines fit.
    """
    groups = [g for g in groups if g[1]]
    if not index_columns and not groups:
        return []
    blocks = select_columns(index_columns, groups, available_width() if max_width is None else max_width)
    specs = _block_specs(blocks)

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


def hyper_edge_set_table_lines(hyper_edge_set: HyperEdgeSet, max_width: int | None = None) -> list[str]:
    """Render ports and features as an aligned text table fitting in ``max_width`` characters."""
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

    rows, ellipsis_at = select_rows(n_rows)
    row_idx = np.asarray(rows, dtype=int)

    def take(array) -> np.ndarray:
        """Pull only the displayed rows to the host (arrays may live on a device)."""
        return np.asarray(array.reshape(-1)[row_idx])

    index_columns = [(name, [cells[r] for r in rows]) for name, cells in index_values]

    has_fictitious = False
    if hyper_edge_set.non_fictitious is not None:
        mask = hyper_edge_set.non_fictitious
        if mask.size == n_rows and not bool(mask.all()):
            has_fictitious = True
            index_columns.append(("", ["" if m else FICTITIOUS_MARKER for m in take(mask)]))

    groups = []
    if hyper_edge_set.port_dict is not None:
        columns = [(k, format_int_column(take(v))) for k, v in sorted(hyper_edge_set.port_dict.items())]
        groups.append(("ports", columns))
    if hyper_edge_set.feature_dict is not None:
        columns = [(k, format_float_column(take(v))) for k, v in sorted(hyper_edge_set.feature_dict.items())]
        groups.append(("features", columns))

    lines = render_grouped_table(index_columns, groups, ellipsis_at=ellipsis_at, max_width=max_width)
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
        table = hyper_edge_set_table_lines(hyper_edge_set, max_width=available_width() - len(continuation))
        lines.extend(continuation + line for line in table)
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
