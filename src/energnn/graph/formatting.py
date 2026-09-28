# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Text and HTML representations of ``Graph``, ``HyperEdgeSet`` and ``GraphStructure``.

The classes delegate ``__str__``, ``_repr_pretty_`` and ``_repr_html_`` here.
Representations are deliberately short: ``str(graph)`` is one line per
hyper-edge set. Tables are rendered by pandas, either on demand through
``HyperEdgeSet.to_dataframe`` / ``Graph.to_dataframes``, or folded in the
notebook HTML view. Only the rows that are displayed are pulled from the
(possibly device-resident) arrays.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from energnn.graph.graph import Graph
    from energnn.graph.hyper_edge_set import HyperEdgeSet
    from energnn.graph.structure import GraphStructure

MAX_ROWS = 30
FICTITIOUS_MARKER = "×"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def select_rows(n_rows: int, max_rows: int | None) -> np.ndarray:
    """Indices of the rows to display: all of them, or ``max_rows`` split between head and tail."""
    if max_rows is None or n_rows <= max_rows:
        return np.arange(n_rows)
    head = (max_rows + 1) // 2
    return np.concatenate([np.arange(head), np.arange(n_rows - (max_rows - head), n_rows)])


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
        n_real, n_total = int(hyper_edge_set.non_fictitious.sum()), int(hyper_edge_set.non_fictitious.size)
        if n_real != n_total:
            obj_part += f" ({n_real} real)"
    parts.append(obj_part)
    n_ports = len(hyper_edge_set.port_dict) if hyper_edge_set.port_dict is not None else 0
    parts.append(_plural(n_ports, "port"))
    n_feat = len(hyper_edge_set.feature_names) if hyper_edge_set.feature_names is not None else 0
    parts.append(_plural(n_feat, "feature"))
    return " · ".join(parts)


def hyper_edge_set_to_dataframe(hyper_edge_set: HyperEdgeSet, max_rows: int | None = None) -> pd.DataFrame:
    """Ports and features as a ``DataFrame`` with ``("ports", name)`` / ``("features", name)`` columns.

    The index is the object id (plus the batch id for a batched set) and, when some objects are
    fictitious, a marker level. With ``max_rows``, only the head and tail rows are transferred
    and returned; the index keeps the original object ids.
    """
    if hyper_edge_set.is_single:
        n_batch, n_obj = None, hyper_edge_set.n_obj
    elif hyper_edge_set.is_batch:
        n_batch, n_obj = hyper_edge_set.n_batch, hyper_edge_set.n_obj
    else:
        raise ValueError("HyperEdgeSet is neither single nor batched.")

    n_rows = n_obj if n_batch is None else n_batch * n_obj
    rows = select_rows(n_rows, max_rows)

    def take(array) -> np.ndarray:
        return np.asarray(array.reshape(-1)[rows])

    levels: list[tuple[str, np.ndarray]] = []
    if n_batch is not None:
        levels.append(("batch", rows // n_obj))
    levels.append(("obj", rows if n_batch is None else rows % n_obj))
    mask = hyper_edge_set.non_fictitious
    if mask is not None and mask.size == n_rows and not bool(mask.all()):
        levels.append(("", np.where(take(mask), "", FICTITIOUS_MARKER)))
    index = pd.MultiIndex.from_arrays([v for _, v in levels], names=[k for k, _ in levels])

    columns: dict[tuple[str, str], np.ndarray] = {}
    if hyper_edge_set.port_dict is not None:
        columns.update({("ports", k): take(v) for k, v in sorted(hyper_edge_set.port_dict.items())})
    if hyper_edge_set.feature_dict is not None:
        columns.update({("features", k): take(v) for k, v in sorted(hyper_edge_set.feature_dict.items())})
    return pd.DataFrame(columns, index=index)


def format_hyper_edge_set(hyper_edge_set: HyperEdgeSet, max_rows: int | None = MAX_ROWS) -> str:
    """Summary line followed by the pandas table (head/tail rows beyond ``max_rows``)."""
    df = hyper_edge_set_to_dataframe(hyper_edge_set, max_rows=max_rows)
    lines = [f"HyperEdgeSet · {hyper_edge_set_summary(hyper_edge_set)}", repr(df)]
    n_rows = hyper_edge_set.n_obj * (hyper_edge_set.n_batch if hyper_edge_set.is_batch else 1)
    if len(df) < n_rows:
        lines.append(f"[{len(df)} of {n_rows} rows shown]")
    return "\n".join(lines)


def html_hyper_edge_set(hyper_edge_set: HyperEdgeSet, max_rows: int | None = MAX_ROWS) -> str:
    """Notebook view: summary line and the pandas HTML table."""
    df = hyper_edge_set_to_dataframe(hyper_edge_set, max_rows=max_rows)
    summary = html.escape(f"HyperEdgeSet · {hyper_edge_set_summary(hyper_edge_set)}")
    return f"<div><b>{summary}</b>{df.to_html()}</div>"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def graph_summary(graph: Graph) -> str:
    """One-line description: batch size, number of hyper-edge sets and addresses."""
    sets = graph.hyper_edge_sets or {}
    parts = [_plural(len(sets), "hyper-edge set")]
    if sets and graph.is_batch:
        parts.insert(0, f"batch of {next(iter(sets.values())).n_batch}")
    if graph.true_shape is not None:
        addresses = np.unique(np.asarray(graph.true_shape.addresses).reshape(-1))
        if addresses.size == 1:
            parts.append(f"{int(addresses[0])} addresses")
        elif addresses.size > 1:
            parts.append(f"{int(addresses.min())}–{int(addresses.max())} addresses")
    return " · ".join(parts)


def format_graph(graph: Graph) -> str:
    """Header line and one summary line per hyper-edge set, sorted by name."""
    sets = sorted(graph.hyper_edge_sets.items()) if graph.hyper_edge_sets else []
    lines = [f"Graph · {graph_summary(graph)}"]
    name_width = max((len(name) for name, _ in sets), default=0)
    for i, (name, hyper_edge_set) in enumerate(sets):
        branch = "└─ " if i == len(sets) - 1 else "├─ "
        lines.append(f"{branch}{name.ljust(name_width)}  {hyper_edge_set_summary(hyper_edge_set)}")
    return "\n".join(lines)


def html_graph(graph: Graph, max_rows: int | None = MAX_ROWS) -> str:
    """Notebook view: header, then one folded ``<details>`` block per hyper-edge set holding its table."""
    sets = sorted(graph.hyper_edge_sets.items()) if graph.hyper_edge_sets else []
    parts = [f"<div><b>{html.escape(f'Graph · {graph_summary(graph)}')}</b>"]
    for name, hyper_edge_set in sets:
        title = html.escape(f"{name} · {hyper_edge_set_summary(hyper_edge_set)}")
        table = hyper_edge_set_to_dataframe(hyper_edge_set, max_rows=max_rows).to_html()
        parts.append(f"<details><summary>{title}</summary>{table}</details>")
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# GraphStructure
# ---------------------------------------------------------------------------


def format_graph_structure(structure: GraphStructure) -> str:
    """Header line and one line per hyper-edge set listing its ports and features."""
    items = list(structure.hyper_edge_sets.items())
    lines = [f"GraphStructure · {_plural(len(items), 'hyper-edge set')}"]
    name_width = max((len(name) for name, _ in items), default=0)
    port_cells = ["ports: " + (", ".join(s.port_list) if s.port_list else "—") for _, s in items]
    port_width = max((len(cell) for cell in port_cells), default=0)
    for i, ((name, edge_structure), ports) in enumerate(zip(items, port_cells)):
        branch = "└─ " if i == len(items) - 1 else "├─ "
        features = "features: " + (", ".join(edge_structure.feature_list) if edge_structure.feature_list else "—")
        lines.append(f"{branch}{name.ljust(name_width)}  {ports.ljust(port_width)}  {features}")
    return "\n".join(lines)
