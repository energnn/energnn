# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Tests for the text rendering of Graph, HyperEdgeSet and GraphStructure."""

from unittest import mock

import numpy as np
import pytest
from IPython.lib.pretty import pretty

from energnn.graph import Graph, GraphShape, GraphStructure, HyperEdgeSet, collate_graphs, collate_hyper_edge_sets
from energnn.graph.backend import NumpyBackend
from energnn.graph.formatting import (
    ELLIPSIS,
    FICTITIOUS_MARKER,
    MAX_ROWS,
    format_float_column,
    format_graph,
    format_graph_structure,
    format_hyper_edge_set,
    format_int_column,
    hyper_edge_set_summary,
    hyper_edge_set_table_lines,
    render_grouped_table,
    select_columns,
    select_rows,
)
from energnn.graph.structure import HyperEdgeSetStructure

WIDE = 10_000


def make_hes(backend, n_obj=3, n_features=2, non_fictitious=None):
    xp = backend.xp
    return HyperEdgeSet(
        backend=backend,
        port_dict={"a": xp.arange(n_obj), "b": (xp.arange(n_obj) + 1) % n_obj},
        feature_array=xp.asarray(np.arange(n_obj * n_features, dtype=float).reshape(n_obj, n_features) / 2),
        feature_names={f"f{i}": i for i in range(n_features)},
        non_fictitious=xp.asarray(np.ones(n_obj, bool) if non_fictitious is None else non_fictitious),
    )


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------


def test_format_int_column():
    assert format_int_column(np.array([0, 12, -3])) == ["0", "12", "-3"]


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0, 0.5, 2.0], ["1.0", "0.5", "2.0"]),
        ([1.0, 2.0], ["1", "2"]),
        ([0.125, 3.0], ["0.125", "3.000"]),
        ([1234567.0, 12.25, 100.0], ["1.23457e+06", "1.22500e+01", "1.00000e+02"]),
        ([1e-7, 1.0], ["1e-07", "1e+00"]),
        ([np.nan, 1.5, np.inf], ["nan", "1.5", "inf"]),
        ([np.nan], ["nan"]),
        ([], []),
    ],
)
def test_format_float_column_shares_one_format_per_column(values, expected):
    assert format_float_column(np.array(values, dtype=float)) == expected


# ---------------------------------------------------------------------------
# Row and column selection
# ---------------------------------------------------------------------------


def test_select_rows_no_truncation():
    assert select_rows(5, max_rows=10) == ([0, 1, 2, 3, 4], None)


def test_select_rows_truncates_head_and_tail():
    rows, ellipsis_at = select_rows(100, max_rows=5)
    assert rows == [0, 1, 2, 98, 99]
    assert ellipsis_at == 3


def _wide_table(n_columns=6, n_rows=2):
    index = [("", [str(r) for r in range(n_rows)])]
    ports = [(f"p{i}", ["123"] * n_rows) for i in range(2)]
    feats = [(f"f{i}", ["0.5000"] * n_rows) for i in range(n_columns)]
    return index, [("ports", ports), ("features", feats)]


def test_select_columns_keeps_everything_when_it_fits():
    index, groups = _wide_table()
    blocks = select_columns(index, groups, WIDE)
    assert [name for name, _ in blocks] == ["", "ports", "features"]
    assert [c for _, cols in blocks for c, _ in cols] == ["", "p0", "p1", "f0", "f1", "f2", "f3", "f4", "f5"]


def test_select_columns_elides_middle_columns():
    index, groups = _wide_table()
    blocks = select_columns(index, groups, max_width=40)
    names = [c for _, cols in blocks for c, _ in cols]
    assert names[0] == ""  # index always kept
    assert names.count(ELLIPSIS) == 1
    assert names[1] == "p0" and names[-1] == "f5"  # head and tail kept
    assert len(names) < 9
    assert all(len(line) <= 40 for line in render_grouped_table(index, groups, max_width=40))


def test_select_columns_keeps_at_least_one_data_column():
    index, groups = _wide_table()
    blocks = select_columns(index, groups, max_width=1)
    names = [c for _, cols in blocks for c, _ in cols]
    assert names == ["", "p0", ELLIPSIS]


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def test_render_grouped_table_layout():
    index = [("", ["0", "1"])]
    groups = [("ports", [("a", ["0", "1"])]), ("features", [("x", ["1.0", "2.5"])])]
    lines = render_grouped_table(index, groups, max_width=WIDE)
    assert lines == [
        "  │ ports │ features",
        "  │     a │        x",
        "──┼───────┼─────────",
        "0 │     0 │      1.0",
        "1 │     1 │      2.5",
    ]


def test_render_grouped_table_ellipsis_row_and_empty_groups():
    index = [("", ["0", "1", "2"])]
    groups = [("ports", []), ("features", [("x", ["1", "2", "3"])])]
    lines = render_grouped_table(index, groups, ellipsis_at=1, max_width=WIDE)
    assert "ports" not in lines[0]
    assert lines[4].strip().startswith(ELLIPSIS)
    assert len(lines) == 3 + 3 + 1


def test_render_grouped_table_empty():
    assert render_grouped_table([], [], max_width=WIDE) == []


# ---------------------------------------------------------------------------
# HyperEdgeSet
# ---------------------------------------------------------------------------


def test_hyper_edge_set_summary(backend):
    assert hyper_edge_set_summary(make_hes(backend)) == "3 objects · 2 ports · 2 features"
    hes = make_hes(backend, non_fictitious=[True, False, False])
    assert hyper_edge_set_summary(hes) == "3 objects (1 real) · 2 ports · 2 features"
    batched = collate_hyper_edge_sets([make_hes(backend), make_hes(backend)])
    assert hyper_edge_set_summary(batched) == "batch of 2 · 3 objects · 2 ports · 2 features"


def test_hyper_edge_set_str_single(backend):
    hes = make_hes(backend, non_fictitious=[True, True, False])
    text = str(hes)
    assert text.splitlines()[0] == "HyperEdgeSet · 3 objects (2 real) · 2 ports · 2 features"
    assert text.splitlines()[-1] == f"{FICTITIOUS_MARKER} fictitious object"
    marked = [line for line in text.splitlines() if line.startswith(f"2  {FICTITIOUS_MARKER}")]
    assert len(marked) == 1
    assert "0.5" in text and "1.5" in text and "2.5" in text


def test_hyper_edge_set_str_without_fictitious_marker(backend):
    text = str(make_hes(backend))
    assert FICTITIOUS_MARKER not in text


def test_hyper_edge_set_str_batched(backend):
    batched = collate_hyper_edge_sets([make_hes(backend), make_hes(backend)])
    lines = str(batched).splitlines()
    assert "batch  obj" in lines[2]
    assert len(lines) == 1 + 3 + 6


def test_hyper_edge_set_table_truncates_rows(backend):
    n = 5 * MAX_ROWS
    lines = hyper_edge_set_table_lines(make_hes(backend, n_obj=n), max_width=WIDE)
    body = lines[3:]
    assert len(body) == MAX_ROWS + 1
    assert body[0].split()[0] == "0"
    assert body[-1].split()[0] == str(n - 1)
    assert ELLIPSIS in body[(MAX_ROWS + 1) // 2]


def test_hyper_edge_set_table_formats_only_displayed_rows(backend):
    n = 5 * MAX_ROWS
    with mock.patch("energnn.graph.formatting.format_float_column", wraps=format_float_column) as spy:
        hyper_edge_set_table_lines(make_hes(backend, n_obj=n), max_width=WIDE)
    assert spy.call_count == 2
    assert all(len(call.args[0]) == MAX_ROWS for call in spy.call_args_list)


def test_hyper_edge_set_table_fits_in_width(backend):
    hes = make_hes(backend, n_features=20)
    for width in (30, 50, 80):
        assert all(len(line) <= width for line in hyper_edge_set_table_lines(hes, max_width=width))


def test_hyper_edge_set_without_ports_or_features():
    backend = NumpyBackend()
    only_ports = HyperEdgeSet(
        backend=backend,
        port_dict={"a": np.arange(2)},
        feature_array=None,
        feature_names=None,
        non_fictitious=np.ones(2, bool),
    )
    assert "0 features" in str(only_ports) and "│ features" not in str(only_ports)
    only_feats = HyperEdgeSet(
        backend=backend,
        port_dict=None,
        feature_array=np.zeros((2, 1)),
        feature_names={"x": 0},
        non_fictitious=np.ones(2, bool),
    )
    assert "0 ports" in str(only_feats) and "│ ports" not in str(only_feats)


def test_hyper_edge_set_repr_pretty_matches_str(backend):
    hes = make_hes(backend)
    assert pretty(hes) == str(hes) == format_hyper_edge_set(hes)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def make_graph(backend, hyper_edge_sets):
    xp = backend.xp
    shape = GraphShape(
        backend=backend,
        hyper_edge_sets={k: xp.asarray(v.n_obj) for k, v in hyper_edge_sets.items()},
        addresses=xp.asarray(3),
    )
    return Graph(
        backend=backend,
        hyper_edge_sets=hyper_edge_sets,
        true_shape=shape,
        current_shape=shape,
        non_fictitious_addresses=xp.asarray(np.ones(3, bool)),
    )


def test_graph_str_tree(backend):
    graph = make_graph(backend, {"line": make_hes(backend), "bus": make_hes(backend, n_obj=2)})
    lines = str(graph).splitlines()
    assert lines[0] == "Graph · 2 hyper-edge sets · 3 addresses"
    assert lines[1] == "│"
    assert lines[2].startswith("├─ bus · 2 objects")
    assert all(line.startswith("│") for line in lines[3:9])  # 5 table lines + blank branch line
    assert lines[9].startswith("└─ line · 3 objects")
    assert all(line.startswith("   ") for line in lines[10:])
    assert str(graph) == format_graph(graph) == pretty(graph)


def test_graph_str_batched(backend):
    g = make_graph(backend, {"bus": make_hes(backend)})
    batched = collate_graphs([g, g])
    assert str(batched).splitlines()[0].startswith("Graph · batch of 2 · 1 hyper-edge set · 3 addresses")


# ---------------------------------------------------------------------------
# GraphStructure
# ---------------------------------------------------------------------------


def test_graph_structure_str():
    structure = GraphStructure(
        {
            "bus": HyperEdgeSetStructure(port_list=["a"], feature_list=["x", "y"]),
            "line": HyperEdgeSetStructure(port_list=["a", "b"], feature_list=None),
        }
    )
    assert str(structure).splitlines() == [
        "GraphStructure · 2 hyper-edge sets",
        "├─ bus    ports: a      features: x, y",
        "└─ line   ports: a, b   features: —",
    ]
    assert pretty(structure) == str(structure) == format_graph_structure(structure)


def test_graph_structure_str_empty():
    assert str(GraphStructure({})) == "GraphStructure · 0 hyper-edge sets"
