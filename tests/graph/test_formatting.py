# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Tests for the text/HTML representations of Graph, HyperEdgeSet and GraphStructure."""

import numpy as np
import pandas as pd
from IPython.lib.pretty import pretty

from energnn.graph import Graph, GraphShape, GraphStructure, HyperEdgeSet, collate_graphs, collate_hyper_edge_sets
from energnn.graph.formatting import FICTITIOUS_MARKER, MAX_ROWS, hyper_edge_set_summary, select_rows
from energnn.graph.structure import HyperEdgeSetStructure


def make_hes(backend, n_obj=3, n_features=2, non_fictitious=None):
    xp = backend.xp
    return HyperEdgeSet(
        backend=backend,
        port_dict={"a": xp.arange(n_obj), "b": (xp.arange(n_obj) + 1) % n_obj},
        feature_array=xp.asarray(np.arange(n_obj * n_features, dtype=float).reshape(n_obj, n_features) / 2),
        feature_names={f"f{i}": i for i in range(n_features)},
        non_fictitious=xp.asarray(np.ones(n_obj, bool) if non_fictitious is None else np.array(non_fictitious)),
    )


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


def test_select_rows():
    assert select_rows(5, None).tolist() == [0, 1, 2, 3, 4]
    assert select_rows(5, 10).tolist() == [0, 1, 2, 3, 4]
    assert select_rows(100, 5).tolist() == [0, 1, 2, 98, 99]


# ---------------------------------------------------------------------------
# HyperEdgeSet
# ---------------------------------------------------------------------------


def test_hyper_edge_set_summary(backend):
    assert hyper_edge_set_summary(make_hes(backend)) == "3 objects · 2 ports · 2 features"
    hes = make_hes(backend, non_fictitious=[True, False, False])
    assert hyper_edge_set_summary(hes) == "3 objects (1 real) · 2 ports · 2 features"
    batched = collate_hyper_edge_sets([make_hes(backend), make_hes(backend)])
    assert hyper_edge_set_summary(batched) == "batch of 2 · 3 objects · 2 ports · 2 features"


def test_to_dataframe_single(backend):
    df = make_hes(backend).to_dataframe()
    assert df.columns.tolist() == [("ports", "a"), ("ports", "b"), ("features", "f0"), ("features", "f1")]
    assert df.index.names == ["obj"]
    assert df.index.get_level_values("obj").tolist() == [0, 1, 2]
    assert df[("ports", "b")].tolist() == [1, 2, 0]
    assert df[("features", "f1")].tolist() == [0.5, 1.5, 2.5]


def test_to_dataframe_batched(backend):
    df = collate_hyper_edge_sets([make_hes(backend), make_hes(backend, n_obj=3)]).to_dataframe()
    assert df.index.names == ["batch", "obj"]
    assert df.index.get_level_values("batch").tolist() == [0, 0, 0, 1, 1, 1]
    assert df.index.get_level_values("obj").tolist() == [0, 1, 2, 0, 1, 2]


def test_to_dataframe_fictitious_marker(backend):
    df = make_hes(backend, non_fictitious=[True, True, False]).to_dataframe()
    assert df.index.names == ["obj", ""]
    assert df.index.get_level_values("").tolist() == ["", "", FICTITIOUS_MARKER]
    assert "" not in make_hes(backend).to_dataframe().index.names


def test_to_dataframe_max_rows_keeps_original_ids(backend):
    n = 5 * MAX_ROWS
    df = make_hes(backend, n_obj=n).to_dataframe(max_rows=MAX_ROWS)
    assert len(df) == MAX_ROWS
    ids = df.index.get_level_values("obj").tolist()
    assert ids[:3] == [0, 1, 2] and ids[-1] == n - 1
    assert df[("ports", "a")].tolist() == ids


def test_to_dataframe_without_ports_or_features(backend):
    xp = backend.xp
    only_ports = HyperEdgeSet(
        backend=backend,
        port_dict={"a": xp.arange(2)},
        feature_array=None,
        feature_names=None,
        non_fictitious=xp.asarray(np.ones(2, bool)),
    )
    assert only_ports.to_dataframe().columns.tolist() == [("ports", "a")]
    only_feats = HyperEdgeSet(
        backend=backend,
        port_dict=None,
        feature_array=xp.zeros((2, 1)),
        feature_names={"x": 0},
        non_fictitious=xp.asarray(np.ones(2, bool)),
    )
    assert only_feats.to_dataframe().columns.tolist() == [("features", "x")]


def test_hyper_edge_set_str(backend):
    hes = make_hes(backend, non_fictitious=[True, True, False])
    lines = str(hes).splitlines()
    assert lines[0] == "HyperEdgeSet · 3 objects (2 real) · 2 ports · 2 features"
    assert lines[1:] == repr(hes.to_dataframe()).splitlines()
    assert "rows shown" not in str(hes)


def test_hyper_edge_set_str_truncates(backend):
    n = 5 * MAX_ROWS
    text = str(make_hes(backend, n_obj=n))
    assert text.splitlines()[-1] == f"[{MAX_ROWS} of {n} rows shown]"
    assert len(text.splitlines()) < MAX_ROWS + 10


def test_hyper_edge_set_repr_pretty_and_html(backend):
    hes = make_hes(backend)
    assert pretty(hes) == str(hes)
    html = hes._repr_html_()
    assert html.startswith("<div><b>HyperEdgeSet · 3 objects")
    assert "<table" in html


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def test_graph_str_is_one_line_per_set(backend):
    graph = make_graph(backend, {"line": make_hes(backend), "bus": make_hes(backend, n_obj=2)})
    assert str(graph).splitlines() == [
        "Graph · 2 hyper-edge sets · 3 addresses",
        "├─ bus   2 objects · 2 ports · 2 features",
        "└─ line  3 objects · 2 ports · 2 features",
    ]
    assert pretty(graph) == str(graph)


def test_graph_str_batched(backend):
    g = make_graph(backend, {"bus": make_hes(backend)})
    assert str(collate_graphs([g, g])).splitlines()[0] == "Graph · batch of 2 · 1 hyper-edge set · 3 addresses"


def test_graph_to_dataframes_and_html(backend):
    graph = make_graph(backend, {"line": make_hes(backend), "bus": make_hes(backend, n_obj=2)})
    dfs = graph.to_dataframes()
    assert list(dfs) == ["bus", "line"]
    assert all(isinstance(df, pd.DataFrame) for df in dfs.values())
    assert len(dfs["bus"]) == 2
    html = graph._repr_html_()
    assert html.count("<details>") == 2
    assert "<summary>bus · 2 objects" in html


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
        "├─ bus   ports: a     features: x, y",
        "└─ line  ports: a, b  features: —",
    ]
    assert pretty(structure) == str(structure)
    assert str(GraphStructure({})) == "GraphStructure · 0 hyper-edge sets"
