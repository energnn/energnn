# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Tests for merge_graphs / merge_hyper_edge_sets: joining graphs that describe the same objects."""

import jax
import numpy as np
import pytest

from energnn.graph import (
    Graph,
    GraphShape,
    HyperEdgeSet,
    JaxBackend,
    NumpyBackend,
    collate_graphs,
    merge_graphs,
    merge_hyper_edge_sets,
    separate_graphs,
)


def context(backend, n_bus=3):
    """A context: buses with a port and one feature, lines between them."""
    hes = {
        "bus": HyperEdgeSet.from_dict(
            port_dict={"id": np.arange(n_bus)}, feature_dict={"p": np.arange(n_bus, dtype=float)}, backend=backend
        ),
        "line": HyperEdgeSet.from_dict(
            port_dict={"from": np.arange(n_bus - 1), "to": np.arange(1, n_bus)},
            feature_dict={"b": np.ones(n_bus - 1)},
            backend=backend,
        ),
    }
    return Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=n_bus, backend=backend)


def decision(backend, n_bus=3, name="theta", scale=10.0):
    """A decision: buses with one feature, no ports, no addresses."""
    hes = {"bus": HyperEdgeSet.from_dict(feature_dict={name: scale * np.arange(n_bus, dtype=float)}, backend=backend)}
    return Graph.from_dict(hyper_edge_set_dict=hes, n_addresses=0, backend=backend)


def features(graph, name):
    return {k: np.asarray(v).tolist() for k, v in graph.hyper_edge_sets[name].feature_dict.items()}


# ---------------------------------------------------------------------------
# Hyper-edge sets
# ---------------------------------------------------------------------------


def test_merge_hyper_edge_sets_unions_ports_and_features(backend):
    left = context(backend).hyper_edge_sets["bus"]
    right = HyperEdgeSet.from_dict(
        port_dict={"id": np.arange(3), "zone": np.array([0, 0, 1])},
        feature_dict={"q": np.array([1.0, 2.0, 3.0])},
        backend=backend,
    )
    merged = merge_hyper_edge_sets(left, right)
    assert sorted(merged.port_dict) == ["id", "zone"]
    assert {k: np.asarray(v).tolist() for k, v in merged.feature_dict.items()} == {"p": [0.0, 1.0, 2.0], "q": [1.0, 2.0, 3.0]}
    assert merged.n_obj == 3 and merged.is_single
    assert np.asarray(merged.feature_array).shape == (3, 2)


def test_merge_hyper_edge_sets_rejects_different_objects(backend):
    left = context(backend).hyper_edge_sets["bus"]
    with pytest.raises(ValueError, match="different objects"):
        merge_hyper_edge_sets(left, decision(backend, n_bus=4).hyper_edge_sets["bus"])
    wrong_port = HyperEdgeSet.from_dict(port_dict={"id": np.array([2, 1, 0])}, backend=backend)
    with pytest.raises(ValueError, match="Port 'id'"):
        merge_hyper_edge_sets(left, wrong_port)
    other_mask = HyperEdgeSet.from_dict(feature_dict={"x": np.zeros(3)}, backend=backend)
    other_mask.non_fictitious = backend.xp.array([1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="fictitious"):
        merge_hyper_edge_sets(left, other_mask)


def test_merge_hyper_edge_sets_feature_collisions(backend):
    left = context(backend).hyper_edge_sets["bus"]
    right = HyperEdgeSet.from_dict(feature_dict={"p": np.array([5.0, 6.0, 7.0])}, backend=backend)
    with pytest.raises(ValueError, match="suffixes"):
        merge_hyper_edge_sets(left, right)
    merged = merge_hyper_edge_sets(left, right, suffixes=("", "_pred"))
    assert {k: np.asarray(v).tolist() for k, v in merged.feature_dict.items()} == {
        "p": [0.0, 1.0, 2.0],
        "p_pred": [5.0, 6.0, 7.0],
    }
    with pytest.raises(ValueError, match="collide"):
        merge_hyper_edge_sets(left, right, suffixes=("_x", "_x"))


def test_merge_hyper_edge_sets_suffixes_every_name_not_only_collisions(backend):
    left = context(backend).hyper_edge_sets["bus"]
    right = HyperEdgeSet.from_dict(feature_dict={"q": np.array([5.0, 6.0, 7.0])}, backend=backend)
    merged = merge_hyper_edge_sets(left, right, suffixes=("_ctx", "_dec"))
    assert sorted(merged.feature_names) == ["p_ctx", "q_dec"]


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------


def test_merge_graphs_attaches_a_decision_to_its_context(backend):
    ctx = context(backend)
    merged = merge_graphs(ctx, decision(backend))
    assert sorted(merged.hyper_edge_sets) == ["bus", "line"]
    assert features(merged, "bus") == {"p": [0.0, 1.0, 2.0], "theta": [0.0, 10.0, 20.0]}
    assert features(merged, "line") == {"b": [1.0, 1.0]}  # classes of one side only are kept as is
    assert np.asarray(merged.hyper_edge_sets["bus"].port_dict["id"]).tolist() == [0, 1, 2]
    # the address registry is the context's one (the decision has none)
    assert np.asarray(merged.non_fictitious_addresses).shape == (3,)
    assert int(merged.true_shape.addresses) == 3 and int(merged.current_shape.addresses) == 3
    assert int(merged.true_shape.hyper_edge_sets["bus"]) == 3 and int(merged.true_shape.hyper_edge_sets["line"]) == 2
    # inputs untouched
    assert features(ctx, "bus") == {"p": [0.0, 1.0, 2.0]}


def test_merge_graphs_chains_with_suffixes_and_the_method(backend):
    merged = (
        context(backend)
        .merge(decision(backend, name="p", scale=10.0), suffixes=("", "_pred"))
        .merge(decision(backend, name="p", scale=100.0), suffixes=("", "_target"))
    )
    assert features(merged, "bus") == {"p": [0.0, 1.0, 2.0], "p_pred": [0.0, 10.0, 20.0], "p_target": [0.0, 100.0, 200.0]}


def test_merge_graphs_takes_the_addresses_of_the_side_that_declares_them(backend):
    ctx = context(backend)
    merged = merge_graphs(decision(backend), ctx)  # left has no addresses: the right registry is used
    assert np.asarray(merged.non_fictitious_addresses).tolist() == [1.0, 1.0, 1.0]
    assert int(merged.true_shape.addresses) == 3 and int(merged.current_shape.addresses) == 3
    extra = Graph.from_dict(
        hyper_edge_set_dict={"sub": HyperEdgeSet.from_dict(port_dict={"bus": np.array([0, 4])}, backend=backend)},
        n_addresses=5,
        backend=backend,
    )
    with pytest.raises(ValueError, match="address masks"):
        merge_graphs(ctx, extra)  # both declare addresses, with different registries


def test_merge_graphs_rejects_incompatible_inputs(backend):
    ctx = context(backend)
    with pytest.raises(ValueError, match="Class 'bus'"):
        merge_graphs(ctx, decision(backend, n_bus=4))
    with pytest.raises(ValueError, match="single graph with a batched"):
        merge_graphs(ctx, collate_graphs([decision(backend), decision(backend)]))
    other_mask = Graph.from_dict(
        hyper_edge_set_dict={"sub": HyperEdgeSet.from_dict(port_dict={"bus": np.array([0, 2])}, backend=backend)},
        n_addresses=3,
        backend=backend,
    )
    other_mask.non_fictitious_addresses = backend.xp.array([1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="address masks"):
        merge_graphs(ctx, other_mask)


def test_merge_graphs_converts_backends():
    merged = merge_graphs(context(JaxBackend()), decision(NumpyBackend()))
    assert isinstance(merged._backend, JaxBackend)
    assert features(merged, "bus") == {"p": [0.0, 1.0, 2.0], "theta": [0.0, 10.0, 20.0]}
    bus = merge_hyper_edge_sets(context(NumpyBackend()).hyper_edge_sets["bus"], decision(JaxBackend()).hyper_edge_sets["bus"])
    assert isinstance(bus._backend, NumpyBackend) and isinstance(bus.feature_array, np.ndarray)


def test_merge_graphs_on_batches(backend):
    batch = collate_graphs([context(backend), context(backend)])
    decisions = collate_graphs([decision(backend), decision(backend, scale=20.0)])
    merged = merge_graphs(batch, decisions)
    assert merged.is_batch
    assert np.asarray(merged.hyper_edge_sets["bus"].feature_array).shape == (2, 3, 2)
    assert np.asarray(merged.non_fictitious_addresses).shape == (2, 3)
    first, second = separate_graphs(merged)
    assert features(first, "bus") == {"p": [0.0, 1.0, 2.0], "theta": [0.0, 10.0, 20.0]}
    assert features(second, "bus") == {"p": [0.0, 1.0, 2.0], "theta": [0.0, 20.0, 40.0]}


def test_merge_graphs_keeps_padding_and_flat_features(backend):
    ctx = context(backend)
    dec = decision(backend)
    target = GraphShape(
        backend=backend,
        hyper_edge_sets={"bus": backend.xp.array(5), "line": backend.xp.array(4)},
        addresses=backend.xp.array(6),
    )
    ctx.pad(target)
    dec.pad(GraphShape(backend=backend, hyper_edge_sets={"bus": backend.xp.array(5)}, addresses=backend.xp.array(0)))
    merged = merge_graphs(ctx, dec)
    assert merged.hyper_edge_sets["bus"].n_obj == 5 and int(merged.true_shape.hyper_edge_sets["bus"]) == 3
    assert int(merged.current_shape.addresses) == 6 and int(merged.true_shape.addresses) == 3
    assert np.asarray(merged.non_fictitious_addresses).tolist() == [1, 1, 1, 0, 0, 0]
    flat = np.asarray(merged.feature_flat_array)
    assert flat.shape == (5 * 2 + 4,)  # bus: p and theta over 5 padded objects, line: b over 4
    merged.unpad()
    assert merged.hyper_edge_sets["bus"].n_obj == 3


def test_merged_graph_is_a_valid_pytree(backend):
    merged = merge_graphs(context(backend), decision(backend))
    roundtrip = jax.tree.map(lambda x: x, merged)
    assert features(roundtrip, "bus") == features(merged, "bus")
    assert sorted(roundtrip.hyper_edge_sets["bus"].feature_names) == ["p", "theta"]
