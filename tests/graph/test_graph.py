# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from energnn.graph.backend import JaxBackend, NumpyBackend
from energnn.graph.graph import (
    Graph,
    check_hyper_edge_set_dict_type,
    check_valid_addresses,
    collate_graphs,
    concatenate_graphs,
    get_statistics,
    separate_graphs,
)
from energnn.graph.hyper_edge_set import HyperEdgeSet
from energnn.graph.shape import GraphShape
from tests.graph.utils import assert_graphs_equal, make_graph_with_registry, make_simple_edge


# ---------------------------------------------------------------------------
# Backend-parametrized tests
# ---------------------------------------------------------------------------


def test_from_dict_and_basic_props(backend):
    g = make_graph_with_registry(n_addresses=5, n_obj=3, backend=backend)
    assert isinstance(g.true_shape, GraphShape)
    assert isinstance(g.current_shape, GraphShape)
    assert g.is_single is True
    assert g.is_batch is False
    assert len(g.non_fictitious_addresses) == 5


def test_backend_stored_correctly(backend):
    g = make_graph_with_registry(n_addresses=4, n_obj=2, backend=backend)
    assert g._backend == backend


def test_is_batch_detection_after_collation(backend):
    g1 = make_graph_with_registry(n_addresses=3, n_obj=2, backend=backend)
    g2 = make_graph_with_registry(n_addresses=3, n_obj=2, backend=backend)
    batch = collate_graphs([g1, g2])
    assert batch.is_batch is True
    assert batch.is_single is False
    assert batch.non_fictitious_addresses.ndim == 2


def test_collate_preserves_subclass_type(backend):
    g1 = make_graph_with_registry(n_addresses=3, n_obj=2, backend=backend)
    g2 = make_graph_with_registry(n_addresses=3, n_obj=2, backend=backend)
    batch = collate_graphs([g1, g2])
    assert type(batch) is type(g1)


def test_feature_flat_array_getter_and_setter(backend):
    e1 = make_simple_edge(n_obj=2, backend=backend)
    e2 = make_simple_edge(n_obj=2, backend=backend)
    g = Graph.from_dict(hyper_edge_set_dict={"a": e1, "b": e2}, n_addresses=4, backend=backend)
    flat = g.feature_flat_array
    assert flat.ndim == 1
    xp = backend.xp
    new_flat = xp.array(np.array(flat) + 1.0)
    g.feature_flat_array = new_flat
    np.testing.assert_allclose(np.array(g.feature_flat_array), np.array(new_flat))
    with pytest.raises(ValueError):
        g.feature_flat_array = xp.array(np.array(new_flat)[:-1])


def test_pad_and_unpad_graph(backend):
    g = make_graph_with_registry(n_addresses=5, n_obj=2, backend=backend)
    xp = backend.xp
    target_edges = {k: xp.array(int(v) + 3) for k, v in g.current_shape.hyper_edge_sets.items()}
    target_addresses = xp.array(int(g.current_shape.addresses) + 4)
    target_shape = GraphShape(backend=backend, hyper_edge_sets=target_edges, addresses=target_addresses)
    g.pad(target_shape)
    for k in target_edges:
        assert g.hyper_edge_sets[k].n_obj == int(target_edges[k])
    assert len(g.non_fictitious_addresses) == int(target_addresses)
    g.unpad()
    for k, v in g.true_shape.hyper_edge_sets.items():
        assert g.hyper_edge_sets[k].n_obj == int(v)
    assert len(g.non_fictitious_addresses) == int(g.true_shape.addresses)


def test_count_connected_components(backend):
    e = HyperEdgeSet.from_dict(
        port_dict={"u": np.array([0, 2], dtype=np.float32), "v": np.array([1, 2], dtype=np.float32)},
        feature_dict={"val": np.array([0.1, 0.2], dtype=np.float32)},
        backend=backend,
    )
    g = Graph.from_dict(hyper_edge_set_dict={"e": e}, n_addresses=3, backend=backend)
    n_comp, labels = g.count_connected_components()
    assert n_comp == 2
    assert labels.shape[0] == 3
    labels_np = np.array(labels)
    assert labels_np[0] == labels_np[1]
    assert labels_np[2] != labels_np[0]


def test_offset_addresses(backend):
    g1 = make_graph_with_registry(n_addresses=4, n_obj=2, backend=backend)
    orig = {k: np.array(v).copy() for k, v in g1.hyper_edge_sets["etype"].port_dict.items()}
    g1.offset_addresses(10)
    for k in orig:
        np.testing.assert_allclose(np.array(g1.hyper_edge_sets["etype"].port_dict[k]), orig[k] + 10)
    assert len(g1.non_fictitious_addresses) == 4


def test_quantiles_single(backend):
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    e = HyperEdgeSet.from_dict(
        port_dict={"a": np.arange(arr.size)},
        feature_dict={"f0": arr},
        backend=backend,
    )
    g = Graph.from_dict(hyper_edge_set_dict={"E": e}, n_addresses=10, backend=backend)
    q = g.quantiles(q_list=[0.0, 50.0, 100.0])
    assert np.isclose(float(q["E/f0/0.0th-percentile"]), 0.0)
    assert np.isclose(float(q["E/f0/50.0th-percentile"]), 2.0)
    assert np.isclose(float(q["E/f0/100.0th-percentile"]), 4.0)


def test_collate_and_separate_graphs_roundtrip(backend):
    g1 = make_graph_with_registry(n_addresses=4, n_obj=2, backend=backend)
    g2 = make_graph_with_registry(n_addresses=4, n_obj=2, backend=backend)
    batch = collate_graphs([g1, g2])
    separated = separate_graphs(batch)
    assert isinstance(separated, list)
    assert len(separated) == 2
    assert separated[0].true_shape.hyper_edge_sets.keys() == g1.true_shape.hyper_edge_sets.keys()
    assert len(separated[0].non_fictitious_addresses) == len(g1.non_fictitious_addresses)


def test_concatenate_graphs(backend):
    g1 = make_graph_with_registry(n_addresses=4, n_obj=2, backend=backend)
    g2 = make_graph_with_registry(n_addresses=3, n_obj=3, backend=backend)
    cat = concatenate_graphs([g1, g2])
    assert len(cat.non_fictitious_addresses) == len(g1.non_fictitious_addresses) + len(g2.non_fictitious_addresses)
    assert int(cat.true_shape.addresses) == int(g1.true_shape.addresses) + int(g2.true_shape.addresses)


def test_get_statistics(backend):
    e1 = HyperEdgeSet.from_dict(
        port_dict={"a": np.array([0, 1])},
        feature_dict={"x": np.array([1.0, 2.0])},
        backend=backend,
    )
    e2 = HyperEdgeSet.from_dict(
        port_dict={"a": np.array([0, 1])},
        feature_dict={"x": np.array([2.0, 4.0])},
        backend=backend,
    )
    g1 = Graph.from_dict(hyper_edge_set_dict={"T": e1}, n_addresses=2, backend=backend)
    g2 = Graph.from_dict(hyper_edge_set_dict={"T": e2}, n_addresses=2, backend=backend)
    stats = get_statistics(g1, axis=None, norm_graph=g2)

    arr = np.array([1.0, 2.0])
    np.testing.assert_allclose(float(stats["T/x/rmse"]), np.sqrt(np.mean(arr**2)), rtol=1e-5)
    np.testing.assert_allclose(float(stats["T/x/mae"]), np.mean(np.abs(arr)), rtol=1e-5)
    np.testing.assert_allclose(float(stats["T/x/mean"]), np.mean(arr), rtol=1e-5)
    np.testing.assert_allclose(float(stats["T/x/std"]), np.std(arr), rtol=1e-5)
    assert "T/x/nrmse" in stats and "T/x/nmae" in stats


def test_to_backend_conversion(backend):
    g = make_graph_with_registry(n_addresses=4, n_obj=2, backend=backend)
    other = NumpyBackend() if isinstance(backend, JaxBackend) else JaxBackend()
    converted = g.to_backend(other)
    assert converted._backend == other
    np.testing.assert_allclose(
        np.array(converted.non_fictitious_addresses),
        np.array(g.non_fictitious_addresses),
    )


def test_numpy_jax_roundtrip():
    np_g = make_graph_with_registry(n_addresses=5, n_obj=4, backend=NumpyBackend())
    jax_g = Graph.from_numpy_graph(np_g)
    assert isinstance(jax_g._backend, JaxBackend)
    np_round = jax_g.to_numpy_graph()
    assert isinstance(np_round._backend, NumpyBackend)
    assert_graphs_equal(np_g, np_round)


def test_check_edge_dict_type_errors(backend):
    with pytest.raises(TypeError):
        check_hyper_edge_set_dict_type("not a dict")
    with pytest.raises(TypeError):
        check_hyper_edge_set_dict_type({"a": 123})


def test_check_valid_addresses_out_of_range_raises(backend):
    e = make_simple_edge(n_obj=2, backend=backend)
    e.port_dict["dst"] = backend.xp.array(np.array([0, 10], dtype=np.float32))
    with pytest.raises(AssertionError):
        check_valid_addresses({"e": e}, 5)


# ---------------------------------------------------------------------------
# JAX-specific tests (not parametrized)
# ---------------------------------------------------------------------------


def test_jax_from_numpy_graph_types():
    """Graph.to_backend(JaxBackend) returns a Graph whose arrays are jax arrays."""
    import jax.numpy as jnp

    np_g = make_graph_with_registry(n_addresses=5, n_obj=4, backend=NumpyBackend())
    jg = Graph.from_numpy_graph(np_g, dtype="float32")
    assert isinstance(jg, Graph)
    assert isinstance(jg.hyper_edge_sets["etype"], HyperEdgeSet)
    assert isinstance(jg.true_shape, GraphShape)
    assert isinstance(jg.current_shape, GraphShape)
    assert isinstance(jg.feature_flat_array, jnp.ndarray)


def test_jax_pytree_flatten_unflatten():
    """Graph with JaxBackend survives JAX tree_flatten / tree_unflatten."""
    import jax

    np_g = make_graph_with_registry(n_addresses=4, n_obj=3, backend=NumpyBackend())
    jg = np_g.to_backend(JaxBackend())
    leaves, treedef = jax.tree_util.tree_flatten(jg)
    recon = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(recon, Graph)
    np_round = recon.to_numpy_graph()
    assert_graphs_equal(np_g, np_round)


def test_jax_jit_identity():
    """Graph with JaxBackend can be passed through jax.jit."""
    import jax

    np_g = make_graph_with_registry(n_addresses=4, n_obj=2, backend=NumpyBackend())
    jg = np_g.to_backend(JaxBackend())

    @jax.jit
    def identity(g):
        return g

    result = identity(jg)
    assert isinstance(result, Graph)
    assert isinstance(result.hyper_edge_sets["etype"], HyperEdgeSet)
