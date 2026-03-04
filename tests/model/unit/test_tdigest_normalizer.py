#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import jax
import jax.numpy as jnp

import energnn.model.normalizer.tdigest_normalizer as tdn
from energnn.model.normalizer.tdigest_normalizer import (
    TDigestModule,
    TDigestNormalizer,
)
from energnn.graph.jax import JaxEdge, JaxGraph
from tests.utils import TestProblemLoader

# TDigestNormalizer relies on float32 explicitly in io_callback.
jax.config.update("jax_enable_x64", False)

# make deterministic
np.random.seed(0)

# small fixture graphs (used by some tests)
n = 10
pb_loader = TestProblemLoader(seed=0)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)  # single example usable in tests


def test_merge_equal_quantiles_host_no_repeats():
    # p grid of length 3, two features with strictly increasing quantiles
    p = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    q = np.array([[0.0, -1.0], [0.5, 0.0], [1.0, 1.0]], dtype=np.float32)  # (3,2)
    # broadcast p to (3,2)
    p_2d = np.stack([p, p], axis=1)
    p_out, q_out = tdn._merge_equal_quantiles_host(p_2d, q)
    assert p_out.shape == (3, 2)
    assert q_out.shape == (3, 2)
    # distinct quantiles -> p_out should be identical to p broadcasted across features
    np.testing.assert_allclose(p_out[:, 0], p, rtol=0, atol=1e-6)
    np.testing.assert_allclose(p_out[:, 1], p, rtol=0, atol=1e-6)
    np.testing.assert_allclose(q_out, q.astype(np.float32), rtol=0, atol=0)


def test_merge_equal_quantiles_host_with_duplicates():
    # create q where q[1]==q[2] for the first feature -> merging should average p[1] & p[2]
    p = np.array([0.0, 0.4, 0.6, 1.0], dtype=np.float32)
    q = np.array(
        [
            [0.0, 10.0],
            [0.5, 20.0],
            [0.5, 30.0],  # duplicate in column 0 with previous row
            [1.0, 40.0],
        ],
        dtype=np.float32,
    )  # shape (4,2)
    # broadcast p to (4,2)
    p_2d = np.stack([p, p], axis=1)
    p_out, q_out = tdn._merge_equal_quantiles_host(p_2d, q)
    assert p_out.shape == (4, 2)
    assert q_out.shape == (4, 2)
    # for column 0, rows 1 and 2 have identical q values -> they should receive the average p = (0.4+0.6)/2 = 0.5
    assert np.isclose(p_out[1, 0], 0.5, atol=1e-6)
    assert np.isclose(p_out[2, 0], 0.5, atol=1e-6)
    # q_out should equal q cast to float32
    np.testing.assert_allclose(q_out, q.astype(np.float32), rtol=0, atol=1e-6)


def test_ingest_new_data_shapes_and_fp():
    # batch: N rows, F features = 2
    batch = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [3.0, 40.0]], dtype=np.float32)
    F = batch.shape[1]
    max_centroids = np.array([50] * F, dtype=np.int32)
    min_val = np.array([np.nan] * F, dtype=np.float32)
    max_val = np.array([np.nan] * F, dtype=np.float32)
    centroids_m = np.zeros((50, F), dtype=np.float32)
    centroids_c = np.zeros((50, F), dtype=np.float32)
    K = 3
    fp = np.linspace(-1, 1, K)[:, None] + np.zeros((1, F), dtype=np.float32)
    xp = np.zeros((K, F), dtype=np.float32)
    mask = np.ones((batch.shape[0], 1), dtype=np.float32)

    res = tdn._ingest_new_data(max_centroids, min_val, max_val, centroids_m, centroids_c, fp, xp, batch, mask)
    (
        new_max_centroids,
        new_min_val,
        new_max_val,
        new_centroids_m,
        new_centroids_c,
        new_fp,
        new_xp,
    ) = res

    # shapes
    assert new_xp.shape == (K, F)
    assert new_fp.shape == (K, F)
    assert new_xp.dtype == np.float32
    assert new_fp.dtype == np.float32
    # new_fp should lie in [-1, 1]
    assert np.all(new_fp >= -1.0 - 1e-6)
    assert np.all(new_fp <= 1.0 + 1e-6)


def test_ingest_new_data_quantile_values_basic():
    # prepare a column with constant values and a column with increasing values
    col0 = np.zeros((8,), dtype=np.float32)  # constant -> quantiles same 0
    col1 = np.arange(8, dtype=np.float32)  # increasing -> quantiles should follow distribution
    batch = np.stack([col0, col1], axis=1)  # shape (8,2)
    F = batch.shape[1]
    max_centroids = np.array([100] * F, dtype=np.int32)
    min_val = np.array([np.nan] * F, dtype=np.float32)
    max_val = np.array([np.nan] * F, dtype=np.float32)
    centroids_m = np.zeros((100, F), dtype=np.float32)
    centroids_c = np.zeros((100, F), dtype=np.float32)
    K = 5
    fp = np.linspace(-1, 1, K)[:, None] + np.zeros((1, F), dtype=np.float32)
    xp = np.zeros((K, F), dtype=np.float32)
    mask = np.ones((batch.shape[0], 1), dtype=np.float32)

    res = tdn._ingest_new_data(max_centroids, min_val, max_val, centroids_m, centroids_c, fp, xp, batch, mask)
    new_xp = res[6]

    # col0: all quantiles equal to 0
    np.testing.assert_allclose(new_xp[:, 0], 0.0, atol=1e-6)
    # col1: quantiles should be in ascending order and within min/max of col1
    assert np.all(np.diff(new_xp[:, 1]) >= -1e-6)
    assert new_xp[0, 1] >= col1.min() - 1e-6
    assert new_xp[-1, 1] <= col1.max() + 1e-6


def test_tdigest_module_initial_shapes():
    mod = TDigestModule(in_size=3, update_limit=5, n_breakpoints=4, max_centroids=20, use_running_average=False)
    # xp and fp shapes
    K = mod.n_breakpoints
    assert mod.xp_var[...].shape == (K, 3)
    assert mod.fp_var[...].shape == (K, 3)
    # digest centroids shapes
    assert mod.centroids_m_var[...].shape == (mod.max_centroids, 3)
    assert mod.centroids_c_var[...].shape == (mod.max_centroids, 3)


def test_tdigest_module_call_updates_and_maps_values():
    """
    Call the module and verify xp/fp set and output mapping in [-1,1].
    """
    mod = TDigestModule(in_size=2, update_limit=5, n_breakpoints=4, max_centroids=20, use_running_average=False)
    # input x shape (n_items, F)
    x = jnp.array(np.random.normal(size=(5, 2)), dtype=jnp.float32)
    non_fictitious = jnp.ones((5, 1), dtype=jnp.float32)

    out = mod(x, non_fictitious)
    # updates incremented
    assert mod.updates[...] == 1
    # xp and fp now set (originally zeros for xp)
    assert mod.xp_var[...].shape[1] == 2
    assert mod.fp_var[...].shape[1] == 2
    assert not np.allclose(np.array(mod.xp_var[...]), 0.0)
    # output has same shape as input
    assert out.shape == x.shape
    assert np.all(np.isfinite(np.array(out)))


def test_tdigest_normalizer_apply_preserves_none_feature_edges(monkeypatch):
    # Build graph with one edge having None features and another with features
    node_edge_with_none = JaxEdge(
        address_dict=jax_context.edges["source"].address_dict,
        feature_array=None,
        feature_names=None,
        non_fictitious=jax_context.edges["source"].non_fictitious,
    )
    edge_with_feat = JaxEdge(
        address_dict=jax_context.edges["arrow"].address_dict,
        feature_array=jnp.ones((jax_context.edges["arrow"].feature_array.shape[0], 2), dtype=jnp.float32),
        feature_names={"f1": jnp.array(0), "f2": jnp.array(1)},
        non_fictitious=jax_context.edges["arrow"].non_fictitious,
    )
    g = JaxGraph(
        edges={"source": node_edge_with_none, "arrow": edge_with_feat},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    normalizer = TDigestNormalizer(
        in_structure=pb_loader.context_structure, update_limit=1, n_breakpoints=3, max_centroids=8, use_running_average=False
    )
    # patch TDigestModule.__call__ to return input*2 (simulate normalization)
    monkeypatch.setattr(TDigestModule, "__call__", lambda self, x, nf: x * 2.0 * nf)
    out_graph, _ = normalizer(graph=g, get_info=False)

    # source edge had None -> must remain None
    assert out_graph.edges["source"].feature_array is None
    # edge with features must be multiplied by 2 and masked by non_fictitious
    mask2 = np.array(edge_with_feat.non_fictitious)
    expected = np.array(edge_with_feat.feature_array) * mask2[..., None] * 2.0
    np.testing.assert_allclose(np.array(out_graph.edges["arrow"].feature_array), expected, rtol=1e-6, atol=1e-6)
