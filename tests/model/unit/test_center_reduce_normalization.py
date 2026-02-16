#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph, JaxEdge, JaxGraphShape
from energnn.gnn.normalizer.center_reduce_normalization import GraphCenterReduceNorm, EdgeCenterReduceNorm
from tests.utils import TestProblemLoader

# make deterministic
np.random.seed(0)
pb_loader = TestProblemLoader(seed=0)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)


def test_edge_center_reduce_norm_unbatched_input():
    x = jnp.array([[0.0, 0.0], [2.0, 2.0]], dtype=jnp.float32)
    mod = EdgeCenterReduceNorm(
        n_features=2, update_limit=10, beta_1=0.9, beta_2=0.999, epsilon=1e-6, use_running_average=False
    )

    out = mod(x)
    out_np = np.array(out)

    mean_hat = np.array(mod.mean[...]) / (1.0 - mod.beta_1**mod.updates)
    var_hat = np.array(mod.var[...]) / (1.0 - mod.beta_2**mod.updates)

    # Broadcast mean_hat/var_hat to x's shape and compute expected value
    expected = (np.array(x) - mean_hat[None, :]) / (np.sqrt(var_hat)[None, :] + mod.epsilon)

    np.testing.assert_allclose(out_np, expected, atol=1e-6, rtol=1e-6)


def test_edge_center_reduce_norm_batched_normalizes_across_batch_and_items():
    # batched input (B, N, F)
    B = 3
    single = jnp.array([[0.0, 0.0], [2.0, 2.0]], dtype=jnp.float32)
    x = jnp.stack([single] * B, axis=0)  # shape (B, 2, 2)
    mod = EdgeCenterReduceNorm(n_features=2, update_limit=10, beta_1=0.9, beta_2=0.999, use_running_average=False)

    out = mod(x)
    out_np = np.array(out)

    # For batched, the module computes current_mean/current_var across axes (0,1),
    # and then updates mod.mean.value and mod.var.value and increments mod.updates.
    # Compute mean_hat/var_hat same way as the module uses them:
    mean_hat = np.array(mod.mean[...]) / (1.0 - mod.beta_1**mod.updates)
    var_hat = np.array(mod.var[...]) / (1.0 - mod.beta_2**mod.updates)

    # expected broadcasting across (B,N,F)
    expected = (np.array(x) - mean_hat[None, None, :]) / (np.sqrt(var_hat)[None, None, :] + mod.epsilon)

    np.testing.assert_allclose(out_np, expected, atol=1e-6, rtol=1e-6)


def test_edge_center_reduce_norm_updates_counters_and_variables():
    x = jnp.array([[1.0, 3.0], [2.0, 4.0]], dtype=jnp.float32)
    mod = EdgeCenterReduceNorm(n_features=2, update_limit=10, use_running_average=False)
    # initial updates should be 0
    assert mod.updates == 0
    mean_before = mod.mean[...].copy()
    var_before = mod.var[...].copy()

    _ = mod(x)
    # after one call updates increments
    assert mod.updates == 1
    # mean/var values changed from initial (zeros/ones)
    assert not np.allclose(np.array(mean_before), np.array(mod.mean[...]))
    assert not np.allclose(np.array(var_before), np.array(mod.var[...]))


def test_edge_center_reduce_norm_respects_update_limit_and_use_running_average():
    x = jnp.array([[0.0, 0.0], [2.0, 2.0]], dtype=jnp.float32)

    # Case A: use_running_average=True and update_limit=1 -> only first call should update
    modA = EdgeCenterReduceNorm(n_features=2, update_limit=1, use_running_average=True)
    _ = modA(x)
    mean_after_first = modA.mean[...].copy()
    var_after_first = modA.var[...].copy()
    _ = modA(x)  # second call should not change mean/var
    mean_after_second = modA.mean[...].copy()
    var_after_second = modA.var[...].copy()
    np.testing.assert_allclose(np.array(mean_after_first), np.array(mean_after_second))
    np.testing.assert_allclose(np.array(var_after_first), np.array(var_after_second))

    # Case B: use_running_average=False -> updates continue even if updates >= update_limit
    modB = EdgeCenterReduceNorm(n_features=2, update_limit=1, use_running_average=False)
    _ = modB(x)
    mean_after_first_b = modB.mean[...].copy()
    _ = modB(x)
    mean_after_second_b = modB.mean[...].copy()
    # should be different because updates still applied
    assert not np.allclose(np.array(mean_after_first_b), np.array(mean_after_second_b))


def test_edge_center_reduce_norm_raises_on_invalid_input_dim():
    mod = EdgeCenterReduceNorm(n_features=2, update_limit=10)
    # invalid ndim = 1
    with pytest.raises(ValueError):
        mod(jnp.array([1.0, 2.0]))
    # invalid ndim = 4
    with pytest.raises(ValueError):
        mod(jnp.zeros((1, 2, 3, 4)))


# -----------------------------
# GraphCenterReduceNorm integration
# -----------------------------
def test_initialize_from_example_creates_modules_for_edges_with_features():
    gnorm = GraphCenterReduceNorm(update_limit=10)
    # context has edges with feature_arrays -> initialize
    gnorm.initialize_from_example(jax_context)
    # expect at least 'source' in edge_keys (our test context has 'source' edge)
    assert "source" in gnorm.edge_keys
    # module attribute exists
    assert hasattr(gnorm, "norm_source")


def test_call_creates_modules_on_the_fly_if_missing_and_applies():
    # create GraphCenterReduceNorm without calling initialize
    gnorm = GraphCenterReduceNorm(update_limit=10)
    # pick a context that has edges; since norm_x doesn't exist initially, __call__ should create it
    assert not hasattr(gnorm, "norm_source")
    normalized_context, info = gnorm(context=jax_context, get_info=True)
    # module should now exist
    assert hasattr(gnorm, "norm_source")
    # returned normalized_context is a JaxGraph and info contains quantiles
    assert isinstance(normalized_context, JaxGraph)
    assert "input_graph" in info and "output_graph" in info


def test_graph_center_reduce_norm_applies_per_edge_and_preserves_masks_and_shapes():
    gnorm = GraphCenterReduceNorm(update_limit=10)
    # use context with an actual non_fict mask
    ctx = jax_context
    # ensure some fictitious addresses exist by toggling mask (but our loader uses non_fictitious default all ones)
    # call normalization
    normalized_ctx, _ = gnorm(context=ctx, get_info=False)
    # shapes preserved
    for k in ctx.edges.keys():
        orig = ctx.edges[k]
        normed = normalized_ctx.edges[k]
        if orig.feature_array is None:
            assert normed.feature_array is None
        else:
            assert normed.feature_array.shape == orig.feature_array.shape
            # masked rows (if any) should be zeroed: test by using non_fictitious mask
            mask = np.array(orig.non_fictitious)
            arr_norm = np.array(normed.feature_array)
            # where mask == 0 rows should be all zeros
            for i, m in enumerate(mask):
                if m == 0:
                    assert np.allclose(arr_norm[i], 0.0)


def test_graph_center_reduce_norm_noop_for_none_or_empty_edges():
    # Build a JaxGraph with one edge having feature_array = None and another with empty feature rows shape[-2]==0
    # We'll reuse shapes from existing context to construct shapes
    # edge_none should be preserved and no module created for it.
    edge_none = JaxEdge(feature_array=None, feature_names=None, non_fictitious=jnp.array([]), address_dict=None)
    # empty feature array: shape (0, F)
    empty_arr = jnp.zeros((0, 3))
    edge_empty = JaxEdge(feature_array=empty_arr, feature_names=None, non_fictitious=jnp.array([]), address_dict=None)

    # Build minimal true/current shape objects to satisfy JaxGraph constructor
    true_shape = JaxGraphShape(edges={"node": jnp.array(0)}, addresses=jnp.array(0))
    current_shape = JaxGraphShape(edges={"node": jnp.array(0)}, addresses=jnp.array(0))

    g = JaxGraph(
        edges={"none": edge_none, "empty": edge_empty},
        non_fictitious_addresses=jnp.array([]),
        true_shape=true_shape,
        current_shape=current_shape,
    )
    gnorm = GraphCenterReduceNorm(update_limit=5)
    out_g, _ = gnorm(context=g, get_info=False)
    # ensure edges preserved and no module created for those keys (since none had no items)
    assert hasattr(out_g, "edges")
    assert "none" in out_g.edges and "empty" in out_g.edges
    # modules not created
    assert not hasattr(gnorm, "norm_none")
    assert not hasattr(gnorm, "norm_empty")


def test_graph_center_reduce_norm_get_info_returns_quantiles():
    gnorm = GraphCenterReduceNorm(update_limit=10)
    normalized_ctx, info = gnorm(context=jax_context, get_info=True)
    assert isinstance(info, dict)
    assert "input_graph" in info
    assert "output_graph" in info


def test_graph_center_reduce_norm_batched_forward_compatibility():
    # Use the batched context jax_context_batch; ensure call doesn't error
    gnorm = GraphCenterReduceNorm(update_limit=10)
    # call on batched graph created by loader
    normalized_batch, info = gnorm(context=jax_context_batch, get_info=False)
    assert isinstance(normalized_batch, JaxGraph)
    # for at least one edge shapes should match
    for k in jax_context_batch.edges.keys():
        assert normalized_batch.edges[k].feature_array.shape == jax_context_batch.edges[k].feature_array.shape


# -------------------------
# state and keys tests
# -------------------------
def test_edge_keys_and_updates_increment_when_adding_edges():
    gnorm = GraphCenterReduceNorm(update_limit=5)
    # initially empty
    assert gnorm.edge_keys == tuple()
    # call on context - should create modules and populate edge_keys
    _ = gnorm(context=jax_context, get_info=False)
    assert len(gnorm.edge_keys) > 0
    # check that individual edge modules have updates attribute and it increments after call
    for key in gnorm.edge_keys:
        mod = getattr(gnorm, f"norm_{key}")
        # call on the same edge array directly to increment its updates
        arr = jax_context.edges[key].feature_array
        _ = mod(arr)
        assert mod.updates >= 1


# ensure repeated initialization does not crash and ignores empty edges
def test_initialize_from_example_ignores_zero_length_edges():
    gnorm = GraphCenterReduceNorm(update_limit=5)
    # construct a context with an edge having zero rows
    empty_arr = jnp.zeros((0, 2))
    edge_empty = JaxEdge(feature_array=empty_arr, feature_names=None, non_fictitious=jnp.array([]), address_dict=None)
    true_shape = JaxGraphShape(edges={"empty": jnp.array(0)}, addresses=jnp.array(0))
    current_shape = JaxGraphShape(edges={"empty": jnp.array(0)}, addresses=jnp.array(0))
    g = JaxGraph(
        edges={"empty": edge_empty}, non_fictitious_addresses=jnp.array([]), true_shape=true_shape, current_shape=current_shape
    )
    # should not raise
    gnorm.initialize_from_example(g)
    # no norm created
    assert not hasattr(gnorm, "norm_empty")


# -------------------------
# small robustness tests
# -------------------------
def test_multiple_edges_initialization_and_call_applies_each():
    # Use real jax_context which has multiple edges
    gnorm = GraphCenterReduceNorm(update_limit=10)
    gnorm.initialize_from_example(jax_context)
    # ensure modules created for all present edges with features
    for key in list(jax_context.edges.keys()):
        if jax_context.edges[key].feature_array is not None and jax_context.edges[key].feature_array.shape[-2] > 0:
            assert hasattr(gnorm, f"norm_{key}")

    # call and ensure no error and outputs present
    out, _ = gnorm(context=jax_context, get_info=False)
    for key in out.edges.keys():
        assert out.edges[key].feature_array is not None or jax_context.edges[key].feature_array is None
