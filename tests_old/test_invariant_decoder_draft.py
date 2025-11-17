#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from energnn.gnn.decoder import (
    AttentionInvariantDecoder,
    InvariantDecoder,
    MeanInvariantDecoder,
    SumInvariantDecoder,
    ZeroInvariantDecoder,
)
from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph
from tests.utils import TestProblemLoader


# Prepare deterministic data and loader
np.random.seed(0)
n = 10
pb_loader = TestProblemLoader(
    dataset_size=8,
    n_batch=4,
    context_edge_params={
        "node": {"n_obj": n, "feature_list": ["a", "b"], "address_list": ["0"]},
        "edge": {"n_obj": n, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
    },
    oracle_edge_params={
        "node": {"n_obj": n, "feature_list": ["e"]},
        "edge": {"n_obj": n, "feature_list": ["f"]},
    },
    n_addr=n,
    shuffle=True,
)
pb_batch = next(iter(pb_loader))
context_batch, _ = pb_batch.get_context()
jax_context_batch = JaxGraph.from_numpy_graph(context_batch)
context = separate_graphs(context_batch)[0]
jax_context = JaxGraph.from_numpy_graph(context)
coordinates = jnp.array(np.random.uniform(size=(10, 7)))
coordinates_batch = jnp.array(np.random.uniform(size=(4, 10, 7)))


# ---------- util helpers ----------
def assert_vmap_jit_consistent(fn_apply, params, ctx_batch, coords_batch, rtol=1e-3, atol=1e-3):
    """
    Check that vmapped and vmapped+jit versions produce consistent shapes/values (within tolerance).
    fn_apply: callable (params, ctx, coords, get_info) -> (out, info)
    """
    apply_vmap = jax.vmap(fn_apply, in_axes=[None, 0, 0, None], out_axes=0)
    out1, info1 = apply_vmap(params, ctx_batch, coords_batch, False)
    out2, info2 = apply_vmap(params, ctx_batch, coords_batch, True)
    out3, info3 = jax.jit(apply_vmap)(params, ctx_batch, coords_batch, False)
    out4, info4 = jax.jit(apply_vmap)(params, ctx_batch, coords_batch, True)

    # Compare shapes first
    # out can be arrays or JaxGraph for some decoders (here invariant decoders return arrays)
    assert type(out1) == type(out2) == type(out3) == type(out4)
    np_out1 = np.array(out1)
    np_out3 = np.array(out3)
    assert np_out1.shape == np.array(out2).shape == np_out3.shape == np.array(out4).shape

    # numerical closeness between jitted and non-jitted (some differences possible)
    np.testing.assert_allclose(np_out1, np_out3, rtol=rtol, atol=atol)
    # info when get_info True should be compatible (here usually empty dict)
    assert info1 == {}
    assert info3 == {}
    assert info2 == info4


# ---------- ZeroInvariantDecoder ----------
def test_zero_invariant_decoder_single_and_batch_shapes_and_values():
    decoder = ZeroInvariantDecoder()
    # single
    rng = jax.random.PRNGKey(0)
    params = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=5)
    out, info = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=True)
    # zero vector and shape
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (5,)
    np.testing.assert_allclose(np.array(out), np.zeros((5,)))
    assert info == {}

    # batch: vmapped/jit should work
    def fn(p, ctx, coords, gi):
        return decoder.apply(p, context=ctx, coordinates=coords, get_info=gi)

    assert_vmap_jit_consistent(fn, params, jax_context_batch, coordinates_batch)


def test_zero_invariant_decoder_init_deterministic():
    decoder = ZeroInvariantDecoder()
    rng = jax.random.PRNGKey(1)
    p1 = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=3)
    p2 = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=3)
    chex.assert_trees_all_equal(p1, p2)


# ---------- SumInvariantDecoder ----------
def test_sum_invariant_decoder_basic_and_masking():
    decoder = SumInvariantDecoder(
        psi_hidden_size=[8], psi_out_size=6, psi_activation=nn.relu, phi_hidden_size=[8], phi_activation=nn.relu
    )
    rng = jax.random.PRNGKey(2)
    params = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=4)

    # single forward
    out, info = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=True)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (4,)
    assert info == {}

    # modify mask: set all addresses to zero -> output may change. We check output is consistent with mask applied:
    ctx_masked = jax_context
    mask_zero = jnp.zeros_like(jax_context.non_fictitious_addresses)
    ctx_masked = JaxGraph(
        edges=ctx_masked.edges,
        true_shape=ctx_masked.true_shape,
        current_shape=ctx_masked.current_shape,
        non_fictitious_addresses=mask_zero,
    )
    out_masked, _ = decoder.apply(params, context=ctx_masked, coordinates=coordinates)
    # With mask all zeros, psi * mask -> zeros, sum -> zeros, phi(zeros) deterministic: should be finite vector.
    assert np.all(np.isfinite(np.array(out_masked)))
    # shape preserved
    assert out_masked.shape == out.shape

    # vmap/jit compatibility
    def fn(p, ctx, coords, gi):
        return decoder.apply(p, context=ctx, coordinates=coords, get_info=gi)

    assert_vmap_jit_consistent(fn, params, jax_context_batch, coordinates_batch)


def test_sum_invariant_decoder_init_deterministic():
    decoder = SumInvariantDecoder(
        psi_hidden_size=[4], psi_out_size=3, psi_activation=nn.relu, phi_hidden_size=[4], phi_activation=nn.relu
    )
    rng = jax.random.PRNGKey(11)
    p1 = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=2)
    p2 = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=2)
    chex.assert_trees_all_equal(p1, p2)


# ---------- MeanInvariantDecoder ----------
def test_mean_invariant_decoder_shape_and_mask_behavior():
    decoder = MeanInvariantDecoder(
        psi_hidden_size=[8], psi_out_size=5, psi_activation=nn.relu, phi_hidden_size=[8], phi_activation=nn.relu
    )
    rng = jax.random.PRNGKey(3)
    params = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=6)

    # According to implementation, output is phi(numerator/denominator) * expand_dims(non_fictitious_addresses, -1)
    # -> shape (n_addresses, out_size)
    out, info = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=True)
    assert isinstance(out, jnp.ndarray)
    assert out.shape[0] == jax_context.non_fictitious_addresses.shape[0]
    assert out.shape[1] == 6
    assert info == {}

    # If mask is all zeros, output should be zeros (numerically stable)
    ctx_all_zero = JaxGraph(
        edges=jax_context.edges,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
        non_fictitious_addresses=jnp.zeros_like(jax_context.non_fictitious_addresses),
    )
    out_zero_mask, _ = decoder.apply(params, context=ctx_all_zero, coordinates=coordinates)
    assert np.allclose(np.array(out_zero_mask), 0.0, atol=1e-6)

    # vmap/jit compatibility
    def fn(p, ctx, coords, gi):
        return decoder.apply(p, context=ctx, coordinates=coords, get_info=gi)

    assert_vmap_jit_consistent(fn, params, jax_context_batch, coordinates_batch)


def test_mean_invariant_decoder_init_deterministic():
    decoder = MeanInvariantDecoder(
        psi_hidden_size=[4], psi_out_size=3, psi_activation=nn.relu, phi_hidden_size=[4], phi_activation=nn.relu
    )
    rng = jax.random.PRNGKey(12)
    p1 = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=2)
    p2 = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=2)
    chex.assert_trees_all_equal(p1, p2)


# ---------- AttentionInvariantDecoder ----------
def test_attention_invariant_decoder_heads_and_shapes():
    decoder = AttentionInvariantDecoder(
        n=3,
        v_hidden_size=[8],
        v_activation=nn.relu,
        v_out_size=2,
        s_hidden_size=[8],
        s_activation=nn.relu,
        psi_hidden_size=[8],
        psi_activation=nn.relu,
    )
    rng = jax.random.PRNGKey(4)
    params = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=5)

    out, info = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=True)
    assert isinstance(out, jnp.ndarray)
    # value_list -> n vectors each length v_out_size -> concatenated -> length n*v_out_size
    assert out.shape == (5,)
    assert info == {}

    # With vmap/jit wrapper
    def fn(p, ctx, coords, gi):
        return decoder.apply(p, context=ctx, coordinates=coords, get_info=gi)

    assert_vmap_jit_consistent(fn, params, jax_context_batch, coordinates_batch)


def test_attention_invariant_decoder_init_deterministic():
    decoder = AttentionInvariantDecoder(
        n=2,
        v_hidden_size=[4],
        v_activation=nn.relu,
        v_out_size=3,
        s_hidden_size=[4],
        s_activation=nn.relu,
        psi_hidden_size=[4],
        psi_activation=nn.relu,
    )
    rng = jax.random.PRNGKey(13)
    p1 = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=2)
    p2 = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=2)
    chex.assert_trees_all_equal(p1, p2)


# ---------- small edge-case tests ----------
def test_mean_decoder_all_masked_stability():
    # Confirm numeric stability: no NaNs when all addresses are masked out
    decoder = MeanInvariantDecoder(
        psi_hidden_size=[2], psi_out_size=2, psi_activation=nn.relu, phi_hidden_size=[2], phi_activation=nn.relu
    )
    rng = jax.random.PRNGKey(7)
    params = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=3)
    ctx_all_zero = JaxGraph(
        edges=jax_context.edges,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
        non_fictitious_addresses=jnp.zeros_like(jax_context.non_fictitious_addresses),
    )
    out, _ = decoder.apply(params, context=ctx_all_zero, coordinates=coordinates)
    assert not np.any(np.isnan(np.array(out)))
    assert np.allclose(np.array(out), 0.0, atol=1e-6)
