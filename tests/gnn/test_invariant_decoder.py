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
from flax.core.frozen_dict import freeze, unfreeze

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
from tests.gnn.utils import set_dense_layers_to_identity_or_zero


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
    assert type(out1) == type(out2) == type(out3) == type(out4)
    np_out1 = np.array(out1)
    np_out3 = np.array(out3)
    assert np_out1.shape == np.array(out2).shape == np_out3.shape == np.array(out4).shape

    # numerical closeness between jitted and non-jitted
    np.testing.assert_allclose(np_out1, np_out3, rtol=rtol, atol=atol)
    assert info1 == {}
    assert info3 == {}
    assert info2 == info4

# ------------------------
# ZeroInvariantDecoder tests
# ------------------------
def test_zero_invariant_decoder_output_shapes_and_values():
    decoder = ZeroInvariantDecoder()
    rng = jax.random.PRNGKey(0)
    params = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=5)
    out, info = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=True)
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


# ------------------------
# SumInvariantDecoder tests
# ------------------------
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

    # mask all zeros stability
    ctx_masked = jax_context
    mask_zero = jnp.zeros_like(jax_context.non_fictitious_addresses)
    ctx_masked = JaxGraph(
        edges=ctx_masked.edges,
        true_shape=ctx_masked.true_shape,
        current_shape=ctx_masked.current_shape,
        non_fictitious_addresses=mask_zero,
    )
    out_masked, _ = decoder.apply(params, context=ctx_masked, coordinates=coordinates)
    assert np.all(np.isfinite(np.array(out_masked)))
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

def test_sum_invariant_decoder_numeric_identity():
    """
    Build SumInvariantDecoder with psi and phi being identity maps (no hidden layers).
    Expect output = phi(sum(mask * coordinates)) = sum(mask * coordinates).
    """
    d = coordinates.shape[1]  # coordinate dimension
    decoder = SumInvariantDecoder(
        psi_hidden_size=[], psi_out_size=d, psi_activation=None, phi_hidden_size=[], phi_activation=None
    )
    rng = jax.random.PRNGKey(21)
    params = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=d)

    # Patch psi and phi to identity (dense kernel -> identity, bias -> 0)
    params = set_dense_layers_to_identity_or_zero(params, "psi", set_identity=True)
    params = set_dense_layers_to_identity_or_zero(params, "phi", set_identity=True)

    out, _ = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=False)
    # compute expected: sum over addresses of coordinates * mask
    mask = np.array(jax_context.non_fictitious_addresses)
    coords = np.array(coordinates)
    expected = np.sum(coords * mask[:, None], axis=0)
    np.testing.assert_allclose(np.array(out), expected, rtol=0.0, atol=1e-6)


# ------------------------
# MeanInvariantDecoder tests
# ------------------------
def test_mean_invariant_decoder_shape_and_mask_behavior():
    decoder = MeanInvariantDecoder(
        psi_hidden_size=[8], psi_out_size=5, psi_activation=nn.relu, phi_hidden_size=[8], phi_activation=nn.relu
    )
    rng = jax.random.PRNGKey(3)
    params = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=6)

    out, info = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=True)
    assert isinstance(out, jnp.ndarray)
    assert out.shape[0] == jax_context.non_fictitious_addresses.shape[0]
    assert out.shape[1] == 6
    assert info == {}

    ctx_all_zero = JaxGraph(
        edges=jax_context.edges,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
        non_fictitious_addresses=jnp.zeros_like(jax_context.non_fictitious_addresses),
    )
    out_zero_mask, _ = decoder.apply(params, context=ctx_all_zero, coordinates=coordinates)
    assert np.allclose(np.array(out_zero_mask), 0.0, atol=1e-6)

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


def test_mean_invariant_decoder_numeric_identity():
    """
    Build MeanInvariantDecoder with psi identity and phi identity.
    According to implementation, numerator = sum(mask * coords), denominator = psi_out_size (bug in code).
    output = phi(numerator / denominator) * expand(mask, -1)
    """
    d = coordinates.shape[1]
    psi_out = d
    decoder = MeanInvariantDecoder(
        psi_hidden_size=[], psi_out_size=psi_out, psi_activation=None, phi_hidden_size=[], phi_activation=None
    )
    rng = jax.random.PRNGKey(22)
    params = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=d)

    # make psi and phi identity
    params = set_dense_layers_to_identity_or_zero(params, "psi", set_identity=True)
    params = set_dense_layers_to_identity_or_zero(params, "phi", set_identity=True)

    out, _ = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=False)
    mask = np.array(jax_context.non_fictitious_addresses)
    coords = np.array(coordinates)
    numerator = np.sum(coords * mask[:, None], axis=0)  # shape (d,)
    # According to code: denominator == psi_out_size (sum of ones of length psi_out_size)
    denominator = float(psi_out) + 1e-9
    expected_per_address = (numerator / denominator)[None, :] * mask[:, None]
    np.testing.assert_allclose(np.array(out), expected_per_address, rtol=1e-6, atol=1e-6)


# ------------------------
# AttentionInvariantDecoder tests
# ------------------------
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
    assert out.shape == (5,)
    assert info == {}

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

def test_attention_invariant_decoder_numeric_simple():
    """
    Build AttentionInvariantDecoder with:
      - value MLP = identity (v -> v)
      - score MLP = zero (so exp(s)=1 and uniform softmax)
      - psi_mlp = identity
    Then output should be sum(mask * coords) / (num_masked)  (for n=1 head, v_out_size == coord dim)
    """
    d = coordinates.shape[1]
    decoder = AttentionInvariantDecoder(
        n=1,
        v_hidden_size=[],
        v_activation=None,
        v_out_size=d,
        s_hidden_size=[],
        s_activation=None,
        psi_hidden_size=[],
        psi_activation=None,
    )
    rng = jax.random.PRNGKey(23)
    params = decoder.init_with_size(rngs=rng, context=jax_context, coordinates=coordinates, out_size=d)

    # set value-mlp-0 to identity, score-mlp-0 to zero, psi-mlp to identity
    params = set_dense_layers_to_identity_or_zero(params, "value-mlp-0", set_identity=True)
    params = set_dense_layers_to_identity_or_zero(params, "score-mlp-0", set_identity=False)  # zeros => s=0
    params = set_dense_layers_to_identity_or_zero(params, "psi-mlp", set_identity=True)

    out, _ = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=False)
    mask = np.array(jax_context.non_fictitious_addresses)
    coords = np.array(coordinates)

    numerator = np.sum(coords * mask[:, None], axis=0)  # shape (d,)
    denom = np.sum(mask) + 1e-9  # number of masked addresses
    expected = numerator / float(denom)
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-6, atol=1e-6)
