#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import chex
import jax
import jax.numpy as jnp

from energnn.model.decoder.invariant_decoder import (
    SumInvariantDecoder,
    MeanInvariantDecoder,
    InvariantDecoder,
)
from energnn.model.utils import MLP
from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph
from tests.utils import TestProblemLoader


np.random.seed(0)
pb_loader = TestProblemLoader(seed=0, batch_size=4, n_max=10)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)
coordinates = jnp.array(np.random.uniform(size=(10, 7)))
coordinates_batch = jnp.array(np.random.uniform(size=(4, 10, 7)))


def assert_vmap_jit_consistent(decoder: InvariantDecoder, ctx_batch: JaxGraph, coords_batch: jax.Array, rtol=1e-3, atol=1e-3):
    """
    Vmap over (batched) graphs and coordinates and compare jit vs non-jit and get_info variations.
    Works for nnx decoders that are already built in __init__ (no lazy RNG consumption on first call).
    """

    def apply_fn(graph, coords, get_info):
        return decoder(graph=graph, coordinates=coords, get_info=get_info)

    apply_vmap = jax.vmap(apply_fn, in_axes=(0, 0, None), out_axes=0)

    out1, info1 = apply_vmap(ctx_batch, coords_batch, False)
    out2, info2 = apply_vmap(ctx_batch, coords_batch, True)
    out3, info3 = jax.jit(apply_vmap)(ctx_batch, coords_batch, False)
    out4, info4 = jax.jit(apply_vmap)(ctx_batch, coords_batch, True)

    # Compare types & shapes
    assert type(out1) == type(out2) == type(out3) == type(out4)
    np_out1 = np.array(out1)
    np_out3 = np.array(out3)
    assert np_out1.shape == np.array(out2).shape == np_out3.shape == np.array(out4).shape

    # numerical closeness between jitted and non-jitted
    np.testing.assert_allclose(np_out1, np_out3, rtol=rtol, atol=atol)
    assert info1 == {}
    assert info3 == {}
    assert info2 == info4


# -------------------------
# SumInvariantDecoder tests
# -------------------------
def test_sum_invariant_decoder_basic_and_masking():
    psi = MLP(in_size=7, hidden_sizes=[8], out_size=6, activation=None, seed=2)
    phi = MLP(in_size=6, hidden_sizes=[8], out_size=4, activation=None, seed=2)
    decoder = SumInvariantDecoder(psi=psi, phi=phi)

    # single forward
    out, info = decoder(graph=jax_context, coordinates=coordinates, get_info=True)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (4,)
    assert info == {}

    # mask all zeros stability: when mask is zero, numerator=0 -> phi(0) should be finite
    ctx_masked = JaxGraph(
        edges=jax_context.edges,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
        non_fictitious_addresses=jnp.zeros_like(jax_context.non_fictitious_addresses),
    )
    out_masked, _ = decoder(graph=ctx_masked, coordinates=coordinates, get_info=False)
    assert np.all(np.isfinite(np.array(out_masked)))
    assert out_masked.shape == out.shape

    # vmap/jit compatibility
    assert_vmap_jit_consistent(decoder, ctx_batch=jax_context_batch, coords_batch=coordinates_batch)


def test_sum_invariant_decoder_init_deterministic():
    psi1 = MLP(in_size=7, hidden_sizes=[4], out_size=3, activation=None, seed=11)
    phi1 = MLP(in_size=3, hidden_sizes=[4], out_size=2, activation=None, seed=11)
    dec1 = SumInvariantDecoder(psi=psi1, phi=phi1)

    psi2 = MLP(in_size=7, hidden_sizes=[4], out_size=3, activation=None, seed=11)
    phi2 = MLP(in_size=3, hidden_sizes=[4], out_size=2, activation=None, seed=11)
    dec2 = SumInvariantDecoder(psi=psi2, phi=phi2)

    out1, _ = dec1(graph=jax_context, coordinates=coordinates, get_info=False)
    out2, _ = dec2(graph=jax_context, coordinates=coordinates, get_info=False)
    chex.assert_trees_all_equal(out1, out2)


def test_sum_invariant_decoder_numeric_identity():
    """
    Replace psi and phi by identity functions and check output == sum(mask * coords).
    """
    d = coordinates.shape[1]  # coordinate dimension
    psi = MLP(in_size=d, hidden_sizes=[], out_size=d, activation=None, seed=21)
    phi = MLP(in_size=d, hidden_sizes=[], out_size=d, activation=None, seed=21)
    decoder = SumInvariantDecoder(psi=psi, phi=phi)

    # patch psi and phi to identity functions
    decoder.psi = lambda x: x
    decoder.phi = lambda x: x

    out, _ = decoder(graph=jax_context, coordinates=coordinates, get_info=False)
    mask = np.array(jax_context.non_fictitious_addresses)
    coords_np = np.array(coordinates)
    expected = np.sum(coords_np * mask[:, None], axis=0)
    np.testing.assert_allclose(np.array(out), expected, rtol=0.0, atol=1e-6)


# -------------------------
# MeanInvariantDecoder tests (nnx returns a global vector)
# -------------------------
def test_mean_invariant_decoder_shape_and_mask_behavior():
    psi = MLP(in_size=7, hidden_sizes=[8], out_size=5, activation=None, seed=3)
    phi = MLP(in_size=5, hidden_sizes=[8], out_size=6, activation=None, seed=3)
    decoder = MeanInvariantDecoder(psi=psi, phi=phi)

    out, info = decoder(graph=jax_context, coordinates=coordinates, get_info=True)
    assert isinstance(out, jnp.ndarray)
    # NNX Mean returns a global vector of size out_size
    assert out.shape == (6,)
    assert info == {}

    # all-zero mask => numerator=0 => phi(0) should be finite (and for identity phi returns 0)
    ctx_all_zero = JaxGraph(
        edges=jax_context.edges,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
        non_fictitious_addresses=jnp.zeros_like(jax_context.non_fictitious_addresses),
    )
    out_zero_mask, _ = decoder(graph=ctx_all_zero, coordinates=coordinates)
    assert np.all(np.isfinite(np.array(out_zero_mask)))
    assert out_zero_mask.shape == out.shape

    # vmap/jit compatibility
    assert_vmap_jit_consistent(decoder, ctx_batch=jax_context_batch, coords_batch=coordinates_batch)


def test_mean_invariant_decoder_init_deterministic():
    psi1 = MLP(in_size=7, hidden_sizes=[4], out_size=3, activation=None, seed=12)
    phi1 = MLP(in_size=3, hidden_sizes=[4], out_size=2, activation=None, seed=12)
    dec1 = MeanInvariantDecoder(psi=psi1, phi=phi1)

    psi2 = MLP(in_size=7, hidden_sizes=[4], out_size=3, activation=None, seed=12)
    phi2 = MLP(in_size=3, hidden_sizes=[4], out_size=2, activation=None, seed=12)
    dec2 = MeanInvariantDecoder(psi=psi2, phi=phi2)
    out1, _ = dec1(graph=jax_context, coordinates=coordinates, get_info=False)
    out2, _ = dec2(graph=jax_context, coordinates=coordinates, get_info=False)
    chex.assert_trees_all_equal(out1, out2)


def test_mean_invariant_decoder_numeric_identity():
    """
    psi = identity, phi = identity => output = numerator / denominator
    where numerator = sum(mask * coords) and denominator = sum(mask) + 1e-9
    """
    d = coordinates.shape[1]
    psi = MLP(in_size=d, hidden_sizes=[], out_size=d, activation=None, seed=22)
    phi = MLP(in_size=d, hidden_sizes=[], out_size=d, activation=None, seed=22)
    decoder = MeanInvariantDecoder(psi=psi, phi=phi)

    decoder.psi = lambda x: x
    decoder.phi = lambda x: x

    out, _ = decoder(graph=jax_context, coordinates=coordinates, get_info=False)
    mask = np.array(jax_context.non_fictitious_addresses)
    coords_np = np.array(coordinates)
    numerator = np.sum(coords_np * mask[:, None], axis=0)
    denominator = float(np.sum(mask)) + 1e-9
    expected = numerator / denominator
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-6, atol=1e-6)
