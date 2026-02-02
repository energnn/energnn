#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import jax.numpy as jnp
from flax import nnx

from energnn.model.decoder import AttentionInvariantDecoder, MLPEquivariantDecoder, MeanInvariantDecoder, SumInvariantDecoder
from ...utils import test_context, test_context_batch, test_coordinates, test_coordinates_batch


def test_mlp_equivariant_decoder():
    decoder = MLPEquivariantDecoder(
        out_structure={"node": {"e": jnp.array([0])}, "edge": {"f": jnp.array([0])}},
        hidden_size=[16],
        activation=nnx.relu,
        seed=64,
    )
    output, _ = decoder(graph=test_context, coordinates=test_coordinates, get_info=False)

    def f(x, h, get_info):
        return decoder(graph=x, coordinates=h, get_info=get_info)

    decoder_vmap = nnx.vmap(f, in_axes=(0, 0, None), out_axes=0)

    output_batch, _ = decoder_vmap(test_context_batch, test_coordinates_batch, False)

    # Assert that the vmapped decoder output is the same as the non-batched decoder output
    assert jnp.allclose(output.feature_flat_array, output_batch.feature_flat_array[0])


def test_sum_invariant_decoder():
    decoder = SumInvariantDecoder(
        psi_hidden_size=[16],
        psi_out_size=32,
        psi_activation=nnx.relu,
        phi_hidden_size=[16],
        phi_activation=nnx.relu,
        out_size=8,
        seed=64,
    )
    output, _ = decoder(graph=test_context, coordinates=test_coordinates, get_info=False)

    def f(x, h, get_info):
        return decoder(graph=x, coordinates=h, get_info=get_info)

    decoder_vmap = nnx.vmap(f, in_axes=(0, 0, None), out_axes=0)

    output_batch, _ = decoder_vmap(test_context_batch, test_coordinates_batch, False)
    assert jnp.allclose(output, output_batch[0])


def test_mean_invariant_decoder():
    decoder = MeanInvariantDecoder(
        psi_hidden_size=[16],
        psi_out_size=32,
        psi_activation=nnx.relu,
        phi_hidden_size=[16],
        phi_activation=nnx.relu,
        out_size=8,
        seed=64,
    )
    output, _ = decoder(graph=test_context, coordinates=test_coordinates, get_info=False)

    def f(x, h, get_info):
        return decoder(graph=x, coordinates=h, get_info=get_info)

    decoder_vmap = nnx.vmap(f, in_axes=(0, 0, None), out_axes=0)

    output_batch, _ = decoder_vmap(test_context_batch, test_coordinates_batch, False)
    assert jnp.allclose(output, output_batch[0])


def test_attention_invariant_decoder():
    decoder = AttentionInvariantDecoder(
        n=2,
        v_hidden_size=[16],
        v_activation=nnx.relu,
        v_out_size=32,
        s_hidden_size=[16],
        s_activation=nnx.relu,
        psi_hidden_size=[16],
        psi_activation=nnx.relu,
        out_size=8,
        seed=64,
    )
    output, _ = decoder(graph=test_context, coordinates=test_coordinates, get_info=False)

    def f(x, h, get_info):
        return decoder(graph=x, coordinates=h, get_info=get_info)

    decoder_vmap = nnx.vmap(f, in_axes=(0, 0, None), out_axes=0)

    output_batch, _ = decoder_vmap(test_context_batch, test_coordinates_batch, False)
    assert jnp.allclose(output, output_batch[0])
