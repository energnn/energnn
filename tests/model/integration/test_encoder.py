#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import jax.numpy as jnp
from flax import nnx

from energnn.model.encoder import IdentityEncoder, MLPEncoder
from ...utils import test_context, test_context_batch


def test_identity_encoder():
    encoder = IdentityEncoder()
    output, _ = encoder(graph=test_context, get_info=False)
    assert jnp.allclose(output.feature_flat_array, test_context.feature_flat_array)

    def f(x, get_info):
        return encoder(graph=x, get_info=get_info)

    encoder_vmap = nnx.vmap(f, in_axes=(0, None), out_axes=0)
    output_batch, _ = encoder_vmap(test_context_batch, False)
    assert jnp.allclose(output_batch.feature_flat_array, test_context_batch.feature_flat_array)

    assert jnp.allclose(output.feature_flat_array, output_batch.feature_flat_array[0])


def test_mlp_encoder():
    encoder = MLPEncoder(
        hidden_size=[16],
        out_size=8,
        activation=nnx.relu,
        seed=64,
    )
    output, _ = encoder(graph=test_context, get_info=False)

    def f(x, get_info):
        return encoder(graph=x, get_info=get_info)

    encoder_vmap = nnx.vmap(f, in_axes=(0, None), out_axes=0)
    output_batch, _ = encoder_vmap(test_context_batch, False)

    # Assert that the vmapped decoder output is the same as the non-batched decoder output
    assert jnp.allclose(output.feature_flat_array, output_batch.feature_flat_array[0])
