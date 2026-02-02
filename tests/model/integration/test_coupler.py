#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import diffrax
import jax.numpy as jnp
from flax import nnx

from energnn.model.coupler import LocalSumMessageFunction, NeuralODECoupler
from ...utils import test_context, test_context_batch


def test_neural_ode_coupler():
    coupler = NeuralODECoupler(
        phi_hidden_size=[16],
        phi_activation=nnx.relu,
        phi_final_activation=nnx.tanh,
        message_functions=[
            LocalSumMessageFunction(out_size=4, hidden_size=[8], activation=nnx.relu, final_activation=nnx.tanh, seed=64),
            LocalSumMessageFunction(out_size=4, hidden_size=[8], activation=nnx.relu, final_activation=nnx.tanh, seed=65),
        ],
        latent_dimension=8,
        dt=0.1,
        stepsize_controller=diffrax.ConstantStepSize(),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        solver=diffrax.Euler(),
        max_steps=1000,
        seed=64,
    )
    output, _ = coupler(graph=test_context, get_info=False)

    def f(x, get_info):
        return coupler(graph=x, get_info=get_info)

    decoder_vmap = nnx.vmap(f, in_axes=(0, None), out_axes=0)

    output_batch, _ = decoder_vmap(test_context_batch, False)

    # Assert that the vmapped decoder output is the same as the non-batched coupled output
    assert jnp.allclose(output, output_batch[0])
