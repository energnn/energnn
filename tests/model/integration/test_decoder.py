# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import diffrax
from flax import nnx

from energnn.model import LocalSumMessageFunction, MLP, MLPEquivariantDecoder, NeuralODECoupler
from tests.utils import TestProblemLoader


def test_mlp_equivariant_decoder():
    loader = TestProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    coupler = NeuralODECoupler(
        phi=MLP(in_size=4, hidden_sizes=[], out_size=4, seed=64, final_activation=nnx.tanh),
        message_functions=[
            LocalSumMessageFunction(
                in_graph_structure=loader.context_structure,
                in_array_size=4,
                out_size=4,
                hidden_sizes=[4],
                activation=nnx.leaky_relu,
                final_activation=nnx.tanh,
                outer_activation=nnx.tanh,
                seed=64,
            )
        ],
        dt=0.25,
        stepsize_controller=diffrax.ConstantStepSize(),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        solver=diffrax.Euler(),
        max_steps=10,
    )

    decoder = MLPEquivariantDecoder(
        in_array_size=4,
        in_graph_structure=loader.context_structure,
        out_structure=loader.decision_structure,
        hidden_sizes=[4],
        activation=nnx.leaky_relu,
        seed=64,
    )

    def f(coupler, decoder, graph):
        coordinates, _ = coupler(graph=graph, get_info=False)
        return decoder(graph=graph, coordinates=coordinates, get_info=False)

    encoder_vmap = nnx.jit(nnx.vmap(f, in_axes=(None, None, 0), out_axes=0))

    encoded_batch, _ = encoder_vmap(coupler=coupler, decoder=decoder, graph=context_batch)
