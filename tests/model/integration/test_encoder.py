# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from flax import nnx

from energnn.model.encoder import IdentityEncoder, MLPEncoder
from tests.utils import TestProblemLoader


def test_identity_encoder():
    loader = TestProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    encoder = IdentityEncoder()

    def f(encoder, graph):
        return encoder(graph=graph, get_info=False)

    encoder_vmap = nnx.jit(nnx.vmap(f, in_axes=(None, 0), out_axes=0))

    encoded_batch, _ = encoder_vmap(encoder=encoder, graph=context_batch)


def test_mlp_encoder():
    loader = TestProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    encoder = MLPEncoder(
        in_structure=loader.context_structure,
        hidden_sizes=[],
        final_activation=nnx.leaky_relu,
        out_size=4,
        seed=64,
    )

    def f(encoder, graph):
        return encoder(graph=graph, get_info=False)

    encoder_vmap = nnx.jit(nnx.vmap(f, in_axes=(None, 0), out_axes=0))

    encoded_batch, _ = encoder_vmap(encoder=encoder, graph=context_batch)
