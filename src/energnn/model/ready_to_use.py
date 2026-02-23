# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from flax import nnx

from energnn.graph import GraphStructure
from energnn.model.coupler import LocalSumMessageFunction, RecurrentCoupler
from energnn.model.decoder import MLPEquivariantDecoder
from energnn.model.encoder import MLPEncoder
from energnn.model.normalizer import TDigestNormalizer
from energnn.model.simple_gnn import SimpleGNN
from energnn.model.utils import MLP


class ReadyRecurrentEquivariantGNN(SimpleGNN):

    def __init__(
        self,
        in_structure: GraphStructure,
        out_structure: GraphStructure,
        n_breakpoints: int,
        latent_dimension: int,
        hidden_sizes: list[int],
        n_steps: int = 5,
        seed: int = 0,
    ):

        rngs = nnx.Rngs(seed)

        normalizer = TDigestNormalizer(in_structure=in_structure, n_breakpoints=n_breakpoints, update_limit=1000)

        encoder = MLPEncoder(
            in_structure=in_structure,
            hidden_sizes=hidden_sizes,
            activation=nnx.leaky_relu,
            out_size=latent_dimension,
            use_bias=True,
            final_activation=None,
            rngs=rngs,
        )

        message_function = LocalSumMessageFunction(
            in_graph_structure=in_structure,
            in_array_size=latent_dimension,
            hidden_sizes=hidden_sizes,
            activation=nnx.leaky_relu,
            out_size=latent_dimension,
            use_bias=True,
            final_activation=None,
            outer_activation=nnx.tanh,
            encoded_feature_size=latent_dimension,
            rngs=rngs,
        )

        phi = MLP(
            in_size=latent_dimension,
            hidden_sizes=[],
            activation=nnx.leaky_relu,
            out_size=latent_dimension,
            use_bias=True,
            final_activation=nnx.tanh,
            rngs=rngs,
        )

        coupler = RecurrentCoupler(
            phi=phi,
            message_functions=[message_function],
            n_steps=n_steps,
        )

        decoder = MLPEquivariantDecoder(
            in_graph_structure=in_structure,
            in_array_size=latent_dimension,
            hidden_sizes=hidden_sizes,
            activation=nnx.leaky_relu,
            out_structure=out_structure,
            use_bias=True,
            final_activation=None,
            encoded_feature_size=latent_dimension,
            rngs=rngs,
        )

        super().__init__(
            normalizer=normalizer,
            encoder=encoder,
            coupler=coupler,
            decoder=decoder,
        )


class TinyRecurrentEquivariantGNN(ReadyRecurrentEquivariantGNN):

    def __init__(self, in_structure: GraphStructure, out_structure: GraphStructure, seed: int = 0):
        super().__init__(
            in_structure=in_structure,
            out_structure=out_structure,
            n_breakpoints=10,
            latent_dimension=4,
            hidden_sizes=[],
            n_steps=5,
            seed=seed,
        )


class SmallRecurrentEquivariantGNN(ReadyRecurrentEquivariantGNN):

    def __init__(self, in_structure: GraphStructure, out_structure: GraphStructure, seed: int = 0):
        super().__init__(
            in_structure=in_structure,
            out_structure=out_structure,
            n_breakpoints=20,
            latent_dimension=8,
            hidden_sizes=[16],
            n_steps=10,
            seed=seed,
        )


class MediumRecurrentEquivariantGNN(ReadyRecurrentEquivariantGNN):

    def __init__(self, in_structure: GraphStructure, out_structure: GraphStructure, seed: int = 0):
        super().__init__(
            in_structure=in_structure,
            out_structure=out_structure,
            n_breakpoints=50,
            latent_dimension=16,
            hidden_sizes=[32],
            n_steps=20,
            seed=seed,
        )


class LargeRecurrentEquivariantGNN(ReadyRecurrentEquivariantGNN):

    def __init__(self, in_structure: GraphStructure, out_structure: GraphStructure, seed: int = 0):
        super().__init__(
            in_structure=in_structure,
            out_structure=out_structure,
            n_breakpoints=100,
            latent_dimension=32,
            hidden_sizes=[64],
            n_steps=50,
            seed=seed,
        )


class ExtraLargeRecurrentEquivariantGNN(ReadyRecurrentEquivariantGNN):

    def __init__(self, in_structure: GraphStructure, out_structure: GraphStructure, seed: int = 0):
        super().__init__(
            in_structure=in_structure,
            out_structure=out_structure,
            n_breakpoints=200,
            latent_dimension=64,
            hidden_sizes=[128, 128],
            n_steps=200,
            seed=seed,
        )
