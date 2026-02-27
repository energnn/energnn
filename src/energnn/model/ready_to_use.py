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
    """
    Tiny ready-to-use equivariant GNN, with basic recurrent message passing.

    - Normalizer: TDigestNormalizer with 10 breakpoints.
    - Encoder: MLPEncoder with 0 hidden layer and output of size 4.
    - Coupler: RecurrentCoupler with 5 steps, and latent dimension 4.
    - Decoder: MLPEquivariantDecoder with 0 hidden layer.

    :param in_structure: Structure of the input graph.
    :type in_structure: GraphStructure
    :param out_structure: Structure of the output graph.
    :type out_structure: GraphStructure
    :param seed: Seed for RNG streams.
    :type seed: int
    """

    def __init__(self, *, in_structure: GraphStructure, out_structure: GraphStructure, seed: int = 0):
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
    """
    Small ready-to-use equivariant GNN, with basic recurrent message passing.

    - Normalizer: TDigestNormalizer with 20 breakpoints.
    - Encoder: MLPEncoder with 1 hidden layer of size 16 and output of size 8.
    - Coupler: RecurrentCoupler with 10 steps, hidden layers of size 16 and latent dimension 8.
    - Decoder: MLPEquivariantDecoder with 1 hidden layer of size 16.

    :param in_structure: Structure of the input graph.
    :type in_structure: GraphStructure
    :param out_structure: Structure of the output graph.
    :type out_structure: GraphStructure
    :param seed: Seed for RNG streams.
    :type seed: int
    """

    def __init__(self, *, in_structure: GraphStructure, out_structure: GraphStructure, seed: int = 0):
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
    """
    Medium ready-to-use equivariant GNN, with basic recurrent message passing.

    - Normalizer: TDigestNormalizer with 50 breakpoints.
    - Encoder: MLPEncoder with 1 hidden layer of size 32 and output of size 16.
    - Coupler: RecurrentCoupler with 20 steps, hidden layers of size 32 and latent dimension 16.
    - Decoder: MLPEquivariantDecoder with 1 hidden layer of size 32.

    :param in_structure: Structure of the input graph.
    :type in_structure: GraphStructure
    :param out_structure: Structure of the output graph.
    :type out_structure: GraphStructure
    :param seed: Seed for RNG streams.
    :type seed: int
    """

    def __init__(self, *, in_structure: GraphStructure, out_structure: GraphStructure, seed: int = 0):
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
    """
    Large ready-to-use equivariant GNN, with basic recurrent message passing.

    - Normalizer: TDigestNormalizer with 100 breakpoints.
    - Encoder: MLPEncoder with 1 hidden layer of size 64 and output of size 32.
    - Coupler: RecurrentCoupler with 50 steps, hidden layers of size 64 and latent dimension 32.
    - Decoder: MLPEquivariantDecoder with 1 hidden layer of size 64.

    :param in_structure: Structure of the input graph.
    :type in_structure: GraphStructure
    :param out_structure: Structure of the output graph.
    :type out_structure: GraphStructure
    :param seed: Seed for RNG streams.
    :type seed: int
    """

    def __init__(self, *, in_structure: GraphStructure, out_structure: GraphStructure, seed: int = 0):
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
    """
    Extra large ready-to-use equivariant GNN, with basic recurrent message passing.

    - Normalizer: TDigestNormalizer with 200 breakpoints.
    - Encoder: MLPEncoder with 2 hidden layers of size 128 and 128 and output of size 64.
    - Coupler: RecurrentCoupler with 200 steps, 2 hidden layers of size 128 and 128 and latent dimension 64.
    - Decoder: MLPEquivariantDecoder with 2 hidden layer of size 128 and 128.

    :param in_structure: Structure of the input graph.
    :type in_structure: GraphStructure
    :param out_structure: Structure of the output graph.
    :type out_structure: GraphStructure
    :param seed: Seed for RNG streams.
    :type seed: int
    """

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
