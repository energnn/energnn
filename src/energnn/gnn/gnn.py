# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import jax.random

from energnn.gnn.coupler import Coupler
from energnn.gnn.decoder import EquivariantDecoder, InvariantDecoder
from energnn.gnn.encoder import Encoder
from energnn.graph.jax import JaxGraph

ENCODER = "encoder"
COUPLER = "coupler"
DECODER = "decoder"


class EquivariantGNN:
    r"""
    Equivariant Graph Neural Network, that maps context graphs :math:`x` to decisions graphs :math:`\hat{y}`.

    This GNN pipeline consists of three stages:
    1. **Encoding**: Embed address and edge features into a latent space via `Encoder`.
    2. **Coupling**: Associate graph addresses with continuous coordinates via `Coupler`.
    3. **Decoding**: Produce an output graph structure using `EquivariantDecoder`.

    :param Encoder encoder: Embeds input features into a latent space.
    :param Coupler coupler: Associates addresses with latent coordinates.
    :param EquivariantDecoder decoder: Decodes address coordinates into a meaningful output.
    """

    encoder: Encoder
    coupler: Coupler
    decoder: EquivariantDecoder

    def __init__(self, *, encoder: Encoder, coupler: Coupler, decoder: EquivariantDecoder):
        """Construct an EquivariantGNN instance with specified encoder, coupler, and decoder."""
        self.encoder = encoder
        self.coupler = coupler
        self.decoder = decoder

    def init(self, *, rngs: jax.Array, context: JaxGraph, out_structure: dict[str, int]) -> dict:
        """
        Initializes the weights of the encoder, coupler and decoder based on the `context` and `decision_structure`.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph context.
        :param out_structure: Desired structure of the output graph.
        :return: Dictionary of initialized parameters for 'encoder', 'coupler', and 'decoder'.
        """
        params = {}
        rng_e, rng_c, rng_d = jax.random.split(rngs, num=3)
        (context, info), params[ENCODER] = self.encoder.init_with_output(rngs=rng_e, context=context)
        (coordinates, info), params[COUPLER] = self.coupler.init_with_output(rngs=rng_c, context=context)
        params[DECODER] = self.decoder.init_with_structure(
            rngs=rng_d, context=context, coordinates=coordinates, out_structure=out_structure
        )
        return params

    def apply(self, params: dict, *, context: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Applies the encoder, the coupler and the decoder to the context graph :math:`x`.

        :param params: Parameter dictionary.
        :param context: Input graph context.
        :param get_info: Flag to return intermediate diagnostics.
        :return: Tuple (`output_graph`, `info`) where:
                 - output_graph: The output or decision graph.
                 - info: Nested dict of intermediate info from each stage under keys 'encoder', 'coupler', 'decoder'.
        """
        info = {}
        context, info["encoder"] = self.encoder.apply(params[ENCODER], context=context, get_info=get_info)
        coordinates, info["coupler"] = self.coupler.apply(params[COUPLER], context=context, get_info=get_info)
        output_graph, info["decoder"] = self.decoder.apply(
            params[DECODER], context=context, coordinates=coordinates, get_info=get_info
        )
        return output_graph, info


class InvariantGNN:
    r"""Invariant Graph Neural Network, that maps context graphs :math:`x` to decision vector :math:`\hat{y}`.

    This GNN pipeline consists of three stages:
    1. **Encoding**: Embed address and edge features into a latent space via `Encoder`.
    2. **Coupling**: Associate graph addresses with continuous coordinates via `Coupler`.
    3. **Decoding**: Produce an output graph structure using `InvariantDecoder`.

    :param Encoder encoder: Embeds input features into a latent space.
    :param Coupler coupler: Associates addresses with latent coordinates.
    :param InvariantDecoder decoder: Decodes address coordinates into a meaningful output.
    """

    encoder: Encoder
    coupler: Coupler
    decoder: InvariantDecoder

    def __init__(self, *, encoder: Encoder, coupler: Coupler, decoder: InvariantDecoder):
        """Construct an InvariantGNN instance with specified encoder, coupler, and decoder."""
        self.encoder = encoder
        self.coupler = coupler
        self.decoder = decoder

    def init(self, *, rngs: jax.Array, context: JaxGraph, out_size: int) -> dict:
        """
        Initializes the weights of the encoder, coupler and decoder based on the `context` and `decision_structure`.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph context.
        :param out_size: Size of the output decision vector.
        :return: Dictionary of initialized parameters for 'encoder', 'coupler', and 'decoder'.
        """
        params = {}
        rng_e, rng_c, rng_d = jax.random.split(rngs, num=3)
        (context, info), params[ENCODER] = self.encoder.init_with_output(rngs=rng_e, context=context)
        (coordinates, info), params[COUPLER] = self.coupler.init_with_output(rngs=rng_c, context=context)
        params[DECODER] = self.decoder.init_with_size(rngs=rng_d, context=context, coordinates=coordinates, out_size=out_size)
        return params

    def apply(self, params: dict, *, context: JaxGraph, get_info: bool = False) -> tuple[jax.Array, dict]:
        """
        Applies the encoder, the coupler and the decoder to the context graph :math:`x`.

        :param params: Parameter dictionary.
        :param context: Input graph context.
        :param get_info: Flag to return intermediate diagnostics.
        :return: Tuple (`output_vector`, `info`) where:
                 - output_vector: JAX array of shape `(out_size,)`.
                 - info: Nested dict of info under keys 'encoder', 'coupler', 'decoder'.
        """
        info = {}
        context, info["encoder"] = self.encoder.apply(params[ENCODER], context=context, get_info=get_info)
        coordinates, info["coupler"] = self.coupler.apply(params[COUPLER], context=context, get_info=get_info)
        output_graph, info["decoder"] = self.decoder.apply(
            params[DECODER], context=context, coordinates=coordinates, get_info=get_info
        )
        return output_graph, info
