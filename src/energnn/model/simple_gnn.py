# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#

import jax
from flax import nnx

from energnn.graph import JaxGraph
from .coupler import Coupler
from .decoder import Decoder
from .encoder import Encoder
from .normalizer import Normalizer


class SimpleGNN(nnx.Module):
    """
    Simple Graph Neural Network (GNN) model designed to handle Hyper Heterogeneous Multi Graphs (H2MGs).

    The model consists of a normalization step, an encoding step, a coupling step, and a decoding step.
    The decoder can either be invariant or equivariant, depending on the task requirements.

    :param normalizer: Normalization module that maps the input feature distribution to a learning-compatible distribution.
    :param encoder: Encoder module that produces a graph where hyper-edge features are embedded into a latent space.
    :param coupler: Coupler module that outputs latent coordinates for each address present in the input graph.
    :param decoder: Decoder module that maps latent coordinates and encoded graph to a meaningful output.
    """

    def __init__(self, normalizer: Normalizer, encoder: Encoder, coupler: Coupler, decoder: Decoder):
        self.normalizer = normalizer
        self.encoder = encoder
        self.coupler = coupler
        self.decoder = decoder

    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph | jax.Array, dict]:
        """
        Processes a given graph through a sequence of steps: normalization, encoding, coupling,
        and decoding. The method applies a series of transformations to the input graph and
        returns a decoded graph / array along with optional processing information.

        :param graph: The input graph to be processed.
        :param get_info: A boolean indicating whether detailed processing information should
            be returned. Defaults to False.
        :return: A tuple consisting of the processed decoded graph / array and an optional dictionary
            with detailed information about each processing step if `get_info` is True.
        """
        info = {}
        normalized_graph, info["normalization"] = self.normalizer(graph=graph, get_info=get_info)
        encoded_graph, info["encoding"] = self.encoder(graph=normalized_graph, get_info=get_info)
        latent_coordinates, info["coupling"] = self.coupler(graph=encoded_graph, get_info=get_info)
        output, info["decoding"] = self.decoder(coordinates=latent_coordinates, graph=encoded_graph, get_info=get_info)
        return output, info

    def forward_batch(self, *, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph | jax.Array, dict]:
        """Applies the model to a batch of graphs.

        Only the encoder, coupler, and decoder modules are vmapped, while the normalization module is not.

        :param graph: Batch of input graphs.
        :param get_info: Whether to return additional information about the processing steps.
        """

        def apply_core(encoder, coupler, decoder, graph, get_info):
            info = {}
            encoded_graph, info["encoding"] = encoder(graph=graph, get_info=get_info)
            latent_coordinates, info["coupling"] = coupler(graph=encoded_graph, get_info=get_info)
            output, info["decoding"] = decoder(coordinates=latent_coordinates, graph=encoded_graph, get_info=get_info)
            return output, info

        normalized_graph, info_norm = self.normalizer(graph=graph, get_info=get_info)
        output, info_core = jax.vmap(apply_core, in_axes=[None, None, None, 0, None], out_axes=0)(
            self.encoder, self.coupler, self.decoder, normalized_graph, get_info
        )
        return output, info_norm | info_core
