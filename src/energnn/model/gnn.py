# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import jax
from flax import nnx

from energnn.graph import Graph
from .coupler import Coupler
from .decoder import Decoder
from .encoder import Encoder
from .normalizer import Normalizer


class GNN(nnx.Module):
    """
    Simple Graph Neural Network (GNN) model designed to handle Hyper Heterogeneous Multi Graphs (H2MGs).

    The model consists of a normalization step, an encoding step, a coupling step, and a decoding step.
    The decoder can either be invariant or equivariant, depending on the task requirements.

    :param normalizer: Maps the input features to a learning-compatible range.
    :type normalizer: Normalizer
    :param encoder: Embeds hyper-edge set features into a latent space.
    :type encoder: Encoder
    :param coupler: Outputs latent coordinates for each address present in the input graph.
    :type coupler: Coupler
    :param decoder: Maps latent coordinates and encoded graph to a meaningful output.
    :type decoder: Decoder
    """

    def __init__(self, normalizer: Normalizer, encoder: Encoder, coupler: Coupler, decoder: Decoder):
        self.normalizer = normalizer
        self.encoder = encoder
        self.coupler = coupler
        self.decoder = decoder

    def __call__(self, graph: Graph, step_with_metrics: bool = False) -> tuple[Graph | jax.Array, dict]:
        """
        Processes a given graph through a sequence of steps: normalization, encoding, coupling,
        and decoding. The method applies a series of transformations to the input graph and
        returns a decoded graph / array along with optional processing information.

        :param graph: The input graph to be processed.
        :param step_with_metrics: Whether this step collects metrics. Forwarded unchanged to every block; each block
            returns metrics only on such steps and only if it was built with `return_metrics=True`. Defaults to False.
        :return: A tuple consisting of the processed decoded graph / array and a dictionary of metrics per block.
        """
        metrics = {}
        normalized_graph, metrics["normalization"] = self.normalizer(graph=graph, step_with_metrics=step_with_metrics)
        encoded_graph, metrics["encoding"] = self.encoder(graph=normalized_graph, step_with_metrics=step_with_metrics)
        latent_coordinates, metrics["coupling"] = self.coupler(graph=encoded_graph, step_with_metrics=step_with_metrics)
        output, metrics["decoding"] = self.decoder(
            coordinates=latent_coordinates, graph=encoded_graph, step_with_metrics=step_with_metrics
        )
        return output, metrics

    def forward_batch(self, *, graph: Graph, step_with_metrics: bool = False) -> tuple[Graph | jax.Array, dict]:
        """Applies the model to a batch of graphs.

        Only the encoder, coupler, and decoder modules are vmapped, while the normalization module is not.

        :param graph: Batch of input graphs.
        :param step_with_metrics: Whether this step collects metrics, forwarded unchanged to every block.
        """

        def apply_core(encoder, coupler, decoder, graph, step_with_metrics):
            metrics = {}
            encoded_graph, metrics["encoding"] = encoder(graph=graph, step_with_metrics=step_with_metrics)
            latent_coordinates, metrics["coupling"] = coupler(graph=encoded_graph, step_with_metrics=step_with_metrics)
            output, metrics["decoding"] = decoder(
                coordinates=latent_coordinates, graph=encoded_graph, step_with_metrics=step_with_metrics
            )
            return output, metrics

        normalized_graph, metrics_norm = self.normalizer(graph=graph, step_with_metrics=step_with_metrics)
        output, metrics_core = jax.vmap(apply_core, in_axes=[None, None, None, 0, None], out_axes=0)(
            self.encoder, self.coupler, self.decoder, normalized_graph, step_with_metrics
        )
        return output, metrics_norm | metrics_core
