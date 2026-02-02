# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from energnn.graph.jax.graph import JaxEdge, JaxGraph, JaxGraphShape
from energnn.model.utils import MLP, gather
from .decoder import Decoder


class EquivariantDecoder(Decoder, ABC):

    def __init__(self, *, out_structure: dict, seed: int = 0):
        super().__init__(seed=seed)
        self.out_structure = nnx.data(out_structure)


class MLPEquivariantDecoder(EquivariantDecoder):
    r"""Equivariant decoder that applies class-specific MLPs over edge features and latent coordinates.

    .. math::
        \forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, \hat{y}_e = \phi_\theta^c(x_e, h_e),

    where :math:`\phi_\theta^c` is a class specific MLP.

    :param out_structure: Output structure of the decoder.
    :param activation: Activation of the MLP :math:`\phi_\theta^c`.
    :param hidden_size: Hidden size of the MLP :math:`\phi_\theta^c`.
    """

    def __init__(self, *, out_structure: dict, activation: Callable, hidden_size: list[int], seed: int = 0):
        super().__init__(seed=seed, out_structure=out_structure)
        self.activation = activation
        self.hidden_size = hidden_size
        self.mlp_dict: dict = {}

    def _build_missing_mlps(self) -> None:
        """Creates an MLP for each edge class appearing in the graph."""
        for k, d in self.out_structure.items():
            if k not in self.mlp_dict:
                self.mlp_dict[k] = MLP(
                    hidden_size=self.hidden_size, out_size=len(d), activation=self.activation, rngs=self.rngs
                )
        return None

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[JaxGraph, dict]:

        self._build_missing_mlps()

        def apply_over_edge(edge_mlp_names):
            edge, mlp, feature_names = edge_mlp_names

            decoder_input = []
            for _, address_array in edge.address_dict.items():
                decoder_input.append(gather(coordinates=coordinates, addresses=address_array))
            if edge.feature_array is not None:
                decoder_input.append(edge.feature_array)
            decoder_input = jnp.concatenate(decoder_input, axis=-1)
            decoder_output = mlp(decoder_input)
            decoder_output = decoder_output * jnp.expand_dims(edge.non_fictitious, -1)
            return JaxEdge(
                feature_array=decoder_output,
                feature_names=feature_names,
                non_fictitious=edge.non_fictitious,
                address_dict=None,
            )

        edge_mlp_names_dict = {
            k: (edge, self.mlp_dict[k], self.out_structure[k]) for k, edge in graph.edges.items() if k in self.out_structure
        }
        edge_dict = jax.tree.map(apply_over_edge, edge_mlp_names_dict, is_leaf=(lambda x: isinstance(x, tuple)))
        true_shape = JaxGraphShape(
            edges={key: value for key, value in graph.true_shape.edges.items() if key in self.out_structure},
            addresses=jnp.array(0),
        )
        current_shape = JaxGraphShape(
            edges={key: value for key, value in graph.current_shape.edges.items() if key in self.out_structure},
            addresses=jnp.array(0),
        )

        output_graph = JaxGraph(
            edges=edge_dict,
            non_fictitious_addresses=jnp.array([]),
            true_shape=true_shape,
            current_shape=current_shape,
        )

        return output_graph, {}
