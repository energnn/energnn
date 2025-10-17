# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from energnn.gnn.utils import MLP, gather
from energnn.graph.jax.graph import JaxEdge, JaxGraph, JaxGraphShape


class EquivariantDecoder(ABC):
    """
    Interface for equivariant decoders.

    Subclasses must implement methods to initialize weight and apply the decoder
    to a JaxGraph object.

    :param out_structure: Output structure of the decoder.
    :param out_structure: dict
    """

    out_structure: dict = field(default_factory=dict)

    def init_with_structure(self, *, rngs: jax.Array, context: JaxGraph, coordinates: jax.Array, out_structure: dict) -> dict:
        """
        Set the output structure of the decoder and return initialized decoder weights.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph.
        :param coordinates: Coordinates stored as JAX array.
        :param out_structure: Size of the output vector.
        :return: Initialized parameters.
        """
        self.out_structure = out_structure
        return self.init(rngs=rngs, context=context, coordinates=coordinates)

    @abstractmethod
    def init(self, *, rngs: jax.Array, context: JaxGraph, coordinates: jax.Array) -> dict:
        """
        Should return initialized decoder weights.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph.
        :param coordinates: Coordinates stored as JAX array.
        :return: Initialized parameters.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def init_with_output(self, *, context: JaxGraph, coordinates: jax.Array) -> tuple[JaxGraph, dict]:
        """Should return initialized decoder weights and decision graph.

        :param context: Input graph.
        :param coordinates: Coordinates stored as JAX array.
        :return: Initialized parameters and decision vector

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, params, *, context: JaxGraph, coordinates: jax.Array, get_info: bool) -> tuple[JaxGraph, dict]:
        """
        Should return initialized decision graph.

        :param params: Parameters.
        :param context: Input graph to decode.
        :param coordinates: Coordinates stored as JAX array.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple(encoded graph, info).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError


class ZeroEquivariantDecoder(nn.Module, EquivariantDecoder):
    r"""Zero equivariant decoder that returns only zeros.

    .. math::
        \forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, \hat{y}_e = [0, \dots, 0]

    :param out_structure: Output structure of the decoder.
    """

    out_structure: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[JaxGraph, dict]:
        edge_dict = {}
        for key, edge in context.edges.items():
            if key in self.out_structure:
                n_obj = edge.feature_array.shape[0]
                feature_names = self.out_structure[key]  # .unfreeze()
                feature_array = jnp.zeros([n_obj, len(feature_names)])
                edge_dict[key] = JaxEdge(
                    feature_array=feature_array,
                    feature_names=feature_names,
                    non_fictitious=edge.non_fictitious,
                    address_dict=None,
                )
        true_shape = JaxGraphShape(
            edges={key: value for key, value in context.true_shape.edges.items() if key in self.out_structure},
            addresses=jnp.array(0),
        )
        current_shape = JaxGraphShape(
            edges={key: value for key, value in context.current_shape.edges.items() if key in self.out_structure},
            addresses=jnp.array(0),
        )

        output_graph = JaxGraph(
            edges=edge_dict,
            non_fictitious_addresses=jnp.array([]),
            true_shape=true_shape,
            current_shape=current_shape,
        )
        return output_graph, {}


class MLPEquivariantDecoder(nn.Module, EquivariantDecoder):
    r"""Equivariant decoder that applies class-specific MLPs over edge features and latent coordinates.

    .. math::
        \forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, \hat{y}_e = \phi_\theta^c(x_e, h_e),

    where :math:`\phi_\theta^c` is a class specific MLP.

    :param out_structure: Output structure of the decoder.
    :param activation: Activation of the MLP :math:`\phi_\theta^c`.
    :param hidden_size: Hidden size of the MLP :math:`\phi_\theta^c`.
    :param final_kernel_zero_init: If true, initializes the last kernel to zero.
    """

    # TODO : comment s'assurer que le décodeur renvoie des 0 sur les objets fictifs ?

    activation: Callable[[jax.Array], jax.Array]
    hidden_size: list[int]
    out_structure: dict = field(default_factory=dict)
    final_kernel_zero_init: bool = False

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[JaxGraph, dict]:
        edge_dict = {}

        def get_MLP(edge_output_features, x):
            return MLP(
                hidden_size=self.hidden_size,
                out_size=len(edge_output_features),
                activation=self.activation,
                final_kernel_zero_init=self.final_kernel_zero_init,
                name=x,
            )

        mlp_dict = {key: get_MLP(value, key) for key, value in self.out_structure.items()}

        def gather_inputs(edge):
            decoder_input = []
            for _, address_array in edge.address_dict.items():
                decoder_input.append(gather(coordinates=coordinates, addresses=address_array))

                # decoder_input.append(coordinates.at[address_array.astype(int)].get(mode="drop", fill_value=0))
                # decoder_input.append(coordinates.at[address_array.astype(int)].get(mode="drop", fill_value=0))
            if edge.feature_array is not None:
                decoder_input.append(edge.feature_array)
            return jnp.concatenate(decoder_input, axis=-1)

        edge_dict = {key: value for key, value in context.edges.items() if key in self.out_structure}
        decoder_input_dict = jax.tree.map(gather_inputs, edge_dict, is_leaf=(lambda x: isinstance(x, JaxEdge)))

        def apply_mlp(edge, feature_names, decoder_input, mlp):
            decoder_output = mlp(decoder_input)
            decoder_output = decoder_output * jnp.expand_dims(edge.non_fictitious, -1)
            return JaxEdge(
                feature_array=decoder_output,
                feature_names=feature_names,
                non_fictitious=edge.non_fictitious,
                address_dict=None,
            )

        edge_dict = jax.tree.map(
            apply_mlp,
            edge_dict,
            self.out_structure.unfreeze(),
            decoder_input_dict,
            mlp_dict,
            is_leaf=(lambda x: isinstance(x, JaxEdge)),
        )

        true_shape = JaxGraphShape(
            edges={key: value for key, value in context.true_shape.edges.items() if key in self.out_structure},
            addresses=jnp.array(0),
        )
        current_shape = JaxGraphShape(
            edges={key: value for key, value in context.current_shape.edges.items() if key in self.out_structure},
            addresses=jnp.array(0),
        )

        output_graph = JaxGraph(
            edges=edge_dict,
            non_fictitious_addresses=jnp.array([]),
            true_shape=true_shape,
            current_shape=current_shape,
        )

        # output_graph = self.put_nans_graph(output_graph=output_graph)

        return output_graph, {}
