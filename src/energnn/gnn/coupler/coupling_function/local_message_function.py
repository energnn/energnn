# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod
from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
import jax.random

from energnn.gnn.utils import MLP, gather, scatter_add
from energnn.graph.jax import JaxGraph

MAX_INTEGER = 2147483647


class LocalMessageFunction(ABC):
    """
    Interface for the local message function.

    Subclasses must implement methods to initialize weights and apply the function to a JaxGraph object.
    """

    @abstractmethod
    def init(self, *, rngs: jax.Array, context: JaxGraph, coordinates: jax.Array) -> dict:
        """
        Should return initialized the local message function weights.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph for applying the function
        :param coordinates: Coordinates stored as JAX array.
        :return: Initialized weights.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def init_with_output(self, *, rngs: jax.Array, context: JaxGraph, coordinates: jax.Array) -> tuple[jax.Array, dict]:
        """
        Should return initialized function weights and local message.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph.
        :param coordinates: Coordinates stored as JAX array.
        :return: Initialized weights and self message.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, params: dict, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False
    ) -> tuple[jax.Array, dict]:
        """
        Should return local message.

        :param params: Parameters.
        :param context: The input graph.
        :param coordinates: Coordinates stored as JAX array.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple(local message, info).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError


class EmptyLocalMessageFunction(nn.Module, LocalMessageFunction):
    r"""
    Empty Local Message Function that returns nothing.

    This class implements a placeholder local message function that returns an empty feature array.
    """

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        n_addr = coordinates.shape[0]
        return jnp.empty(shape=(n_addr, 0)), {}


class IdentityLocalMessageFunction(nn.Module, LocalMessageFunction):
    r"""
    Identity local message function module for GNN message passing.

    This module returns the node features unchanged as the local message.
    It implements the identity mapping on node features:

    .. math::
        h^\rightarrow_a = h_a
    """

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        return coordinates, {}


class SumLocalMessageFunction(nn.Module, LocalMessageFunction):
    r"""
    Local sum-based message function module for GNN message passing.

    This module aggregates messages from each node's local neighborhood by applying
    a class- and port-specific MLP :math:`\xi^{c,o}_\theta` to edge features and neighbor coordinates,
    summing the results across all incoming ports, and applying a final activation :math:`\sigma`.

    The operation is defined as:

    .. math::
        h^\rightarrow_a = \sigma \left( \sum_{(c,e,o)\in \mathcal{N}_a(x)} \xi^{c,o}_\theta(h_e, x_e)\right),

    where :math:`\xi^{c,o}_\theta` is a class-specific and port-specific MLP, and :math:`\sigma` is an
    element-wise activation function.

    :param list[int] hidden_size: Hidden size of the MLPs :math:`\xi^{c,o}_\theta`.
    :param flax.linen.activation activation: Activation function for the MLPs :math:`\xi^{c,o}_\theta`.
    :param int out_size: Local message size.
    :param flax.linen.activation final_activation: Activation function :math:`\sigma` applied over the output.
    """

    out_size: int
    hidden_size: list[int]
    activation: Callable[[jax.Array], jax.Array]
    final_activation: Callable[[jax.Array], jax.Array]

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        neighbour_message = jnp.zeros((coordinates.shape[0], self.out_size))

        def get_mlps(edge_key, address_key):
            return MLP(
                hidden_size=self.hidden_size,
                out_size=self.out_size,
                activation=self.activation,
                name=f"{edge_key}-{address_key}-local_message_mlp",
            )

        mlp_tree = {
            key: {address_key: get_mlps(key, address_key) for address_key in edge.address_dict.keys()}
            for key, edge in context.edges.items()
        }

        def get_messages(edge, mlp_subtree):
            edge_message_input = []

            if edge.feature_names is not None:
                edge_message_input.append(edge.feature_array)

            for address_key, address_array in edge.address_dict.items():
                edge_message_input.append(gather(coordinates=coordinates, addresses=address_array))

            edge_message_input = jnp.concatenate(edge_message_input, axis=-1)

            messages_subtree = {}

            for address_key, address_array in edge.address_dict.items():
                messages_subtree[address_key] = mlp_subtree[address_key](edge_message_input)
                messages_subtree[address_key] = messages_subtree[address_key] * jnp.expand_dims(edge.non_fictitious, -1)

            return messages_subtree

        def sum_messages_for_addresses(latent_coordinates, edge_mlp):
            edge, mlp_subtree = edge_mlp
            message_subtree = get_messages(edge, mlp_subtree)
            for address_key, edge_message in message_subtree.items():
                latent_coordinates = scatter_add(
                    accumulator=latent_coordinates, increment=edge_message, addresses=edge.address_dict[address_key]
                )

            return latent_coordinates

        edge_mlp_tree = {edge_key: (edge, mlp_tree[edge_key]) for edge_key, edge in context.edges.items()}
        neighbour_message = jax.tree.reduce(
            sum_messages_for_addresses, edge_mlp_tree, initializer=neighbour_message, is_leaf=lambda x: isinstance(x, tuple)
        )

        return self.final_activation(neighbour_message), {}
