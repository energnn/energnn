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

from energnn.gnn.utils import MLP
from energnn.graph.jax import JaxGraph


class SelfMessageFunction(ABC):
    """
    Interface for self message functions.

    Subclasses must implement methods to initialize weights and apply the function to a JaxGraph object.
    """

    @abstractmethod
    def init(self, *, rngs: jax.Array, context: JaxGraph, coordinates: jax.Array) -> dict:
        """
        Should return initialized function weights.

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
        Should return initialized function weights and self message.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph.
        :param coordinates: Coordinates stored as JAX array.
        :return: Initialized weights and self message.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, params, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """
        Should return self message.

        :param params: Parameters.
        :param context: The input graph.
        :param coordinates: Coordinates stored as JAX array.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple(self message, info).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError


class EmptySelfMessageFunction(nn.Module, SelfMessageFunction):
    """
    Empty self message function that returns nothing.

    This class implements a placeholder self-message function that returns an empty feature array.
    """

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        n_addr = coordinates.shape[0]
        return jnp.empty(shape=(n_addr, 0)), {}


class IdentitySelfMessageFunction(nn.Module, SelfMessageFunction):
    r"""
    Identity self-message function module for GNN message passing.

    This module returns the node features unchanged as the self-message.
    It implements the identity mapping on node features:

    .. math::
        h^\circlearrowleft_a=h_a
    """

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        return coordinates, {}


class MLPSelfMessageFunction(nn.Module, SelfMessageFunction):
    r"""
    MLP-based self-message function module for GNN message passing.

    This module applies a trainable multi-layer perceptron :math:`\psi_\theta` to each node's feature vector.
    The operation is defined as:
    .. math::
        h^\circlearrowleft_a = \psi_\theta(h_a),

    where :math:`\psi_\theta` is a trainable MLP.

    :param hidden_size: Hidden size of MLP :math:`\psi_\theta`.
    :param out_size: Output size of MLP :math:`\psi_\theta`.
    :param flax.linen.activation activation: Activation function of MLP :math:`\psi_\theta`.
    :param flax.linen.activation final_layer_activation: Activation function applied over the MLP output.
    """

    hidden_size: list[int]
    out_size: int
    activation: Callable[[jax.Array], jax.Array]
    final_layer_activation: Callable[[jax.Array], jax.Array]  # TODO: add the correct type

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        self_mlp = MLP(hidden_size=self.hidden_size, out_size=self.out_size, activation=self.activation, name="self_mlp")
        self_message = self_mlp(coordinates)
        out = self.final_layer_activation(self_message)
        out = out * jnp.expand_dims(context.non_fictitious_addresses, -1)
        return out, {}
