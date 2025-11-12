# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod

import flax.linen as nn
import jax.numpy as jnp
import jax.random

from energnn.graph.jax import JaxGraph

MAX_INTEGER = 2147483647


class RemoteMessageFunction(ABC):
    """
    Interface for the remote message function.

    Subclasses must implement methods to initialize weights and apply the function to a JaxGraph object.
    """

    @abstractmethod
    def init(self, *, rngs: jax.Array, context: JaxGraph, coordinates: jax.Array) -> dict:
        """
        Should return initialized the remote message function weights.

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
        Should return initialized function weights and remote message.

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
        Should return remote message.

        :param params: Parameters.
        :param context: The input graph.
        :param coordinates: Coordinates stored as JAX array.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple(remote message, info).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError


class EmptyRemoteMessageFunction(nn.Module, RemoteMessageFunction):
    """
    Empty remote message function that returns nothing.

    This class implements a placeholder remote message function that returns an empty feature array.
    """

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        n_addr = coordinates.shape[0]
        return jnp.empty(shape=(n_addr, 0)), {}


class IdentityRemoteMessageFunction(nn.Module, RemoteMessageFunction):
    r"""
    Identity remote message function module for GNN message passing.

    This module returns the node features unchanged as the remote-message.
    It implements the identity mapping on node features:
    .. math::
        h^\leadsto_a = h_a
    """

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        return coordinates, {}
