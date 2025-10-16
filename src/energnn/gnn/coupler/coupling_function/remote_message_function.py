#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from abc import ABC, abstractmethod
from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
import jax.random

from energnn.gnn.utils import MLP
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


class LinearAttentionRemoteMessageFunction(nn.Module, RemoteMessageFunction):
    r"""
    Linearized attention-based remote message function for GNNs.

    Implements the linear attention mechanism as described in (https://arxiv.org/abs/2006.16236).

    .. math::
        &q^i_a = q^i_\theta(h_a) , \\
                                            &k^i_a = k^i_\theta(h_a) ,\\
                                            &v^i_a = v^i_\theta(h_a), \\
                                            &v'^i_a = \frac{\phi(q^i_a)^T \sum_{a'\in \mathcal{A}(x)} \phi(k^i_{a'}) v^i_{a'}}
                            {\phi(q^i_a)^T \sum_{a'\in \mathcal{A}(x)} \phi(k^i_{a'})}, \\
        &h^\leadsto_a = \psi_\theta(v'^1_a, \dots, v'^n_a),

    where :math:`(q^i_\theta)_i` (query), :math:`(k^i_\theta)_i` (key), :math:`(v^i_\theta)_i` (value)
    and :math:`\psi_\theta` (outer) are trainable MLPs, and :math:`\phi` is a positive kernel function.

    :param n_heads: Number of attention heads :math:`n`.

    :param qk_size: Dimension of the query and key vectors.

    :param q_hidden_size: Hidden dimensions of the query MLPs :math:`(q^i_\theta)_i`.
    :param flax.linen.activation q_activation: Activation function for the query MLPs :math:`(q^i_\theta)_i`.

    :param k_hidden_size: Hidden dimensions of the key MLPs :math:`(k^i_\theta)_i`.
    :param flax.linen.activation k_activation: Activation function for the key MLPs :math:`(k^i_\theta)_i`.

    :param v_hidden_size: Hidden dimensions of the value MLPs :math:`(v^i_\theta)_i`.
    :param flax.linen.activation v_activation: Activation function for the value MLPs :math:`(v^i_\theta)_i`.
    :param v_out_size: Output size of the value MLPs :math:`(v^i_\theta)_i`.

    :param psi_hidden_size: Hidden dimensions of the outer MLP :math:`\psi_\theta`.
    :param flax.linen.activation psi_activation: Activation function for the outer MLP :math:`\psi_\theta`

    :param str kernel_name: Name of the kernel function (only "elu" is implemented yet).
    :param out_size: Dimension of the remote message.
    """

    out_size: int

    n_heads: int
    qk_size: int
    q_hidden_size: list[int]
    q_activation: Callable[[jax.Array], jax.Array]
    k_hidden_size: list[int]
    k_activation: Callable[[jax.Array], jax.Array]
    v_hidden_size: list[int]
    v_activation: Callable[[jax.Array], jax.Array]
    v_out_size: int
    psi_hidden_size: list[int]
    psi_activation: Callable[[jax.Array], jax.Array]
    kernel_name: str = "elu"

    def elu_kernel(self, x: jax.Array) -> jax.Array:
        r"""
        ELU-based positive kernel function: :math:`\phi(x) = elu(x) + 1`.

        :param x: Input array.
        :return: Transformed array.
        """
        return nn.elu(x) + 1

    def kernel(self, x: jax.Array) -> jax.Array:
        """
        Select and apply the positive feature kernel to x.

        :param x: Input array.
        :return: Kernel-transformed array.
        :raises NotImplementedError: If kernel_name is unsupported.
        """
        if self.kernel_name == "elu":
            return self.elu_kernel(x)
        raise NotImplementedError(f"Kernel must be in ['elu'], got {self.kernel_name}")

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """
        Compute the remote message for each node using linear attention across all nodes.

        :return: Tuple (remote_message, info_dict).
        """

        head_outputs: list[jax.Array] = []
        info: dict = {}

        for i in range(self.n_heads):

            # Project the latent coordinates to query, key and value
            query_mlp = MLP(
                hidden_size=self.q_hidden_size,
                out_size=self.qk_size,
                activation=self.q_activation,
                name=f"query-mlp-{i}",
            )
            key_mlp = MLP(
                hidden_size=self.k_hidden_size,
                out_size=self.qk_size,
                activation=self.k_activation,
                name=f"key-mlp-{i}",
            )
            value_mlp = MLP(
                hidden_size=self.v_hidden_size,
                out_size=self.v_out_size,
                activation=self.v_activation,
                name=f"value-mlp-{i}",
            )
            query = query_mlp(coordinates)
            query = query * jnp.expand_dims(context.non_fictitious_addresses, -1)
            key = key_mlp(coordinates)
            key = key * jnp.expand_dims(context.non_fictitious_addresses, -1)
            value = value_mlp(coordinates)
            value = value * jnp.expand_dims(context.non_fictitious_addresses, -1)

            # We write: attention = softmax(query @ key / sqrt(dk)) @ value
            # By linearizing the softmax (replace it with a dot product of kernel)
            # attention = query @ (key.T @ value)
            # By storing key @ value, we can reuse it for all queries
            key_kernel = self.kernel(key)
            query_kernel = self.kernel(query)

            kv = key_kernel.T @ value  # (dk, n) @ (n, dv) = (dk, dv)
            sum_k = jnp.sum(key_kernel, axis=0)  # (dk,)

            numerator = query_kernel @ kv  # (n, dk) @ (dk, dv) = (n, dv)
            denominator = query_kernel @ sum_k[..., None]  # (n, dk) @ (dk, 1) = (n, 1)

            denominator = jnp.where(denominator == 0, 1e-6, denominator)  # can be dropped because kernel > 0
            head_output = numerator / denominator

            head_outputs.append(head_output)

        # Concatenate the heads and apply a final MLP
        concatenated = jnp.concatenate(head_outputs, axis=-1)
        final_mlp = MLP(
            hidden_size=self.psi_hidden_size,
            out_size=self.out_size,
            activation=self.psi_activation,
            name="psi-mlp",
        )
        concatenated = final_mlp(concatenated)
        concatenated = concatenated * jnp.expand_dims(context.non_fictitious_addresses, -1)

        return concatenated, info
