#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from abc import ABC, abstractmethod
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from energnn.gnn.utils import MLP
from energnn.graph.jax import JaxGraph


class InvariantDecoder(ABC):
    """
    Interface for invariant decoders.

    Subclasses must implement methods to initialize parameters and apply the decoder
    to a JaxGraph object

    :param out_size: Size of the output vector.
    """

    out_size: int = 0

    def init_with_size(self, *, rngs: jax.Array, context: JaxGraph, coordinates: jax.Array, out_size: int):
        """
        Set the size of the decoder output and return initialized decoder weights.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph.
        :param coordinates: Coordinates stored as JAX array.
        :param out_size: Size of the output vector.
        :return: Initialized parameters.
        """
        self.out_size = out_size
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
    def init_with_output(self, *, context: JaxGraph, coordinates: jax.Array) -> tuple[jax.Array, dict]:
        """
        Should return initialized decoder weights and decision vector.

        :param context: Input graph.
        :param coordinates: Coordinates stored as JAX array.
        :return: Initialized parameters and decision vector

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, params, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """
        Should return decision vector.

        :param params: Parameters.
        :param context: Input graph to decode.
        :param coordinates: Coordinates stored as JAX array.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple(decision vector, info).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError


class ZeroInvariantDecoder(nn.Module, InvariantDecoder):
    r"""
    Zero invariant decoder that returns a vector of zeros.

    .. math::
        \hat{y} = [0, \dots, 0]

    :param out_size: Size of the output vector.
    """

    out_size: int = 0

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        return jnp.zeros([self.out_size]), {}


class SumInvariantDecoder(nn.Module, InvariantDecoder):
    r"""
    Sum invariant decoder, that sums the information of all addresses.

    .. math::
        \hat{y} = \phi_\theta \left( \sum_{a \in \mathcal{A}(x)} \psi_\theta(h_a)\right),

    where :math:`\phi_\theta` (outer) and :math:`\psi_\theta` (inner) are both trainable MLPs.

    :param psi_hidden_size: List of hidden sizes of inner MLP :math:`\psi_\theta`.
    :param psi_out_size: Output size of inner MLP :math:`\psi_\theta`.
    :param psi_activation: Activation function of inner MLP :math:`\psi_\theta`.
    :param phi_hidden_size: List of hidden sizes of outer MLP :math:`\phi_\theta`.
    :param phi_activation: Activation function of outer MLP :math:`\phi_\theta`.
    :param out_size: Output size of the decoder.
    """

    psi_hidden_size: list[int]
    psi_out_size: int
    psi_activation: Callable[[jax.Array], jax.Array]
    phi_hidden_size: list[int]
    phi_activation: Callable[[jax.Array], jax.Array]
    out_size: int = 0

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        psi = MLP(hidden_size=self.psi_hidden_size, out_size=self.psi_out_size, activation=self.psi_activation, name="psi")
        phi = MLP(hidden_size=self.phi_hidden_size, out_size=self.out_size, activation=self.phi_activation, name="phi")

        h = psi(coordinates)
        h = h * jnp.expand_dims(context.non_fictitious_addresses, -1)
        h = jnp.sum(h, axis=0)

        return phi(h), {}


class MeanInvariantDecoder(nn.Module, InvariantDecoder):
    r"""
    Mean invariant decoder, that averages the information of all addresses.

    .. math::
        \hat{y} = \phi_\theta \left( \frac{1}{\vert \mathcal{A}(x) \vert} \sum_{a \in \mathcal{A}(x)} \psi_\theta(h_a) \right),

    where :math:`\phi_\theta` (outer) and :math:`\psi_\theta` (inner) are both trainable MLPs.

    :param psi_hidden_size: List of hidden sizes of inner MLP :math:`\psi_\theta`.
    :param flax.linen.activation psi_activation: Activation function of inner MLP :math:`\psi_\theta`.
    :param psi_out_size: Output size of inner MLP :math:`\psi_\theta`.
    :param phi_hidden_size: List of hidden sizes of outer MLP :math:`\phi_\theta`.
    :param flax.linen.activation phi_activation: Activation function of outer MLP :math:`\phi_\theta`.
    :param out_size: Output size of the decoder.
    """

    psi_hidden_size: list[int]
    psi_out_size: int
    psi_activation: Callable[[jax.Array], jax.Array]
    phi_hidden_size: list[int]
    phi_activation: Callable[[jax.Array], jax.Array]
    out_size: int = 0

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        psi = MLP(hidden_size=self.psi_hidden_size, out_size=self.psi_out_size, activation=self.psi_activation, name="psi")
        phi = MLP(hidden_size=self.phi_hidden_size, out_size=self.out_size, activation=self.phi_activation, name="phi")

        numerator = psi(coordinates)
        numerator = numerator * jnp.expand_dims(context.non_fictitious_addresses, -1)
        numerator = jnp.sum(numerator, axis=0)

        denominator = jnp.sum(numerator * 0 + 1, axis=0) + 1e-9

        return phi(numerator / denominator) * jnp.expand_dims(context.non_fictitious_addresses, -1), {}


class AttentionInvariantDecoder(nn.Module, InvariantDecoder):
    r"""Attention invariant decoder, that weights addresses contribution with an attention mechanism.

    .. math::
        &v_a^i = v^i_\theta(h_a) \\
        &s_a^i = s^i_\theta(h_a) \\
        &\alpha^i_a = \frac{\exp(s_a^i)}{ \sum_{a' \in \mathcal{A}(x) } \exp(s^i_{a'}) } \\
        &{v'}_a^i = \sum_{a' \in \mathcal{A}(x)} \alpha_a^i v_a^i \\
        &\hat{y} = \psi_\theta({v'}_a^1, \dots, {v'}_a^n)

    where :math:`(v^i_\theta)_i` (value), :math:`(s^i_\theta)_i` (score) and :math:`\psi_\theta` (outer) are trainable MLPs.


    :param n: Number of attention heads.
    :param v_hidden_size: List of hidden sizes of MLPs :math:`(v_\theta)_i`.
    :param flax.linen.activation v_activation: Activation function of value MLPs :math:`(v_\theta)_i`.
    :param v_out_size: Output size of value MLPs :math:`(v_\theta)_i`.
    :param s_hidden_size: List of hidden sizes of score MLP :math:`(s^i_\theta)_i`.
    :param flax.linen.activation s_activation: Activation function of score MLP :math:`(s^i_\theta)_i`.
    :param psi_hidden_size: List of hidden sizes of outer MLP :math:`\psi_\theta`.
    :param flax.linen.activation psi_activation: Activation function of outer MLP :math:`\phi_\theta`.
    :param out_size: Output size of the decoder.
    """

    v_hidden_size: list[int]
    v_activation: Callable[[jax.Array], jax.Array]
    v_out_size: int
    s_hidden_size: list[int]
    s_activation: Callable[[jax.Array], jax.Array]
    psi_hidden_size: list[int]
    psi_activation: Callable[[jax.Array], jax.Array]
    out_size: int = 0
    n: int = 1

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        value_list = []
        for i in range(self.n):

            v_mlp = MLP(
                hidden_size=self.v_hidden_size,
                out_size=self.v_out_size,
                activation=self.v_activation,
                name="value-mlp-{}".format(i),
            )
            s_mlp = MLP(
                hidden_size=self.s_hidden_size, out_size=1, activation=self.s_activation, name="score-mlp-{}".format(i)
            )

            v = v_mlp(coordinates)
            s = s_mlp(coordinates)

            numerator = v * jnp.exp(s)
            numerator = numerator * jnp.expand_dims(context.non_fictitious_addresses, -1)
            numerator = jnp.sum(numerator, axis=0)

            denominator = jnp.exp(s)
            denominator = denominator * jnp.expand_dims(context.non_fictitious_addresses, -1)
            denominator = jnp.sum(denominator, axis=0) + 1e-9

            value_list.append(numerator / denominator)

        value_vec = jnp.concatenate(value_list, axis=0)
        psi_mlp = MLP(hidden_size=self.psi_hidden_size, out_size=self.out_size, activation=self.psi_activation, name="psi-mlp")
        return psi_mlp(value_vec), {}
