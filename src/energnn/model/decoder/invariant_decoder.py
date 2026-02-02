# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx

from energnn.graph.jax import JaxGraph
from energnn.model.utils import Activation, MLP
from .decoder import Decoder


class InvariantDecoder(Decoder, ABC):

    def __init__(self, *, seed: int = 0, out_size: int = 64):
        super().__init__(seed=seed)
        self.out_size = int(out_size)

    @abstractmethod
    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """
        Should return decision vector.

        :param graph: Input graph to decode.
        :param coordinates: Coordinates stored as JAX array.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple(decision vector, info).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError


class SumInvariantDecoder(InvariantDecoder):
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
    :param built: Boolean to indicate whether the decoder is built.
    :param rngs: ``nnx.Rngs`` or integer seed used to initialize MLPs.
    """

    def __init__(
        self,
        *,
        psi_hidden_size: list[int],
        psi_out_size: int,
        psi_activation: Activation,
        phi_hidden_size: list[int],
        phi_activation: Activation,
        out_size: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed, out_size=out_size)

        self.psi_hidden_size = [int(h) for h in psi_hidden_size]
        self.psi_out_size = int(psi_out_size)
        self.psi_activation = psi_activation
        self.phi_hidden_size = [int(h) for h in phi_hidden_size]
        self.phi_activation = phi_activation

        self.psi = MLP(
            hidden_size=self.psi_hidden_size,
            out_size=self.psi_out_size,
            activation=self.psi_activation,
            rngs=self.rngs,
        )
        self.phi = MLP(
            hidden_size=self.phi_hidden_size,
            out_size=self.out_size,
            activation=self.phi_activation,
            rngs=self.rngs,
        )

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        h = self.psi(coordinates)
        h = h * jnp.expand_dims(graph.non_fictitious_addresses, -1)
        h = jnp.sum(h, axis=0)
        out = self.phi(h)
        return out, {}


class MeanInvariantDecoder(InvariantDecoder):
    r"""
    Mean invariant decoder, that averages the information of all addresses.

    .. math::
        \hat{y} = \phi_\theta \left( \frac{1}{\vert \mathcal{A}(x) \vert} \sum_{a \in \mathcal{A}(x)} \psi_\theta(h_a) \right),

    where :math:`\phi_\theta` (outer) and :math:`\psi_\theta` (inner) are both trainable MLPs.

    :param psi_hidden_size: List of hidden sizes of inner MLP :math:`\psi_\theta`.
    :param psi_activation: Activation function of inner MLP :math:`\psi_\theta`.
    :param psi_out_size: Output size of inner MLP :math:`\psi_\theta`.
    :param phi_hidden_size: List of hidden sizes of outer MLP :math:`\phi_\theta`.
    :param phi_activation: Activation function of outer MLP :math:`\phi_\theta`.
    :param out_size: Output size of the decoder.
    """

    def __init__(
        self,
        *,
        psi_hidden_size: list[int],
        psi_out_size: int,
        psi_activation: Activation,
        phi_hidden_size: list[int],
        phi_activation: Activation,
        out_size: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed, out_size=out_size)

        self.psi_hidden_size = [int(h) for h in psi_hidden_size]
        self.psi_out_size = int(psi_out_size)
        self.psi_activation = psi_activation
        self.phi_hidden_size = [int(h) for h in phi_hidden_size]
        self.phi_activation = phi_activation

        self.psi = MLP(
            hidden_size=self.psi_hidden_size,
            out_size=self.psi_out_size,
            activation=self.psi_activation,
            rngs=self.rngs,
        )
        self.phi = MLP(
            hidden_size=self.phi_hidden_size,
            out_size=self.out_size,
            activation=self.phi_activation,
            rngs=self.rngs,
        )

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        numerator = self.psi(coordinates)
        numerator = numerator * jnp.expand_dims(graph.non_fictitious_addresses, -1)
        numerator = jnp.sum(numerator, axis=0)
        denominator = jnp.sum(graph.non_fictitious_addresses, axis=0) + 1e-9
        return self.phi(numerator / denominator), {}


class AttentionInvariantDecoder(InvariantDecoder):
    r"""Attention invariant decoder, that weights addresses contribution with an attention mechanism.

    .. math::
        &v_a^i = v^i_\theta(h_a) \\
        &s_a^i = s^i_\theta(h_a) \\
        &\alpha^i_a = \frac{\exp(s_a^i)}{ \sum_{a' \in \mathcal{A}(x) } \exp(s^i_{a'}) } \\
        &{v'}^i = \sum_{a \in \mathcal{A}(x)} \alpha_a^i v_a^i \\
        &\hat{y} = \psi_\theta({v'}^1, \dots, {v'}^n)

    where :math:`(v^i_\theta)_i` (value), :math:`(s^i_\theta)_i` (score) and :math:`\psi_\theta` (outer) are trainable MLPs.


    :param n: Number of attention heads.
    :param v_hidden_size: List of hidden sizes of MLPs :math:`(v_\theta)_i`.
    :param v_activation: Activation function of value MLPs :math:`(v_\theta)_i`.
    :param v_out_size: Output size of value MLPs :math:`(v_\theta)_i`.
    :param s_hidden_size: List of hidden sizes of score MLP :math:`(s^i_\theta)_i`.
    :param s_activation: Activation function of score MLP :math:`(s^i_\theta)_i`.
    :param psi_hidden_size: List of hidden sizes of outer MLP :math:`\psi_\theta`.
    :param psi_activation: Activation function of outer MLP :math:`\phi_\theta`.
    :param out_size: Output size of the decoder.
    """

    def __init__(
        self,
        *,
        v_hidden_size: list[int],
        v_activation: Activation,
        v_out_size: int,
        s_hidden_size: list[int],
        s_activation: Activation,
        psi_hidden_size: list[int],
        psi_activation: Activation,
        out_size: int = 0,
        n: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed, out_size=out_size)

        self.v_hidden_size = [int(h) for h in v_hidden_size]
        self.v_activation = v_activation
        self.v_out_size = int(v_out_size)
        self.s_hidden_size = [int(h) for h in s_hidden_size]
        self.s_activation = s_activation
        self.psi_hidden_size = [int(h) for h in psi_hidden_size]
        self.psi_activation = psi_activation
        self.out_size = int(out_size)
        self.n = int(n)

        self.v_mlps = nnx.List(
            [
                MLP(hidden_size=self.v_hidden_size, out_size=self.v_out_size, activation=self.v_activation, rngs=self.rngs)
                for i in range(self.n)
            ]
        )

        self.s_mlps = nnx.List(
            [
                MLP(hidden_size=self.s_hidden_size, out_size=1, activation=self.s_activation, rngs=self.rngs)
                for i in range(self.n)
            ]
        )

        self.psi = MLP(
            hidden_size=self.psi_hidden_size,
            out_size=self.out_size,
            activation=self.psi_activation,
            rngs=self.rngs,
        )

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        value_list: list[jax.Array] = []
        mask = jnp.expand_dims(graph.non_fictitious_addresses, -1)

        for i in range(self.n):
            v = self.v_mlps[i](coordinates)
            s = self.s_mlps[i](coordinates)

            numerator = v * jnp.exp(s)
            numerator = numerator * mask
            numerator = jnp.sum(numerator, axis=0)

            denominator = jnp.exp(s) * mask
            denominator = jnp.sum(denominator, axis=0) + 1e-9

            value_list.append(numerator / denominator)

        value_vec = jnp.concatenate(value_list, axis=0)
        out = self.psi(value_vec)
        return out, {}
