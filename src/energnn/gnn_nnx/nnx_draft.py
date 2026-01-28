# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from energnn.gnn_nnx.utils import MLP
from energnn.graph.jax import JaxGraph


Activation = Callable[[jax.Array], jax.Array]


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


class ZeroInvariantDecoder(nnx.Module, InvariantDecoder):
    r"""
    Zero invariant decoder that returns a vector of zeros.

    .. math::
        \hat{y} = [0, \dots, 0]

    :param out_size: Size of the output vector.
    """

    out_size: int = 0

    def __init__(self, *, out_size: int = 0) -> None:
        self.out_size = int(out_size)

    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        return jnp.zeros([self.out_size]), {}


class SumInvariantDecoder(nnx.Module, InvariantDecoder):
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
    :param rngs: nnx.Rngs ou int seed pour initialisation des MLPs.
    """

    def __init__(
        self,
        psi_hidden_size: list[int],
        psi_out_size: int,
        psi_activation: Activation,
        phi_hidden_size: list[int],
        phi_activation: Activation,
        *,
        out_size: int = 0,
        rngs: nnx.Rngs | int | None = None,
        built: bool = False,
    ) -> None:
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        self.psi_hidden_size = [int(h) for h in psi_hidden_size]
        self.psi_out_size = int(psi_out_size)
        self.psi_activation = psi_activation
        self.phi_hidden_size = [int(h) for h in phi_hidden_size]
        self.phi_activation = phi_activation
        self.out_size = int(out_size)
        self.rngs = rngs
        self.built = built

        self.psi = MLP(
            hidden_size=self.psi_hidden_size,
            out_size=self.psi_out_size,
            activation=self.psi_activation,
            rngs=self.rngs,
            name="psi",
        )
        self.phi = MLP(
            hidden_size=self.phi_hidden_size,
            out_size=self.out_size,
            activation=self.phi_activation,
            rngs=self.rngs,
            name="phi",
        )

    def build_from_sample(self, coordinates: jax.Array):
        self.psi.build_from_sample(coordinates)
        self.phi.build_from_sample(coordinates)
        self.built = True

    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        if not self.built:
            self.psi.build_from_sample(coordinates)
            self.phi.build_from_sample(coordinates)

        h = self.psi(coordinates)
        h = h * jnp.expand_dims(context.non_fictitious_addresses, -1)
        h = jnp.sum(h, axis=0)

        out = self.phi(h)
        return out, {}


class MeanInvariantDecoder(nnx.Module, InvariantDecoder):
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
    :param built: Boolean to indicate whether the decoder is built.
    :param rngs: nnx.Rngs ou int seed.
    """

    def __init__(
        self,
        psi_hidden_size: list[int],
        psi_out_size: int,
        psi_activation: Activation,
        phi_hidden_size: list[int],
        phi_activation: Activation,
        *,
        out_size: int = 0,
        rngs: nnx.Rngs | int | None = None,
        name: str | None = None,
        built: bool = False,
    ) -> None:
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        self.psi_hidden_size = [int(h) for h in psi_hidden_size]
        self.psi_out_size = int(psi_out_size)
        self.psi_activation = psi_activation
        self.phi_hidden_size = [int(h) for h in phi_hidden_size]
        self.phi_activation = phi_activation
        self.out_size = int(out_size)
        self.rngs = rngs
        self.built = built
        self.name = name

        self.psi = MLP(
            hidden_size=self.psi_hidden_size,
            out_size=self.psi_out_size,
            activation=self.psi_activation,
            rngs=self.rngs,
            name="psi",
        )
        self.phi = MLP(
            hidden_size=self.phi_hidden_size,
            out_size=self.out_size,
            activation=self.phi_activation,
            rngs=self.rngs,
            name="phi",
        )

    def build_from_sample(self, coordinates: jax.Array):
        self.psi.build_from_sample(coordinates)
        self.phi.build_from_sample(coordinates)
        self.built = True

    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        if not self.built:
            self.psi.build_from_sample(coordinates)
            self.phi.build_from_sample(coordinates)

        numerator = self.psi(coordinates)  # (addresses, psi_out_size)
        mask = jnp.expand_dims(context.non_fictitious_addresses, -1)
        numerator = numerator * mask
        numerator = jnp.sum(numerator, axis=0)

        # denominator = number of non-fictitious addresses (scalar per feature)
        denominator = jnp.sum(mask, axis=0) + 1e-9  # shape (1, psi_out_size) -> broadcast
        # compute mean
        mean_vec = numerator / denominator.squeeze(0)

        out = self.phi(mean_vec)  # shape (out_size,)
        # expand out to per-address if caller expects broadcasted output (original Linen returned expanded)
        out = out * jnp.expand_dims(context.non_fictitious_addresses, -1)
        return out, {}


class AttentionInvariantDecoder(nnx.Module, InvariantDecoder):
    """
    Attention invariant decoder.

    :param v_hidden_size: hidden sizes for value MLPs.
    :param v_activation: activation for value MLPs.
    :param v_out_size: output size for value MLPs.
    :param s_hidden_size: hidden sizes for score MLPs (outputs scalar score).
    :param s_activation: activation for score MLPs.
    :param psi_hidden_size: hidden sizes for final psi MLP.
    :param psi_activation: activation for final psi MLP.
    :param out_size: final output size.
    :param n: number of attention heads.
    :param rngs: nnx.Rngs or int seed.
    """

    def __init__(
        self,
        v_hidden_size: list[int],
        v_activation: Activation,
        v_out_size: int,
        s_hidden_size: list[int],
        s_activation: Activation,
        psi_hidden_size: list[int],
        psi_activation: Activation,
        *,
        out_size: int = 0,
        n: int = 1,
        rngs: nnx.Rngs | int | None = None,
        name: str | None = None,
    ) -> None:
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        self.v_hidden_size = [int(h) for h in v_hidden_size]
        self.v_activation = v_activation
        self.v_out_size = int(v_out_size)
        self.s_hidden_size = [int(h) for h in s_hidden_size]
        self.s_activation = s_activation
        self.psi_hidden_size = [int(h) for h in psi_hidden_size]
        self.psi_activation = psi_activation
        self.out_size = int(out_size)
        self.n = int(n)
        self.rngs = rngs
        self.name = name

        # create lists of MLPs for value (v) and score (s)
        self.v_mlps: list[MLP] = [
            MLP(
                hidden_size=self.v_hidden_size,
                out_size=self.v_out_size,
                activation=self.v_activation,
                rngs=self.rngs,
                name=f"value-mlp-{i}",
            )
            for i in range(self.n)
        ]
        self.s_mlps: list[MLP] = [
            MLP(
                hidden_size=self.s_hidden_size, out_size=1, activation=self.s_activation, rngs=self.rngs, name=f"score-mlp-{i}"
            )
            for i in range(self.n)
        ]

        self.psi = MLP(
            hidden_size=self.psi_hidden_size,
            out_size=self.out_size,
            activation=self.psi_activation,
            rngs=self.rngs,
            name="psi-mlp",
        )

    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        # ensure lazy sub-MLPs are initialized
        for mlp in self.v_mlps:
            mlp.build_from_sample(coordinates)
        for mlp in self.s_mlps:
            mlp.build_from_sample(coordinates)
        self.psi.build_from_sample(coordinates)

        value_list = []
        mask = jnp.expand_dims(context.non_fictitious_addresses, -1)
        for i in range(self.n):
            v = self.v_mlps[i](coordinates)  # (addresses, v_out_size)
            s = self.s_mlps[i](coordinates)  # (addresses, 1)

            numerator = v * jnp.exp(s)
            numerator = numerator * mask
            numerator = jnp.sum(numerator, axis=0)  # (v_out_size,)

            denominator = jnp.exp(s) * mask
            denominator = jnp.sum(denominator, axis=0) + 1e-9  # shape (1, v_out_size) if broadcast ; safe

            value_list.append(numerator / denominator)

        value_vec = jnp.concatenate(value_list, axis=0)  # concat heads -> (n * v_out_size,)
        out = self.psi(value_vec)
        return out, {}
