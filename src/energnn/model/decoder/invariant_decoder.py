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

from energnn.graph import JaxGraph
from energnn.model.utils import MLP
from .decoder import Decoder


class InvariantDecoder(Decoder, ABC):
    """Abstract base class for invariant decoders that produce global outputs.

    Invariant decoders aggregate information from all addresses in a permutation-invariant
    manner to produce a single global output vector.
    """

    @abstractmethod
    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """Decode latent coordinates into a global decision vector.

        :param graph: Input graph to decode.
        :param coordinates: Coordinates stored as JAX array.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple containing decision vector and info dictionary.
        :raises NotImplementedError: If subclass does not override this method.
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

    def __init__(self, *, psi: MLP, phi: MLP) -> None:
        super().__init__()
        self.psi = psi
        self.phi = phi

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

    def __init__(self, *, psi: MLP, phi: MLP) -> None:
        super().__init__()
        self.psi = psi
        self.phi = phi

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        numerator = self.psi(coordinates)
        numerator = numerator * jnp.expand_dims(graph.non_fictitious_addresses, -1)
        numerator = jnp.sum(numerator, axis=0)
        denominator = jnp.sum(graph.non_fictitious_addresses, axis=0) + 1e-9
        return self.phi(numerator / denominator), {}


