# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import field
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from energnn.gnn_nnx.utils import MLP, gather
from energnn.graph.jax.graph import JaxEdge, JaxGraph, JaxGraphShape

Activation = Callable[[jax.Array], jax.Array]


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


class ZeroEquivariantDecoder(nnx.Module):
    r"""Zero equivariant decoder that returns only zeros.

    .. math::
        \forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, \hat{y}_e = [0, \dots, 0]

    :param out_structure: Output structure of the decoder.
    """

    def __init__(self, out_structure: dict | None = None) -> None:
        self.out_structure = out_structure or {}

    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[JaxGraph, dict]:
        edge_dict: dict = {}
        for key, edge in context.edges.items():
            if key in self.out_structure:
                n_obj = int(edge.feature_array.shape[0])
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


class MLPEquivariantDecoder(nnx.Module):
    r"""Equivariant decoder that applies class-specific MLPs over edge features and latent coordinates.

    .. math::
        \forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, \hat{y}_e = \phi_\theta^c(x_e, h_e),

    where :math:`\phi_\theta^c` is a class specific MLP.

    :param out_structure: Output structure of the decoder.
    :param activation: Activation of the MLP :math:`\phi_\theta^c`.
    :param hidden_size: Hidden size of the MLP :math:`\phi_\theta^c`.
    :param final_kernel_zero_init: If true, initializes the last kernel to zero.
    :param rngs: ``nnx.Rngs`` or integer seed used to initialize MLPs.
    """

    def __init__(
        self,
        *,
        hidden_size: list[int],
        out_structure: dict | None = None,
        activation: Activation | None = None,
        final_kernel_zero_init: bool = False,
        rngs: nnx.Rngs | int | None = None,
    ) -> None:
        self.hidden_size = [int(h) for h in hidden_size]
        self.out_structure = out_structure or {}
        self.activation = activation
        self.final_kernel_zero_init = bool(final_kernel_zero_init)
        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)
        self.rngs = rngs

        self.mlps: nnx.Dict = nnx.Dict()

    def _build_mlps_for_context(self, context: JaxGraph, coordinates: jax.Array) -> nnx.Dict:

        coord_dim = int(coordinates.shape[-1])

        for k, feature_names in self.out_structure.items():
            if k not in self.mlps:
                self.mlps[k] = MLP(
                    hidden_size=self.hidden_size,
                    out_size=len(feature_names),
                    activation=self.activation,
                    final_kernel_zero_init=self.final_kernel_zero_init,
                    rngs=self.rngs,
                    name=k,
                )

            edge = context.edges.get(k)
            if edge is not None:
                n_addr_arrays = len(edge.address_dict) if edge.address_dict is not None else 0
                feat_dim = int(edge.feature_array.shape[-1]) if (edge.feature_array is not None) else 0
                sample_dim = n_addr_arrays * coord_dim + feat_dim

                if sample_dim <= 0:
                    continue

                sample = jnp.ones((sample_dim,))
                self.mlps[k].build_from_sample(sample)

        return self.mlps

    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[JaxGraph, dict]:

        info: dict = {}

        mlp_dict = self._build_mlps_for_context(context, coordinates)

        # restrict to the edges we will decode (keys present in out_structure)
        edge_dict = {k: e for k, e in context.edges.items() if k in self.out_structure}

        def gather_inputs(edge: JaxEdge) -> jax.Array:
            decoder_input: list[jax.Array] = []
            for _, address_array in edge.address_dict.items():
                decoder_input.append(gather(coordinates=coordinates, addresses=address_array))
            if edge.feature_array is not None:
                decoder_input.append(edge.feature_array)
            return jnp.concatenate(decoder_input, axis=-1)

        decoder_input_dict = jax.tree.map(gather_inputs, edge_dict, is_leaf=(lambda x: isinstance(x, JaxEdge)))

        # convert mlp storage (nnx.Dict) -> plain dict aligned with edge_dict keys
        plain_mlps = {k: mlp_dict[k] for k in edge_dict.keys()}

        def apply_mlp(edge: JaxEdge, feature_names, decoder_input, mlp) -> JaxEdge:
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
            plain_mlps,
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

        return output_graph, info
