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
from energnn.graph.jax import JaxEdge, JaxGraph

MAX_INTEGER = 2147483647


class Encoder(ABC):
    r"""
    Interface for the graph encoder :math:`E_\theta`.

    Subclasses must implement methods to initialize parameters and apply the encoder
    to a JaxGraph object.
    """

    @abstractmethod
    def __init__(self):
        """
        Abstract constructor.
        Implementations may define module parameters or internal state.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def init(self, *, rngs: jax.Array, context: JaxGraph) -> dict:
        """
        Should return initialized encoder weights.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph.
        :return: Initialized parameters.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def init_with_output(self, *, rngs: jax.Array, context: JaxGraph) -> tuple[tuple[JaxGraph, dict], dict]:
        """
        Initialize encoder parameters and return encoded graph and parameters.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph.
        :return: Tuple ((encoded graph, encoder parameters), other info dict).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, params: dict, context: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Apply encoder to input graph and return encoded `context`.

        :param params: Parameters.
        :param context: Input graph to encode.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple (encoded graph, info).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError


class IdentityEncoder(Encoder):
    r"""
    Identity encoder that returns the input graph unchanged.

    .. math::
        \tilde{x} = x
    """

    def __init__(self):
        pass

    def init(self, *, rngs: jax.Array, context: JaxGraph) -> dict:
        """Return empty parameters (no learnable weights)."""
        return {}

    def init_with_output(self, *, rngs: jax.Array, context: JaxGraph) -> tuple[tuple[JaxGraph, dict], dict]:
        """
        Initialize the encoder and returns the output graph (unmodified graph) and empty parameter dicts.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph.
        :return: ((input graph, empty dict), empty dict)
        """
        return (context, {}), {}

    def apply(self, params: dict, context: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """Apply the identity encoder and return the input graph without changes.

        :param params: Parameters.
        :param context: Input graph to encode.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Input graph and empty info dict.
        """
        return context, {}


class MLPEncoder(nn.Module, Encoder):
    r"""
    Encoder that applies class-specific Multi Layer Perceptrons.

    .. math::
        \begin{align}
        &\forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, & \tilde{x}_e = \phi_\theta^c(x_e),
        \end{align}

    where :math:`({\phi}_{\theta}^c)_{c\in C}` are a set of class-specific MLPs.

    :param hidden_size: Hidden sizes of MLPs :math:`({\phi}_{\theta}^c)_{c\in C}`.
    :param out_size: Output size of MLPs :math:`({\phi}_{\theta}^c)_{c\in C}`.
    :param flax.linen.activation activation: Activation functions of MLPs :math:`({\phi}_{\theta}^c)_{c\in C}`.
    """

    hidden_size: list[int]
    out_size: int
    activation: Callable[[jax.Array], jax.Array]

    @nn.compact
    def __call__(self, *, context: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Apply the Multi Layer Perceptron neural network to edges of an input graph and return the corresponding graph.

        Each edge type (key in `context.edges`) gets its own MLP.

        :param context: Input graph with edges to encode.
        :param get_info: Flag to return additional information for tracking purpose.
        :return: Encoded graph and additional info dictionary.
        """
        info: dict = {}

        feature_names = {f"lat_{i}": jnp.array(i) for i in range(self.out_size)}

        mlp_dict = {
            k: MLP(hidden_size=self.hidden_size, out_size=self.out_size, activation=self.activation, name=k)
            for k in context.edges
        }

        def apply_mlp(edge, mlp):
            """Apply the MLP to an edge"""
            if edge.feature_array is not None:
                encoded_array = mlp(edge.feature_array)
                edge = JaxEdge(
                    feature_array=encoded_array,
                    feature_names=feature_names,
                    non_fictitious=edge.non_fictitious,
                    address_dict=edge.address_dict,
                )
            else:
                edge = JaxEdge(
                    feature_array=None, feature_names=None, non_fictitious=edge.non_fictitious, address_dict=edge.address_dict
                )

            return edge

        encoded_edge_dict = jax.tree.map(
            apply_mlp,
            context.edges,
            mlp_dict,
            is_leaf=(lambda x: isinstance(x, JaxEdge)),
        )

        encoded_context = JaxGraph(
            edges=encoded_edge_dict,
            non_fictitious_addresses=context.non_fictitious_addresses,
            true_shape=context.true_shape,
            current_shape=context.current_shape,
        )

        return encoded_context, info
