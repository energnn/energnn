# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random
from flax import nnx

from energnn.graph.jax import JaxEdge, JaxGraph
from energnn.model.utils import MLP
from .encoder import Encoder

Activation = Callable[[jax.Array], jax.Array]


class MLPEncoder(Encoder):
    r"""
    Encoder that applies class-specific Multi Layer Perceptrons.

    .. math::
        \begin{align}
        &\forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, & \tilde{x}_e = \phi_\theta^c(x_e),
        \end{align}

    where :math:`({\phi}_{\theta}^c)_{c\in C}` is a set of class-specific MLPs.

    :param hidden_size: Hidden sizes for each MLP.
    :param out_size: Output size for each MLP.
    :param activation: Activation function to use inside MLPs.
    :param rngs: nnx.Rngs or integer seed used to derive RNG streams for per-type MLPs.
    :return: NNX Module that encodes edges using class-specific MLPs.
    """

    def __init__(self, *, hidden_size: list[int], out_size: int, activation: Activation = jax.nn.relu, seed: int = 0) -> None:
        super().__init__(seed=seed)

        self.hidden_size = [int(h) for h in hidden_size]
        self.out_size = int(out_size)
        self.activation = activation
        self.mlp_dict: dict = {}
        self.feature_names = nnx.Dict({f"lat_{i}": jnp.array(i) for i in range(self.out_size)})

    def _build_missing_mlps(self, graph: JaxGraph) -> None:
        """Creates an MLP for each edge class appearing in the graph."""
        for k, edge in graph.edges.items():
            if k not in self.mlp_dict:
                self.mlp_dict[k] = MLP(
                    hidden_size=self.hidden_size, out_size=self.out_size, activation=self.activation, rngs=self.rngs
                )
        return None

    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Apply the Multi Layer Perceptron neural network to edges of an input graph and return the corresponding graph.

        Each edge type (key in `context.edges`) gets its own MLP.

        :param graph: Input graph with edges to encode.
        :param get_info: Flag to return additional information for tracking purpose.
        :return: Encoded graph and additional info dictionary.
        """
        self._build_missing_mlps(graph)

        edge_mlp_dict = {k: (edge, self.mlp_dict[k]) for k, edge in graph.edges.items()}

        def apply_mlp(edge_mlp: tuple[JaxEdge, MLP]) -> JaxEdge:
            """Apply the MLP to an edge."""
            edge, mlp = edge_mlp
            if edge.feature_array is not None:
                feature_array, feature_names = mlp(edge.feature_array), self.feature_names
            else:
                feature_array, feature_names = None, None
            return JaxEdge(
                feature_array=feature_array,
                feature_names=feature_names,
                non_fictitious=edge.non_fictitious,
                address_dict=edge.address_dict,
            )

        encoded_edge_dict = jax.tree.map(apply_mlp, edge_mlp_dict, is_leaf=(lambda x: isinstance(x, tuple)))

        encoded_context = JaxGraph(
            edges=encoded_edge_dict,
            non_fictitious_addresses=graph.non_fictitious_addresses,
            true_shape=graph.true_shape,
            current_shape=graph.current_shape,
        )

        return encoded_context, {}
