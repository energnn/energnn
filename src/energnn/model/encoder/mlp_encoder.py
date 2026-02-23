# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import jax
import jax.numpy as jnp
import jax.random
from flax import nnx
from flax.nnx import initializers
from flax.typing import Initializer

from energnn.graph import GraphStructure
from energnn.graph.jax import JaxEdge, JaxGraph
from energnn.model.utils import Activation, MLP
from .encoder import Encoder


class MLPEncoder(Encoder):
    r"""
    Encoder that applies class-specific Multi Layer Perceptrons.

    .. math::
        \begin{align}
        &\forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c_x, & \tilde{x}_e = \phi_\theta^c(x_e),
        \end{align}

    where :math:`({\phi}_{\theta}^c)_{c\in C}` is a set of class-specific MLPs.

    :param input_structure: Input graph structure.
    :param hidden_sizes: Hidden sizes of MLPs :math:`({\phi}_{\theta}^c)_{c\in C}`.
    :param activation: Activation functions of MLPs :math:`({\phi}_{\theta}^c)_{c\in C}`.
    :param out_size: Output size of MLPs :math:`({\phi}_{\theta}^c)_{c\in C}`.
    :param use_bias: Whether to use bias in MLPs :math:`({\phi}_{\theta}^c)_{c\in C}`.
    :param kernel_init: Kernel initializer for MLPs :math:`({\phi}_{\theta}^c)_{c\in C}`.
    :param bias_init: Bias initializer for MLPs :math:`({\phi}_{\theta}^c)_{c\in C}`.
    :param final_activation: Final activation function for MLPs :math:`({\phi}_{\theta}^c)_{c\in C}`.
    :param seed: Seed for RNG streams for weight initialization.
    """

    def __init__(
        self,
        *,
        in_structure: GraphStructure,
        hidden_sizes: list[int],
        activation: Activation | None = nnx.relu,
        out_size: int = 1,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()

        if out_size <= 0:
            raise ValueError(f"out_size must be positive, got {out_size}")
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError(f"All hidden sizes must be positive, got {hidden_sizes}")

        self.in_structure = in_structure
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.activation = activation
        self.out_size = int(out_size)
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.final_activation = final_activation

        self.mlp_dict = self._build_mlp_dict(seed=seed)
        self.feature_names = nnx.data({f"lat_{i}": jnp.array(i) for i in range(self.out_size)})

    def _build_mlp_dict(self, seed: int = 0) -> dict[str, MLP]:
        """Creates an MLP for each edge class appearing in the input structure, initialized with the given seed."""
        rngs = nnx.Rngs(seed)  # Create a single rng stream and pass it to all MLP blocks.
        mlp_dict = {}
        for edge_key, edge_structure in self.in_structure.edges.items():
            if edge_structure.feature_list is not None and len(edge_structure.feature_list) > 0:
                in_size = len(edge_structure.feature_list)
                mlp_dict[edge_key] = MLP(
                    in_size=in_size,
                    hidden_sizes=self.hidden_sizes,
                    activation=self.activation,
                    out_size=self.out_size,
                    use_bias=self.use_bias,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    final_activation=self.final_activation,
                    rngs=rngs,
                )
        return nnx.data(mlp_dict)

    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Apply the class-specific MLPs to the input graph and return the encoded graph.

        :param graph: Input graph with edges to encode.
        :param get_info: Flag to return additional information for tracking purpose.
        :return: Encoded graph and additional info dictionary.
        :raises KeyError: If an edge class in the graph is not present in the encoder's MLP dictionary.
        """

        # Verify all edge keys have corresponding MLPs
        missing_keys = set(graph.edges.keys()) - set(self.mlp_dict.keys())
        if missing_keys:
            raise KeyError(
                f"Graph contains edge classes {missing_keys} that were not present in the input structure. "
                f"Available edge classes: {set(self.mlp_dict.keys())}"
            )

        edge_mlp_dict = {k: (edge, self.mlp_dict[k]) for k, edge in graph.edges.items() if k in self.mlp_dict.keys()}

        def apply_mlp(edge_mlp: tuple[JaxEdge, MLP]) -> JaxEdge:
            """Apply the MLP to the edge."""
            edge, mlp = edge_mlp
            if edge.feature_array is not None:
                mask = jnp.expand_dims(edge.non_fictitious, -1)
                feature_array, feature_names = mlp(edge.feature_array) * mask, self.feature_names
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
