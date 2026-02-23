# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from flax.typing import Initializer

from energnn.graph import GraphStructure
from energnn.graph.jax.graph import JaxEdge, JaxGraph, JaxGraphShape
from energnn.model.utils import Activation, MLP, gather
from .decoder import Decoder


class EquivariantDecoder(Decoder, ABC):
    """Abstract base class for equivariant decoders that produce per-edge outputs.

    Equivariant decoders preserve the structure of the input graph and produce
    predictions for each edge in a permutation-equivariant manner.
    """

    def __init__(self, *, out_structure: GraphStructure):
        super().__init__()
        self.out_structure = out_structure


class MLPEquivariantDecoder(EquivariantDecoder):
    r"""Equivariant decoder that applies class-specific MLPs over edge features and latent coordinates.

    .. math::
        \forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c_x, \hat{y}_e = \phi_\theta^c(x_e, h_e),

    where :math:`\phi_\theta^c` is a class specific MLP.

    :param in_graph_structure: Input graph structure.
    :param in_array_size: Size of the input coordinate arrays.
    :param hidden_sizes: Hidden sizes of the MLPs :math:`\phi_\theta^c`.
    :param activation: Activation of the MLP :math:`\phi_\theta^c`.
    :param out_structure: Graph structure of the output.
    :param use_bias: Whether to use bias in the MLPs :math:`\phi_\theta^c`.
    :param kernel_init: Kernel initializer for the MLPs :math:`\phi_\theta^c`.
    :param bias_init: Bias initializer for the MLPs :math:`\phi_\theta^c`.
    :param final_activation: Activation of the final layer of the MLPs :math:`\phi_\theta^c`.
    :param encoded_feature_size: None if the input data has not been encoded, otherwise the size of the encoded features.
    :param seed: Seed for RNG streams for weight initialization.
    """

    def __init__(
        self,
        *,
        in_graph_structure: GraphStructure,
        in_array_size: int,
        hidden_sizes: list[int],
        activation: Activation,
        out_structure: GraphStructure,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        encoded_feature_size: int | None = None,
        seed: int | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(out_structure=out_structure)

        if in_array_size <= 0:
            raise ValueError(f"in_array_size must be positive, got {in_array_size}")
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError(f"All hidden sizes must be positive, got {hidden_sizes}")
        if encoded_feature_size is not None and encoded_feature_size <= 0:
            raise ValueError(f"encoded_feature_size must be positive or None, got {encoded_feature_size}")

        self.in_graph_structure = in_graph_structure
        self.in_array_size = in_array_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.out_structure = out_structure
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.final_activation = final_activation
        self.encoded_feature_size = encoded_feature_size

        self.mlp_dict = self._build_mlp_dict(seed=seed, rngs=rngs)
        self.feature_names_dict = nnx.data(
            {
                k: {kk: jnp.array([i]) for i, kk in enumerate(v.feature_list)}
                for k, v in self.out_structure.edges.items()
                if v.feature_list is not None
            }
        )

    def _build_mlp_dict(self, seed: int = 0, rngs: nnx.Rngs | None = None) -> dict[str, MLP]:
        if rngs is None:
            rngs = nnx.Rngs(seed)
        elif seed is not None:
            raise ValueError("Seed must be None when rngs are provided.")
        mlp_dict = {}

        for edge_key, out_edge_structure in self.out_structure.edges.items():
            assert edge_key in self.in_graph_structure.edges.keys()
            in_edge_structure = self.in_graph_structure.edges[edge_key]
            assert len(in_edge_structure.address_list) > 0
            n_ports = len(in_edge_structure.address_list)
            in_size = self.in_array_size * n_ports
            if in_edge_structure.feature_list is not None and len(in_edge_structure.feature_list) > 0:
                if self.encoded_feature_size is not None:
                    in_size += self.encoded_feature_size
                else:
                    in_size += len(in_edge_structure.feature_list)

            assert out_edge_structure.feature_list is not None and len(out_edge_structure.feature_list) > 0
            out_size = len(out_edge_structure.feature_list)

            mlp_dict[edge_key] = MLP(
                in_size=in_size,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                out_size=out_size,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                final_activation=self.final_activation,
                rngs=rngs,
            )
        return nnx.data(mlp_dict)

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """Decode latent coordinates into an output graph.

        :param graph: Encoded graph providing context for decoding.
        :param coordinates: Latent coordinates array.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple of decoded graph and info dictionary.
        :raises KeyError: If an edge class in the graph is not present in the decoder's MLP dictionary.
        """

        def apply_over_edge(edge_mlp_names):
            edge, mlp, feature_names = edge_mlp_names

            decoder_input = []
            for _, address_array in edge.address_dict.items():
                decoder_input.append(gather(coordinates=coordinates, addresses=address_array))
            if edge.feature_array is not None:
                decoder_input.append(edge.feature_array)
            decoder_input = jnp.concatenate(decoder_input, axis=-1)
            decoder_output = mlp(decoder_input)
            decoder_output = decoder_output * jnp.expand_dims(edge.non_fictitious, -1)
            return JaxEdge(
                feature_array=decoder_output,
                feature_names=feature_names,
                non_fictitious=edge.non_fictitious,
                address_dict=None,
            )

        edge_mlp_names_dict = {
            k: (edge, self.mlp_dict[k], self.feature_names_dict[k]) for k, edge in graph.edges.items() if k in self.mlp_dict
        }
        edge_dict = jax.tree.map(apply_over_edge, edge_mlp_names_dict, is_leaf=(lambda x: isinstance(x, tuple)))
        true_shape = JaxGraphShape(
            edges={key: value for key, value in graph.true_shape.edges.items() if key in self.feature_names_dict},
            addresses=jnp.array(0),
        )
        current_shape = JaxGraphShape(
            edges={key: value for key, value in graph.current_shape.edges.items() if key in self.feature_names_dict},
            addresses=jnp.array(0),
        )

        output_graph = JaxGraph(
            edges=edge_dict,
            non_fictitious_addresses=jnp.array([]),
            true_shape=true_shape,
            current_shape=current_shape,
        )

        return output_graph, {}
