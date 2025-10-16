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

from energnn.gnn.utils import MLP, gather, scatter_add
from energnn.graph.jax import JaxGraph

MAX_INTEGER = 2147483647


class LocalMessageFunction(ABC):
    """
    Interface for the local message function.

    Subclasses must implement methods to initialize weights and apply the function to a JaxGraph object.
    """

    @abstractmethod
    def init(self, *, rngs: jax.Array, context: JaxGraph, coordinates: jax.Array) -> dict:
        """
        Should return initialized the local message function weights.

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
        Should return initialized function weights and local message.

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
        Should return local message.

        :param params: Parameters.
        :param context: The input graph.
        :param coordinates: Coordinates stored as JAX array.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple(local message, info).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError


class EmptyLocalMessageFunction(nn.Module, LocalMessageFunction):
    r"""
    Empty Local Message Function that returns nothing.

    This class implements a placeholder local message function that returns an empty feature array.
    """

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        n_addr = coordinates.shape[0]
        return jnp.empty(shape=(n_addr, 0)), {}


class IdentityLocalMessageFunction(nn.Module, LocalMessageFunction):
    r"""
    Identity local message function module for GNN message passing.

    This module returns the node features unchanged as the local message.
    It implements the identity mapping on node features:
    .. math::
        h^\rightarrow_a = h_a
    """

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        return coordinates, {}


class SumLocalMessageFunction(nn.Module, LocalMessageFunction):
    r"""
    Local sum-based message function module for GNN message passing.

    This module aggregates messages from each node's local neighborhood by applying
    a class- and port-specific MLP :math:`\xi^{c,o}_\theta` to edge features and neighbor coordinates,
    summing the results across all incoming ports, and applying a final activation :math:`\sigma`.

    The operation is defined as:

    .. math::
        h^\rightarrow_a = \sigma \left( \sum_{(c,e,o)\in \mathcal{N}_a(x)} \xi^{c,o}_\theta(h_e, x_e)\right),

    where :math:`\xi^{c,o}_\theta` is a class-specific and port-specific MLP, and :math:`\sigma` is an
    element-wise activation function.

    :param list[int] hidden_size: Hidden size of the MLPs :math:`\xi^{c,o}_\theta`.
    :param flax.linen.activation activation: Activation function for the MLPs :math:`\xi^{c,o}_\theta`.
    :param int out_size: Local message size.
    :param flax.linen.activation final_activation: Activation function :math:`\sigma` applied over the output.
    """

    out_size: int
    hidden_size: list[int]
    activation: Callable[[jax.Array], jax.Array]
    final_activation: Callable[[jax.Array], jax.Array]

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        neighbour_message = jnp.zeros((coordinates.shape[0], self.out_size))

        def get_mlps(edge_key, address_key):
            return MLP(
                hidden_size=self.hidden_size,
                out_size=self.out_size,
                activation=self.activation,
                name=f"{edge_key}-{address_key}-local_message_mlp",
            )

        mlp_tree = {
            key: {address_key: get_mlps(key, address_key) for address_key in edge.address_dict.keys()}
            for key, edge in context.edges.items()
        }

        def get_messages(edge, mlp_subtree):
            edge_message_input = []

            if edge.feature_names is not None:
                edge_message_input.append(edge.feature_array)

            for address_key, address_array in edge.address_dict.items():
                edge_message_input.append(gather(coordinates=coordinates, addresses=address_array))

            edge_message_input = jnp.concatenate(edge_message_input, axis=-1)

            messages_subtree = {}

            for address_key, address_array in edge.address_dict.items():
                messages_subtree[address_key] = mlp_subtree[address_key](edge_message_input)
                messages_subtree[address_key] = messages_subtree[address_key] * jnp.expand_dims(edge.non_fictitious, -1)

            return messages_subtree

        def sum_messages_for_addresses(latent_coordinates, edge_mlp):
            edge, mlp_subtree = edge_mlp
            message_subtree = get_messages(edge, mlp_subtree)
            for address_key, edge_message in message_subtree.items():
                latent_coordinates = scatter_add(
                    accumulator=latent_coordinates, increment=edge_message, addresses=edge.address_dict[address_key]
                )

            return latent_coordinates

        edge_mlp_tree = {edge_key: (edge, mlp_tree[edge_key]) for edge_key, edge in context.edges.items()}
        neighbour_message = jax.tree.reduce(
            sum_messages_for_addresses, edge_mlp_tree, initializer=neighbour_message, is_leaf=lambda x: isinstance(x, tuple)
        )

        return self.final_activation(neighbour_message), {}


class AttentionLocalMessageFunction(nn.Module, LocalMessageFunction):
    r"""
    Attention-based local message function module for GNN message passing.

    This module implements a multi-head attention over each node's local neighborhood :math:`\mathcal{N}_a(x)`.
    The operation computes as:

    .. math::
        & v^{c,o,i}_e = v^{c,o,i}_\theta(x_e, h_e) \\
        & s^{c,o,i}_e = s^{c,o,i}_\theta(x_e,h_e) \\
        & \alpha_e^{c,o,i} = \frac{\exp(s^{c,o,i}_e)}{\sum_{(c',e',o')\in \mathcal{N}_a(x)} \exp(s^{c',o',i}_{e'})} \\
        & v'^{i}_a = \sum_{(c,e,o)\in \mathcal{N}_a(x)} \alpha_e^{c,o,i} v^{c,o,i}_e \\
        & h^\rightarrow_a = \psi_\theta (v'^{1}_a, \dots, v'^{n}_a)

    where :math:`n` is the number of attention heads,
    :math:`v^{c,o,i}_\theta` (*value*) and :math:`s^{c,o, i}_\theta` (*score*)
    are class-specific, port-specific and head-specific MLPs,
    and :math:`\psi_\theta` is the outer MLP.

    :param n_heads: Number of attention heads :math:`n`.
    :param value_hidden_size: Hidden size of value MLPs :math:`(v^{c,o,i}_\theta)_{c,o,i}`.
    :param flax.linen.activation value_activation: Activation function for the value MLPs :math:`(v^{c,o,i}_\theta)_{c,o,i}`.
    :param value_out_size: Output size of the value MLPs :math:`(v^{c,o,i}_\theta)_{c,o,i}`.
    :param score_hidden_size: Hidden size of the score MLPs :math:`(s^{c,o,i}_\theta)_{c,o,i}`.
    :param flax.linen.activation score_activation: Activation function for the score MLPs :math:`(s^{c,o,i}_\theta)_{c,o,i}`.
    :param psi_hidden_size: Hidden size of outer MLP :math:`\psi_\theta`.
    :param flax.linen.activation psi_activation: Activation function for the outer MLP :math:`\psi_\theta`.
    :param out_size: Local message dimension.
    """

    n_heads: int
    value_hidden_size: list[int]
    value_activation: Callable[[jax.Array], jax.Array]
    value_out_size: int
    score_hidden_size: list[int]
    score_activation: Callable[[jax.Array], jax.Array]
    psi_hidden_size: list[int]
    psi_activation: Callable[[jax.Array], jax.Array]
    out_size: int

    @nn.compact
    def __call__(self, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        v_prime_list = []
        for i in range(self.n_heads):

            # Get tree of Value MLPs
            value_mlp_tree = {
                k: {
                    kk: MLP(
                        hidden_size=self.value_hidden_size,
                        out_size=self.value_out_size,
                        activation=self.value_activation,
                        name=f"value_{k}_{kk}_{i}",
                    )
                    for kk in e.address_dict.keys()
                }
                for k, e in context.edges.items()
            }

            # Get tree of Score MLPs
            score_mlp_tree = {
                k: {
                    kk: MLP(
                        hidden_size=self.score_hidden_size,
                        out_size=1,
                        activation=self.score_activation,
                        name=f"score_{k}_{kk}_{i}",
                    )
                    for kk in e.address_dict.keys()
                }
                for k, e in context.edges.items()
            }

            def get_value_score(edge, value_mlp_dict, score_mlp_dict):

                mlp_input = []
                if (edge.feature_names is not None) and (edge.feature_array.size != 0):
                    mlp_input.append(edge.feature_array)
                for address_key, address_array in edge.address_dict.items():
                    if address_array.size != 0:
                        mlp_input.append(gather(coordinates=coordinates, addresses=address_array))

                if mlp_input:
                    mlp_input = jnp.concatenate(mlp_input, axis=-1)
                    value_dict = {k: mlp(mlp_input) for k, mlp in value_mlp_dict.items()}
                    score_dict = {k: mlp(mlp_input) for k, mlp in score_mlp_dict.items()}
                else:
                    value_dict, score_dict = None, None

                return value_dict, score_dict

            def sum(num_den, edge_mlp):
                numerator, denominator = num_den
                edge, value_mlp_dict, score_mlp_dict = edge_mlp
                value_dict, score_dict = get_value_score(edge, value_mlp_dict, score_mlp_dict)

                if value_dict is not None:
                    for address_key in value_dict:
                        value = value_dict[address_key] * jnp.expand_dims(edge.non_fictitious, -1)
                        score = score_dict[address_key] * jnp.expand_dims(edge.non_fictitious, -1)
                        address_array = edge.address_dict[address_key]
                        numerator = scatter_add(
                            accumulator=numerator, increment=value * jnp.exp(score), addresses=address_array
                        )
                        denominator = scatter_add(accumulator=denominator, increment=jnp.exp(score), addresses=address_array)
                return numerator, denominator

            edge_mlp_tree = {k: (e, value_mlp_tree[k], score_mlp_tree[k]) for k, e in context.edges.items()}
            numerator = jnp.zeros((coordinates.shape[0], self.value_out_size))
            denominator = jnp.zeros((coordinates.shape[0], 1))
            numerator, denominator = jax.tree.reduce(
                sum, edge_mlp_tree, initializer=(numerator, denominator), is_leaf=lambda x: isinstance(x, tuple)
            )
            numerator = numerator * jnp.expand_dims(context.non_fictitious_addresses, -1)
            denominator = denominator * jnp.expand_dims(context.non_fictitious_addresses, -1) + 1e-9
            v_prime_list.append(numerator / denominator)

        psi = MLP(hidden_size=self.psi_hidden_size, out_size=self.out_size, activation=self.psi_activation, name="psi")
        v_prime_concat = jnp.concatenate(v_prime_list, axis=1)
        output = psi(v_prime_concat)
        output = output * jnp.expand_dims(context.non_fictitious_addresses, -1)
        return output, {}


# class GATv2LocalMessageFunction(nn.Module, LocalMessageFunction):
#     r"""GATv2 implementation of local attention (Legacy).
#
#     The attention mechanism follows the GATv2 paper (https://arxiv.org/abs/2105.14491) adapted to the hypergraph setting.
#     For every address in the graph, we compute an attention score between the address's latent coordinates
#     and the connected hyperedge features, and a value message as well.
#     Since the attention mechanism is sparsified by the local nature, it is possible to accumulate the denominator of
#     the softmax and the numerator separately without having to store all of the intermediate values.
#
#     Using the notation used in the paper, for a given layer :math:`l`, and address :math:`i` with neighborhood :math:`N(i)`,
#     the outputed latent representation :math:`h` is computed as follows:
#
#     .. math::
#
#         h^{(l+1)}_i = \sum_{j \in N(i)} {\alpha}_{i, j} V_{j {\rightarrow} i}
#
#
#     With the attention score denoted by :math:`a` and the learned layer matrices :math:`W_{\text{left}}`
#     and :math:`W_{\text{right}}`, the vector message :math:`V` and attention score :math:`\alpha` are then defined as :
#
#     .. math::
#
#         \alpha_{i, j} = \frac{e_{i, j}}{\sum_{j' \in N(a)} e_{i, j'}}
#
#         e_{i, j} = \frac{\exp(a^T \text{LeakyReLU}(W_{\text{left}} \cdot h^{(l)}_i + W_\text{right} \cdot x_i))}{\sqrt{d_k}}
#
#         V_{j \rightarrow i} = \text{MLP}({h^{(l)}_i}, \text{concat}(\{{h^{(l)}_j}, j\in N(i) \}), x_i)
#
#     Implementation V_{j \rightarrow i} is implemented so as to handle differently hyper-edge of order 1 and 2.
#
#     .. warning::
#
#         Differently from the implementation from GATv2, where the same :math:`W` is used for :math:`h^{(l)}_i`
#         and :math:`x_i`, we implemented GATv2 with two elements :math:`W_{\text{left}}` and :math:`W_\text{right}`
#         later combined.
#
#     :param list[int] hidden_size: Hidden size of the MLPs.
#     :param int out_size: Final output size (latent coordinates dimension).
#     :param flax.linen.activation activation: Activation function for the MLPs.
#     :param flax.linen.activation final_activation: Activation function for the final output.
#     :param int qk_size: Dimension of the query and key vectors.
#     :param int n_heads: Number of attention heads.
#
#
#     :returns:
#         - **neighbour_message** (jax.Array):   The new latent coordinates.
#         - **info** (dict): Dictionary containing auxiliary information logged.
#     """
#
#     hidden_size: list[int]
#     out_size: int
#     activation: Callable[[jax.Array], jax.Array]
#     final_activation: Callable[[jax.Array], jax.Array]
#     qk_size: int
#     n_heads: int
#
#     def get_mlps(self, edge_key, address_key, feature_array, out_size):
#         attn_mlps = []
#         for i in range(self.n_heads):
#             attn_mlp_l = MLP(
#                 hidden_size=[],
#                 out_size=self.qk_size,
#                 activation=None,
#                 name="attn_mlp_l-{}-{}-{}".format(edge_key, address_key, i),
#             )
#
#             if feature_array is not None:
#                 attn_mlp_r = MLP(
#                     hidden_size=[],
#                     out_size=self.qk_size,
#                     activation=None,
#                     name="attn_mlp_r-{}-{}-{}".format(edge_key, address_key, i),
#                 )
#             else:
#                 attn_mlp_r = None
#             local_attn = MLP(
#                 hidden_size=[],
#                 out_size=1,
#                 activation=None,
#                 name="local_attention-{}-{}-{}".format(edge_key, address_key, i),
#             )
#
#             # different message for every attention head
#             value_mlp = MLP(
#                 hidden_size=self.hidden_size,
#                 out_size=out_size,
#                 activation=self.activation,
#                 name="value-{}-{}-{}".format(edge_key, address_key, i),
#             )
#             attn_mlps.append((attn_mlp_l, attn_mlp_r, local_attn, value_mlp))
#         return attn_mlps
#
#     def get_attention_messages(self, latent_coordinates, edge, attn_mlps, softmax_sums, update_latent_coordinates):
#         if edge.feature_array is not None:
#             key_input = edge.feature_array
#
#         for address_key, address_array in edge.address_dict.items():
#             clean_address_array = address_array.astype(int)
#             query_input = latent_coordinates.at[clean_address_array].get(mode="drop", fill_value=0.0)
#             if edge.feature_array is not None:
#                 value_input = jnp.concatenate([query_input, key_input], axis=-1)
#             else:
#                 key_input = None
#                 value_input = query_input
#
#             for i in range(self.n_heads):
#                 # with correspondance in the doc and paper description:
#                 # - attn_mlp_l : W_left
#                 # - attn_mlp_r : W_right
#                 # - local_attn : a
#                 # - value_mlp : the final message computation
#                 attn_mlp_l, attn_mlp_r, local_attn, value_mlp = attn_mlps[address_key][i]
#
#                 logits_l = attn_mlp_l(query_input)  # query (address embeddings)
#                 if key_input is not None:
#                     logits_r = attn_mlp_r(key_input)  # key (hyperedge features)
#                     logits = logits_l + logits_r
#                 else:
#                     logits = logits_l
#                 logits = nn.leaky_relu(logits, negative_slope=0.2)
#                 logits = local_attn(logits) / jnp.sqrt(self.qk_size)
#
#                 exp_logits = jnp.exp(logits)
#                 value = self.activation(value_mlp(value_input))  # Compute the new message for the given head
#                 new_message = exp_logits * value  # keep the precomputed message
#                 new_message = jnp.where(jnp.isnan(value), jnp.nan, new_message)
#
#                 softmax_sums[i] = softmax_sums[i].at[clean_address_array].add(exp_logits.squeeze(-1), mode="drop")
#                 update_latent_coordinates[i] = (
#                     update_latent_coordinates[i].at[clean_address_array].add(new_message, mode="drop")
#                 )
#
#     @nn.compact
#     def __call__(self, *, input_graph: Graph, latent_coordinates: jax.Array, get_info: bool = False)
#     -> Tuple[jax.Array, dict]:
#         neighbour_message = jnp.zeros((latent_coordinates.shape[0], self.out_size))
#
#         attn_mlp_tree = {
#             key: {
#                 address_key: self.get_mlps(key, address_key, edge.feature_array, latent_coordinates.shape[1])
#                 for address_key in edge.address_dict.keys()
#             }
#             for key, edge in input_graph.edges.items()
#         }
#
#         # Intialize the accumulator for the sum used in the softmax.
#         # This is useful because we have sparse attention on graphs.
#
#         # Initialize softmax sum, per head, with a zero array of size the number of addresses
#         softmax_sums = [jnp.zeros(latent_coordinates.shape[0]) for _ in range(self.n_heads)]
#         # Initialize the accumulator for the new output values (messages) for each head
#         update_latent_coordinates = [jnp.zeros_like(latent_coordinates) for _ in range(self.n_heads)]
#
#         def sum_messages_for_addresses(edge_mlps):
#             edge, attn_mlp_subtree = edge_mlps
#             self.get_attention_messages(latent_coordinates, edge, attn_mlp_subtree, softmax_sums, update_latent_coordinates)
#
#         edge_mlp_tree = {edge_key: (edge, attn_mlp_tree[edge_key]) for edge_key, edge in input_graph.edges.items()}
#
#         jax.tree.map(sum_messages_for_addresses, edge_mlp_tree, is_leaf=lambda x: isinstance(x, Tuple))
#
#         for i in range(self.n_heads):
#             update_latent_coordinates[i] = update_latent_coordinates[i] / softmax_sums[i][..., None]
#
#         neighbour_message = jnp.concatenate(update_latent_coordinates, axis=-1)
#         if self.n_heads > 1:
#             final_local_mlp = MLP(
#                 hidden_size=[],
#                 out_size=latent_coordinates.shape[1],
#                 activation=None,
#                 name="final_mlp_local",
#             )
#             neighbour_message = final_local_mlp(neighbour_message)
#
#         return self.final_activation(neighbour_message), {}
