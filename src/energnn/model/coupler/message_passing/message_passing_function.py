# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from flax.typing import Dtype, Initializer

from energnn.graph import GraphStructure, Graph, HyperEdgeSet
from energnn.model.utils import Activation, MLP, gather, scatter_add, scatter_max


class MessagePassingFunction(nnx.Module, ABC):
    r"""Interface for a message function :math:`\xi_\theta` in a GNN message passing scheme."""

    @abstractmethod
    def __call__(self, *, graph: Graph, coordinates: jax.Array, step_with_metrics: bool = False) -> tuple[jax.Array, dict]:
        """Should take as input a tuple (graph, coordinates) and return new coordinates."""
        raise NotImplementedError


def _resolve_rngs(seed: int | None, rngs: nnx.Rngs | None) -> nnx.Rngs:
    """Returns the RNG streams for weight initialization, built from the seed if no rngs are provided."""
    if rngs is None:
        if seed is None:
            raise ValueError("Either seed or rngs must be provided.")
        return nnx.Rngs(seed)
    if seed is not None:
        raise ValueError("Seed must be None when rngs are provided.")
    return rngs


def _build_scatter_ports(
    in_graph_structure: GraphStructure, port_scatter_blacklist: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Maps each hyper-edge set key that has ports to its ordered list of non-blacklisted ports."""
    scatter_ports = {}
    for key, hyper_edge_set_structure in in_graph_structure.hyper_edge_sets.items():
        if hyper_edge_set_structure.port_list is not None and len(hyper_edge_set_structure.port_list) > 0:
            scatter_ports[key] = [
                port_key
                for port_key in hyper_edge_set_structure.port_list
                if port_key not in port_scatter_blacklist.get(key, [])
            ]
    return scatter_ports


def _build_hyper_edge_mlps(
    *,
    in_graph_structure: GraphStructure,
    scatter_ports: dict[str, list[str]],
    in_array_size: int,
    hidden_sizes: list[int],
    out_size: int,
    rngs: nnx.Rngs,
    activation: Activation = nnx.relu,
    use_bias: bool = True,
    kernel_init: Initializer = initializers.lecun_normal(),
    bias_init: Initializer = initializers.zeros_init(),
    final_activation: Activation | None = None,
    encoded_feature_size: int | None = None,
    fuse_ports: bool = False,
    dtype: Dtype | None = None,
) -> dict:
    """
    Builds one MLP per hyper-edge class and scatter port, taking as input the features and the port
    coordinates of a hyper-edge. If ``fuse_ports`` is True, builds instead one MLP per class, whose output
    of size ``out_size * n_ports`` is the concatenation of the outputs of its scatter ports.
    """
    mlp_tree: dict = {}
    for key, ports in scatter_ports.items():
        hyper_edge_set_structure = in_graph_structure.hyper_edge_sets[key]
        in_size = in_array_size * len(hyper_edge_set_structure.port_list or [])
        if hyper_edge_set_structure.feature_list is not None and len(hyper_edge_set_structure.feature_list) > 0:
            if encoded_feature_size is not None:
                in_size += encoded_feature_size
            else:
                in_size += len(hyper_edge_set_structure.feature_list)

        def build_mlp(mlp_out_size: int) -> MLP:
            return MLP(
                in_size=in_size,
                hidden_sizes=hidden_sizes,
                activation=activation,
                out_size=mlp_out_size,
                use_bias=use_bias,
                kernel_init=kernel_init,
                bias_init=bias_init,
                final_activation=final_activation,
                dtype=dtype,
                rngs=rngs,
            )

        if fuse_ports:
            if len(ports) > 0:
                mlp_tree[key] = build_mlp(out_size * len(ports))
        else:
            mlp_tree[key] = {port_key: build_mlp(out_size) for port_key in ports}
    return nnx.data(mlp_tree)


def _build_address_mlp(
    *,
    in_array_size: int,
    hidden_sizes: list[int],
    out_size: int,
    rngs: nnx.Rngs,
    activation: Activation = nnx.relu,
    use_bias: bool = True,
    kernel_init: Initializer = initializers.lecun_normal(),
    bias_init: Initializer = initializers.zeros_init(),
    final_activation: Activation | None = None,
) -> MLP:
    """Builds an MLP shared by all addresses, taking as input the coordinates of an address."""
    return MLP(
        in_size=in_array_size,
        hidden_sizes=hidden_sizes,
        activation=activation,
        out_size=out_size,
        use_bias=use_bias,
        kernel_init=kernel_init,
        bias_init=bias_init,
        final_activation=final_activation,
        rngs=rngs,
    )


def _masked_edge_input(
    hyper_edge_set: HyperEdgeSet, coordinates: jax.Array, dtype: Dtype | None = None
) -> tuple[jax.Array, jax.Array]:
    """
    Concatenates the features and the port coordinates of each hyper-edge, masked if fictitious.
    If ``dtype`` is provided, the features and the mask are cast to it.
    """
    input_array = []
    if hyper_edge_set.feature_names is not None and len(hyper_edge_set.feature_names) > 0:
        feature_array = hyper_edge_set.feature_array
        input_array.append(feature_array if dtype is None else feature_array.astype(dtype))
    for port_array in (hyper_edge_set.port_dict or {}).values():
        input_array.append(gather(coordinates=coordinates, addresses=port_array))
    non_fictitious_mask = jnp.expand_dims(hyper_edge_set.non_fictitious, -1)
    if dtype is not None:
        non_fictitious_mask = non_fictitious_mask.astype(dtype)
    return jnp.concatenate(input_array, axis=-1) * non_fictitious_mask, non_fictitious_mask


def _softmax_aggregation(
    scored_messages: dict[str, dict[str, tuple[jax.Array, jax.Array, jax.Array]]],
    n_addresses: int,
    out_size: int,
    n_heads: int,
    eps: float,
) -> jax.Array:
    """
    Sums the values received by each address, weighted per head by the softmax of their scores.
    ``scored_messages[class_key][port_name]`` is the tuple (score, value, port_array) of the hyper-edges
    of this class through this port, of shapes (n_edges, n_heads), (n_edges, n_heads, head_size) and (n_edges,).
    """
    head_size = out_size // n_heads
    neg_inf = jnp.float32(-1.0e30)

    def max_over_ports(accumulator, port_messages):
        """Accumulates the per-receiver, per-head maximum score over one port."""
        score, _, port_array = port_messages
        return scatter_max(accumulator=accumulator, increment=score, addresses=port_array)

    max_scores = jax.tree.reduce(
        max_over_ports,
        scored_messages,
        initializer=jnp.full((n_addresses, n_heads), neg_inf),
        is_leaf=lambda x: isinstance(x, tuple),
    )

    def weight_over_ports(accumulator, port_messages):
        """Accumulates the softmax-weighted value sum and weight sum over one port."""
        value_accumulator, weight_accumulator = accumulator
        score, value, port_array = port_messages
        weight = jnp.exp(score - gather(coordinates=max_scores, addresses=port_array))
        weighted_value = (weight[:, :, None] * value).reshape(-1, out_size)
        value_accumulator = scatter_add(accumulator=value_accumulator, increment=weighted_value, addresses=port_array)
        weight_accumulator = scatter_add(accumulator=weight_accumulator, increment=weight, addresses=port_array)
        return value_accumulator, weight_accumulator

    value_sum, weight_sum = jax.tree.reduce(
        weight_over_ports,
        scored_messages,
        initializer=(jnp.zeros((n_addresses, out_size)), jnp.zeros((n_addresses, n_heads))),
        is_leaf=lambda x: isinstance(x, tuple),
    )

    output = value_sum.reshape(n_addresses, n_heads, head_size) / (weight_sum[:, :, None] + eps)
    return output.reshape(n_addresses, out_size)


class LocalSumMessagePassingFunction(MessagePassingFunction):
    r"""
    Local sum-based message function module for GNN message passing.

    This module aggregates messages from each node's local neighborhood by applying
    a class- and port-specific MLP :math:`\xi^{c,o}_\theta` to hyper-edge features and neighbor coordinates,
    summing the results across all incoming ports, and applying a final activation :math:`\sigma`.

    For each address :math:`a`, the output is defined as:

    .. math::
        \psi_\theta(h,x)_a = \sigma \left( \sum_{(c,e,o)\in \mathcal{N}_x(a)} \xi^{c,o}_\theta(h_e, x_e)\right),

    where :math:`\xi^{c,o}_\theta` is a class-specific and port-specific MLP, :math:`\sigma` is an
    element-wise activation function, and :math:`h_e := (h_{o(e)})_{o \in {\mathcal{O}^c}}` is the concatenation of
    port coordinates of hyper-edge :math:`e`.

    When ``fuse_ports`` is set to True, the port-specific MLPs of a class are fused into a single
    class-specific MLP :math:`\xi^{c}_\theta` that predicts the messages of all non-blacklisted ports
    at once: its output of size ``out_size * n_ports`` is split into one chunk per port, which is then
    scattered to the corresponding addresses:

    .. math::
        \psi_\theta(h,x)_a = \sigma \left( \sum_{(c,e,o)\in \mathcal{N}_x(a)} \left[\xi^{c}_\theta(h_e, x_e)\right]_o\right),

    where :math:`\left[\cdot\right]_o` denotes the output chunk associated with port :math:`o`.
    Hidden layers are then shared between the ports of a same class, which reduces both the parameter
    count and the amount of computation.

    .. note::
        Enabling ``fuse_ports`` changes the parameter structure of the module: checkpoints saved
        with ``fuse_ports=False`` (e.g. from older models, where this mode did not exist) cannot be
        loaded with ``fuse_ports=True``, and vice versa.

    :param in_graph_structure: Input graph structure.
    :param in_array_size: Size of the input coordinate arrays.
    :param hidden_sizes: Hidden sizes of the MLPs :math:`\xi^{c,o}_\theta`.
    :param activation: Activation function for the MLPs :math:`\xi^{c,o}_\theta`.
    :param out_size: Size of the message associated to each port.
    :param use_bias: Whether to use bias in the MLPs :math:`\xi^{c,o}_\theta`.
    :param kernel_init: Kernel initializer for the MLPs :math:`\xi^{c,o}_\theta`.
    :param bias_init: Bias initializer for the MLPs :math:`\xi^{c,o}_\theta`.
    :param final_activation: Final activation function for the MLPs :math:`\xi^{c,o}_\theta`.
    :param outer_activation: Activation function :math:`\sigma` applied over the output.
    :param encoded_feature_size: None if the input data has not been encoded, otherwise the size of the encoded features.
    :param port_scatter_blacklist: Dictionary mapping hyper-edge set keys to lists of port keys to be excluded from the sum.
    :param fuse_ports: If True, use a single MLP per class predicting the messages of all its
        non-blacklisted ports, instead of one MLP per class and port.
    :param dtype: Computation dtype of the message passing (e.g. ``jnp.bfloat16`` for mixed
        precision): coordinates and features are cast to this dtype before being gathered and fed
        to the MLPs, and messages are aggregated in this dtype. MLP parameters are stored in
        float32 regardless, and the output is cast back to the coordinates dtype. None (default)
        keeps the computation in the input dtype, i.e. full float32.
    :param scatter_dtype: Dtype in which the messages are accumulated by the scatter-add. None
        (default) accumulates in the computation dtype. On GPUs without hardware atomic add for
        the computation dtype (e.g. bfloat16 before compute capability 9.0), the scatter-add is
        emulated and slow: setting ``scatter_dtype=jnp.float32`` together with
        ``dtype=jnp.bfloat16`` keeps the MLPs in bfloat16 while accumulating in float32.
    :param seed: Seed for RNG streams for weight initialization.
    """

    def __init__(
        self,
        in_graph_structure: GraphStructure,
        in_array_size: int,
        hidden_sizes: list[int],
        activation: Activation = nnx.relu,
        out_size: int = 1,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        outer_activation: Activation = nnx.tanh,
        encoded_feature_size: int | None = None,
        port_scatter_blacklist: dict[str, list[str]] | None = None,
        fuse_ports: bool = False,
        dtype: Dtype | None = None,
        scatter_dtype: Dtype | None = None,
        seed: int | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        self.in_graph_structure = in_graph_structure
        self.in_array_size = in_array_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.out_size = out_size
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.final_activation = final_activation
        self.outer_activation = outer_activation
        self.encoded_feature_size = encoded_feature_size
        if port_scatter_blacklist is None:
            self.port_scatter_blacklist = {}
        else:
            self.port_scatter_blacklist = port_scatter_blacklist
        self.fuse_ports = fuse_ports
        self.dtype = dtype
        self.scatter_dtype = scatter_dtype
        self.active_ports = _build_scatter_ports(in_graph_structure, self.port_scatter_blacklist)
        self.mlp_tree = _build_hyper_edge_mlps(
            in_graph_structure=in_graph_structure,
            scatter_ports=self.active_ports,
            in_array_size=in_array_size,
            hidden_sizes=hidden_sizes,
            out_size=out_size,
            rngs=_resolve_rngs(seed, rngs),
            activation=activation,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            final_activation=final_activation,
            encoded_feature_size=encoded_feature_size,
            fuse_ports=fuse_ports,
            dtype=dtype,
        )

    def __call__(self, *, graph: Graph, coordinates: jax.Array, step_with_metrics: bool = False) -> tuple[jax.Array, dict]:

        out_dtype = coordinates.dtype
        compute_coordinates = coordinates if self.dtype is None else coordinates.astype(self.dtype)
        scatter_dtype = compute_coordinates.dtype if self.scatter_dtype is None else self.scatter_dtype

        def sum_over_edges(_accumulator, edge_mlp_tuple):
            """Sums the messages predicted by class-specific MLPs through ports of a hyper-edge set."""
            key, hyper_edge_set, mlp_or_dict = edge_mlp_tuple

            masked_input, non_fictitious_mask = _masked_edge_input(hyper_edge_set, compute_coordinates, self.dtype)

            if self.fuse_ports:
                output_array = (mlp_or_dict(masked_input) * non_fictitious_mask).astype(scatter_dtype)
                for i, port_name in enumerate(self.active_ports[key]):
                    increment = output_array[..., i * self.out_size : (i + 1) * self.out_size]
                    _accumulator = scatter_add(
                        accumulator=_accumulator, increment=increment, addresses=hyper_edge_set.port_dict[port_name]
                    )
            else:
                for port_name, mlp in mlp_or_dict.items():
                    increment = (mlp(masked_input) * non_fictitious_mask).astype(scatter_dtype)
                    _accumulator = scatter_add(
                        accumulator=_accumulator, increment=increment, addresses=hyper_edge_set.port_dict[port_name]
                    )
            return _accumulator

        initializer = jnp.zeros((coordinates.shape[0], self.out_size), dtype=scatter_dtype)
        edge_mlp_dict = {
            key: (key, hyper_edge_set, self.mlp_tree[key])
            for key, hyper_edge_set in graph.hyper_edge_sets.items()
            if key in self.mlp_tree
        }
        accumulator = jax.tree.reduce(
            sum_over_edges,
            edge_mlp_dict,
            initializer=initializer,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        return self.outer_activation(accumulator).astype(out_dtype), {}


class GATv2MessagePassingFunction(MessagePassingFunction):
    r"""
    GATv2 attention message function module for GNN message passing.

    This module aggregates messages from each address's local neighborhood with
    dynamic attention weights (Brody et al. 2022). A class- and port-specific MLP
    produces, for each head, an additive score and a value from the hyper-edge
    input :math:`(h_e, x_e)`; the scores are softmax-normalized over the incoming
    neighbors of an address, and the output is the attention-weighted sum of the
    values, concatenated across heads.

    For each address :math:`a`, the output is defined as:

    .. math::
        \alpha^{i}_{c,e,o} = \operatorname*{softmax}_{(c,e,o) \in \mathcal{N}_x(a)}
            s^{c,o,i}_\theta(h_e, x_e), \qquad
        \psi_\theta(h,x)_a = \sigma\!\left(
            \bigoplus_{i=1}^{H}
            \sum_{(c,e,o) \in \mathcal{N}_x(a)} \alpha^{i}_{c,e,o}\,
            \xi^{c,o,i}_\theta(h_e, x_e)
        \right),

    where :math:`s^{c,o,i}_\theta` and :math:`\xi^{c,o,i}_\theta` are the score and
    value heads of a shared class- and port-specific MLP, :math:`H` is the number
    of heads, :math:`\bigoplus` is concatenation over heads, :math:`\sigma` is an
    element-wise activation function, :math:`h_e := (h_{o(e)})_{o \in {\mathcal{O}^c}}`
    is the concatenation of port coordinates of hyper-edge :math:`e`, and :math:`x_e`
    are the features of hyper-edge :math:`e`.

    :param in_graph_structure: Input graph structure.
    :param in_array_size: Size of the input coordinate arrays.
    :param hidden_sizes: Hidden sizes of the MLPs.
    :param n_heads: Number of attention heads; ``out_size`` must be divisible by it.
    :param activation: Activation function for the MLPs.
    :param out_size: Output size per address, concatenated over heads.
    :param use_bias: Whether to use bias in the MLPs.
    :param kernel_init: Kernel initializer for the MLPs.
    :param bias_init: Bias initializer for the MLPs.
    :param final_activation: Final activation function for the MLPs.
    :param outer_activation: Activation function :math:`\sigma` applied over the output.
    :param encoded_feature_size: None if the input data has not been encoded, otherwise the size of the encoded features.
    :param port_scatter_blacklist: Dictionary mapping hyper-edge set keys to lists of port keys to be excluded from the sum.
    :param eps: Numerical stability term in the softmax denominator.
    :param seed: Seed for RNG streams for weight initialization.
    """

    def __init__(
        self,
        in_graph_structure: GraphStructure,
        in_array_size: int,
        hidden_sizes: list[int],
        n_heads: int = 1,
        activation: Activation = nnx.leaky_relu,
        out_size: int = 1,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        outer_activation: Activation = nnx.tanh,
        encoded_feature_size: int | None = None,
        port_scatter_blacklist: dict[str, list[str]] | None = None,
        eps: float = 1e-9,
        seed: int | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        if n_heads < 1:
            raise ValueError(f"n_heads ({n_heads}) must be at least 1.")
        if out_size % n_heads != 0:
            raise ValueError(f"out_size ({out_size}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.out_size = out_size
        self.outer_activation = outer_activation
        self.eps = eps
        # A shared MLP per (class, port) outputs n_heads scores and out_size value channels.
        self.mlp_tree = _build_hyper_edge_mlps(
            in_graph_structure=in_graph_structure,
            scatter_ports=_build_scatter_ports(in_graph_structure, port_scatter_blacklist or {}),
            in_array_size=in_array_size,
            hidden_sizes=hidden_sizes,
            out_size=out_size + n_heads,
            rngs=_resolve_rngs(seed, rngs),
            activation=activation,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            final_activation=final_activation,
            encoded_feature_size=encoded_feature_size,
        )

    def __call__(self, *, graph: Graph, coordinates: jax.Array, step_with_metrics: bool = False) -> tuple[jax.Array, dict]:
        n_addresses = coordinates.shape[0]
        value_size = self.out_size // self.n_heads
        neg_inf = jnp.float32(-1.0e30)

        def scores_and_values(hyper_edge_set, mlp_dict):
            """Evaluates the shared MLP once per port, returning masked scores and values per edge."""
            masked_input, non_fictitious_mask = _masked_edge_input(hyper_edge_set, coordinates)

            def score_and_value(mlp, port_array):
                """Evaluates the fused MLP for one port, returning its masked score and value."""
                output = mlp(masked_input)
                score = jnp.where(non_fictitious_mask, output[:, : self.n_heads], neg_inf)
                value = output[:, self.n_heads :].reshape(-1, self.n_heads, value_size) * non_fictitious_mask[:, :, None]
                return score, value, port_array

            return {
                port_name: score_and_value(mlp, hyper_edge_set.port_dict[port_name]) for port_name, mlp in mlp_dict.items()
            }

        scored_messages = {
            key: scores_and_values(hyper_edge_set, self.mlp_tree[key])
            for key, hyper_edge_set in graph.hyper_edge_sets.items()
            if key in self.mlp_tree
        }

        output = _softmax_aggregation(scored_messages, n_addresses, self.out_size, self.n_heads, self.eps)
        return self.outer_activation(output), {}


class MultiHeadQKVMessagePassingFunction(MessagePassingFunction):
    r"""
    Multi-head query-key-value attention message function module for GNN message passing.

    This module aggregates messages from each address's local neighborhood with
    query-key-value attention. A shared MLP produces, for each head, a query
    from the coordinates of the receiving address, and a class- and port-specific MLP
    produces, for each head, a key and a value from the hyper-edge input :math:`(h_e, x_e)`.
    Each value is weighted by the dot product of its key with the query of the
    receiving address, and the output is the weighted sum of the values, concatenated
    across heads.

    For each address :math:`a`, the output is defined as:

    .. math::
        \alpha^{i}_{c,e,o} = \langle q^{i}_\theta(h_a), k^{c,o,i}_\theta(h_e, x_e) \rangle, \qquad
        \psi_\theta(h,x)_a = \sigma\!\left(
            \bigoplus_{i=1}^{H}
            \sum_{(c,e,o) \in \mathcal{N}_x(a)} \alpha^{i}_{c,e,o}\,
            \xi^{c,o,i}_\theta(h_e, x_e)
        \right),

    where :math:`q^{i}_\theta` is the query head of a shared MLP,
    :math:`k^{c,o,i}_\theta` and :math:`\xi^{c,o,i}_\theta` are the key and value heads
    of a shared class- and port-specific MLP, :math:`d` is the size of a query or key head, :math:`H`
    is the number of heads, :math:`\bigoplus` is concatenation over heads, :math:`\sigma`
    is an element-wise activation function, :math:`h_e := (h_{o(e)})_{o \in {\mathcal{O}^c}}`
    is the concatenation of port coordinates of hyper-edge :math:`e`, and :math:`x_e` are
    the features of hyper-edge :math:`e`. If ``use_softmax`` is True, the scores are replaced
    by scaled dot-product attention weights (Vaswani et al. 2017)
    :math:`\operatorname*{softmax}_{(c,e,o) \in \mathcal{N}_x(a)} \alpha^{i}_{c,e,o} / \sqrt{d}`.

    :param in_graph_structure: Input graph structure.
    :param in_array_size: Size of the input coordinate arrays.
    :param hidden_sizes: Hidden sizes of the MLPs.
    :param n_heads: Number of attention heads; ``out_size`` and ``key_size`` must be divisible by it.
    :param activation: Activation function for the MLPs.
    :param out_size: Output size per address, concatenated over heads, which is also the size of the values.
    :param key_size: Size of the queries and keys, concatenated over heads; ``out_size`` if None.
    :param use_bias: Whether to use bias in the MLPs.
    :param kernel_init: Kernel initializer for the MLPs.
    :param bias_init: Bias initializer for the MLPs.
    :param final_activation: Final activation function for the MLPs.
    :param outer_activation: Activation function :math:`\sigma` applied over the output.
    :param use_softmax: Whether the scores are softmax-normalized, otherwise used as unnormalized weights.
    :param encoded_feature_size: None if the input data has not been encoded, otherwise the size of the encoded features.
    :param port_scatter_blacklist: Dictionary mapping hyper-edge set keys to lists of port keys to be excluded from the sum.
    :param eps: Numerical stability term in the softmax denominator.
    :param seed: Seed for RNG streams for weight initialization.
    """

    def __init__(
        self,
        in_graph_structure: GraphStructure,
        in_array_size: int,
        hidden_sizes: list[int],
        n_heads: int = 1,
        activation: Activation = nnx.leaky_relu,
        out_size: int = 1,
        key_size: int | None = None,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        outer_activation: Activation = nnx.tanh,
        use_softmax: bool = False,
        encoded_feature_size: int | None = None,
        port_scatter_blacklist: dict[str, list[str]] | None = None,
        eps: float = 1e-9,
        seed: int | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        rngs = _resolve_rngs(seed, rngs)
        if n_heads < 1:
            raise ValueError(f"n_heads ({n_heads}) must be at least 1.")
        if out_size % n_heads != 0:
            raise ValueError(f"out_size ({out_size}) must be divisible by n_heads ({n_heads}).")
        if key_size is None:
            key_size = out_size
        if key_size % n_heads != 0:
            raise ValueError(f"key_size ({key_size}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.out_size = out_size
        self.key_size = key_size
        self.outer_activation = outer_activation
        self.use_softmax = use_softmax
        self.eps = eps
        self.query_mlp = _build_address_mlp(
            in_array_size=in_array_size,
            hidden_sizes=hidden_sizes,
            out_size=key_size,
            rngs=rngs,
            activation=activation,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            final_activation=final_activation,
        )
        # A shared MLP per (class, port) outputs key_size key channels and out_size value channels.
        self.mlp_tree = _build_hyper_edge_mlps(
            in_graph_structure=in_graph_structure,
            scatter_ports=_build_scatter_ports(in_graph_structure, port_scatter_blacklist or {}),
            in_array_size=in_array_size,
            hidden_sizes=hidden_sizes,
            out_size=key_size + out_size,
            rngs=rngs,
            activation=activation,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            final_activation=final_activation,
            encoded_feature_size=encoded_feature_size,
        )

    def __call__(self, *, graph: Graph, coordinates: jax.Array, step_with_metrics: bool = False) -> tuple[jax.Array, dict]:
        n_addresses = coordinates.shape[0]
        key_head_size = self.key_size // self.n_heads
        value_head_size = self.out_size // self.n_heads
        neg_inf = jnp.float32(-1.0e30)
        queries = self.query_mlp(coordinates).reshape(n_addresses, self.n_heads, key_head_size)

        def scores_and_values(hyper_edge_set, mlp_dict):
            """Evaluates the shared MLP once per port, returning scores and masked values per edge."""
            masked_input, non_fictitious_mask = _masked_edge_input(hyper_edge_set, coordinates)

            def score_and_value(mlp, port_array):
                """Evaluates the fused MLP for one port, returning its score and its masked value."""
                output = mlp(masked_input)
                attention_key = output[:, : self.key_size].reshape(-1, self.n_heads, key_head_size)
                value = output[:, self.key_size :].reshape(-1, self.n_heads, value_head_size)
                value = value * non_fictitious_mask[:, :, None]
                query = gather(coordinates=queries, addresses=port_array)
                score = jnp.sum(query * attention_key, axis=-1)
                if self.use_softmax:
                    score = jnp.where(non_fictitious_mask, score / jnp.sqrt(key_head_size), neg_inf)
                return score, value, port_array

            return {
                port_name: score_and_value(mlp, hyper_edge_set.port_dict[port_name]) for port_name, mlp in mlp_dict.items()
            }

        scored_messages = {
            key: scores_and_values(hyper_edge_set, self.mlp_tree[key])
            for key, hyper_edge_set in graph.hyper_edge_sets.items()
            if key in self.mlp_tree
        }

        if self.use_softmax:
            output = _softmax_aggregation(scored_messages, n_addresses, self.out_size, self.n_heads, self.eps)
            return self.outer_activation(output), {}

        def dot_product_over_ports(accumulator, port_messages):
            """Accumulates the dot-product-weighted value sum over one port."""
            score, value, port_array = port_messages
            weighted_value = (score[:, :, None] * value).reshape(-1, self.out_size)
            return scatter_add(accumulator=accumulator, increment=weighted_value, addresses=port_array)

        value_sum = jax.tree.reduce(
            dot_product_over_ports,
            scored_messages,
            initializer=jnp.zeros((n_addresses, self.out_size)),
            is_leaf=lambda x: isinstance(x, tuple),
        )
        return self.outer_activation(value_sum), {}


class IdentityMessagePassingFunction(MessagePassingFunction):
    r"""
    Identity local message function module for GNN message passing.

    This module returns the node features unchanged as the local message.
    It implements the identity mapping on node features:

    .. math::
        h^\rightarrow_a = h_a
    """

    def __init__(self):
        pass

    def __call__(self, *, graph: Graph, coordinates: jax.Array, step_with_metrics: bool = False) -> tuple[jax.Array, dict]:
        return coordinates, {}
