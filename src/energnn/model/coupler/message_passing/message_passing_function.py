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

from energnn.graph import GraphStructure, Graph
from energnn.model.utils import Activation, MLP, gather, scatter_add


class MessagePassingFunction(nnx.Module, ABC):
    r"""Interface for a message function :math:`\xi_\theta` in a GNN message passing scheme."""

    @abstractmethod
    def __call__(self, *, graph: Graph, coordinates: jax.Array, step_with_metrics: bool = False) -> tuple[jax.Array, dict]:
        """Should take as input a tuple (graph, coordinates) and return new coordinates."""
        raise NotImplementedError


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

        self.active_ports = self._build_active_ports()
        self.mlp_tree = self._build_mlp_tree(seed=seed, rngs=rngs)

    def _build_active_ports(self) -> dict[str, list[str]]:
        """Maps each hyper-edge set key to its ordered list of non-blacklisted ports."""
        active_ports = {}
        for key, hyper_edge_set_structure in self.in_graph_structure.hyper_edge_sets.items():
            if hyper_edge_set_structure.port_list is not None and len(hyper_edge_set_structure.port_list) > 0:
                active_ports[key] = [
                    port_key
                    for port_key in hyper_edge_set_structure.port_list
                    if port_key not in self.port_scatter_blacklist.get(key, [])
                ]
        return active_ports

    def _build_mlp_tree(self, seed: int | None = 0, rngs: nnx.Rngs | None = None) -> dict:
        if rngs is None:
            rngs = nnx.Rngs(seed)
        elif seed is not None:
            raise ValueError("Seed must be None when rngs are provided.")
        mlp_tree: dict = {}

        for key, hyper_edge_set_structure in self.in_graph_structure.hyper_edge_sets.items():
            port_list = hyper_edge_set_structure.port_list
            active_ports = self.active_ports.get(key, [])
            if port_list is None or len(port_list) == 0:
                continue

            in_size = self.in_array_size * len(port_list)
            if hyper_edge_set_structure.feature_list is not None and len(hyper_edge_set_structure.feature_list) > 0:
                if self.encoded_feature_size is not None:
                    in_size += self.encoded_feature_size
                else:
                    in_size += len(hyper_edge_set_structure.feature_list)

            def build_mlp(mlp_out_size: int) -> MLP:
                return MLP(
                    in_size=in_size,
                    hidden_sizes=self.hidden_sizes,
                    activation=self.activation,
                    out_size=mlp_out_size,
                    use_bias=self.use_bias,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    final_activation=self.final_activation,
                    dtype=self.dtype,
                    rngs=rngs,
                )

            if self.fuse_ports:
                if len(active_ports) > 0:
                    mlp_tree[key] = build_mlp(self.out_size * len(active_ports))
            else:
                mlp_tree[key] = {port_key: build_mlp(self.out_size) for port_key in active_ports}
        return nnx.data(mlp_tree)

    def __call__(self, *, graph: Graph, coordinates: jax.Array, step_with_metrics: bool = False) -> tuple[jax.Array, dict]:

        out_dtype = coordinates.dtype
        compute_coordinates = coordinates if self.dtype is None else coordinates.astype(self.dtype)
        scatter_dtype = compute_coordinates.dtype if self.scatter_dtype is None else self.scatter_dtype

        def sum_over_edges(_accumulator, edge_mlp_tuple):
            """Sums the messages predicted by class-specific MLPs through ports of a hyper-edge set."""
            key, hyper_edge_set, mlp_or_dict = edge_mlp_tuple

            input_array = []
            if hyper_edge_set.feature_names is not None:
                feature_array = hyper_edge_set.feature_array
                input_array.append(feature_array if self.dtype is None else feature_array.astype(self.dtype))
            for port_name, port_array in hyper_edge_set.port_dict.items():
                input_array.append(gather(coordinates=compute_coordinates, addresses=port_array))
            input_array = jnp.concatenate(input_array, axis=-1)
            non_fictitious_mask = jnp.expand_dims(hyper_edge_set.non_fictitious, -1)
            if self.dtype is not None:
                non_fictitious_mask = non_fictitious_mask.astype(self.dtype)

            if self.fuse_ports:
                output_array = (mlp_or_dict(input_array * non_fictitious_mask) * non_fictitious_mask).astype(scatter_dtype)
                for i, port_name in enumerate(self.active_ports[key]):
                    increment = output_array[..., i * self.out_size : (i + 1) * self.out_size]
                    _accumulator = scatter_add(
                        accumulator=_accumulator, increment=increment, addresses=hyper_edge_set.port_dict[port_name]
                    )
            else:
                for port_name, mlp in mlp_or_dict.items():
                    increment = (mlp(input_array * non_fictitious_mask) * non_fictitious_mask).astype(scatter_dtype)
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
