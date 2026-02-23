from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from flax.typing import Initializer

from energnn.graph import GraphStructure, JaxGraph
from energnn.model.utils import Activation, MLP, gather, scatter_add


class MessageFunction(nnx.Module, ABC):
    r"""Interface for a message function :math:`\xi_\theta` in a GNN message passing scheme."""

    @abstractmethod
    def __call__(self, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """Should take as input a tuple (graph, coordinates) and return new coordinates."""
        raise NotImplementedError


class LocalSumMessageFunction(MessageFunction):
    r"""
    Local sum-based message function module for GNN message passing.

    This module aggregates messages from each node's local neighborhood by applying
    a class- and port-specific MLP :math:`\xi^{c,o}_\theta` to edge features and neighbor coordinates,
    summing the results across all incoming ports, and applying a final activation :math:`\sigma`.

    For each address :math:`a`, the output is defined as:

    .. math::
        \psi_\theta(h,x)_a = \sigma \left( \sum_{(c,e,o)\in \mathcal{N}_x(a)} \xi^{c,o}_\theta(h_e, x_e)\right),

    where :math:`\xi^{c,o}_\theta` is a class-specific and port-specific MLP, :math:`\sigma` is an
    element-wise activation function, and :math:`h_e := (h_{o(e)})_{o \in {\mathcal{O}^c}}` is the concatenation of
    port coordinates of edge :math:`e`.

    :param in_graph_structure: Input graph structure.
    :param in_array_size: Size of the input coordinate arrays.
    :param hidden_sizes: Hidden sizes of the MLPs :math:`\xi^{c,o}_\theta`.
    :param activation: Activation function for the MLPs :math:`\xi^{c,o}_\theta`.
    :param out_size: Output size of the MLPs :math:`\xi^{c,o}_\theta`.
    :param use_bias: Whether to use bias in the MLPs :math:`\xi^{c,o}_\theta`.
    :param kernel_init: Kernel initializer for the MLPs :math:`\xi^{c,o}_\theta`.
    :param bias_init: Bias initializer for the MLPs :math:`\xi^{c,o}_\theta`.
    :param final_activation: Final activation function for the MLPs :math:`\xi^{c,o}_\theta`.
    :param outer_activation: Activation function :math:`\sigma` applied over the output.
    :param encoded_feature_size: None if the input data has not been encoded, otherwise the size of the encoded features.
    :param port_scatter_blacklist: Dictionary mapping edge keys to lists of port keys to be excluded from the sum.
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
        seed: int = 0,
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

        self.mlp_tree = self._build_mlp_tree(seed=seed)

    def _build_mlp_tree(self, seed: int = 0) -> dict[str, dict[str, MLP]]:
        rngs = nnx.Rngs(seed)
        mlp_tree = {}

        for edge_key, edge_structure in self.in_graph_structure.edges.items():
            if edge_structure.address_list is not None and len(edge_structure.address_list) > 0:
                n_ports = len(edge_structure.address_list)
                in_size = self.in_array_size * n_ports
                if edge_structure.feature_list is not None and len(edge_structure.feature_list) > 0:
                    if self.encoded_feature_size is not None:
                        in_size += self.encoded_feature_size
                    else:
                        in_size += len(edge_structure.feature_list)

                if edge_key not in mlp_tree.keys():
                    mlp_tree[edge_key] = {}

                for port_key in edge_structure.address_list:
                    if port_key not in self.port_scatter_blacklist.get(edge_key, []):
                        mlp_tree[edge_key][port_key] = MLP(
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
        return nnx.data(mlp_tree)

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        def sum_over_edges(_accumulator, edge_mlp_tuple):
            """Sums the output of class and port specific MLPs through ports of all edges in the graph."""
            edge, mlp_dict = edge_mlp_tuple

            input_array = []
            if edge.feature_names is not None:
                input_array.append(edge.feature_array)
            for port_name, port_array in edge.address_dict.items():
                input_array.append(gather(coordinates=coordinates, addresses=port_array))
            input_array = jnp.concatenate(input_array, axis=-1)
            non_fictitious_mask = jnp.expand_dims(edge.non_fictitious, -1)

            def sum_over_ports(__accumulator: jax.Array, mlp_port: tuple[MLP, jax.Array]) -> jax.Array:
                """Sums the outputs of port-specific MLPs through ports of a given edge."""
                mlp, _port_array = mlp_port
                increment = mlp(input_array * non_fictitious_mask) * non_fictitious_mask
                return scatter_add(accumulator=__accumulator, increment=increment, addresses=_port_array)

            mlp_port_dict = {port_name: (mlp, edge.address_dict[port_name]) for port_name, mlp in mlp_dict.items()}
            return jax.tree.reduce(
                sum_over_ports, mlp_port_dict, initializer=_accumulator, is_leaf=lambda x: isinstance(x, tuple)
            )

        initializer = jnp.zeros((coordinates.shape[0], self.out_size))
        edge_mlp_dict = {edge_key: (edge, self.mlp_tree[edge_key]) for edge_key, edge in graph.edges.items()}
        accumulator = jax.tree.reduce(
            sum_over_edges,
            edge_mlp_dict,
            initializer=initializer,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        return self.outer_activation(accumulator), {}


class IdentityMessageFunction(MessageFunction):
    r"""
    Identity local message function module for GNN message passing.

    This module returns the node features unchanged as the local message.
    It implements the identity mapping on node features:

    .. math::
        h^\rightarrow_a = h_a
    """

    def __init__(self):
        pass

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        return coordinates, {}
