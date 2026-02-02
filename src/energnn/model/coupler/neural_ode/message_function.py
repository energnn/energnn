from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from energnn.graph import JaxGraph
from energnn.model.utils import MLP, gather, scatter_add


class MessageFunction(nnx.Module, ABC):
    r"""Interface for a message function :math:`\xi_\theta` in a GNN message passing scheme.

    It should take as input a tuple (graph, coordinates) and return new coordinates.
    """

    @abstractmethod
    def __call__(self, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        raise NotImplementedError


class LocalSumMessageFunction(MessageFunction):
    r"""
    Local sum-based message function module for GNN message passing.

    This module aggregates messages from each node's local neighborhood by applying
    a class- and port-specific MLP :math:`\xi^{c,o}_\theta` to edge features and neighbor coordinates,
    summing the results across all incoming ports, and applying a final activation :math:`\sigma`.

    For each address :math:`a`, the output is defined as:

    .. math::
        h'_a = \sigma \left( \sum_{(c,e,o)\in \mathcal{N}_x(a)} \xi^{c,o}_\theta(h_e, x_e)\right),

    where :math:`\xi^{c,o}_\theta` is a class-specific and port-specific MLP, and :math:`\sigma` is an
    element-wise activation function.

    :param hidden_size: Hidden size of the MLPs :math:`\xi^{c,o}_\theta`.
    :param activation: Activation function for the MLPs :math:`\xi^{c,o}_\theta`.
    :param out_size: Local message size.
    :param final_activation: Activation function :math:`\sigma` applied over the output.
    """

    def __init__(
        self,
        out_size: int,
        hidden_size: list[int],
        activation: Callable[[jax.Array], jax.Array],
        final_activation: Callable[[jax.Array], jax.Array],
        seed: int = 0,
    ):
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.final_activation = final_activation
        self.mlp_tree: dict = {}
        # self.mlp_tree: nnx.Dict = nnx.Dict()

        self.rngs = nnx.Rngs(seed)

    def _build_missing_mlps(self, graph: JaxGraph, coordinates: jax.Array):
        """Creates MLPs to the mlp tree for each of the `(edge, port)` pairs appearing in the graph."""
        for edge_key, edge in graph.edges.items():

            n_ports = len(edge.address_dict)
            latent_dim = coordinates.shape[-1]
            in_size = latent_dim * n_ports
            if edge.feature_array is not None:
                n_features = edge.feature_array.shape[-1]
                in_size += n_features

            if edge_key not in self.mlp_tree.keys():
                self.mlp_tree[edge_key]: dict = {}
                # self.mlp_tree[edge_key]: nnx.Dict = nnx.Dict()
            for port_key in edge.address_dict.keys():
                if port_key not in self.mlp_tree[edge_key].keys():
                    self.mlp_tree[edge_key][port_key] = MLP(
                        # in_size=in_size,
                        hidden_size=self.hidden_size,
                        out_size=self.out_size,
                        activation=self.activation,
                        rngs=self.rngs,
                    )

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        # Add missing MLPs to the mlp tree.
        self._build_missing_mlps(graph=graph, coordinates=coordinates)

        def sum_over_edges(accumulator, edge_mlp_tuple):
            """Sums the output of class and port specific MLPs through ports of all edges in the graph."""
            edge, mlp_dict = edge_mlp_tuple

            input_array = []
            if edge.feature_names is not None:
                input_array.append(edge.feature_array)
            for port_name, port_array in edge.address_dict.items():
                input_array.append(gather(coordinates=coordinates, addresses=port_array))
            input_array = jnp.concatenate(input_array, axis=-1)
            non_fictitious_mask = jnp.expand_dims(edge.non_fictitious, -1)

            def sum_over_ports(accumulator: jax.Array, mlp_port: tuple[MLP, jax.Array]) -> jax.Array:
                """Sums the outputs of port specific MLPs through ports of a given edge."""
                mlp, port_array = mlp_port
                increment = mlp(input_array) * non_fictitious_mask
                return scatter_add(accumulator=accumulator, increment=increment, addresses=edge.address_dict[port_name])

            mlp_port_dict = {port_name: (mlp, edge.address_dict[port_name]) for port_name, mlp in mlp_dict.items()}
            accumulator = jax.tree.reduce(
                sum_over_ports, mlp_port_dict, initializer=accumulator, is_leaf=lambda x: isinstance(x, tuple)
            )
            return accumulator

        initializer = jnp.zeros((coordinates.shape[0], self.out_size))
        edge_mlp_dict = {edge_key: (edge, self.mlp_tree[edge_key]) for edge_key, edge in graph.edges.items()}
        accumulator = jax.tree.reduce(
            sum_over_edges,
            edge_mlp_dict,
            initializer=initializer,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        return self.final_activation(accumulator), {}


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
