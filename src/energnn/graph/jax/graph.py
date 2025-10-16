#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import Device
from jax.tree_util import register_pytree_node_class

from energnn.graph.graph import Graph
from energnn.graph.jax.edge import JaxEdge
from energnn.graph.jax.shape import JaxGraphShape
from energnn.graph.jax.utils import jnp_to_np, np_to_jnp

EDGES = "edges"
TRUE_SHAPE = "true_shape"
CURRENT_SHAPE = "current_shape"
NON_FICTITIOUS_ADDRESSES = "non_fictitious_addresses"


@register_pytree_node_class
class JaxGraph(dict):
    """
    Jax implementation of Hyper Heterogeneous Multi Graph (H2MG).

    Store edges, shapes, and address masks for single or batched graphs.

    :param edges: Dictionary of edges contained in the graph.
    :param true_shape: True shape of the graph, not altered by padding.
    :param current_shape: Current shape of the graph, consistent with padding.
    :param non_fictitious_addresses: Mask filled with ones for real addresses, and zeros otherwise.
    """

    def __init__(
        self,
        *,
        edges: dict[str, JaxEdge],
        true_shape: JaxGraphShape,
        current_shape: JaxGraphShape,
        non_fictitious_addresses: jax.Array,
    ) -> None:
        super().__init__()
        self[EDGES] = edges
        self[TRUE_SHAPE] = true_shape
        self[CURRENT_SHAPE] = current_shape
        self[NON_FICTITIOUS_ADDRESSES] = non_fictitious_addresses

    @property
    def true_shape(self) -> JaxGraphShape:
        """
        True shape of the graph with the real number of objects for each edge
        class as well as the size of the registry stored in a GraphShape object.
        There is no setter for this property.

        :return: a graph shape of true sizes
        """
        return self[TRUE_SHAPE]

    @property
    def current_shape(self) -> JaxGraphShape:
        """
        Current shape of the graph taking into accounts fake padding objects.

        :return: a graph shape of current sizes
        """
        return self[CURRENT_SHAPE]

    def tree_flatten(self):
        """
        Flatten the JaxGraph for JAX PyTree compatibility.

        :returns: flat children and auxiliary data (the keys order).
        """
        children = self.values()
        aux = self.keys()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> JaxGraph:
        """
        Reconstruct a JaxGraph from flattened data, required for JAX compatibility.

        :param aux_data: sequence of keys matching children order.
        :param children: sequence of array values.
        :return: a reconstructed JaxGraph instance.
        """
        d = dict(zip(aux_data, children))
        return cls(
            edges=d[EDGES],
            true_shape=d[TRUE_SHAPE],
            current_shape=d[CURRENT_SHAPE],
            non_fictitious_addresses=d[NON_FICTITIOUS_ADDRESSES],
        )

    @property
    def edges(self) -> dict[str, JaxEdge]:
        """
        Get dictionary of edge instances.

        :return: dict of edge class to Edge
        """
        return self[EDGES]

    @edges.setter
    def edges(self, edge_dict: dict[str, JaxEdge]) -> None:
        """
        Set dictionary of edge instances.

        :param edge_dict: new dictionary of edge instances
        """
        self[EDGES] = edge_dict

    @property
    def non_fictitious_addresses(self) -> jax.Array:
        """
        Get mask filled with ones for real addresses, and zeros otherwise

        :return: array filled with ones and zeros
        """
        return self[NON_FICTITIOUS_ADDRESSES]

    @property
    def feature_flat_array(self) -> jax.Array:
        """
        Returns an array that concatenates all edge features.

        :return: jax array of concatenated features
        """
        values_list = []
        for key, edge in sorted(self.edges.items()):
            if edge.feature_flat_array is not None:
                values_list.append(edge.feature_flat_array)
        return jnp.concatenate(values_list, axis=-1)

    @classmethod
    def from_numpy_graph(cls, graph: Graph, device: Device | None = None, dtype: str = "float32") -> JaxGraph:
        """
        Convert a classical numpy graph to a jax.numpy format for GNN processing.

        This method transforms all array-like attributes of an ``Graph`` object into
        their JAX equivalents, allowing efficient use with JAX transformations and accelerators.

        :param graph: A graph object containing NumPy arrays to convert.
        :param device: Optional JAX device (e.g., CPU, GPU) to place the converted arrays on.
                       If None, JAX uses the default device.
        :param dtype: Desired floating-point precision for converted arrays (e.g., "float32", "float64").
        :return: A JAX-compatible version of the graph, ready for use in GNN pipelines.
        """
        edge_dict = {k: JaxEdge.from_numpy_edge(edge, device=device, dtype=dtype) for k, edge in graph.edges.items()}
        true_shape = JaxGraphShape.from_numpy_shape(graph.true_shape, device=device, dtype=dtype)
        current_shape = JaxGraphShape.from_numpy_shape(graph.current_shape, device=device, dtype=dtype)
        non_fictitious_addresses = np_to_jnp(graph.non_fictitious_addresses, device=device, dtype=dtype)
        return cls(
            edges=edge_dict,
            non_fictitious_addresses=non_fictitious_addresses,
            true_shape=true_shape,
            current_shape=current_shape,
        )

    def to_numpy_graph(self) -> Graph:
        """
        Convert a jax.numpy graph for GNN processing to a classical numpy graph.

        This method transforms the internal JAX arrays of the graph back into standard
        NumPy arrays, enabling compatibility with non-JAX components.

        :return: A classical ``Graph`` object with NumPy arrays.
        """
        edge_dict = {k: edge.to_numpy_edge() for k, edge in self.edges.items()}
        true_shape = self.true_shape.to_numpy_shape()
        current_shape = self.current_shape.to_numpy_shape()
        non_fictitious_addresses = jnp_to_np(self.non_fictitious_addresses)
        return Graph(
            edges=edge_dict,
            non_fictitious_addresses=non_fictitious_addresses,
            true_shape=true_shape,
            current_shape=current_shape,
        )

    def quantiles(self, q_list: list[float] = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]) -> dict[str, jax.Array]:
        """
        Computes quantiles of edge features.

        Warning : assumes that the graph is single, and not batched. Will be vmapped.

        :param q_list: percentiles to compute
        :return: mapping "edge/feature/percentile" to values
        """
        info = {}
        for object_name, edge in self.edges.items():
            if edge.feature_names is not None:
                for feature_name, i in edge.feature_names.items():
                    array = edge.feature_array[..., jnp.array(i, dtype=int)]
                    if jnp.size(array) > 0:
                        for q in q_list:
                            value = jnp.nanpercentile(array, q=q)
                            info[f"{object_name}/{feature_name}/{q}th-percentile"] = value
        return info

    # @feature_flat_array.setter  # TODO : voir si on arrive à s'en débarasser
    # def feature_flat_array(self, value: jax.Array) -> None:
    #     """Updates the flat array contained in the H2MG."""
    #     if jnp.any(self.feature_flat_array.shape != value.shape):
    #         raise ValueError("Invalid array shape.")
    #     i = 0
    #     if self.edges is not None:
    #         for key, edge in sorted(self.edges.items()):
    #             if edge.feature_names is not None:
    #                 length = jnp.shape(edge.feature_flat_array)[-1]
    #                 if length > 0:
    #                     self.edges[key].feature_flat_array = value[..., i : i + length]  # Slice over last axis
    #                     i += length
    #     else:
    #         raise ValueError("This graph does not contain any edge, and can't be cast as a flat array.")
