# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Device
from jax.tree_util import register_pytree_node_class

from energnn.graph.graph import Graph
from energnn.graph.jax.hyper_edge_set import JaxHyperEdgeSet
from energnn.graph.jax.shape import JaxGraphShape
from energnn.graph.jax.utils import jnp_to_np, np_to_jnp

HYPER_EDGE_SETS = "hyper_edge_sets"
TRUE_SHAPE = "true_shape"
CURRENT_SHAPE = "current_shape"
NON_FICTITIOUS_ADDRESSES = "non_fictitious_addresses"


@register_pytree_node_class
class JaxGraph(dict):
    """
    Jax implementation of Hyper Heterogeneous Multi Graph (H2MG).

    Stores hyper-edge sets, shapes, and address masks for single or batched graphs.

    :param hyper_edge_sets: Dictionary of hyper-edge sets contained in the graph.
    :param true_shape: True shape of the graph, not altered by padding.
    :param current_shape: Current shape of the graph, consistent with padding.
    :param non_fictitious_addresses: Mask filled with ones for real addresses, and zeros otherwise.
    """

    def __init__(
        self,
        *,
        hyper_edge_sets: dict[str, JaxHyperEdgeSet],
        true_shape: JaxGraphShape,
        current_shape: JaxGraphShape,
        non_fictitious_addresses: jax.Array,
    ) -> None:
        super().__init__()
        self[HYPER_EDGE_SETS] = hyper_edge_sets
        self[TRUE_SHAPE] = true_shape
        self[CURRENT_SHAPE] = current_shape
        self[NON_FICTITIOUS_ADDRESSES] = non_fictitious_addresses

    @classmethod
    def from_dict(cls, *, hyper_edge_set_dict: dict[str, JaxHyperEdgeSet], n_addresses: jnp.ndarray) -> Graph:
        """
        Builds a graph from a dictionary of :class:`energnn.graph.JaxHyperEdgeSet` and a registry.

        :param hyper_edge_set_dict: Dictionary of hyper-edge sets contained in the graph.
        :param n_addresses: Number of unique addresses that appear in all the hyper-edge sets.
        :return: Graph that contains both the hyper-edge sets and the registry.
        """
        non_fictitious_addresses = jnp.ones(shape=[n_addresses])
        check_hyper_edge_set_dict_type(hyper_edge_set_dict)
        check_valid_addresses(hyper_edge_set_dict, n_addresses)
        true_shape = JaxGraphShape.from_dict(hyper_edge_set_dict=hyper_edge_set_dict,
                                          non_fictitious=non_fictitious_addresses)
        current_shape = true_shape
        return cls(
            hyper_edge_sets=hyper_edge_set_dict,
            true_shape=true_shape,
            current_shape=current_shape,
            non_fictitious_addresses=non_fictitious_addresses,
        )

    @property
    def true_shape(self) -> JaxGraphShape:
        """
        True shape of the graph with the real number of objects for each hyper-edge set
        class as well as the size of the registry stored in a GraphShape object.
        There is no setter for this property.

        :return: A graph shape of true sizes.
        """
        return self[TRUE_SHAPE]

    @property
    def current_shape(self) -> JaxGraphShape:
        """
        The current shape of the graph taking into accounts fake padding objects.

        :return: A graph shape of current sizes.
        """
        return self[CURRENT_SHAPE]

    def tree_flatten(self):
        """
        Flattens the JaxGraph for JAX PyTree compatibility.

        :returns: Flat children and auxiliary data (the keys order).
        """
        children = self.values()
        aux = self.keys()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> JaxGraph:
        """
        Reconstructs a JaxGraph from flattened data, required for JAX compatibility.

        :param aux_data: Sequence of keys matching the order of the children.
        :param children: Sequence of array values.
        :return: A reconstructed JaxGraph instance.
        """
        d = dict(zip(aux_data, children))
        return cls(
            hyper_edge_sets=d[HYPER_EDGE_SETS],
            true_shape=d[TRUE_SHAPE],
            current_shape=d[CURRENT_SHAPE],
            non_fictitious_addresses=d[NON_FICTITIOUS_ADDRESSES],
        )

    @property
    def hyper_edge_sets(self) -> dict[str, JaxHyperEdgeSet]:
        """
        Gets the dictionary of edge instances.

        :return: Dict of hyper-edge set class to JaxHyperEdgeSet.
        """
        return self[HYPER_EDGE_SETS]

    @hyper_edge_sets.setter
    def hyper_edge_sets(self, hyper_edge_set_dict: dict[str, JaxHyperEdgeSet]) -> None:
        """
        Sets the dictionary of hyper-edge sets.

        :param hyper_edge_set_dict: New dictionary of hyper-edge set instances.
        """
        self[HYPER_EDGE_SETS] = hyper_edge_set_dict

    @property
    def non_fictitious_addresses(self) -> jax.Array:
        """
        Gets the mask filled with ones for real addresses, and zeros otherwise.

        :return: Array filled with ones and zeros.
        """
        return self[NON_FICTITIOUS_ADDRESSES]

    @property
    def feature_flat_array(self) -> jax.Array:
        """
        Returns an array that concatenates all hyper-edge set features.

        :return: Jax array of concatenated features.
        """
        values_list = []
        for key, hyper_edge_set in sorted(self.hyper_edge_sets.items()):
            if hyper_edge_set.feature_flat_array is not None:
                values_list.append(hyper_edge_set.feature_flat_array)
        return jnp.concatenate(values_list, axis=-1)

    @classmethod
    def from_numpy_graph(cls, graph: Graph, device: Device | None = None, dtype: str = "float32") -> JaxGraph:
        """
        Convert a classical numpy graph to a jax.numpy format for GNN processing.

        This method transforms all array-like attributes of a ``Graph`` object into
        their JAX equivalents, allowing efficient use with JAX transformations and accelerators.

        :param graph: A graph object containing NumPy arrays to convert.
        :param device: Optional JAX device (e.g., CPU, GPU) to place the converted arrays on.
                       If None, JAX uses the default device.
        :param dtype: Desired floating-point precision for converted arrays (e.g., "float32", "float64").
        :return: A JAX-compatible version of the graph, ready for use in GNN pipelines.
        """
        hyper_edge_sets = {
            k: JaxHyperEdgeSet.from_numpy_hyper_edge_set(hyper_edge_set, device=device, dtype=dtype)
            for k, hyper_edge_set in graph.hyper_edge_sets.items()
        }
        true_shape = JaxGraphShape.from_numpy_shape(graph.true_shape, device=device, dtype=dtype)
        current_shape = JaxGraphShape.from_numpy_shape(graph.current_shape, device=device, dtype=dtype)
        non_fictitious_addresses = np_to_jnp(graph.non_fictitious_addresses, device=device, dtype=dtype)
        return cls(
            hyper_edge_sets=hyper_edge_sets,
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
        hyper_edge_sets = {k: hyper_edge_set.to_numpy_hyper_edge_set() for k, hyper_edge_set in self.hyper_edge_sets.items()}
        true_shape = self.true_shape.to_numpy_shape()
        current_shape = self.current_shape.to_numpy_shape()
        non_fictitious_addresses = jnp_to_np(self.non_fictitious_addresses)
        return Graph(
            hyper_edge_sets=hyper_edge_sets,
            non_fictitious_addresses=non_fictitious_addresses,
            true_shape=true_shape,
            current_shape=current_shape,
        )

    def quantiles(self, q_list: list[float] | None = None) -> dict[str, jax.Array]:
        """
        Computes quantiles of hyper-edge set features.

        Warning : Assumes that the graph is single and not batched. Will be vmapped.

        :param q_list: Percentiles to compute
        :return: Mapping "hyper-edge set/feature/percentile" to values
        """
        if q_list is None:
            q_list = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]
        info = {}
        for object_name, hyper_edge_set in self.hyper_edge_sets.items():
            if hyper_edge_set.feature_names is not None:
                for feature_name, i in hyper_edge_set.feature_names.items():
                    array = hyper_edge_set.feature_array[..., jnp.array(i, dtype=int)]
                    if jnp.size(array) > 0:
                        for q in q_list:
                            value = jnp.nanpercentile(array, q=q)
                            info[f"{object_name}/{feature_name}/{q}th-percentile"] = value
        return info

    def __str__(self):
        numpy_graph = self.to_numpy_graph()
        return str(numpy_graph)
