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

from energnn.graph.jax.utils import jnp_to_np, np_to_jnp
from energnn.graph.shape import GraphShape
from energnn.graph.jax.hyper_edge_set import JaxHyperEdgeSet

HYPER_EDGE_SETS = "hyper_edge_sets"
ADDRESSES = "addresses"


@register_pytree_node_class
class JaxGraphShape(dict):
    """
    PyTree container for storing the number of objects in each class, and addresses in the graph.

    This class inherits from `dict` and stores two keys:
    :param hyper_edge_sets: Dictionary of that contains the number of objects for each class.
    :param addresses: Number of addresses in the graph.

    The PyTree methods ``tree_flatten`` and ``tree_unflatten`` make this object
    compatible with JAX transformations (jit, vmap, etc.).
    """

    def __init__(self, *, hyper_edge_sets: dict[str, jax.Array], addresses: jax.Array):
        super().__init__()
        self[HYPER_EDGE_SETS] = hyper_edge_sets
        self[ADDRESSES] = addresses

    @classmethod
    def from_dict(cls, hyper_edge_set_dict: dict[str, JaxHyperEdgeSet], non_fictitious: jax.Array) -> JaxGraphShape:
        """
        Builds a new JaxGraphShape object from a hyper-edge set dictionary and registry.

        :param hyper_edge_set_dict: Mapping from a hyper-edge set class name to a `JaxHyperEdgeSet` instance.
        :param non_fictitious: Optional numpy array whose last dimension indicates registry size.
        :return: New JaxGraphShape instance.
        """
        hyper_edge_set_shape_dict = {k: jnp.array(v.n_obj) for (k, v) in hyper_edge_set_dict.items()}
        if non_fictitious is not None:
            addresses = jnp.array(non_fictitious.shape[0])
        else:
            addresses = jnp.array([0])
        return cls(hyper_edge_sets=hyper_edge_set_shape_dict, addresses=addresses)

    def to_jsonable_dict(self):
        """
        Serialize JaxGraphShape to JSON-friendly dict.

        :return: Dict with 'JaxHyperEdgeSet' mapping to ints and 'addresses' as int.
        """
        return {HYPER_EDGE_SETS: {k: int(v) for k, v in self.hyper_edge_sets.items()}, ADDRESSES: int(self.addresses)}

    @classmethod
    def from_jsonable_dict(cls, count_shape: dict) -> JaxGraphShape:
        """
        Deserialize JaxGraphShape from a JSON-friendly dictionary.

        :param count_shape: Dict with 'hyper_edge_sets' and 'addresses'.
        :return: Reconstructed JaxGraphShape.
        """
        hyper_edge_sets = {k: jnp.array(v) for k, v in count_shape[HYPER_EDGE_SETS].items()}
        addresses = jnp.array(count_shape[ADDRESSES])
        return cls(hyper_edge_sets=hyper_edge_sets, addresses=addresses)

    @classmethod
    def max(cls, a: JaxGraphShape, b: JaxGraphShape) -> JaxGraphShape:
        """
        Returns the maximum shape of 2 graph shapes.

        :param a: A first graph shape.
        :param b: A second graph shape.
        :return: A graph shape with maxima per hyper-edge set class and addresses.
        """
        hyper_edge_set_classes = set(list(a.hyper_edge_sets.keys()) + list(b.hyper_edge_sets.keys()))
        hyper_edge_set_shape_max = {}
        for hyper_edge_set_class in hyper_edge_set_classes:
            hyper_edge_set_shape_max[hyper_edge_set_class] = jnp.maximum(
                a.hyper_edge_sets.get(hyper_edge_set_class, -jnp.inf), b.hyper_edge_sets.get(hyper_edge_set_class, -jnp.inf)
            )
        addresses = jnp.maximum(a.addresses, b.addresses)
        return cls(hyper_edge_sets=hyper_edge_set_shape_max, addresses=addresses)

    @classmethod
    def sum(cls, a: JaxGraphShape, b: JaxGraphShape) -> JaxGraphShape:
        """
        Returns the sum shape of 2 graph shapes.

        :param a: A first graph shape.
        :param b: A second graph shape.
        :return: A graph shape with summed counts per hyper-edge set class and addresses.
        """
        hyper_edge_set_classes = set(list(a.hyper_edge_sets.keys()) + list(b.hyper_edge_sets.keys()))
        hyper_edge_set_shape_max = {}
        for hyper_edge_set_class in hyper_edge_set_classes:
            hyper_edge_set_shape_max[hyper_edge_set_class] = a.hyper_edge_sets.get(
                hyper_edge_set_class, 0
            ) + b.hyper_edge_sets.get(hyper_edge_set_class, 0)
        addresses = a.addresses + b.addresses
        return cls(hyper_edge_sets=hyper_edge_set_shape_max, addresses=addresses)

    def tree_flatten(self):
        """
        Flatten the JaxGraphShape for JAX PyTree compatibility.

        :returns: Flat children and auxiliary data (the keys order).
        """
        children = self.values()
        aux = self.keys()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> JaxGraphShape:
        """
        Reconstruct a JaxGraphShape from flattened data, required for JAX compatibility.

        :param aux_data: Sequence of keys matching the order of the children.
        :param children: Sequence of array values.
        :return: A reconstructed JaxGraphShape instance.
        """
        d = dict(zip(aux_data, children))
        return cls(hyper_edge_sets=d[HYPER_EDGE_SETS], addresses=d[ADDRESSES])

    @property
    def hyper_edge_sets(self) -> dict[str, jax.Array]:
        """Dictionary of edge shapes."""
        return self[HYPER_EDGE_SETS]

    @property
    def addresses(self) -> jax.Array:
        """Number of addresses in the graph."""
        return self[ADDRESSES]

    @classmethod
    def from_numpy_shape(cls, shape: GraphShape, device: Device | None = None, dtype: str = "float32") -> JaxGraphShape:
        """
        Convert a classical numpy shape to a jax.numpy format for GNN processing.

        This method transforms all array-like attributes of a ``GraphShape`` object into
        their JAX equivalents, allowing efficient use with JAX transformations and accelerators.

        :param shape: A shape object containing NumPy arrays to convert.
        :param device: Optional JAX device (e.g., CPU, GPU) to place the converted arrays on.
                       If None, JAX uses the default device.
        :param dtype: Desired floating-point precision for converted arrays (e.g., "float32", "float64").
        :return: A JAX-compatible version of the shape, ready for use in GNN pipelines.
        """
        hyper_edge_sets = np_to_jnp(shape.hyper_edge_sets, device=device, dtype=dtype)
        addresses = np_to_jnp(shape.addresses, device=device, dtype=dtype)
        return cls(hyper_edge_sets=hyper_edge_sets, addresses=addresses)

    def to_numpy_shape(self) -> GraphShape:
        """
        Convert a jax.numpy shape for GNN processing to a classical numpy shape.

        This method transforms the internal JAX arrays of the shape back into standard
        NumPy arrays, enabling compatibility with non-JAX components.

        :return: A classical ``GraphShape`` object with NumPy arrays.
        """
        hyper_edge_sets = jnp_to_np(self.hyper_edge_sets)
        addresses = jnp_to_np(self.addresses)
        return GraphShape(hyper_edge_sets=hyper_edge_sets, addresses=addresses)

    @property
    def array(self) -> jax.Array:
        """Concatenated hyper-edge set shapes as a single jax array."""
        return jnp.stack([v for v in self.hyper_edge_sets.values()], axis=-1)

    @property
    def is_single(self) -> bool:
        """True if the jax array is 1-D."""
        return len(self.array.shape) == 1

    @property
    def is_batch(self) -> bool:
        """True if the jax array is 2-D."""
        return len(self.array.shape) == 2

    @property
    def n_batch(self) -> int:
        """
        Return the batch size.

        :raises ValueError: If JaxGraphShape is not batched.
        """
        if not self.is_batch:
            raise ValueError("JaxGraphShape is not batched.")
        return self.array.shape[0]

def collate_shapes_jax(shape_list: list[JaxGraphShape]) -> JaxGraphShape:
    """
    Batches a list of JaxGraphShape into one batched JaxGraphShape.

    :param shape_list: List of JaxGraphShape objects (must share hyper-edge set keys).
    :return: Batched JaxGraphShape with stacked arrays.
    :raises ValueError: If the input list is empty.
    """
    if not shape_list:
        raise ValueError("Empty shape list provided to collate_shapes.")

    hyper_edge_set_shape_batch = {
        k: jnp.stack([s.hyper_edge_sets[k] for s in shape_list], axis=0) for k in shape_list[0].hyper_edge_sets
    }
    addresses_batch = jnp.stack([s.addresses for s in shape_list], axis=0)
    return JaxGraphShape(hyper_edge_sets=hyper_edge_set_shape_batch, addresses=addresses_batch)


def separate_shapes_jax(shape_batch: JaxGraphShape) -> list[JaxGraphShape]:
    """
    Splits a batched JaxGraphShape into individual JaxGraphShape instances.

    :param shape_batch: JaxGraphShape with 2D hyper-edge sets and address arrays.
    :return: List of JaxGraphShape (one per batch).
    :raises ValueError: If input is not batched.
    """
    if not shape_batch.is_batch:
        raise ValueError("Input JaxGraphShape must be batched for separation.")

    addresses_list = jnp.unstack(shape_batch.addresses, axis=0)
    a = {k: jnp.unstack(shape_batch.hyper_edge_sets[k]) for k in shape_batch.hyper_edge_sets}
    hyper_edge_set_list = [dict(zip(a, t)) for t in zip(*a.values())]

    shape_list = []
    for a, e in zip(addresses_list, hyper_edge_set_list):
        shape = JaxGraphShape(hyper_edge_sets=e, addresses=a)
        shape_list.append(shape)
    return shape_list


def max_shape_jax(graph_shape_list: list[JaxGraphShape]) -> JaxGraphShape:
    """
    Returns the maximum jax graph shape from a list of jax graph shapes.

    If some objects do not appear in some shapes, then those objects
    are systematically included in the output.

    :param graph_shape_list: List of jax graph shapes to be compared.
    :return: JaxGraphShape with maxima per hyper-edge set class and addresses.
    :raises ValueError: If the list is empty or contains non-JaxGraphShape.
    """
    if not graph_shape_list:
        raise ValueError("Empty input list given for max_shape.")

    max_graph_shape = graph_shape_list[0]
    for graph_shape in graph_shape_list:
        if not isinstance(graph_shape, JaxGraphShape):
            raise ValueError("Invalid input in graph_list, expected JaxGraphShape.")
        max_graph_shape = JaxGraphShape.max(max_graph_shape, graph_shape)
    return max_graph_shape


def sum_shapes_jax(graph_shape_list: list[JaxGraphShape]) -> JaxGraphShape:
    """
    Returns the sum jax graph shape from a list of jax graph shapes.

    :param graph_shape_list: List of jax graph shapes to be summed.
    :return: JaxGraphShape with summed counts per hyper-edge set class and addresses.
    :raises ValueError: If the list is empty or contains non-JaxGraphShape.
    """
    if not graph_shape_list:
        raise ValueError("Empty input list given for sum_shapes.")

    sum_graph_shape = graph_shape_list[0]
    for graph_shape in graph_shape_list[1:]:
        if not isinstance(graph_shape, JaxGraphShape):
            raise ValueError("Invalid input in graph_list, expected JaxGraphShape.")
        sum_graph_shape = JaxGraphShape.sum(sum_graph_shape, graph_shape)
    return sum_graph_shape