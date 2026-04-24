# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import numpy as np
from typing import Any

from jax.tree_util import register_pytree_node_class

from energnn.graph.backend import Backend, NumpyBackend, JaxBackend

HYPER_EDGE_SETS = "hyper_edge_sets"
ADDRESSES = "addresses"


class GraphShape(dict):
    """
    Represents the shape of a graph, including counts of hyper-edge sets per class and registry size.

    This class extends `dict` and maintains two keys:
    - ``HYPER_EDGE_SETS``: dict mapping hyper-edge set class names to count arrays.
    - ``ADDRESSES``: array representing the number of non-fictitious nodes.

    :param hyper_edge_sets: Dictionary of that contains the number of objects for each class.
    :param addresses: Number of addresses in the graph.
    :param backend: Backend used for array operations (defaults to NumpyBackend).
    """

    def __init__(self, *, hyper_edge_sets: dict[str, Any], addresses: Any, backend: Backend | None = None):
        super().__init__()
        self[HYPER_EDGE_SETS] = hyper_edge_sets
        self[ADDRESSES] = addresses
        self._backend = backend or NumpyBackend()

    @classmethod
    def from_dict(cls, hyper_edge_set_dict: dict[str, Any], non_fictitious: Any, backend: Backend | None = None) -> GraphShape:
        """
        Builds a new GraphShape object from a hyper-edge set dictionary and registry.

        :param hyper_edge_set_dict: Mapping from a hyper-edge set class name to a `HyperEdgeSet` instance.
        :param non_fictitious: Optional array whose last dimension indicates registry size.
        :param backend: Backend used for array operations.
        :return: New GraphShape instance.
        """
        backend = backend or NumpyBackend()
        hyper_edge_set_shape_dict = {k: backend.array(v.n_obj) for (k, v) in hyper_edge_set_dict.items()}
        if non_fictitious is not None:
            addresses = backend.array(backend.shape(non_fictitious)[0])
        else:
            addresses = backend.array([0])
        return cls(hyper_edge_sets=hyper_edge_set_shape_dict, addresses=addresses, backend=backend)

    def to_jsonable_dict(self):
        """
        Serialize GraphShape to JSON-friendly dict.

        :return: Dict with 'HyperEdgeSet' mapping to ints and 'addresses' as int.
        """
        return {HYPER_EDGE_SETS: {k: int(v) for k, v in self.hyper_edge_sets.items()}, ADDRESSES: int(self.addresses)}

    @classmethod
    def from_jsonable_dict(cls, count_shape: dict, backend: Backend | None = None) -> GraphShape:
        """
        Deserialize GraphShape from a JSON-friendly dictionary.

        :param count_shape: Dict with 'hyper_edge_sets' and 'addresses'.
        :param backend: Backend used for array operations.
        :return: Reconstructed GraphShape.
        """
        backend = backend or NumpyBackend()
        hyper_edge_sets = {k: backend.array(v) for k, v in count_shape[HYPER_EDGE_SETS].items()}
        addresses = backend.array(count_shape[ADDRESSES])
        return cls(hyper_edge_sets=hyper_edge_sets, addresses=addresses, backend=backend)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GraphShape):
            return False
        if self.keys() != other.keys():
            return False

        # Check addresses
        try:
            if not np.all(self.addresses == other.addresses):
                return False
        except Exception:
            return False

        # Check hyper_edge_sets
        self_hes = self.hyper_edge_sets
        other_hes = other.hyper_edge_sets
        if self_hes.keys() != other_hes.keys():
            return False
        for k in self_hes:
            try:
                if not np.all(self_hes[k] == other_hes[k]):
                    return False
            except Exception:
                return False
        return True

    @classmethod
    def max(cls, a: GraphShape, b: GraphShape) -> GraphShape:
        """
        Returns the maximum shape of 2 graph shapes.

        :param a: A first graph shape.
        :param b: A second graph shape.
        :return: A graph shape with maxima per hyper-edge set class and addresses.
        """
        backend = a._backend
        hyper_edge_set_classes = set(list(a.hyper_edge_sets.keys()) + list(b.hyper_edge_sets.keys()))
        hyper_edge_set_shape_max = {}
        for hyper_edge_set_class in hyper_edge_set_classes:
            val_a = a.hyper_edge_sets.get(hyper_edge_set_class, backend.array(-float("inf")))
            val_b = b.hyper_edge_sets.get(hyper_edge_set_class, backend.array(-float("inf")))
            hyper_edge_set_shape_max[hyper_edge_set_class] = backend.maximum(val_a, val_b)
        addresses = backend.maximum(a.addresses, b.addresses)
        return cls(hyper_edge_sets=hyper_edge_set_shape_max, addresses=addresses, backend=backend)

    @classmethod
    def sum(cls, a: GraphShape, b: GraphShape) -> GraphShape:
        """
        Returns the sum shape of 2 graph shapes.

        :param a: A first graph shape.
        :param b: A second graph shape.
        :return: A graph shape with summed counts per hyper-edge set class and addresses.
        """
        backend = a._backend
        hyper_edge_set_classes = set(list(a.hyper_edge_sets.keys()) + list(b.hyper_edge_sets.keys()))
        hyper_edge_set_shape_sum = {}
        for hyper_edge_set_class in hyper_edge_set_classes:
            hyper_edge_set_shape_sum[hyper_edge_set_class] = a.hyper_edge_sets.get(
                hyper_edge_set_class, 0
            ) + b.hyper_edge_sets.get(hyper_edge_set_class, 0)
        addresses = a.addresses + b.addresses
        return cls(hyper_edge_sets=hyper_edge_set_shape_sum, addresses=addresses, backend=backend)

    @property
    def hyper_edge_sets(self) -> dict[str, Any]:
        """Dictionary of hyper-edge set shapes."""
        return self[HYPER_EDGE_SETS]

    @property
    def addresses(self) -> Any:
        """Registry shape."""
        return self[ADDRESSES]

    @property
    def array(self) -> Any:
        """Concatenated hyper-edge set shapes as a single array."""
        return self._backend.stack([v for v in self.hyper_edge_sets.values()], axis=-1)

    @property
    def is_single(self) -> bool:
        """True if the array is 1-D."""
        return len(self._backend.shape(self.array)) == 1

    @property
    def is_batch(self) -> bool:
        """True if the array is 2-D."""
        return len(self._backend.shape(self.array)) == 2

    @property
    def n_batch(self) -> int:
        """
        Return the batch size.

        :raises ValueError: If GraphShape is not batched.
        """
        if not self.is_batch:
            raise ValueError("GraphShape is not batched.")
        return self._backend.shape(self.array)[0]


@register_pytree_node_class
class JaxGraphShape(GraphShape):
    def __init__(self, *, hyper_edge_sets: dict[str, Any], addresses: Any, backend: Backend | None = None):
        # Ignore provided backend and enforce JAX backend
        super().__init__(hyper_edge_sets=hyper_edge_sets, addresses=addresses, backend=JaxBackend())

    def tree_flatten(self):
        """
        Flatten the JaxGraphShape for JAX PyTree compatibility.

        :returns: Flat children and auxiliary data.
        """
        # Move all arrays to children. hes_values contains the per-edge-set shapes.
        hes_keys = sorted(self.hyper_edge_sets.keys())
        hes_values = tuple(self.hyper_edge_sets[k] for k in hes_keys)

        children = (self.addresses, hes_values)
        aux = (tuple(hes_keys), self._backend)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> JaxGraphShape:
        """
        Reconstruct a JaxGraphShape from flattened data.
        """
        aux_list = list(aux_data)
        if len(aux_list) != 2:
            raise ValueError("aux_data must have 2 elements: (hes_keys, backend)")
        addresses, hes_values = children
        hes_keys, backend = aux_list
        hyper_edge_sets = dict(zip(hes_keys, hes_values))
        return cls(hyper_edge_sets=hyper_edge_sets, addresses=addresses)

    @classmethod
    def from_numpy_shape(cls, shape: GraphShape, device: Any | None = None, dtype: str = "float32") -> JaxGraphShape:
        from energnn.graph.utils import np_to_jnp
        hyper_edge_sets = np_to_jnp(shape.hyper_edge_sets, device=device, dtype=dtype)
        addresses = np_to_jnp(shape.addresses, device=device, dtype=dtype)
        return cls(hyper_edge_sets=hyper_edge_sets, addresses=addresses)

    def to_numpy_shape(self) -> GraphShape:
        from energnn.graph.utils import jnp_to_np
        hyper_edge_sets = jnp_to_np(self.hyper_edge_sets)
        addresses = jnp_to_np(self.addresses)
        return GraphShape(hyper_edge_sets=hyper_edge_sets, addresses=addresses)


def collate_shapes(shape_list: list[GraphShape]) -> GraphShape:
    """
    Batches a list of GraphShape into one batched GraphShape.

    :param shape_list: List of GraphShape objects (must share hyper-edge set keys).
    :return: Batched GraphShape with stacked arrays.
    :raises ValueError: If the input list is empty.
    """
    if not shape_list:
        raise ValueError("Empty shape list provided to collate_shapes.")

    backend = shape_list[0]._backend
    hyper_edge_set_shape_batch = {
        k: backend.stack([s.hyper_edge_sets[k] for s in shape_list], axis=0) for k in shape_list[0].hyper_edge_sets
    }
    addresses_batch = backend.stack([s.addresses for s in shape_list], axis=0)

    if isinstance(shape_list[0], JaxGraphShape):
        return JaxGraphShape(hyper_edge_sets=hyper_edge_set_shape_batch, addresses=addresses_batch)
    return GraphShape(hyper_edge_sets=hyper_edge_set_shape_batch, addresses=addresses_batch, backend=backend)


def separate_shapes(shape_batch: GraphShape) -> list[GraphShape]:
    """
    Splits a batched GraphShape into individual GraphShape instances.

    :param shape_batch: GraphShape with 2D hyper-edge sets and address arrays.
    :return: List of GraphShape (one per batch).
    :raises ValueError: If input is not batched.
    """
    if not shape_batch.is_batch:
        raise ValueError("Input GraphShape must be batched for separation.")

    backend = shape_batch._backend
    addresses_list = backend.unstack(shape_batch.addresses, axis=0)
    a = {k: backend.unstack(shape_batch.hyper_edge_sets[k], axis=0) for k in shape_batch.hyper_edge_sets}

    keys = list(a.keys())
    n_samples = len(addresses_list)
    hyper_edge_set_list = []
    for i in range(n_samples):
        hyper_edge_set_list.append({k: a[k][i] for k in keys})

    shape_list = []
    for addr, e in zip(addresses_list, hyper_edge_set_list):
        if isinstance(shape_batch, JaxGraphShape):
            shape = JaxGraphShape(hyper_edge_sets=e, addresses=addr)
        else:
            shape = GraphShape(hyper_edge_sets=e, addresses=addr, backend=backend)
        shape_list.append(shape)
    return shape_list


def max_shape(graph_shape_list: list[GraphShape]) -> GraphShape:
    """
    Returns the maximum graph shape from a list of graph shapes.

    If some objects do not appear in some shapes, then those objects
    are systematically included in the output.

    :param graph_shape_list: List of graph shapes to be compared.
    :return: GraphShape with maxima per hyper-edge set class and addresses.
    :raises ValueError: If the list is empty or contains non-GraphShape.
    """
    if not graph_shape_list:
        raise ValueError("Empty input list given for max_shape.")

    max_graph_shape = graph_shape_list[0]
    for graph_shape in graph_shape_list:
        if not isinstance(graph_shape, GraphShape):
            raise ValueError("Invalid input in graph_list, expected GraphShape.")
        max_graph_shape = GraphShape.max(max_graph_shape, graph_shape)

    if isinstance(graph_shape_list[0], JaxGraphShape):
        return JaxGraphShape(hyper_edge_sets=max_graph_shape.hyper_edge_sets, addresses=max_graph_shape.addresses)
    return max_graph_shape


def sum_shapes(graph_shape_list: list[GraphShape]) -> GraphShape:
    """
    Returns the sum graph shape from a list of graph shapes.

    :param graph_shape_list: List of graph shapes to be summed.
    :return: GraphShape with summed counts per hyper-edge set class and addresses.
    :raises ValueError: If the list is empty or contains non-GraphShape.
    """
    if not graph_shape_list:
        raise ValueError("Empty input list given for sum_shapes.")

    sum_graph_shape = graph_shape_list[0]
    for graph_shape in graph_shape_list[1:]:
        if not isinstance(graph_shape, GraphShape):
            raise ValueError("Invalid input in graph_list, expected GraphShape.")
        sum_graph_shape = GraphShape.sum(sum_graph_shape, graph_shape)

    if isinstance(graph_shape_list[0], JaxGraphShape):
        return JaxGraphShape(hyper_edge_sets=sum_graph_shape.hyper_edge_sets, addresses=sum_graph_shape.addresses)
    return sum_graph_shape
