# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import numpy as np

from energnn.graph.hyper_edge_set import HyperEdgeSet

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
    """

    def __init__(self, *, hyper_edge_sets: dict[str, np.ndarray], addresses: np.ndarray):
        super().__init__()
        self[HYPER_EDGE_SETS] = hyper_edge_sets
        self[ADDRESSES] = addresses

    @classmethod
    def from_dict(cls, hyper_edge_set_dict: dict[str, HyperEdgeSet], non_fictitious: np.ndarray) -> GraphShape:
        """
        Builds a new GraphShape object from a hyper-edge set dictionary and registry.

        :param hyper_edge_set_dict: Mapping from a hyper-edge set class name to a `HyperEdgeSet` instance.
        :param non_fictitious: Optional numpy array whose last dimension indicates registry size.
        :return: New GraphShape instance.
        """
        hyper_edge_set_shape_dict = {k: np.array(v.n_obj) for (k, v) in hyper_edge_set_dict.items()}
        if non_fictitious is not None:
            addresses = np.array(non_fictitious.shape[0])
        else:
            addresses = np.array([0])
        return cls(hyper_edge_sets=hyper_edge_set_shape_dict, addresses=addresses)

    def to_jsonable_dict(self):
        """
        Serialize GraphShape to JSON-friendly dict.

        :return: Dict with 'HyperEdgeSet' mapping to ints and 'addresses' as int.
        """
        return {HYPER_EDGE_SETS: {k: int(v) for k, v in self.hyper_edge_sets.items()}, ADDRESSES: int(self.addresses)}

    @classmethod
    def from_jsonable_dict(cls, count_shape: dict) -> GraphShape:
        """
        Deserialize GraphShape from a JSON-friendly dictionary.

        :param count_shape: Dict with 'hyper_edge_sets' and 'addresses'.
        :return: Reconstructed GraphShape.
        """
        hyper_edge_sets = {k: np.array(v) for k, v in count_shape[HYPER_EDGE_SETS].items()}
        addresses = np.array(count_shape[ADDRESSES])
        return cls(hyper_edge_sets=hyper_edge_sets, addresses=addresses)

    @classmethod
    def max(cls, a: GraphShape, b: GraphShape) -> GraphShape:
        """
        Returns the maximum shape of 2 graph shapes.

        :param a: A first graph shape.
        :param b: A second graph shape.
        :return: A graph shape with maxima per hyper-edge set class and addresses.
        """
        hyper_edge_set_classes = set(list(a.hyper_edge_sets.keys()) + list(b.hyper_edge_sets.keys()))
        hyper_edge_set_shape_max = {}
        for hyper_edge_set_class in hyper_edge_set_classes:
            hyper_edge_set_shape_max[hyper_edge_set_class] = np.maximum(
                a.hyper_edge_sets.get(hyper_edge_set_class, -np.inf), b.hyper_edge_sets.get(hyper_edge_set_class, -np.inf)
            )
        addresses = np.maximum(a.addresses, b.addresses)
        return cls(hyper_edge_sets=hyper_edge_set_shape_max, addresses=addresses)

    @classmethod
    def sum(cls, a: GraphShape, b: GraphShape) -> GraphShape:
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

    @property
    def hyper_edge_sets(self) -> dict[str, np.ndarray]:
        """Dictionary of hyper-edge set shapes."""
        return self[HYPER_EDGE_SETS]

    @property
    def addresses(self) -> np.ndarray:
        """Registry shape."""
        return self[ADDRESSES]

    @property
    def array(self) -> np.ndarray:
        """Concatenated hyper-edge set shapes as a single array."""
        return np.stack([v for v in self.hyper_edge_sets.values()], axis=-1)

    @property
    def is_single(self) -> bool:
        """True if the array is 1-D."""
        return len(self.array.shape) == 1

    @property
    def is_batch(self) -> bool:
        """True if the array is 2-D."""
        return len(self.array.shape) == 2

    @property
    def n_batch(self) -> int:
        """
        Return the batch size.

        :raises ValueError: If GraphShape is not batched.
        """
        if not self.is_batch:
            raise ValueError("GraphShape is not batched.")
        return self.array.shape[0]


def collate_shapes(shape_list: list[GraphShape]) -> GraphShape:
    """
    Batches a list of GraphShape into one batched GraphShape.

    :param shape_list: List of GraphShape objects (must share hyper-edge set keys).
    :return: Batched GraphShape with stacked arrays.
    :raises ValueError: If the input list is empty.
    """
    if not shape_list:
        raise ValueError("Empty shape list provided to collate_shapes.")

    hyper_edge_set_shape_batch = {
        k: np.stack([s.hyper_edge_sets[k] for s in shape_list], axis=0) for k in shape_list[0].hyper_edge_sets
    }
    addresses_batch = np.stack([s.addresses for s in shape_list], axis=0)
    return GraphShape(hyper_edge_sets=hyper_edge_set_shape_batch, addresses=addresses_batch)


def separate_shapes(shape_batch: GraphShape) -> list[GraphShape]:
    """
    Splits a batched GraphShape into individual GraphShape instances.

    :param shape_batch: GraphShape with 2D hyper-edge sets and address arrays.
    :return: List of GraphShape (one per batch).
    :raises ValueError: If input is not batched.
    """
    if not shape_batch.is_batch:
        raise ValueError("Input GraphShape must be batched for separation.")

    addresses_list = np.unstack(shape_batch.addresses, axis=0)
    a = {k: np.unstack(shape_batch.hyper_edge_sets[k]) for k in shape_batch.hyper_edge_sets}
    hyper_edge_set_list = [dict(zip(a, t)) for t in zip(*a.values())]

    shape_list = []
    for a, e in zip(addresses_list, hyper_edge_set_list):
        shape = GraphShape(hyper_edge_sets=e, addresses=a)
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
    return sum_graph_shape
