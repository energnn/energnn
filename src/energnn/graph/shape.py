#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from __future__ import annotations

import numpy as np

from energnn.graph.edge import Edge

EDGES = "edges"
ADDRESSES = "addresses"


class GraphShape(dict):
    """
    Represents the shape of a graph, including counts of edges per class and registry size.

    This class extends `dict` and maintains two keys:
    - ``EDGES``: dict mapping edge class names to count arrays.
    - ``ADDRESSES``: array representing number of non-fictitious nodes.

    :param edges: Dictionary of that contains the number of objects for each class.
    :param addresses: Number of addresses in the graph.
    """

    def __init__(self, *, edges: dict[str, np.ndarray], addresses: np.ndarray):
        super().__init__()
        self[EDGES] = edges
        self[ADDRESSES] = addresses

    @classmethod
    def from_dict(cls, edge_dict: dict[str, Edge], non_fictitious: np.ndarray) -> GraphShape:
        """
        Builds a new GraphShape object from edge dictionary and registry.

        :param edge_dict: mapping from edge class name to `Edge` instance.
        :param non_fictitious: optional numpy array whose last dimension indicates registry size.
        :return: new GraphShape instance.
        """
        edge_shape_dict = {k: np.array(v.n_obj) for (k, v) in edge_dict.items()}
        if non_fictitious is not None:
            addresses = np.array(non_fictitious.shape[0])
        else:
            addresses = np.array([0])
        return cls(edges=edge_shape_dict, addresses=addresses)

    def to_jsonable_dict(self):
        """
        Serialize GraphShape to JSON-friendly dict.

        :return: dict with 'edges' mapping to ints and 'addresses' as int.
        """
        return {EDGES: {k: int(v) for k, v in self.edges.items()}, ADDRESSES: int(self.addresses)}

    @classmethod
    def from_jsonable_dict(cls, count_shape: dict) -> GraphShape:
        """
        Deserialize GraphShape from JSON-friendly dict.

        :param count_shape: dict with 'edges' and 'addresses'.
        :return: Reconstructed GraphShape.
        """
        edges = {k: np.array(v) for k, v in count_shape[EDGES].items()}
        addresses = np.array(count_shape[ADDRESSES])
        return cls(edges=edges, addresses=addresses)

    @classmethod
    def max(cls, a: GraphShape, b: GraphShape) -> GraphShape:
        """
        Returns the maximum shape of 2 graph shapes.

        :param a: first graph shape
        :param b: second graph shape
        :return: a graph shape with maxima per edge class and addresses
        """
        edge_classes = set(list(a.edges.keys()) + list(b.edges.keys()))
        edge_shape_max = {}
        for edge_class in edge_classes:
            edge_shape_max[edge_class] = np.maximum(a.edges.get(edge_class, -np.inf), b.edges.get(edge_class, -np.inf))
        addresses = np.maximum(a.addresses, b.addresses)
        return cls(edges=edge_shape_max, addresses=addresses)

    @classmethod
    def sum(cls, a: GraphShape, b: GraphShape) -> GraphShape:
        """
        Returns the sum shape of 2 graph shapes.

        :param a: first graph shape
        :param b: second graph shape
        :return: a graph shape with summed counts per edge class and addresses
        """
        edge_classes = set(list(a.edges.keys()) + list(b.edges.keys()))
        edge_shape_max = {}
        for edge_class in edge_classes:
            edge_shape_max[edge_class] = a.edges.get(edge_class, 0) + b.edges.get(edge_class, 0)
        addresses = a.addresses + b.addresses
        return cls(edges=edge_shape_max, addresses=addresses)

    @property
    def edges(self) -> dict[str, np.ndarray]:
        """Dictionary of edge shapes."""
        return self[EDGES]

    @property
    def addresses(self) -> np.ndarray:
        """Registry shape."""
        return self[ADDRESSES]

    @property
    def array(self) -> np.ndarray:
        """Concatenated edge shapes as a single array."""
        return np.stack([v for v in self.edges.values()], axis=-1)

    @property
    def is_single(self) -> bool:
        """True if array is 1-D:"""
        return len(self.array.shape) == 1

    @property
    def is_batch(self) -> bool:
        """True if array is 2-D:"""
        return len(self.array.shape) == 2

    @property
    def n_batch(self) -> int:
        """
        Return the batch size

        :raises ValueError: If GraphShape is not batched
        """
        if not self.is_batch:
            raise ValueError("GraphShape is not batched.")
        return self.array.shape[0]


def collate_shapes(shape_list: list[GraphShape]) -> GraphShape:
    """
    Batch a list of GraphShape into one batched GraphShape.

    :param shape_list: list of GraphShape objects (must share edge keys)
    :return: batched GraphShape with stacked arrays
    :raises ValueError: if input list is empty
    """
    if not shape_list:
        raise ValueError("Empty shape list provided to collate_shapes.")

    edge_shape_batch = {k: np.stack([s.edges[k] for s in shape_list], axis=0) for k in shape_list[0].edges}
    addresses_batch = np.stack([s.addresses for s in shape_list], axis=0)
    return GraphShape(edges=edge_shape_batch, addresses=addresses_batch)


def separate_shapes(shape_batch: GraphShape) -> list[GraphShape]:
    """
    Split a batched GraphShape into individual GraphShape instances.

    :param shape_batch: GraphShape with 2D edge and address arrays
    :return: list of GraphShape (one per batch)
    :raises ValueError: if input is not batched
    """
    if not shape_batch.is_batch:
        raise ValueError("Input GraphShape must be batched for separation.")

    addresses_list = np.unstack(shape_batch.addresses, axis=0)
    a = {k: np.unstack(shape_batch.edges[k]) for k in shape_batch.edges}
    edges_list = [dict(zip(a, t)) for t in zip(*a.values())]  # TODO : vérifier que ça fonctionne comme on veut.

    shape_list = []
    for a, e in zip(addresses_list, edges_list):
        shape = GraphShape(edges=e, addresses=a)
        shape_list.append(shape)
    return shape_list


def max_shape(graph_shape_list: list[GraphShape]) -> GraphShape:
    """
    Returns the maximum graph shape from a list of graph shapes.

    If some objects do not appear in some shapes, then those objects
    are systematically included in the output.

    :param graph_shape_list: List of graph shapes to be compared.
    :return: GraphShape with maxima per edge class and addresses
    :raises ValueError: if list is empty or contains non-GraphShape
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
    :return: GraphShape with summed counts per edge class and addresses
    :raises ValueError: if list is empty or contains non-GraphShape
    """
    if not graph_shape_list:
        raise ValueError("Empty input list given for sum_shapes.")

    sum_graph_shape = graph_shape_list[0]
    for graph_shape in graph_shape_list[1:]:
        if not isinstance(graph_shape, GraphShape):
            raise ValueError("Invalid input in graph_list, expected GraphShape.")
        sum_graph_shape = GraphShape.sum(sum_graph_shape, graph_shape)
    return sum_graph_shape
