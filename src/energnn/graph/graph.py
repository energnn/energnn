# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations
import pickle as pkl
import numpy as np

from energnn.graph.edge import Edge, collate_edges, concatenate_edges, separate_edges
from energnn.graph.shape import GraphShape, collate_shapes, separate_shapes, sum_shapes

EDGES = "edges"
TRUE_SHAPE = "true_shape"
CURRENT_SHAPE = "current_shape"
NON_FICTITIOUS_ADDRESSES = "non_fictitious_addresses"


class Graph(dict):
    """
    Hyper Heterogeneous Multi-Graph (H2MG) container.

    Stores edges, shapes, and address masks for single or batched graphs.

    :param edges: Dictionary of edges contained in the graph.
    :param true_shape: True shape of the graph, not altered by padding.
    :param current_shape: Current shape of the graph, consistent with padding.
    :param non_fictitious_addresses: Mask filled with ones for real addresses, and zeros otherwise.
    """

    def __init__(
        self,
        *,
        edges: dict[str, Edge],
        true_shape: GraphShape,
        current_shape: GraphShape,
        non_fictitious_addresses: np.ndarray,
    ) -> None:
        super().__init__()
        self[EDGES] = edges
        self[TRUE_SHAPE] = true_shape
        self[CURRENT_SHAPE] = current_shape
        self[NON_FICTITIOUS_ADDRESSES] = non_fictitious_addresses

    @classmethod
    def from_dict(cls, *, edge_dict: dict[str, Edge], registry: np.ndarray | None = None) -> Graph:
        """
        Builds a graph from a dictionary of :class:`argo.graph.Edge` and a registry.

        :param edge_dict: Dictionary of edges contained in the graph.
        :param registry: Contains all unique address identifiers that appear in edges.
        :return: Graph that contains both the edges and the registry.
        """
        # registry = to_numpy(registry)  # TODO : voir si on ne peut pas le foutre à la poubelle.
        # Vérifier plutôt que les addresses sont bien dans n_addresses
        n_addresses = registry.shape[0]
        non_fictitious_addresses = np.ones(shape=[n_addresses])
        check_edge_dict_type(edge_dict)
        check_valid_addresses(edge_dict, n_addresses)
        edges = edge_dict
        true_shape = GraphShape.from_dict(edge_dict=edge_dict, non_fictitious=non_fictitious_addresses)
        current_shape = true_shape
        return cls(
            edges=edges, true_shape=true_shape, current_shape=current_shape, non_fictitious_addresses=non_fictitious_addresses
        )

    @property
    def true_shape(self) -> GraphShape:
        """
        True shape of the graph with the real number of objects for each edge
        class as well as the size of the registry stored in a GraphShape object.
        There is no setter for this property.

        :return: a graph shape of true sizes
        """
        return self[TRUE_SHAPE]

    @property
    def current_shape(self) -> GraphShape:
        """
        Current shape of the graph taking into accounts fake padding objects.

        :return: a graph shape of current sizes
        """
        return self[CURRENT_SHAPE]

    @current_shape.setter
    def current_shape(self, value: GraphShape) -> None:
        """
        Set current shape of the graph taking into accounts fake padding objects.

        :param value: a new graph shape
        """
        self[CURRENT_SHAPE] = value

    @property
    def non_fictitious_addresses(self) -> np.ndarray:
        """
        Get mask filled with ones for real addresses, and zeros otherwise

        :return: array filled with ones and zeros
        """
        return self[NON_FICTITIOUS_ADDRESSES]

    @non_fictitious_addresses.setter
    def non_fictitious_addresses(self, value: np.ndarray):
        """
        Set address mask
        :param value: array filled with ones and zeros
        """
        self[NON_FICTITIOUS_ADDRESSES] = value

    @property
    def edges(self) -> dict[str, Edge]:
        """
        Get dictionary of edge instances.

        :return: dict of edge class to Edge
        """
        return self[EDGES]

    @edges.setter
    def edges(self, edge_dict: dict[str, Edge]) -> None:
        """
        Set dictionary of edge instances.

        :param edge_dict: new dictionary of edge instances
        """
        self[EDGES] = edge_dict

    def __str__(self) -> str:
        r = ""
        for k, v in sorted(self.edges.items()):
            r += "{}\n{}\n".format(k, v)
        # r += "N\n{}".format(self.registry)
        return r

    def to_pickle(self, file_path: str) -> None:
        """Saves a graph as a pickle file.

        :param file_path: destination path
        """
        with open(file_path, "wb") as handle:
            pkl.dump(self, handle, protocol=pkl.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, *, file_path: str) -> Graph:
        """Loads a graph from a pickle file.

        :param file_path: source path
        :return: deserialized Graph
        """
        with open(file_path, "rb") as handle:
            graph = pkl.load(handle)
        return graph

    @property
    def is_batch(self) -> bool:
        """
        Determine if graph edges represent a batch.

        :return: True if all edges are batched and mask is 2-D array when defined.
        """
        for k, e in self.edges.items():
            if not e.is_batch:
                return False
        if (self.non_fictitious_addresses is not None) and (len(self.non_fictitious_addresses.shape) != 2):
            return False
        else:
            return True

    @property
    def is_single(self) -> bool:
        """
        Determine if graph edges represent a single graph.

        :return: True if all edges are single and mask is 1-D array when defined.
        """
        for k, e in self.edges.items():
            if not e.is_single:
                return False
        if (self.non_fictitious_addresses is not None) and (len(self.non_fictitious_addresses.shape) != 1):
            return False
        else:
            return True

    @property
    def feature_flat_array(self) -> np.ndarray:
        """
        Returns an array that concatenates all edge features.

        :return: numpy array of concatenated features
        :raises ValueError: if no edges present
        """
        values_list = []

        # Gather hyper-edges features
        if self.edges is not None:
            for key, edge in sorted(self.edges.items()):
                if edge.feature_flat_array is not None:
                    values_list.append(edge.feature_flat_array)
        else:
            raise ValueError("This graph does not contain any edge, and can't be cast as a flat array.")

        return np.concatenate(values_list, axis=-1)

    @feature_flat_array.setter
    def feature_flat_array(self, value: np.ndarray) -> None:
        """
        Updates the flat array contained in the H2MG.

        :param value: flat feature array
        :raises ValueError: if shape mismatch
        """
        if np.any(self.feature_flat_array.shape != value.shape):
            raise ValueError("Invalid array shape.")
        i = 0
        if self.edges is not None:
            for key, edge in sorted(self.edges.items()):
                if edge.feature_names is not None:
                    length = np.shape(edge.feature_flat_array)[-1]
                    if length > 0:
                        self.edges[key].feature_flat_array = value[..., i : i + length]  # Slice over last axis
                        i += length
        else:
            raise ValueError("This graph does not contain any edge, and can't be cast as a flat array.")

    def pad(self, target_shape: GraphShape) -> None:
        """
        Pad edges and address mask to match target_shape.

        :param target_shape: desired GraphShape with larger dimensions
        :raises ValueError: if Graph is not single
        """
        if not self.is_single:
            raise ValueError("This graph is not single and cannot be padded.")

        for edge_key, edge_shape in target_shape.edges.items():
            self.edges[edge_key].pad(edge_shape)
        self.non_fictitious_addresses = np.pad(
            self.non_fictitious_addresses, [0, int(target_shape.addresses) - int(self.current_shape.addresses)]
        )
        self.current_shape = target_shape

    def unpad(self) -> None:
        """
        Removes padding to restore true_shape

        :raises ValueError: if Graph is not single
        """
        for edge_key, edge_shape in self.true_shape.edges.items():
            self.edges[edge_key].unpad(edge_shape)
        self.non_fictitious_addresses = self.non_fictitious_addresses[: int(self.true_shape.addresses)]
        self.current_shape = self.true_shape

    def count_connected_components(self) -> (int, np.ndarray):
        """
        Counts the amount of connected components, and the component id of each address.

        :return: (num_components, component_labels)
        :raises ValueError: if Graph is not single.
        """

        def _max_propagate(*, graph: Graph, h: np.ndarray) -> np.array:
            """Propagates the max value of addresses through edges."""
            import copy

            h_new = copy.deepcopy(h)
            edge_h = {}
            for edge_key, edge in graph.edges.items():
                edge_h[edge_key] = []
                for address_key, address_array in edge.address_dict.items():
                    edge_h[edge_key].append(h_new[address_array.astype(int)])
                edge_h[edge_key] = np.stack(edge_h[edge_key], axis=0)
                edge_h[edge_key] = np.max(edge_h[edge_key], axis=0)
                for address_key, address_array in edge.address_dict.items():
                    new_val = np.max(
                        np.stack([edge_h[edge_key], h_new[address_array.astype(int)]], axis=0),
                        axis=0,
                    )
                    np.maximum.at(h_new, address_array.astype(int), new_val)
            return h_new

        if not self.is_single:
            raise ValueError("Graph is not single.")

        h = np.arange(len(self.non_fictitious_addresses))
        converged = False
        while not converged:
            h_new = _max_propagate(graph=self, h=h)
            converged = np.all(h_new == h)
            h = h_new

        u, indices = np.unique(h, return_inverse=True)

        return len(u), indices

    def offset_addresses(self, offset: np.ndarray | int) -> None:
        """
        Adds an offset on all addresses. Should only be used before graph concatenation.

        :param offset: integer or array to add to addresses
        """
        # TODO : voir si on peut enlever de la classe.
        for k, e in self.edges.items():
            e.offset_addresses(offset=offset)

    def quantiles(self, q_list: list[float] = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]) -> dict[str, np.ndarray]:
        """Computes quantiles of edge features.

        :param q_list: percentiles to compute
        :return: mapping "edge/feature/percentile" to values
        :raises ValueError: if graph is not single or batched and cannot be quantiled.
        """
        info = {}
        for object_name, edge in self.edges.items():
            if edge.feature_dict is not None:
                for feature_name, array in edge.feature_dict.items():
                    if np.size(array) > 0:
                        for q in q_list:
                            if self.is_single:
                                value = np.nanpercentile(array, q=q)
                            elif self.is_batch:
                                value = np.nanpercentile(array, q=q, axis=1)
                            else:
                                raise ValueError("This graph is not single or batch and cannot be quantiled.")
                            info[f"{object_name}/{feature_name}/{q}th-percentile"] = value
        return info


def collate_graphs(graph_list: list[Graph]) -> Graph:
    """
    Collate a list of Graphs into a single Graph with padded shapes.

    All input graphs must share the same `current_shape`.

    :param graph_list: List of Graph instances to collate.
    :returns: A new Graph whose
              - `true_shape` is the batch of all `true_shape`s,
              - `current_shape` is the batch of all `current_shape`s (they must be identical),
              - `edges` are collated per edge type,
              - `non_fictitious_addresses` stacked along a new batch dimension.

    :raises ValueError: if `graph_list` is an empty list.
    :raises AssertionError: if the `current_shape` differ among inputs.
    """
    if not graph_list:
        raise ValueError("collate_graphs requires at least one Graph.")

    first_graph = graph_list[0]

    # Assert that all current shapes are equal
    current_shape_list = [g.current_shape for g in graph_list]
    current_shape = first_graph.current_shape
    for s in current_shape_list:
        assert s == current_shape
    current_shape_batch = collate_shapes(current_shape_list)

    true_shape_list = [g.true_shape for g in graph_list]
    true_shape_batch = collate_shapes(true_shape_list)

    edge_dict_batch = {}
    for k in first_graph.edges.keys():
        edge_dict_batch[k] = collate_edges([g.edges[k] for g in graph_list])

    if first_graph.non_fictitious_addresses is not None:
        non_fictitious_addresses_batch = np.stack([g.non_fictitious_addresses for g in graph_list], axis=0)
    else:
        non_fictitious_addresses_batch = None

    return Graph(
        edges=edge_dict_batch,
        non_fictitious_addresses=non_fictitious_addresses_batch,
        true_shape=true_shape_batch,
        current_shape=current_shape_batch,
    )


def separate_graphs(graph_batch: Graph) -> list[Graph]:
    """
    Split a batch of collated Graph into a list of single Graphs.

    It reverses the operation of :py:func:`collate_graphs`.

    :param graph_batch: A Graph whose `current_shape` and `true_shape` are batched.
    :returns: List of Graphs, each corresponding to one element in the batch.
    """

    current_shape_list = separate_shapes(graph_batch.current_shape)
    true_shape_list = separate_shapes(graph_batch.true_shape)
    n_batch = len(current_shape_list)

    edge_list_dict = {}
    for k in graph_batch.edges.keys():
        edge_list_dict[k] = separate_edges(graph_batch.edges[k])

    if graph_batch.non_fictitious_addresses is not None:
        non_fictitious_addresses_list = np.unstack(graph_batch.non_fictitious_addresses, axis=0)
    else:
        non_fictitious_addresses_list = [None] * n_batch

    edge_dict_list = [{k: edge_list_dict[k][i] for k in edge_list_dict.keys()} for i in range(n_batch)]

    graph_list = []
    for e, n, t, c in zip(edge_dict_list, non_fictitious_addresses_list, true_shape_list, current_shape_list):
        graph = Graph(edges=e, non_fictitious_addresses=n, true_shape=t, current_shape=c)
        graph_list.append(graph)
    return graph_list


def concatenate_graphs(graph_list: list[Graph]) -> Graph:
    """
    Concatenates multiple graphs into a single graph.

    This function merges a sequence of graphs by combining their non-fictitious addresses,
    edges, and shapes into one unified Graph instance. Address offsets are temporarily applied
    to avoid collisions between vertex indices during edge concatenation, then reverted to preserve
    the integrity of the original Graph objects.

    :param graph_list: A list of Graph instances to be concatenated.
    :return: A new Graph object representing the concatenation of all input graphs.

    :raises ValueError: If `graph_list` is empty.

    :note: The input graphs are temporarily modified to apply address offsets, but are restored
           to their original state before the function returns.
    """
    if not graph_list:
        raise ValueError("graph_list must contain at least one Graph")

    n_addresses_list = [len(graph.non_fictitious_addresses) for graph in graph_list]
    offset_list = [sum(n_addresses_list[:i]) for i in range(len(n_addresses_list))]

    non_fictitious_addresses = np.concatenate([graph.non_fictitious_addresses for graph in graph_list], axis=0)
    true_shape = sum_shapes([graph.true_shape for graph in graph_list])
    current_shape = sum_shapes([graph.current_shape for graph in graph_list])

    # Attention, ça c'est super mauvais ! Ça modifie les éléments de la liste...
    [graph.offset_addresses(offset=offset) for graph, offset in zip(graph_list, offset_list)]
    edges = {k: concatenate_edges([graph.edges[k] for graph in graph_list]) for k in graph_list[0].edges}
    [graph.offset_addresses(offset=-offset) for graph, offset in zip(graph_list, offset_list)]

    return Graph(
        edges=edges, non_fictitious_addresses=non_fictitious_addresses, true_shape=true_shape, current_shape=current_shape
    )


def check_edge_dict_type(edge_dict: dict[str, Edge]) -> None:
    """
    Validate that the provided mapping is a dictionary of Edge instances.

    :param edge_dict: A mapping from string keys to Edge objects.
    :raises TypeError: If `edge_dict` is not a dict or if any value in it is not an Edge.
    """
    if not isinstance(edge_dict, dict):
        raise TypeError("Provided 'edge_dict' is not a 'dict', but a {}.".format(type(edge_dict)))
    for key, edge in edge_dict.items():
        if not isinstance(edge, Edge):
            raise TypeError("Item associated with '{}' key is not an 'Edge'.".format(key))


def check_valid_addresses(edge_dict: dict[str, Edge], n_addresses: np.ndarray) -> None:
    """
    Ensure that all address indices in each Edge are valid with respect to the registry.

    Iterates over all edges in `edge_dict` and, if an edge defines `address_names`,
    checks that its integer-coded addresses do not exceed the provided count array.

    :param edge_dict: Mapping from edge names to Edge objects containing address arrays.
    :param n_addresses: 1D array where each entry gives the number of valid addresses
                        for the corresponding edge.
    :raises AssertionError: If any address in any edge is outside the valid range
                            (i.e., not less than the corresponding entry in `n_addresses`).
    """
    # if registry is not None:
    for edge_name, edge in edge_dict.items():
        if edge.address_names is not None:
            assert np.all(edge.address_array < n_addresses)


def get_statistics(graph: Graph, axis: int | None = None, norm_graph: Graph | None = None) -> dict:
    """
    Extract summary statistics from each feature array in the graph's edges.

    For every feature of every edge in `graph`, computes:
      - Root Mean Squared Error (RMSE)
      - Mean Absolute Error (MAE)
      - First and second moments (mean, standard deviation)
      - Range and quantiles (min, 10th, 25th, 50th, 75th, 90th, max)

    If `norm_graph` is provided, also returns normalized metrics:
      - Normalized RMSE (nrmse)
      - Normalized MAE (nmae)

    :param graph: Graph object containing edges with feature dictionaries.
    :param axis: Axis along which to compute statistics. If None, statistics
                 are computed over the flattened array.
    :param norm_graph: Optional Graph whose features serve as normalization reference.
    :return: A dictionary mapping keys of the form
             ``"{edge_name}/{feature_name}/{stat}"`` to their computed values.
             Values are floats or numpy arrays depending on `axis`.
    """
    info = {}
    for object_name, edge in graph.edges.items():
        if edge.feature_dict is not None:
            for feature_name, array in edge.feature_dict.items():
                if array.size == 0:
                    if axis == 1:
                        array = np.array([[0.0]])
                    else:
                        array = np.array([0.0])

                # Root Mean Squared Error
                rmse = np.sqrt(np.nanmean(array**2, axis=axis))
                info["{}/{}/rmse".format(object_name, feature_name)] = rmse
                if norm_graph is not None:
                    norm_array = norm_graph.edges[object_name].feature_dict[feature_name]
                    norm_array = norm_array - np.nanmean(norm_array)
                    nrmse = rmse / (np.sqrt(np.nanmean(norm_array**2, axis=axis)) + 1e-9)
                    info["{}/{}/nrmse".format(object_name, feature_name)] = nrmse

                # Mean Absolute Error
                mae = np.nanmean(np.abs(array), axis=axis)
                info["{}/{}/mae".format(object_name, feature_name)] = mae
                if norm_graph is not None:
                    norm_array = norm_graph.edges[object_name].feature_dict[feature_name]
                    norm_array = norm_array - np.nanmean(norm_array)
                    nmae = mae / (np.nanmean(np.abs(norm_array), axis=axis) + 1e-9)
                    info["{}/{}/nmae".format(object_name, feature_name)] = nmae

                # Moments
                info["{}/{}/mean".format(object_name, feature_name)] = np.nanmean(array, axis=axis)
                info["{}/{}/std".format(object_name, feature_name)] = np.nanstd(array, axis=axis)

                # Quantiles
                info["{}/{}/max".format(object_name, feature_name)] = np.nanmax(array, axis=axis)
                info["{}/{}/90th".format(object_name, feature_name)] = np.nanpercentile(array, q=90, axis=axis)
                info["{}/{}/75th".format(object_name, feature_name)] = np.nanpercentile(array, q=75, axis=axis)
                info["{}/{}/50th".format(object_name, feature_name)] = np.nanpercentile(array, q=50, axis=axis)
                info["{}/{}/25th".format(object_name, feature_name)] = np.nanpercentile(array, q=25, axis=axis)
                info["{}/{}/10th".format(object_name, feature_name)] = np.nanpercentile(array, q=10, axis=axis)
                info["{}/{}/min".format(object_name, feature_name)] = np.nanmin(array, axis=axis)
    return info
