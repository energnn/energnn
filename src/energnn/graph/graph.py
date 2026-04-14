# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import pickle as pkl
from typing import Any, Sequence

import numpy as np
import jax
from jax.tree_util import register_pytree_node_class

from energnn.graph.backend import Backend, NumpyBackend, JaxBackend
from energnn.graph.hyper_edge_set import (
    HyperEdgeSet,
    JaxHyperEdgeSet,
    collate_hyper_edge_sets,
    concatenate_hyper_edge_sets,
    separate_hyper_edge_sets,
)
from energnn.graph.shape import GraphShape, JaxGraphShape, collate_shapes, separate_shapes, sum_shapes
from energnn.graph.utils import to_numpy

HYPER_EDGE_SETS = "hyper_edge_sets"
TRUE_SHAPE = "true_shape"
CURRENT_SHAPE = "current_shape"
NON_FICTITIOUS_ADDRESSES = "non_fictitious_addresses"


class Graph(dict):
    """
    Hyper Heterogeneous Multi-Graph (H2MG) container.

    Stores hyper-edge sets, shapes, and address masks for single or batched graphs.

    :param hyper_edge_sets: Dictionary of hyper-edge sets contained in the graph.
    :param true_shape: True shape of the graph, not altered by padding.
    :param current_shape: Current shape of the graph, consistent with padding.
    :param non_fictitious_addresses: Mask filled with ones for real addresses, and zeros otherwise.
    :param backend: Backend used for array operations (defaults to NumpyBackend).
    """

    def __init__(
        self,
        *,
        hyper_edge_sets: dict[str, HyperEdgeSet],
        true_shape: GraphShape,
        current_shape: GraphShape,
        non_fictitious_addresses: Any,
        backend: Backend | None = None,
    ) -> None:
        super().__init__()
        self[HYPER_EDGE_SETS] = hyper_edge_sets
        self[TRUE_SHAPE] = true_shape
        self[CURRENT_SHAPE] = current_shape
        self[NON_FICTITIOUS_ADDRESSES] = non_fictitious_addresses
        self._backend = backend or NumpyBackend()

    @classmethod
    def from_dict(cls, *, hyper_edge_set_dict: dict[str, HyperEdgeSet], n_addresses: Any, backend: Backend | None = None) -> Graph:
        """
        Builds a graph from a dictionary of :class:`energnn.graph.HyperEdgeSet` and a number of addresses.

        :param hyper_edge_set_dict: Dictionary of hyper-edge sets contained in the graph.
        :param n_addresses: Number of unique addresses that appear in all the hyper-edge sets.
        :param backend: Backend used for array operations.
        :return: Graph that contains both the hyper-edge sets and the registry.
        """
        backend = backend or NumpyBackend()
        non_fictitious_addresses = backend.ones(shape=[n_addresses])
        check_hyper_edge_set_dict_type(hyper_edge_set_dict)
        check_valid_addresses(hyper_edge_set_dict, n_addresses, backend)
        
        # Use JaxGraphShape if we are creating a JaxGraph
        shape_cls = JaxGraphShape if cls.__name__ == "JaxGraph" else GraphShape
        
        true_shape = shape_cls.from_dict(
            hyper_edge_set_dict=hyper_edge_set_dict, non_fictitious=non_fictitious_addresses, backend=backend
        )
        current_shape = true_shape
        return cls(
            hyper_edge_sets=hyper_edge_set_dict,
            true_shape=true_shape,
            current_shape=current_shape,
            non_fictitious_addresses=non_fictitious_addresses,
            backend=backend,
        )

    @property
    def true_shape(self) -> GraphShape:
        """
        True shape of the graph with the real number of objects for each hyper-edge set
        class as well as the size of the registry stored in a GraphShape object.
        There is no setter for this property.

        :return: A graph shape of true sizes.
        """
        return self[TRUE_SHAPE]

    @property
    def current_shape(self) -> GraphShape:
        """
        The current shape of the graph taking into accounts fake padding objects.

        :return: A graph shape of current sizes.
        """
        return self[CURRENT_SHAPE]

    @current_shape.setter
    def current_shape(self, value: GraphShape) -> None:
        """
        Sets the current shape of the graph taking into accounts fake padding objects.

        :param value: A new graph shape.
        """
        self[CURRENT_SHAPE] = value

    @property
    def non_fictitious_addresses(self) -> Any:
        """
        Mask filled with ones for real addresses, and zeros otherwise.

        :return: Array filled with ones and zeros.
        """
        return self[NON_FICTITIOUS_ADDRESSES]

    @non_fictitious_addresses.setter
    def non_fictitious_addresses(self, value: Any):
        """
        Sets the address mask.
        :param value: Array filled with ones and zeros.
        """
        self[NON_FICTITIOUS_ADDRESSES] = value

    @property
    def hyper_edge_sets(self) -> dict[str, HyperEdgeSet]:
        """
        Dictionary of hyper-edge sets.

        :return: Dict of edge class to Edge.
        """
        return self[HYPER_EDGE_SETS]

    @hyper_edge_sets.setter
    def hyper_edge_sets(self, hyper_edge_set_dict: dict[str, HyperEdgeSet]) -> None:
        """
        Sets the dictionary of hyper-edge sets.

        :param hyper_edge_set_dict: New dictionary of hyper-edge sets.
        """
        self[HYPER_EDGE_SETS] = hyper_edge_set_dict

    def __str__(self) -> str:
        r = ""
        for k, v in sorted(self.hyper_edge_sets.items()):
            r += "{}\n{}\n".format(k, v)
        return r

    def to_pickle(self, file_path: str) -> None:
        """Saves a graph as a pickle file.

        :param file_path: Destination path
        """
        with open(file_path, "wb") as handle:
            pkl.dump(self, handle, protocol=pkl.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, *, file_path: str) -> Graph:
        """Loads a graph from a pickle file.

        :param file_path: Source path.
        :return: Deserialized Graph.
        """
        with open(file_path, "rb") as handle:
            graph = pkl.load(handle)
        return graph

    @property
    def is_batch(self) -> bool:
        """
        Determines if the graph is batched.

        :return: True if all hyper-edge sets are batched and if the non-fictitious mask is a 2-D array when defined.
        """
        for k, e in self.hyper_edge_sets.items():
            if not e.is_batch:
                return False
        if (self.non_fictitious_addresses is not None) and (len(self._backend.shape(self.non_fictitious_addresses)) != 2):
            return False
        else:
            return True

    @property
    def is_single(self) -> bool:
        """
        Determines if the graph is single.

        :return: True if all hyper-edge sets are single and if the non-fictitious mask is a 1-D array when defined.
        """
        for k, e in self.hyper_edge_sets.items():
            if not e.is_single:
                return False
        if (self.non_fictitious_addresses is not None) and (len(self._backend.shape(self.non_fictitious_addresses)) != 1):
            return False
        else:
            return True

    @property
    def feature_flat_array(self) -> Any:
        """
        Returns an array that concatenates features of all hyper-edge sets.

        :return: Array of concatenated features.
        :raises ValueError: If no hyper-edge set is present.
        """
        values_list = []

        # Gather hyper-edges set features
        if self.hyper_edge_sets is not None:
            for key, hyper_edge_set in sorted(self.hyper_edge_sets.items()):
                if hyper_edge_set.feature_flat_array is not None:
                    values_list.append(hyper_edge_set.feature_flat_array)
        else:
            raise ValueError("This graph does not contain any hyper-edge set, and can't be cast as a flat array.")

        return self._backend.concatenate(values_list, axis=-1)

    @feature_flat_array.setter
    def feature_flat_array(self, value: Any) -> None:
        """
        Updates the flat array contained in the H2MG.

        :param value: Flat feature array.
        :raises ValueError: If shapes do not match the current feature flat array.
        """
        if self._backend.any(self._backend.array(self._backend.shape(self.feature_flat_array)) != self._backend.array(self._backend.shape(value))):
            raise ValueError("Invalid array shape.")
        i = 0
        if self.hyper_edge_sets is not None:
            for key, hyper_edge_set in sorted(self.hyper_edge_sets.items()):
                if hyper_edge_set.feature_names is not None:
                    length = self._backend.shape(hyper_edge_set.feature_flat_array)[-1]
                    if length > 0:
                        self.hyper_edge_sets[key].feature_flat_array = value[..., i : i + length]  # Slice over the last axis
                        i += length
        else:
            raise ValueError("This graph does not contain any hyper-edge set, and can't be cast as a flat array.")

    def pad(self, target_shape: GraphShape) -> None:
        """
        Pads hyper-edge sets and address mask to match target_shape.

        :param target_shape: Desired GraphShape with larger dimensions.
        :raises ValueError: If the graph is not single.
        """
        if not self.is_single:
            raise ValueError("This graph is not single and cannot be padded.")

        for key, hyper_edge_set_shape in target_shape.hyper_edge_sets.items():
            self.hyper_edge_sets[key].pad(hyper_edge_set_shape)
        
        diff = int(target_shape.addresses) - int(self.current_shape.addresses)
        self.non_fictitious_addresses = self._backend.np.pad(
            self.non_fictitious_addresses, [0, diff]
        )
        self.current_shape = target_shape

    def unpad(self) -> None:
        """
        Removes padding to restore true_shape.

        :raises ValueError: If the graph is not single.
        """
        for key, hyper_edge_set_shape in self.true_shape.hyper_edge_sets.items():
            self.hyper_edge_sets[key].unpad(hyper_edge_set_shape)
        self.non_fictitious_addresses = self.non_fictitious_addresses[: int(self.true_shape.addresses)]
        self.current_shape = self.true_shape

    def count_connected_components(self) -> tuple[int, Any]:
        """
        Counts connected components, and the component id of each address.

        :return: `(num_components, component_labels)`
        :raises ValueError: If the graph is not single.
        """

        def _max_propagate(*, graph: Graph, h_: Any) -> Any:
            """Propagates the max value of addresses through hyper-edges."""
            import copy

            h_new_ = copy.deepcopy(h_)
            edge_h = {}
            for edge_key, edge in graph.hyper_edge_sets.items():
                edge_h[edge_key] = []
                for address_key, address_array in edge.port_dict.items():
                    edge_h[edge_key].append(h_new_[address_array.astype(int)])
                edge_h[edge_key] = np.stack(edge_h[edge_key], axis=0)
                edge_h[edge_key] = np.max(edge_h[edge_key], axis=0)
                for address_key, address_array in edge.port_dict.items():
                    new_val = np.max(
                        np.stack([edge_h[edge_key], h_new_[address_array.astype(int)]], axis=0),
                        axis=0,
                    )
                    np.maximum.at(h_new_, address_array.astype(int), new_val)
            return h_new_

        if not self.is_single:
            raise ValueError("Graph is not single.")
        
        # This implementation is NumPy specific as it uses np.maximum.at
        # We assume the graph is in NumPy mode if this is called, or we convert to NumPy
        was_jax = False
        if isinstance(self._backend, JaxBackend):
            was_jax = True
            # For simplicity, if it's Jax, we could try to implement it in Jax or just use the current implementation with to_numpy
            # But the existing code is already numpy based.
        
        h = np.arange(len(self.non_fictitious_addresses))
        # Ensure we work with numpy for this part as it uses in-place operations
        # If it was JaxGraph, we'd need a Jax implementation.
        # But JaxGraph didn't override this, so it was already broken for Jax if it was called.
        
        converged = False
        while not converged:
            h_new = _max_propagate(graph=self, h_=h)
            converged = np.all(h_new == h)
            h = h_new

        u, indices = np.unique(h, return_inverse=True)

        return len(u), indices

    def offset_addresses(self, offset: Any | int) -> None:
        """
        Adds an offset on all addresses. Should only be used before graph concatenation.

        :param offset: Integer or array to add to addresses
        """
        for k, e in self.hyper_edge_sets.items():
            e.offset_addresses(offset=offset)

    def quantiles(self, q_list: list[float] | None = None) -> dict[str, Any]:
        """Computes quantiles of hyper-edge set features.

        :param q_list: Percentiles to compute
        :return: Mapping "hyper_edge_set/feature/percentile" to values.
        :raises ValueError: If the graph is not single or batched and cannot be quantiled.
        """
        # If we are in JAX JIT, we cannot compute quantiles and return them in a dict 
        # that will be used for logging (which expects concrete values).
        # We check if any array is a JAX Tracer.
        for hes in self.hyper_edge_sets.values():
            if hes.feature_array is not None and isinstance(hes.feature_array, jax.core.Tracer):
                return {}

        if q_list is None:
            q_list = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]
        info = {}
        for object_name, hyper_edge_sets in self.hyper_edge_sets.items():
            if hyper_edge_sets.feature_dict is not None:
                for feature_name, array in hyper_edge_sets.feature_dict.items():
                    array_np = to_numpy(array)
                    if array_np.size > 0:
                        for q in q_list:
                            if self.is_single:
                                value = np.nanpercentile(array_np, q=q)
                            elif self.is_batch:
                                value = np.nanpercentile(array_np, q=q, axis=1)
                            else:
                                raise ValueError("This graph is not single or batch and cannot be quantiled.")
                            info[f"{object_name}/{feature_name}/{q}th-percentile"] = value
        return info


@register_pytree_node_class
class JaxGraph(Graph):
    def __init__(
        self,
        *,
        hyper_edge_sets: dict[str, HyperEdgeSet],
        true_shape: GraphShape,
        current_shape: GraphShape,
        non_fictitious_addresses: Any,
        backend: Backend | None = None,
    ) -> None:
        # Ignore provided backend and force JAX backend to ensure correct JAX behavior
        super().__init__(
            hyper_edge_sets=hyper_edge_sets,
            true_shape=true_shape,
            current_shape=current_shape,
            non_fictitious_addresses=non_fictitious_addresses,
            backend=JaxBackend(),
        )

    def tree_flatten(self):
        """
        Flattens the JaxGraph for JAX PyTree compatibility.

        Make only tensor-like batched data part of the children so that operations like
        `jax.tree.map(lambda x: x[0], tree)` work seamlessly.
        """
        # Sort hyper_edge_sets for deterministic auxiliary data/children structure
        keys = sorted(self[HYPER_EDGE_SETS].keys())
        hes_tuple = tuple(self[HYPER_EDGE_SETS][k] for k in keys)
        
        # Shapes (TRUE_SHAPE, CURRENT_SHAPE) are part of children as they contain arrays 
        # (JAX Tracers) that must be consistently handled across PyTrees.
        children = (hes_tuple, self[NON_FICTITIOUS_ADDRESSES], self[TRUE_SHAPE], self[CURRENT_SHAPE])
        aux = (tuple(keys), self._backend)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> JaxGraph:
        """
        Reconstructs a JaxGraph from flattened data, required for JAX compatibility.
        """
        keys, backend = aux_data
        hes_tuple, non_fictitious_addresses, true_shape, current_shape = children
        
        hyper_edge_sets = dict(zip(keys, hes_tuple))
        return cls(
            hyper_edge_sets=hyper_edge_sets,
            true_shape=true_shape,
            current_shape=current_shape,
            non_fictitious_addresses=non_fictitious_addresses,
            backend=backend,
        )

    @classmethod
    def from_numpy_graph(cls, graph: Graph, device: Any | None = None, dtype: str = "float32") -> JaxGraph:
        from energnn.graph.utils import np_to_jnp
        hyper_edge_sets = {
            k: JaxHyperEdgeSet.from_numpy_hyper_edge_set(v, device=device, dtype=dtype)
            for k, v in graph.hyper_edge_sets.items()
        }
        true_shape = JaxGraphShape.from_numpy_shape(graph.true_shape, device=device, dtype=dtype)
        current_shape = JaxGraphShape.from_numpy_shape(graph.current_shape, device=device, dtype=dtype)
        non_fictitious_addresses = np_to_jnp(graph.non_fictitious_addresses, device=device, dtype=dtype)
        return cls(
            hyper_edge_sets=hyper_edge_sets,
            true_shape=true_shape,
            current_shape=current_shape,
            non_fictitious_addresses=non_fictitious_addresses,
        )

    def to_numpy_graph(self) -> Graph:
        from energnn.graph.utils import jnp_to_np
        hyper_edge_sets = {k: v.to_numpy_hyper_edge_set() for k, v in self.hyper_edge_sets.items()}
        true_shape = self.true_shape.to_numpy_shape()
        current_shape = self.current_shape.to_numpy_shape()
        non_fictitious_addresses = jnp_to_np(self.non_fictitious_addresses)
        return Graph(
            hyper_edge_sets=hyper_edge_sets,
            true_shape=true_shape,
            current_shape=current_shape,
            non_fictitious_addresses=non_fictitious_addresses,
        )


def collate_graphs(graph_list: list[Graph]) -> Graph:
    """
    Collate a list of Graphs into a single Graph with padded shapes.
    """
    if not graph_list:
        raise ValueError("collate_graphs requires at least one Graph.")

    first_graph = graph_list[0]
    backend = first_graph._backend

    # Assert that all current shapes are equal
    current_shape_list = [g.current_shape for g in graph_list]
    current_shape = first_graph.current_shape
    for s in current_shape_list:
        assert s == current_shape
    current_shape_batch = collate_shapes(current_shape_list)

    true_shape_list = [g.true_shape for g in graph_list]
    true_shape_batch = collate_shapes(true_shape_list)

    hyper_edge_sets_batch = {}
    for k in first_graph.hyper_edge_sets.keys():
        hyper_edge_sets_batch[k] = collate_hyper_edge_sets([g.hyper_edge_sets[k] for g in graph_list])

    if first_graph.non_fictitious_addresses is not None:
        non_fictitious_addresses_batch = backend.stack([g.non_fictitious_addresses for g in graph_list], axis=0)
    else:
        non_fictitious_addresses_batch = None

    if isinstance(first_graph, JaxGraph):
        return JaxGraph(
            hyper_edge_sets=hyper_edge_sets_batch,
            non_fictitious_addresses=non_fictitious_addresses_batch,
            true_shape=true_shape_batch,
            current_shape=current_shape_batch,
        )
    return Graph(
        hyper_edge_sets=hyper_edge_sets_batch,
        non_fictitious_addresses=non_fictitious_addresses_batch,
        true_shape=true_shape_batch,
        current_shape=current_shape_batch,
        backend=backend,
    )


def separate_graphs(graph_batch: Graph) -> list[Graph]:
    """
    Split a batch of collated Graph into a list of single Graphs.
    """
    current_shape_list = separate_shapes(graph_batch.current_shape)
    true_shape_list = separate_shapes(graph_batch.true_shape)
    n_batch = len(current_shape_list)
    backend = graph_batch._backend

    hyper_edge_set_list_dict = {}
    for k in graph_batch.hyper_edge_sets.keys():
        hyper_edge_set_list_dict[k] = separate_hyper_edge_sets(graph_batch.hyper_edge_sets[k])

    if graph_batch.non_fictitious_addresses is not None:
        non_fictitious_addresses_list = backend.unstack(graph_batch.non_fictitious_addresses, axis=0)
    else:
        non_fictitious_addresses_list = [None] * n_batch

    hyper_edge_set_dict_list = [
        {k: hyper_edge_set_list_dict[k][i] for k in hyper_edge_set_list_dict.keys()} for i in range(n_batch)
    ]

    graph_list = []
    for e, n, t, c in zip(hyper_edge_set_dict_list, non_fictitious_addresses_list, true_shape_list, current_shape_list):
        if isinstance(graph_batch, JaxGraph):
            graph = JaxGraph(hyper_edge_sets=e, non_fictitious_addresses=n, true_shape=t, current_shape=c)
        else:
            graph = Graph(hyper_edge_sets=e, non_fictitious_addresses=n, true_shape=t, current_shape=c, backend=backend)
        graph_list.append(graph)
    return graph_list


def concatenate_graphs(graph_list: list[Graph]) -> Graph:
    """
    Concatenates multiple graphs into a single graph.
    """
    if not graph_list:
        raise ValueError("graph_list must contain at least one Graph")

    first_graph = graph_list[0]
    backend = first_graph._backend
    
    n_addresses_list = [len(to_numpy(graph.non_fictitious_addresses)) for graph in graph_list]
    offset_list = [sum(n_addresses_list[:i]) for i in range(len(n_addresses_list))]

    non_fictitious_addresses = backend.concatenate([graph.non_fictitious_addresses for graph in graph_list], axis=0)
    true_shape = sum_shapes([graph.true_shape for graph in graph_list])
    current_shape = sum_shapes([graph.current_shape for graph in graph_list])

    [graph.offset_addresses(offset=offset) for graph, offset in zip(graph_list, offset_list)]
    hyper_edge_sets = {
        k: concatenate_hyper_edge_sets([graph.hyper_edge_sets[k] for graph in graph_list])
        for k in first_graph.hyper_edge_sets
    }
    [graph.offset_addresses(offset=-offset) for graph, offset in zip(graph_list, offset_list)]

    if isinstance(first_graph, JaxGraph):
        return JaxGraph(
            hyper_edge_sets=hyper_edge_sets,
            non_fictitious_addresses=non_fictitious_addresses,
            true_shape=true_shape,
            current_shape=current_shape,
        )
    return Graph(
        hyper_edge_sets=hyper_edge_sets,
        non_fictitious_addresses=non_fictitious_addresses,
        true_shape=true_shape,
        current_shape=current_shape,
        backend=backend,
    )


def check_hyper_edge_set_dict_type(hyper_edge_set_dict: dict[str, HyperEdgeSet]) -> None:
    """
    Validate that the provided mapping is a dictionary of HyperEdgeSet instances.
    """
    if not isinstance(hyper_edge_set_dict, dict):
        raise TypeError("Provided 'hyper_edge_set_dict' is not a 'dict', but a {}.".format(type(hyper_edge_set_dict)))
    for key, hyper_edge_set in hyper_edge_set_dict.items():
        if not isinstance(hyper_edge_set, HyperEdgeSet):
            raise TypeError("Item associated with '{}' key is not an 'hyper_edge_set_dict'.".format(key))


def check_valid_addresses(hyper_edge_set_dict: dict[str, HyperEdgeSet], n_addresses: Any, backend: Backend | None = None) -> None:
    """
    Ensure that all address indices in each HyperEdgeSet are valid with respect to the registry.
    """
    backend = backend or NumpyBackend()
    for key, hyper_edge_set in hyper_edge_set_dict.items():
        if hyper_edge_set.port_names is not None:
            # We use to_numpy only if it's already NumPy for performance, 
            # or keep it as-is for JAX since it's just an assertion
            assert np.all(hyper_edge_set.port_array < n_addresses)


def get_statistics(graph: Graph, axis: int | None = None, norm_graph: Graph | None = None) -> dict:
    """
    Extract summary statistics from each feature array in the graph's hyper-edge sets.
    """
    # This remains NumPy based as it's mostly for reporting
    
    # Convert fictitious features to NaN.
    # We work on a copy to avoid modifying the original graph arrays if they are used elsewhere
    # but the original code modified them in place (partially).
    
    info = {}
    for object_name, hyper_edge_set in graph.hyper_edge_sets.items():
        mask = to_numpy(hyper_edge_set.non_fictitious)
        if hyper_edge_set.feature_array is not None:
            array = to_numpy(hyper_edge_set.feature_array).copy()
            array[mask == 0] = np.nan
            
            if hyper_edge_set.feature_dict is not None:
                # We need to compute stats per feature
                # The original code used feature_dict which we'll reconstruct from our NaN-filled array
                feature_names = hyper_edge_set.feature_names
                for feature_name, v in feature_names.items():
                    idx = int(v)
                    # Extract column
                    feat_array = array[..., idx]
                    
                    if feat_array.size == 0:
                        if axis == 1:
                            feat_array = np.array([[0.0]])
                        else:
                            feat_array = np.array([0.0])

                    # Root Mean Squared Error
                    rmse = np.sqrt(np.nanmean(feat_array**2, axis=axis))
                    info["{}/{}/rmse".format(object_name, feature_name)] = rmse
                    if norm_graph is not None:
                        norm_hes = norm_graph.hyper_edge_sets[object_name]
                        norm_feat_dict = norm_hes.feature_dict
                        if norm_feat_dict is not None and feature_name in norm_feat_dict:
                            norm_array = to_numpy(norm_feat_dict[feature_name])
                            norm_array = norm_array - np.nanmean(norm_array)
                            nrmse = rmse / (np.sqrt(np.nanmean(norm_array**2, axis=axis)) + 1e-9)
                            info["{}/{}/nrmse".format(object_name, feature_name)] = nrmse
                        else:
                            info["{}/{}/nrmse".format(object_name, feature_name)] = None

                    # Mean Absolute Error
                    mae = np.nanmean(np.abs(feat_array), axis=axis)
                    info["{}/{}/mae".format(object_name, feature_name)] = mae
                    if norm_graph is not None:
                        norm_hes = norm_graph.hyper_edge_sets[object_name]
                        norm_feat_dict = norm_hes.feature_dict
                        if norm_feat_dict is not None and feature_name in norm_feat_dict:
                            norm_array = to_numpy(norm_feat_dict[feature_name])
                            norm_array = norm_array - np.nanmean(norm_array)
                            nmae = mae / (np.nanmean(np.abs(norm_array), axis=axis) + 1e-9)
                            info["{}/{}/nmae".format(object_name, feature_name)] = nmae
                        else:
                            info["{}/{}/nmae".format(object_name, feature_name)] = None

                    # Moments
                    info["{}/{}/mean".format(object_name, feature_name)] = np.nanmean(feat_array, axis=axis)
                    info["{}/{}/std".format(object_name, feature_name)] = np.nanstd(feat_array, axis=axis)

                    # Quantiles
                    info["{}/{}/max".format(object_name, feature_name)] = np.nanmax(feat_array, axis=axis)
                    info["{}/{}/90th".format(object_name, feature_name)] = np.nanpercentile(feat_array, q=90, axis=axis)
                    info["{}/{}/75th".format(object_name, feature_name)] = np.nanpercentile(feat_array, q=75, axis=axis)
                    info["{}/{}/50th".format(object_name, feature_name)] = np.nanpercentile(feat_array, q=50, axis=axis)
                    info["{}/{}/25th".format(object_name, feature_name)] = np.nanpercentile(feat_array, q=25, axis=axis)
                    info["{}/{}/10th".format(object_name, feature_name)] = np.nanpercentile(feat_array, q=10, axis=axis)
                    info["{}/{}/min".format(object_name, feature_name)] = np.nanmin(feat_array, axis=axis)
    return info
# Backward compatibility aliases
JaxGraphShape = JaxGraphShape
JaxHyperEdgeSet = JaxHyperEdgeSet
collate_graphs_jax = collate_graphs
separate_graphs_jax = separate_graphs
concatenate_graphs_jax = concatenate_graphs
check_hyper_edge_set_dict_type_jax = check_hyper_edge_set_dict_type
check_valid_addresses_jax = check_valid_addresses
get_statistics_jax = get_statistics
