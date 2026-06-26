# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import numpy as np
from jax.tree_util import register_pytree_node_class

from energnn.graph.backend import Backend, NumpyBackend
from energnn.graph.hyper_edge_set import HyperEdgeSet

HYPER_EDGE_SETS = "hyper_edge_sets"
ADDRESSES = "addresses"


@register_pytree_node_class
class GraphShape(dict):
    """
    Represents the shape of a graph: per-class object counts and registry size.

    :param backend: Array backend.
    :param hyper_edge_sets: Dict mapping hyper-edge class name to count array.
    :param addresses: Number of addresses in the graph.
    """

    def __init__(self, *, backend: Backend | None = None, hyper_edge_sets: dict, addresses) -> None:
        super().__init__()
        self._backend: Backend = backend if backend is not None else NumpyBackend()
        self[HYPER_EDGE_SETS] = hyper_edge_sets
        self[ADDRESSES] = addresses

    # ------------------------------------------------------------------
    # JAX PyTree protocol
    # ------------------------------------------------------------------

    def tree_flatten(self):
        children = list(self.values())
        aux = (tuple(self.keys()), self._backend)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children) -> GraphShape:
        keys, backend = aux_data
        d = dict(zip(keys, children))
        return cls(backend=backend, hyper_edge_sets=d[HYPER_EDGE_SETS], addresses=d[ADDRESSES])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls, hyper_edge_set_dict: dict[str, HyperEdgeSet], non_fictitious, backend: Backend | None = None
    ) -> GraphShape:
        """
        Build a GraphShape from a hyper-edge set dictionary and a non-fictitious mask.

        :param hyper_edge_set_dict: Mapping from class name to HyperEdgeSet.
        :param non_fictitious: Array whose last dimension gives the registry size.
        :param backend: Array backend; inferred from the HyperEdgeSets when not given.
        :return: New GraphShape instance.
        """
        if backend is None:
            if hyper_edge_set_dict:
                backend = next(iter(hyper_edge_set_dict.values()))._backend
            else:
                backend = NumpyBackend()
        xp = backend.xp

        hyper_edge_set_shape_dict = {k: xp.array(v.n_obj) for k, v in hyper_edge_set_dict.items()}
        addresses = xp.array(non_fictitious.shape[0]) if non_fictitious is not None else xp.array([0])
        return cls(backend=backend, hyper_edge_sets=hyper_edge_set_shape_dict, addresses=addresses)

    @classmethod
    def from_jsonable_dict(cls, count_shape: dict, backend: Backend | None = None) -> GraphShape:
        """
        Deserialize GraphShape from a JSON-friendly dictionary.

        :param count_shape: Dict with 'hyper_edge_sets' and 'addresses'.
        :param backend: Array backend; defaults to NumpyBackend.
        :return: Reconstructed GraphShape.
        """
        if backend is None:
            backend = NumpyBackend()
        xp = backend.xp
        hyper_edge_sets = {k: xp.array(v) for k, v in count_shape[HYPER_EDGE_SETS].items()}
        addresses = xp.array(count_shape[ADDRESSES])
        return cls(backend=backend, hyper_edge_sets=hyper_edge_sets, addresses=addresses)

    def to_jsonable_dict(self) -> dict:
        """Serialize GraphShape to a JSON-friendly dict."""
        return {HYPER_EDGE_SETS: {k: int(v) for k, v in self.hyper_edge_sets.items()}, ADDRESSES: int(self.addresses)}

    # ------------------------------------------------------------------
    # Backend conversion
    # ------------------------------------------------------------------

    def to_backend(self, new_backend: Backend) -> GraphShape:
        """Return a copy of this GraphShape with arrays converted to ``new_backend``."""
        hyper_edge_sets_b = {k: new_backend.from_numpy(np.array(v)) for k, v in self.hyper_edge_sets.items()}
        addresses_b = new_backend.from_numpy(np.array(self.addresses))
        return type(self)(backend=new_backend, hyper_edge_sets=hyper_edge_sets_b, addresses=addresses_b)

    @classmethod
    def from_numpy_shape(cls, shape: GraphShape, device=None, dtype: str = "float32") -> GraphShape:
        """Convert a NumPy-backed GraphShape to a JAX-backed one."""
        from energnn.graph.backend import JaxBackend

        return shape.to_backend(JaxBackend(device=device, dtype=dtype))

    def to_numpy_shape(self) -> GraphShape:
        """Convert this GraphShape to a NumPy-backed one."""
        return self.to_backend(NumpyBackend())

    # ------------------------------------------------------------------
    # Binary class-methods (max / sum)
    # ------------------------------------------------------------------

    @classmethod
    def max(cls, a: GraphShape, b: GraphShape) -> GraphShape:
        """Return the element-wise maximum of two GraphShape objects."""
        backend = a._backend
        xp = backend.xp
        classes = set(list(a.hyper_edge_sets.keys()) + list(b.hyper_edge_sets.keys()))
        hes_max = {
            c: xp.maximum(a.hyper_edge_sets.get(c, -xp.inf), b.hyper_edge_sets.get(c, -xp.inf)) for c in classes
        }
        addresses = xp.maximum(a.addresses, b.addresses)
        return cls(backend=backend, hyper_edge_sets=hes_max, addresses=addresses)

    @classmethod
    def sum(cls, a: GraphShape, b: GraphShape) -> GraphShape:
        """Return the element-wise sum of two GraphShape objects."""
        backend = a._backend
        classes = set(list(a.hyper_edge_sets.keys()) + list(b.hyper_edge_sets.keys()))
        hes_sum = {
            c: a.hyper_edge_sets.get(c, 0) + b.hyper_edge_sets.get(c, 0) for c in classes
        }
        addresses = a.addresses + b.addresses
        return cls(backend=backend, hyper_edge_sets=hes_sum, addresses=addresses)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hyper_edge_sets(self) -> dict:
        return self[HYPER_EDGE_SETS]

    @property
    def addresses(self):
        return self[ADDRESSES]

    @property
    def array(self):
        """Concatenated hyper-edge-set shape values as a single array."""
        xp = self._backend.xp
        return xp.stack(list(self.hyper_edge_sets.values()), axis=-1)

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
        """Return the batch size; raises if not batched."""
        if not self.is_batch:
            raise ValueError("GraphShape is not batched.")
        return self.array.shape[0]


# ---------------------------------------------------------------------------
# Module-level batch/collate functions
# ---------------------------------------------------------------------------


def collate_shapes(shape_list: list[GraphShape]) -> GraphShape:
    """
    Batch a list of GraphShape into one batched GraphShape.

    :param shape_list: Non-empty list of GraphShape objects with matching keys.
    :return: Batched GraphShape with stacked arrays.
    :raises ValueError: If the list is empty.
    """
    if not shape_list:
        raise ValueError("Empty shape list provided to collate_shapes.")

    cls = type(shape_list[0])
    backend = shape_list[0]._backend
    xp = backend.xp

    hes_batch = {k: xp.stack([s.hyper_edge_sets[k] for s in shape_list], axis=0) for k in shape_list[0].hyper_edge_sets}
    addresses_batch = xp.stack([s.addresses for s in shape_list], axis=0)
    return cls(backend=backend, hyper_edge_sets=hes_batch, addresses=addresses_batch)


def separate_shapes(shape_batch: GraphShape) -> list[GraphShape]:
    """
    Split a batched GraphShape into individual GraphShape instances.

    :param shape_batch: Batched GraphShape with 2-D hyper-edge-set and address arrays.
    :return: List of single GraphShape instances.
    :raises ValueError: If the input is not batched.
    """
    if not shape_batch.is_batch:
        raise ValueError("Input GraphShape must be batched for separation.")

    cls = type(shape_batch)
    backend = shape_batch._backend
    xp = backend.xp

    addresses_list = xp.unstack(shape_batch.addresses, axis=0)
    a = {k: xp.unstack(shape_batch.hyper_edge_sets[k]) for k in shape_batch.hyper_edge_sets}
    hes_list = [dict(zip(a, t)) for t in zip(*a.values())]

    return [cls(backend=backend, hyper_edge_sets=e, addresses=addr) for addr, e in zip(addresses_list, hes_list)]


def max_shape(graph_shape_list: list[GraphShape]) -> GraphShape:
    """
    Return the element-wise maximum over a list of GraphShape objects.

    :param graph_shape_list: Non-empty list of GraphShape objects.
    :return: GraphShape with per-class maxima.
    :raises ValueError: If the list is empty or contains non-GraphShape items.
    """
    if not graph_shape_list:
        raise ValueError("Empty input list given for max_shape.")

    result = graph_shape_list[0]
    for shape in graph_shape_list:
        if not isinstance(shape, GraphShape):
            raise ValueError("Invalid input in graph_list, expected GraphShape.")
        result = GraphShape.max(result, shape)
    return result


def sum_shapes(graph_shape_list: list[GraphShape]) -> GraphShape:
    """
    Return the element-wise sum over a list of GraphShape objects.

    :param graph_shape_list: Non-empty list of GraphShape objects.
    :return: GraphShape with summed per-class counts.
    :raises ValueError: If the list is empty or contains non-GraphShape items.
    """
    if not graph_shape_list:
        raise ValueError("Empty input list given for sum_shapes.")

    result = graph_shape_list[0]
    for shape in graph_shape_list[1:]:
        if not isinstance(shape, GraphShape):
            raise ValueError("Invalid input in graph_list, expected GraphShape.")
        result = GraphShape.sum(result, shape)
    return result
