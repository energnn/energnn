# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import jax
from jax import Device
from jax.tree_util import register_pytree_node_class

from energnn.graph.jax.utils import jnp_to_np, np_to_jnp
from energnn.graph.shape import GraphShape

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
