# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations
import jax
from jax import Device
from jax.tree_util import register_pytree_node_class

from energnn.graph import GraphShape
from energnn.graph.jax.utils import jnp_to_np, np_to_jnp

EDGES = "edges"
ADDRESSES = "addresses"


@register_pytree_node_class
class JaxGraphShape(dict):
    """
    PyTree container for storing the number of objects in each class, and addresses in the graph.

    This class inherits from `dict` and stores two keys:
    :param edges: Dictionary of that contains the number of objects for each class.
    :param addresses: Number of addresses in the graph.

    The PyTree methods ``tree_flatten`` and ``tree_unflatten`` make this object
    compatible with JAX transformations (jit, vmap, etc.).
    """

    def __init__(self, *, edges: dict[str, jax.Array], addresses: jax.Array):
        super().__init__()
        self[EDGES] = edges
        self[ADDRESSES] = addresses

    def tree_flatten(self):
        """
        Flatten the JaxGraphShape for JAX PyTree compatibility.

        :returns: flat children and auxiliary data (the keys order).
        """
        children = self.values()
        aux = self.keys()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> JaxGraphShape:
        """
        Reconstruct a JaxGraphShape from flattened data, required for JAX compatibility.

        :param aux_data: sequence of keys matching children order.
        :param children: sequence of array values.
        :return: a reconstructed JaxGraphShape instance.
        """
        d = dict(zip(aux_data, children))
        return cls(edges=d[EDGES], addresses=d[ADDRESSES])

    @property
    def edges(self) -> dict[str, jax.Array]:
        """Dictionary of edge shapes."""
        return self[EDGES]

    @property
    def addresses(self) -> jax.Array:
        """Number of addresses in the graph."""
        return self[ADDRESSES]

    @classmethod
    def from_numpy_shape(cls, shape: GraphShape, device: Device | None = None, dtype: str = "float32") -> JaxGraphShape:
        """
        Convert a classical numpy shape to a jax.numpy format for GNN processing.

        This method transforms all array-like attributes of an ``GraphShape`` object into
        their JAX equivalents, allowing efficient use with JAX transformations and accelerators.

        :param shape: A shape object containing NumPy arrays to convert.
        :param device: Optional JAX device (e.g., CPU, GPU) to place the converted arrays on.
                       If None, JAX uses the default device.
        :param dtype: Desired floating-point precision for converted arrays (e.g., "float32", "float64").
        :return: A JAX-compatible version of the shape, ready for use in GNN pipelines.
        """
        edges = np_to_jnp(shape.edges, device=device, dtype=dtype)
        addresses = np_to_jnp(shape.addresses, device=device, dtype=dtype)
        return cls(edges=edges, addresses=addresses)

    def to_numpy_shape(self) -> GraphShape:
        """
        Convert a jax.numpy shape for GNN processing to a classical numpy shape.

        This method transforms the internal JAX arrays of the shape back into standard
        NumPy arrays, enabling compatibility with non-JAX components.

        :return: A classical ``GraphShape`` object with NumPy arrays.
        """
        edges = jnp_to_np(self.edges)
        addresses = jnp_to_np(self.addresses)
        return GraphShape(edges=edges, addresses=addresses)
