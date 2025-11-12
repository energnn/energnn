# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations
from typing import Any, Sequence

import jax
from jax import Device
from jax.tree_util import register_pytree_node_class

from energnn.graph.edge import Edge
from energnn.graph.jax.utils import jnp_to_np, np_to_jnp

FEATURE_ARRAY = "feature_array"
FEATURE_NAMES = "feature_names"
ADDRESS_DICT = "address_dict"
NON_FICTITIOUS = "non_fictitious"


@register_pytree_node_class
class JaxEdge(dict):
    """
    jax implementation of a collection of hyper-edges of the same class, optionally batched.

    Internally this is just a dict storing four entries.

    :param address_dict: Dictionary that contains address keys and address values.
    :param feature_array: Array that contains all hyper-edge features.
    :param feature_names: Dictionary from feature names to  index in `feature_array`.
    :param non_fictitious: Binary mask filled with ones for non fictitious objects.
    """

    # :param shape: Number of hyper-edges present in `JaxEdge`.

    def __init__(
        self,
        *,
        address_dict: dict[str, jax.Array] | None,
        feature_array: jax.Array | None,
        feature_names: dict[str, jax.Array] | None,
        # shape: jax.Array,
        non_fictitious: jax.Array,
    ):
        super().__init__()
        self[ADDRESS_DICT] = address_dict
        self[FEATURE_ARRAY] = feature_array
        self[FEATURE_NAMES] = feature_names
        # self[SHAPE] = shape
        self[NON_FICTITIOUS] = non_fictitious

    def tree_flatten(self) -> tuple:
        """
        Flattens a PyTree, required for JAX compatibility.
        :returns: a tuple of values and keys
        """
        children = self.values()
        aux = self.keys()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data: Sequence[str], children: Sequence[Any]) -> JaxEdge:
        """
        Unflattens a PyTree, required for JAX compatibility.

        This method reconstructs an instance of the class from a flattened PyTree structure.

        :param aux_data: Tuple of keys originally returned by tree_flatten.
        :param children: Sequence of values originally returned by tree_flatten.
        :return: Reconstructed instance of the class (`JaxEdge`).
        :raises KeyError: If expected keys are missing in the zipped dictionary.
        """
        d = dict(zip(aux_data, children))
        return cls(
            address_dict=d[ADDRESS_DICT],
            feature_array=d[FEATURE_ARRAY],
            feature_names=d[FEATURE_NAMES],
            non_fictitious=d[NON_FICTITIOUS],
        )

    @property
    def feature_names(self) -> dict[str, jax.Array] | None:
        return self[FEATURE_NAMES]

    @property
    def address_dict(self) -> dict[str, jax.Array] | None:
        return self[ADDRESS_DICT]

    @property
    def non_fictitious(self) -> jax.Array:
        return self[NON_FICTITIOUS]

    @property
    def feature_array(self) -> jax.Array | None:
        return self[FEATURE_ARRAY]

    @feature_array.setter
    def feature_array(self, value: jax.Array) -> None:
        self[FEATURE_ARRAY] = value

    @property
    def feature_flat_array(self) -> jax.Array | None:
        """
        Returns a flat array by concatenating all features together.

        - single mode: shape (num_objects * num_features,)
        - batch mode:  shape (batch_size, num_objects * num_features).
        """
        if self.feature_names is not None:
            if len(self.feature_array.shape) == 2:
                return self.feature_array.reshape([-1], order="F")
            elif len(self.feature_array.shape) == 3:
                n_batch = self.feature_array.shape[0]
                return self.feature_array.reshape([n_batch, -1], order="F")
            else:
                raise ValueError("Feature array should be of order 2 (single) or 3 (batch).")
        else:
            return None

    # @feature_flat_array.setter
    # def feature_flat_array(self, array: jax.Array) -> None:
    #     """Updates the flat array contained in the Edge."""
    #     if jnp.any(self.feature_flat_array.shape != array.shape):
    #         raise ValueError("Invalid shape.")
    #     if self.feature_names is not None:
    #         if self.is_single:
    #             n_obj = int(self.feature_array.shape[0])
    #             self.feature_array = array.reshape([n_obj, -1], order="F")
    #         elif self.is_batch:
    #             n_batch = int(self.feature_array.shape[0])
    #             n_obj = int(self.feature_array.shape[1])
    #             self.feature_array = array.reshape([n_batch, n_obj, -1], order="F")
    #         else:
    #             raise ValueError("Feature array should be of order 2 (single) or 3 (batch).")

    @classmethod
    def from_numpy_edge(cls, edge: Edge, device: Device | None = None, dtype: str = "float32") -> JaxEdge:
        """
        Convert a classical numpy edge to a jax.numpy format for GNN processing.

        This method transforms all array-like attributes of an ``Edge`` object into
        their JAX equivalents, allowing efficient use with JAX transformations and accelerators.

        :param edge: An edge object containing NumPy arrays to convert.
        :param device: Optional JAX device (e.g., CPU, GPU) to place the converted arrays on.
                       If None, JAX uses the default device.
        :param dtype: Desired floating-point precision for converted arrays (e.g., "float32", "float64").
        :return: A JAX-compatible version of the edge, ready for use in GNN pipelines.
        """
        address_dict = np_to_jnp(edge.address_dict, device=device, dtype=dtype)
        feature_array = np_to_jnp(edge.feature_array, device=device, dtype=dtype)
        feature_names = np_to_jnp(edge.feature_names, device=device, dtype=dtype)
        non_fictitious = np_to_jnp(edge.non_fictitious, device=device, dtype=dtype)
        return cls(
            address_dict=address_dict, feature_array=feature_array, feature_names=feature_names, non_fictitious=non_fictitious
        )

    def to_numpy_edge(self) -> Edge:
        """
        Convert a jax.numpy edge for GNN processing to a classical numpy edge.

        This method transforms the internal JAX arrays of the edge back into standard
        NumPy arrays, enabling compatibility with non-JAX components.

        :return: A classical ``Edge`` object with NumPy arrays.
        """
        address_dict = jnp_to_np(self.address_dict)
        feature_array = jnp_to_np(self.feature_array)
        feature_names = jnp_to_np(self.feature_names)
        non_fictitious = jnp_to_np(self.non_fictitious)
        return Edge(
            address_dict=address_dict, feature_array=feature_array, feature_names=feature_names, non_fictitious=non_fictitious
        )
