# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from typing import Any, Sequence

import jax
from jax import Device
from jax.tree_util import register_pytree_node_class

from energnn.graph.hyper_edge_set import HyperEdgeSet
from energnn.graph.jax.utils import jnp_to_np, np_to_jnp

FEATURE_ARRAY = "feature_array"
FEATURE_NAMES = "feature_names"
PORT_DICT = "port_dict"
NON_FICTITIOUS = "non_fictitious"


@register_pytree_node_class
class JaxHyperEdgeSet(dict):
    """
    jax implementation of a collection of hyper-edges of the same class, optionally batched.

    Internally this is just a dict storing four entries.

    :param port_dict: Dictionary that maps port names to address values.
    :param feature_array: Array that contains all hyper-edge features.
    :param feature_names: Dictionary from feature names to index in `feature_array`.
    :param non_fictitious: Binary mask filled with ones for non-fictitious objects.
    """

    def __init__(
        self,
        *,
        port_dict: dict[str, jax.Array] | None,
        feature_array: jax.Array | None,
        feature_names: dict[str, jax.Array] | None,
        non_fictitious: jax.Array,
    ):
        super().__init__()
        self[PORT_DICT] = port_dict
        self[FEATURE_ARRAY] = feature_array
        self[FEATURE_NAMES] = feature_names
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
    def tree_unflatten(cls, aux_data: Sequence[str], children: Sequence[Any]) -> JaxHyperEdgeSet:
        """
        Unflattens a PyTree, required for JAX compatibility.

        This method reconstructs an instance of the class from a flattened PyTree structure.

        :param aux_data: Tuple of keys originally returned by tree_flatten.
        :param children: Sequence of values originally returned by tree_flatten.
        :return: Reconstructed instance of the class (`JaxHyperEdgeSet`).
        :raises KeyError: If the expected keys are missing in the zipped dictionary.
        """
        d = dict(zip(aux_data, children))
        return cls(
            port_dict=d[PORT_DICT],
            feature_array=d[FEATURE_ARRAY],
            feature_names=d[FEATURE_NAMES],
            non_fictitious=d[NON_FICTITIOUS],
        )

    @property
    def feature_names(self) -> dict[str, jax.Array] | None:
        return self[FEATURE_NAMES]

    @property
    def port_dict(self) -> dict[str, jax.Array] | None:
        return self[PORT_DICT]

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

        - Single mode: shape `(num_objects * num_features,)`
        - Batch mode:  shape `(batch_size, num_objects * num_features)`.
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

    @classmethod
    def from_numpy_hyper_edge_set(
        cls, hyper_edge_set: HyperEdgeSet, device: Device | None = None, dtype: str = "float32"
    ) -> JaxHyperEdgeSet:
        """
        Convert a classical numpy hyper-edge set to a jax.numpy format for GNN processing.

        This method transforms all array-like attributes of a ``HyperEdgeSet`` object into
        their JAX equivalents, allowing efficient use with JAX transformations and accelerators.

        :param hyper_edge_set: A hyper-edge set object containing NumPy arrays to convert.
        :param device: Optional JAX device (e.g., CPU, GPU) to place the converted arrays on.
                       If None, JAX uses the default device.
        :param dtype: Desired floating-point precision for converted arrays (e.g., "float32", "float64").
        :return: A JAX-compatible version of the hyper-edge set, ready for use in GNN pipelines.
        """
        port_dict = np_to_jnp(hyper_edge_set.port_dict, device=device, dtype=dtype)
        feature_array = np_to_jnp(hyper_edge_set.feature_array, device=device, dtype=dtype)
        feature_names = np_to_jnp(hyper_edge_set.feature_names, device=device, dtype=dtype)
        non_fictitious = np_to_jnp(hyper_edge_set.non_fictitious, device=device, dtype=dtype)
        return cls(
            port_dict=port_dict, feature_array=feature_array, feature_names=feature_names, non_fictitious=non_fictitious
        )

    def to_numpy_hyper_edge_set(self) -> HyperEdgeSet:
        """
        Convert a jax.numpy hyper-edge set for GNN processing to a classical numpy hyper-edge set.

        This method transforms the internal JAX arrays of the hyper-edge set back into standard
        NumPy arrays, enabling compatibility with non-JAX components.

        :return: A classical ``HyperEdgeSet`` object with NumPy arrays.
        """
        port_dict = jnp_to_np(self.port_dict)
        feature_array = jnp_to_np(self.feature_array)
        feature_names = jnp_to_np(self.feature_names)
        non_fictitious = jnp_to_np(self.non_fictitious)
        return HyperEdgeSet(
            port_dict=port_dict, feature_array=feature_array, feature_names=feature_names, non_fictitious=non_fictitious
        )
