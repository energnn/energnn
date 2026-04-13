# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import pandas as pd
from jax import Device
from jax.tree_util import register_pytree_node_class

from energnn.graph.hyper_edge_set import HyperEdgeSet
from energnn.graph.jax.utils import jnp_to_np, np_to_jnp
from energnn.graph.utils import to_numpy

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
    ) -> None:
        super().__init__()
        self[PORT_DICT] = port_dict
        self[FEATURE_ARRAY] = feature_array
        self[FEATURE_NAMES] = feature_names
        self[NON_FICTITIOUS] = non_fictitious

    @classmethod
    def from_dict(
        cls,
        *,
        port_dict: dict[str, Any] | None = None,
        feature_dict: dict[str, Any] | None = None,
    ) -> JaxHyperEdgeSet:
        """
        Build a JaxHyperEdgeSet from raw dicts of ports and features.

        Both inputs may be None, in which case the corresponding properties
        are set to None and only `non_fictitious` of length zero is created.

        :param port_dict: Dictionary of ports, each key corresponds to a port name and to the values are the
                             corresponding addresses for each object stored into an array.
        :param feature_dict: Dictionary of features, each key corresponds to a feature name and to the values are the
                             corresponding features for each object stored into an array.
        :returns: A properly structured `JaxHyperEdgeSet` instance.
        :raises ValueError: If ports or features contain NaNs or if shapes mismatch.
        """
        # Convert inputs to pure numpy arrays / dicts
        port_dict = check_dict_or_none_jax(to_numpy(port_dict))
        feature_dict = check_dict_or_none_jax(to_numpy(feature_dict))

        check_valid_ports_jax(port_dict)
        check_no_nan_jax(port_dict=port_dict, feature_dict=feature_dict)

        # Build feature_names and feature_array
        if feature_dict is not None:
            feature_names = {name: idx for idx, name in enumerate(sorted(feature_dict))}
            feature_array = dict2array_jax(feature_dict)
        else:
            feature_names, feature_array = None, None

        # Build a non-fictitious mask.
        shape = build_hyper_edge_set_shape_jax(port_dict=port_dict, feature_dict=feature_dict)
        non_fictitious = jnp.ones(int(shape))

        return cls(
            port_dict=port_dict,
            feature_array=feature_array,
            feature_names=feature_names,
            non_fictitious=non_fictitious,
        )

    def __str__(self) -> str:
        """
        Render the JaxHyperEdgeSet as a pandas DataFrame string.

        If `is_single`, uses a single-level index:
            object_id
        If `is_batch`, uses two-level index:
            batch_id, object_id

        :returns:
            String representation of a `pandas.DataFrame`.
        :raises ValueError:
            If the internal array has unexpected dimensions.
        """
        if self.is_single:
            index = pd.MultiIndex.from_product([range(self.n_obj)], names=["object_id"])
        elif self.is_batch:
            index = pd.MultiIndex.from_product(
                [range(self.n_batch), range(self.n_obj)],
                names=["batch_id", "object_id"],
            )
        else:
            raise ValueError("JaxHyperEdgeSet is neither single nor batched.")

        d = {}
        if self.port_names is not None:
            for k, v in sorted(self.port_dict.items()):
                d[("ports", k)] = v.reshape([-1])
        if self.feature_names is not None:
            for k, v in sorted(self.feature_dict.items()):
                d[("features", k)] = v.reshape([-1])

        return pd.DataFrame(d, index=index).__str__()

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
    def array(self) -> jax.Array:
        """
        Concatenate (features, ports) along the last axis.

        :returns:
            Combined array of shape
            - single: `(n_obj, n_feats + n_ports)`
            - batch: `(batch, n_obj, n_feats + n_ports)`
        """
        array = []
        if self.feature_array is not None:
            array.append(self.feature_array)
        if self.port_array is not None:
            array.append(self.port_array)
        return jnp.concatenate(array, axis=-1)

    @property
    def is_batch(self) -> bool:
        """
        True if `array` is 3-D: `(batch, n_obj, features+ports)`.
        """
        return len(self.array.shape) == 3

    @property
    def is_single(self) -> bool:
        """
        True if `array` is 2-D: `(n_obj, features+ports)`.
        """
        return len(self.array.shape) == 2

    @property
    def n_obj(self) -> int:
        """
        Number of hyper-edges (objects) per instance.
        """
        if self.is_single:
            return int(self.array.shape[0])
        elif self.is_batch:
            return int(self.array.shape[1])
        else:
            raise ValueError("JaxHyperEdgeSet is neither single nor batched.")

    @property
    def n_batch(self) -> int:
        """
        Number of batches. Only valid if `is_batch` is True.
        :raises ValueError: If not a batch.
        """
        if self.is_batch:
            return int(self.array.shape[0])
        else:
            raise ValueError("HyperEdgeSet is not batched.")

    @property
    def feature_names(self) -> dict[str, jax.Array] | None:
        return self[FEATURE_NAMES]

    @property
    def port_array(self) -> jax.Array | None:
        """
        Returns the stacked array of ports, of shape `(n_obj, n_ports)` or `(batch, n_obj, n_ports)`.
        """
        if self.port_dict is None:
            return None
        return dict2array_jax(self.port_dict)

    @property
    def port_names(self) -> dict[str, jax.Array] | None:
        """
        Maps a port name to a column index in `port_array`.
        """
        if self.port_dict is None:
            return None
        return {k: jnp.array(idx) for idx, k in enumerate(sorted(self.port_dict.keys()))}

    @property
    def port_dict(self) -> dict[str, jax.Array] | None:
        return self[PORT_DICT]

    @port_dict.setter
    def port_dict(self, value: dict[str, jax.Array] | None) -> None:
        self[PORT_DICT] = value

    @property
    def non_fictitious(self) -> jax.Array:
        return self[NON_FICTITIOUS]

    @non_fictitious.setter
    def non_fictitious(self, value: jax.Array) -> None:
        self[NON_FICTITIOUS] = value

    @property
    def feature_array(self) -> jax.Array | None:
        return self[FEATURE_ARRAY]

    @feature_array.setter
    def feature_array(self, value: jax.Array) -> None:
        self[FEATURE_ARRAY] = value

    @property
    def feature_dict(self) -> dict[str, jax.Array] | None:
        """
        Unstack `feature_array` into a dict: feature_name --> array.

        :returns: Dict of shape `(n_obj,)` or `(batch, n_obj)` per feature.
        """
        if not self.feature_names:
            return None

        result = dict()
        for k, v in self.feature_names.items():
            # The last axis holds features
            if self.is_batch:
                result[k] = self.feature_array[..., jnp.array(v[0], int)]
            else:
                result[k] = self.feature_array[..., jnp.array(v, int)]
        return result

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

    @feature_flat_array.setter
    def feature_flat_array(self, array: jax.Array) -> None:
        """
        Update the feature array from a flat Fortran-ordered array.

        :param array: Must match the shape of current `.feature_flat_array`.
        :raises ValueError: If shapes mismatch.
        """
        flat = self.feature_flat_array
        if flat is None or flat.shape != array.shape:
            raise ValueError("Shape mismatch for feature_flat_array setter.")
        if self.feature_names is not None:
            if self.is_single:
                self.feature_array = array.reshape([self.n_obj, -1], order="F")
            elif self.is_batch:
                self.feature_array = array.reshape([self.n_batch, self.n_obj, -1], order="F")

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

    def pad(self, target_shape: jax.Array | int) -> None:
        """
        Pad a *single* JaxHyperEdgeSet with a series of zeros for features and max-int for ports
        so that shapes match the `target_shape`.

        :param target_shape: Desired n_obj after padding; must be ≥ current n_obj.
        :raises ValueError: If called on a batch or if target_shape < current n_obj.
        """
        if not self.is_single:
            raise ValueError("JaxHyperEdgeSet is batched, impossible to pad.")

        old_n_obj = self.n_obj

        if old_n_obj > target_shape:
            raise ValueError("Provided target_shape is smaller than current shape, padding is impossible! ")

        # Pad features
        if self.feature_array is not None:
            self.feature_array = jnp.pad(self.feature_array, [(0, int(target_shape) - old_n_obj), (0, 0)])

        # Pad ports
        if self.port_dict is not None:
            for k, v in self.port_dict.items():
                self.port_dict[k] = jnp.pad(v, [0, int(target_shape) - old_n_obj])

        # Pad fictitious mask
        if self.non_fictitious is not None:
            self.non_fictitious = jnp.pad(self.non_fictitious, [0, int(target_shape) - old_n_obj])

    def unpad(self, target_shape: jax.Array | int) -> None:
        """
        Remove all objects beyond the index `target` in a *single* JaxHyperEdgeSet.

        :param target_shape: New n_obj; must be ≤ current n_obj.
        :raises ValueError: If called on a batch or if target_shape > current n_obj.
        """

        if not self.is_single:
            raise ValueError("JaxHyperEdgeSet is batched, impossible to unpad.")

        if self.n_obj < target_shape:
            raise ValueError("Provided target_shape is higher than current shape, unpadding is impossible! ")

        # Unpad features
        if self.feature_array is not None:
            self.feature_array = self.feature_array[: int(target_shape)]

        # Unpad ports
        if self.port_dict is not None:
            for k, v in self.port_dict.items():
                self.port_dict[k] = v[: int(target_shape)]

        # Unpad fictitious mask
        if self.non_fictitious is not None:
            self.non_fictitious = self.non_fictitious[: int(target_shape)]

    def offset_addresses(self, offset: jax.Array | int) -> None:
        """Adds an offset on all addresses. Should only be used before graph concatenation.

        :param offset: Scalar or array to add to each address array.
        """
        self.port_dict = {k: a + jnp.array(offset) for k, a in self.port_dict.items()}


def collate_hyper_edge_sets_jax(hyper_edge_set_list: list[JaxHyperEdgeSet]) -> JaxHyperEdgeSet:
    """
    Collate a list of JaxHyperEdgeSet into a single batched JaxHyperEdgeSet.

    Each JaxHyperEdgeSet in the input list is assumed to have the same feature and port schema.
    This function stacks the per-edge attributes along the 0-th axis.

    :param hyper_edge_set_list: Sequence of JaxHyperEdgeSet objects to batch together. Must be non-empty.
    :return: A single batched JaxHyperEdgeSet.

    :raises IndexError: Raised if `hyper_edge_set_list` is empty.
    :raises ValueError: Raised if not all JaxHyperEdgeSet share the same keys in port_names or feature_names.
    """
    if not hyper_edge_set_list:
        raise IndexError("collate_edges requires at least one JaxHyperEdge to collate.")

    first_hyper_edge_set = hyper_edge_set_list[0]

    # Check the consistency of keys
    for e in hyper_edge_set_list[1:]:
        _check_keys_consistency_jax(first_hyper_edge_set, e)

    # Collate feature arrays
    if first_hyper_edge_set.feature_array is not None:
        feature_array = jnp.stack([e.feature_array for e in hyper_edge_set_list], axis=0)
    else:
        feature_array = None

    # Collate feature names
    if first_hyper_edge_set.feature_names is not None:
        feature_names = {
            k: jnp.stack([e.feature_names[k] for e in hyper_edge_set_list]) for k in first_hyper_edge_set.feature_names
        }
    else:
        feature_names = None

    # Collate port dicts
    if first_hyper_edge_set.port_dict is not None:
        port_dict = {k: jnp.stack([e.port_dict[k] for e in hyper_edge_set_list]) for k in first_hyper_edge_set.port_dict}
    else:
        port_dict = None

    # Collate non-fictitious masks
    if first_hyper_edge_set.non_fictitious is not None:
        non_fictitious = jnp.stack([e.non_fictitious for e in hyper_edge_set_list])
    else:
        non_fictitious = None

    return JaxHyperEdgeSet(
        port_dict=port_dict, feature_array=feature_array, feature_names=feature_names, non_fictitious=non_fictitious
    )


def separate_hyper_edge_sets_jax(hyper_edge_set_batch: JaxHyperEdgeSet) -> list[JaxHyperEdgeSet]:
    """
    Separate a batched JaxHyperEdgeSet into its constituent JaxHyperEdgeSet instances.

    The input JaxHyperEdgeSet must have been created by :py:func:`collate_hyper_edge_sets` or otherwise
    its property "array" must return a 3D array.

    :param hyper_edge_set_batch: The batched JaxHyperEdgeSet to unstack.
    :return: List of JaxHyperEdgeSet instances, each corresponding to one batch element.

    :raises ValueError: If `hyper_edge_set_batch.is_batch` is False.
    """
    if not hyper_edge_set_batch.is_batch:
        raise ValueError("Input is not a batch, impossible to separate.")

    if hyper_edge_set_batch.feature_array is not None:
        feature_array_list = jnp.unstack(hyper_edge_set_batch.feature_array, axis=0)
    else:
        feature_array_list = [None] * hyper_edge_set_batch.n_batch

    if hyper_edge_set_batch.feature_names is not None:
        a = {k: jnp.unstack(hyper_edge_set_batch.feature_names[k]) for k in hyper_edge_set_batch.feature_names}
        feature_names_list = [dict(zip(a, t)) for t in zip(*a.values())]
    else:
        feature_names_list = [None] * hyper_edge_set_batch.n_batch

    if hyper_edge_set_batch.port_dict is not None:
        a = {k: jnp.unstack(hyper_edge_set_batch.port_dict[k]) for k in hyper_edge_set_batch.port_dict}
        port_dict_list = [dict(zip(a, t)) for t in zip(*a.values())]
    else:
        port_dict_list = [None] * hyper_edge_set_batch.n_batch

    if hyper_edge_set_batch.non_fictitious is not None:
        non_fictitious_list = jnp.unstack(hyper_edge_set_batch.non_fictitious, axis=0)
    else:
        non_fictitious_list = [None] * hyper_edge_set_batch.n_batch

    hyper_edge_set_list = []
    for fa, fn, ad, nf in zip(feature_array_list, feature_names_list, port_dict_list, non_fictitious_list):
        hyper_edge_set = JaxHyperEdgeSet(port_dict=ad, feature_array=fa, feature_names=fn, non_fictitious=nf)
        hyper_edge_set_list.append(hyper_edge_set)
    return hyper_edge_set_list


def concatenate_hyper_edge_sets_jax(hyper_edge_set_list: list[JaxHyperEdgeSet]) -> JaxHyperEdgeSet:
    """
    Concatenate several single JaxHyperEdgeSet into one single JaxHyperEdgeSet.

    Unlike :py:func:`collate_hyper_edge_sets`, this does *not* create a batch dimension,
    but simply stacks objects end-to-end.

    :param hyper_edge_set_list: List of single (non-batched) JaxHyperEdgeSet
    :returns: One JaxHyperEdgeSet with n_obj = sum of all inputs’ n_obj
    """
    port_dict = {
        k: jnp.concatenate([hes.port_dict[k] for hes in hyper_edge_set_list]) for k in hyper_edge_set_list[0].port_dict
    }
    feature_array = jnp.concatenate([hes.feature_array for hes in hyper_edge_set_list], axis=0)
    feature_names = hyper_edge_set_list[0].feature_names
    non_fictitious = jnp.concatenate([hes.non_fictitious for hes in hyper_edge_set_list])
    return JaxHyperEdgeSet(
        port_dict=port_dict, feature_array=feature_array, feature_names=feature_names, non_fictitious=non_fictitious
    )


def check_dict_shape_jax(*, d: dict[str, jax.Array] | None, n_objects: int | None) -> int | None:
    """
    Ensure all arrays in a dictionary have the same size on their last axis.

    If `n_objects` is not provided, it is inferred from the first array’s last dimension.
    Otherwise, every array’s last dimension must match the given `n_objects`.

    :param d: Mapping from feature/port name to `jax.numpy` array
                   where each array’s last axis is object-indexed.
    :param n_objects: Optional expected size of the last axis; if None, will be inferred.
    :return: The validated or inferred `n_objects`.

    :raises ValueError: If any array’s last dimension does not match `n_objects`.
    """
    if d is not None:
        if n_objects is None:
            item: jnp.ndarray = next(iter(d.values()))
            n_objects = item.shape[-1]
        for name, arr in d.items():
            if arr.shape[-1] != n_objects:
                raise ValueError(f"Array for key '{name}' has last dimension {arr.shape[-1]}, expected {n_objects}.")
    return n_objects


def build_hyper_edge_set_shape_jax(
    *,
    port_dict: dict[str, jax.Array] | None,
    feature_dict: dict[str, jax.Array] | None,
) -> jax.Array:
    """
    Builds a jax.numpy array representing the number of hyper-edges.

    Validate that `port_dict` and `feature_dict` have consistent sizes
    on their last dimensions and return a scalar numpy array containing that count.

    :param port_dict: Mapping from port names to jax.numpy arrays, or None.
    :param feature_dict: Mapping of feature names to jax.numpy arrays, or None.
    :return: A scalar jax.numpy array of dtype float32 with the number of objects.
    :raises ValueError: If both inputs are None, or if their shapes conflict.
    """
    if port_dict is None and feature_dict is None:
        raise ValueError("At least one of port_dict or feature_dict must be provided.")

    n_objects = check_dict_shape_jax(d=port_dict, n_objects=None)
    n_objects = check_dict_shape_jax(d=feature_dict, n_objects=n_objects)
    return jnp.array(n_objects, dtype=jnp.dtype("float32"))


def dict2array_jax(features_dict: dict[str, jax.Array] | None) -> jax.Array | None:
    """
    Stack a dictionary of jax arrays into a single jax array along the last axis.

    The jax arrays are stacked in alphabetical order of their dictionary keys.

    :param features_dict: Mapping from a feature name to a `jax.numpy` array, or None.
    :return: A stacked jax array with an added last dimension for features, or None.
    """
    if features_dict is None:
        return None
    return jnp.stack([features_dict[k] for k in sorted(features_dict)], axis=-1)


def check_dict_or_none_jax(_input: dict | jnp.ndarray | None) -> dict | None:
    """
    Validate that the input is either a dict or None.

    :param _input: Object to validate
    :return: the input if it was a dict or None
    :raises ValueError: if `_input` is neither dict nor None
    """
    if isinstance(_input, dict):
        return _input
    if _input is None:
        return None
    raise ValueError(f"Expected dict or None, got {type(_input)}")


def check_no_nan_jax(
    *,
    port_dict: dict[str, jax.Array] | None,
    feature_dict: dict[str, jax.Array] | None,
) -> None:
    """
    Ensure there are no NaN values in port or feature arrays.

    :param port_dict: Mapping from port names to jax arrays, or None.
    :param feature_dict: Mapping of feature names to jax arrays, or None.
    :raises ValueError: If any jax array contains NaN.
    """
    for name, arr in (port_dict or {}).items():
        if jnp.any(jnp.isnan(arr)):
            raise ValueError(f"NaN detected in port array for key '{name}'.")
    for name, arr in (feature_dict or {}).items():
        if jnp.any(jnp.isnan(arr)):
            raise ValueError(f"NaN detected in feature array for key '{name}'.")


def check_valid_ports_jax(port_dict: dict[str, jax.Array] | None) -> None:
    """
    Ensure that ports map only to integer-valued addresses.

    :param port_dict: Mapping from port names to jax arrays, or None.
    :raises ValueError: If any port array has entries that are not integer.
    """
    for name, arr in (port_dict or {}).items():
        if not jnp.allclose(arr, jnp.int32(arr)):
            raise ValueError(f"Non-integer values detected in port array for key '{name}'.")


def _check_keys_consistency_jax(hes_1: JaxHyperEdgeSet, hes_2: JaxHyperEdgeSet):
    if (hes_1.port_names is None) != (hes_2.port_names is None):
        raise ValueError("Mismatch in presence of port_names among hyper-edge sets.")
    if (hes_1.feature_names is None) != (hes_2.feature_names is None):
        raise ValueError("Mismatch in presence of feature_names among hyper-edge sets.")
    if hes_1.port_names and hes_1.port_names.keys() != hes_2.port_names.keys():
        raise ValueError("Inconsistent port_names keys among hyper-edge sets.")
    if hes_1.feature_names and hes_1.feature_names.keys() != hes_2.feature_names.keys():
        raise ValueError("Inconsistent feature_names keys among hyper-edge sets.")
