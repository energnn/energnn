# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import jax.numpy as jnp
import pandas as pd
from jax.tree_util import register_pytree_node_class

from energnn.graph.backend import Backend, NumpyBackend, JaxBackend
from energnn.graph.utils import to_numpy

FEATURE_ARRAY = "feature_array"
FEATURE_NAMES = "feature_names"
PORT_DICT = "port_dict"
NON_FICTITIOUS = "non_fictitious"


class HyperEdgeSet(dict):
    """
    A collection of hyper-edges of the same class, optionally batched.

    Internally this is just a dict storing four entries.

    :param port_dict: Mapping from a port name to an array of shape `(n_edges,)` or `(batch, n_edges)`.
    :param feature_array: Array that contains all hyper-edge features.
    :param feature_names: Dictionary from feature names to index in `feature_array`.
    :param non_fictitious: Mask array set to 1 for non-fictitious objects and to 0 for fictitious objects.
    :param backend: Backend used for array operations (defaults to NumpyBackend).
    """

    def __init__(
        self,
        *,
        port_dict: dict[str, Any] | None,
        feature_array: Any | None,
        feature_names: dict[str, Any] | None,
        non_fictitious: Any,
        backend: Backend | None = None,
    ) -> None:
        super().__init__()
        self[PORT_DICT] = port_dict
        self[FEATURE_ARRAY] = feature_array
        self[FEATURE_NAMES] = feature_names
        self[NON_FICTITIOUS] = non_fictitious
        self._backend = backend or NumpyBackend()

    @classmethod
    def from_dict(
        cls,
        *,
        port_dict: dict[str, Any] | None = None,
        feature_dict: dict[str, Any] | None = None,
        backend: Backend | None = None,
    ) -> HyperEdgeSet:
        """
        Build a HyperEdgeSet from raw dicts of ports and features.

        Both inputs may be None, in which case the corresponding properties
        are set to None and only `non_fictitious` of length zero is created.

        :param port_dict: Dictionary of ports, each key corresponds to a port name and to the values are the
                             corresponding addresses for each object stored into an array.
        :param feature_dict: Dictionary of features, each key corresponds to a feature name and to the values are the
                             corresponding features for each object stored into an array.
        :param backend: Backend used for array operations.
        :returns: A properly structured `HyperEdgeSet` instance.
        :raises ValueError: If ports or features contain NaNs or if shapes mismatch.
        """
        backend = backend or NumpyBackend()
        # Convert inputs to pure arrays / dicts
        port_dict = check_dict_or_none(port_dict)
        feature_dict = check_dict_or_none(feature_dict)

        check_valid_ports(port_dict, backend)
        check_no_nan(port_dict=port_dict, feature_dict=feature_dict, backend=backend)

        # Build feature_names and feature_array
        if feature_dict is not None:
            feature_names = {name: backend.array(idx) for idx, name in enumerate(sorted(feature_dict))}
            feature_array = dict2array(feature_dict, backend)
        else:
            feature_names, feature_array = None, None

        # Build a non-fictitious mask.
        shape = build_hyper_edge_set_shape(port_dict=port_dict, feature_dict=feature_dict, backend=backend)
        non_fictitious = backend.ones(int(shape))

        return cls(
            port_dict=port_dict,
            feature_array=feature_array,
            feature_names=feature_names,
            non_fictitious=non_fictitious,
            backend=backend,
        )

    def __str__(self) -> str:
        """
        Render the HyperEdgeSet as a pandas DataFrame string.

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
            raise ValueError("HyperEdgeSet is neither single nor batched.")

        d = {}
        if self.port_names is not None:
            for k, v in sorted(self.port_dict.items()):
                d[("ports", k)] = v.reshape([-1])
        if self.feature_names is not None:
            for k, v in sorted(self.feature_dict.items()):
                d[("features", k)] = v.reshape([-1])

        return pd.DataFrame(d, index=index).__str__()

    @property
    def array(self) -> Any:
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
            # Handle dimension mismatch in batch case if needed
            port_arr = self.port_array
            if self.feature_array is not None:
                feat_ndim = len(self._backend.shape(self.feature_array))
                port_ndim = len(self._backend.shape(port_arr))
                if feat_ndim == 3 and port_ndim == 2:
                    # In some JAX versions/tests, port_dict might be a dict of 1D arrays
                    # causing port_array to be 2D while feature_array is 3D.
                    port_arr = port_arr[..., jnp.newaxis] if isinstance(self, JaxHyperEdgeSet) else np.expand_dims(port_arr, axis=-1)
            array.append(port_arr)
        return self._backend.concatenate(array, axis=-1)

    @property
    def is_batch(self) -> bool:
        """
        True if `array` is 3-D: `(batch, n_obj, features+ports)`.
        """
        return len(self._backend.shape(self.array)) == 3

    @property
    def is_single(self) -> bool:
        """
        True if `array` is 2-D: `(n_obj, features+ports)`.
        """
        shape = self._backend.shape(self.array)
        return len(shape) == 2 or (len(shape) == 1 and self.feature_array is None and self.port_dict is None)

    @property
    def n_obj(self) -> int:
        """
        Number of hyper-edges (objects) per instance.
        """
        shape = self._backend.shape(self.array)
        if len(shape) == 2:
            return int(shape[0])
        elif len(shape) == 3:
            return int(shape[1])
        else:
            raise ValueError("HyperEdgeSet is neither single nor batched.")

    @property
    def n_batch(self) -> int:
        """
        Number of batches. Only valid if `is_batch` is True.
        :raises ValueError: If not a batch.
        """
        if self.is_batch:
            return int(self._backend.shape(self.array)[0])
        else:
            raise ValueError("HyperEdgeSet is not batched.")

    @property
    def feature_array(self) -> Any | None:
        return self[FEATURE_ARRAY]

    @feature_array.setter
    def feature_array(self, value: Any) -> None:
        self[FEATURE_ARRAY] = value

    @property
    def feature_names(self) -> dict[str, Any] | None:
        return self[FEATURE_NAMES]

    @property
    def port_array(self) -> Any | None:
        """
        Returns the stacked array of ports, of shape `(n_obj, n_ports)` or `(batch, n_obj, n_ports)`.
        """
        if self.port_dict is None:
            return None
        return dict2array(self.port_dict, self._backend)

    @property
    def port_names(self) -> dict[str, Any] | None:
        """
        Maps a port name to a column index in `port_array`.
        """
        if self.port_dict is None:
            return None
        return {k: self._backend.array(idx) for idx, k in enumerate(sorted(self.port_dict.keys()))}

    @property
    def port_dict(self) -> dict[str, Any] | None:
        return self[PORT_DICT]

    @port_dict.setter
    def port_dict(self, value: dict[str, Any] | None) -> None:
        self[PORT_DICT] = value

    @property
    def non_fictitious(self) -> Any:
        """
        Mask of shape `(n_obj,)` or `(batch, n_obj)`.
        1 = real hyper-edge, 0 = padded/fictitious.
        """
        return self[NON_FICTITIOUS]

    @non_fictitious.setter
    def non_fictitious(self, value: Any) -> None:
        self[NON_FICTITIOUS] = value

    @property
    def feature_dict(self) -> dict[str, Any] | None:
        """
        Unstack `feature_array` into a dict: feature_name --> array.

        :returns: Dict of shape `(n_obj,)` or `(batch, n_obj)` per feature.
        """
        if not self.feature_names:
            return None

        result = dict()
        for k, v in self.feature_names.items():
            # Use backend indexing and shape logic to avoid to_numpy on tracers
            idx = v[0] if self.is_batch else v
            # If backend is JaxBackend, use jnp.take or slicing
            if self.is_batch:
                result[k] = self.feature_array[..., int(idx)]
            else:
                result[k] = self.feature_array[..., int(idx)]
        return result

    @property
    def feature_flat_array(self) -> Any | None:
        """
        Flatten all features into one long vector per `(batch, )` by Fortran ordering.

        :returns:
            Single instance: 1D array of length `n_obj * n_feats`.
            Batched instance: 2D array of shape `(batch, n_obj * n_feats)`.
        """
        if self.feature_array is None:
            return None

        # Check if feature_array has at least 2 dimensions for single or 3 for batch
        # We check ndim directly as properties might be circular or rely on concatenation
        ndim = len(self._backend.shape(self.feature_array))
        if self.is_batch:
            if ndim < 3:
                raise ValueError("feature_array must have at least 3 dimensions for batched edge set.")
        else:
            if ndim < 2:
                raise ValueError("feature_array must have at least 2 dimensions for single edge set.")

        shape = [self.n_batch, -1] if self.is_batch else [-1]
        # NumPy and JAX both support reshape with order='F' but for JAX it might be different
        # Let's use the underlying np module from the backend
        if isinstance(self, JaxHyperEdgeSet):
             # JAX jnp.reshape doesn't support order='F' directly sometimes or in older versions,
             # but it actually does in modern JAX. However, let's be explicit if needed.
             return self._backend.np.reshape(self.feature_array, shape, order="F")
        return self._backend.np.reshape(self.feature_array, shape, order="F")

    @feature_flat_array.setter
    def feature_flat_array(self, array: Any) -> None:
        """
        Update the feature array from a flat Fortran-ordered array.

        :param array: Must match the shape of current `.feature_flat_array`.
        :raises ValueError: If shapes mismatch.
        """
        flat = self.feature_flat_array
        if flat is None or self._backend.shape(flat) != self._backend.shape(array):
            raise ValueError("Shape mismatch for feature_flat_array setter.")
        if self.feature_names is not None:
            if self.is_single:
                self.feature_array = self._backend.np.reshape(array, [self.n_obj, -1], order="F")
            elif self.is_batch:
                self.feature_array = self._backend.np.reshape(array, [self.n_batch, self.n_obj, -1], order="F")

    def pad(self, target_shape: Any | int) -> None:
        """
        Pad a *single* HyperEdgeSet with a series of zeros for features and max-int for ports
        so that shapes match the `target_shape`.

        :param target_shape: Desired n_obj after padding; must be ≥ current n_obj.
        :raises ValueError: If called on a batch or if target_shape < current n_obj.
        """
        if not self.is_single:
            raise ValueError("HyperEdgeSet is batched, impossible to pad.")

        old_n_obj = self.n_obj

        if old_n_obj > target_shape:
            raise ValueError("Provided target_shape is smaller than current shape, padding is impossible! ")

        # Pad features
        if self.feature_array is not None:
            self.feature_array = self._backend.np.pad(self.feature_array, [(0, int(target_shape) - old_n_obj), (0, 0)])

        # Pad ports
        if self.port_dict is not None:
            for k, v in self.port_dict.items():
                self.port_dict[k] = self._backend.np.pad(v, [0, int(target_shape) - old_n_obj])

        # Pad fictitious mask
        if self.non_fictitious is not None:
            self.non_fictitious = self._backend.np.pad(self.non_fictitious, [0, int(target_shape) - old_n_obj])

    def unpad(self, target_shape: Any | int) -> None:
        """
        Remove all objects beyond the index `target` in a *single* HyperEdgeSet.

        :param target_shape: New n_obj; must be ≤ current n_obj.
        :raises ValueError: If called on a batch or if target_shape > current n_obj.
        """

        if not self.is_single:
            raise ValueError("HyperEdgeSet is batched, impossible to unpad.")

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

    def offset_addresses(self, offset: Any | int) -> None:
        """Adds an offset on all addresses. Should only be used before graph concatenation.

        :param offset: Scalar or array to add to each address array.
        """
        self.port_dict = {k: a + self._backend.array(offset) for k, a in self.port_dict.items()}


@register_pytree_node_class
class JaxHyperEdgeSet(HyperEdgeSet):
    def __init__(
        self,
        *,
        port_dict: dict[str, Any] | None,
        feature_array: Any | None,
        feature_names: dict[str, Any] | None,
        non_fictitious: Any,
        backend: Backend | None = None,
    ) -> None:
        # Ignore provided backend and force JAX backend to ensure correct JAX behavior
        super().__init__(
            port_dict=port_dict,
            feature_array=feature_array,
            feature_names=feature_names,
            non_fictitious=non_fictitious,
            backend=JaxBackend(),
        )

    def tree_flatten(self) -> tuple:
        """
        Flattens a PyTree, required for JAX compatibility.
        :returns: a tuple of values and keys
        """
        # Sort port_dict and feature_names for deterministic auxiliary data/children structure
        port_keys = sorted(self[PORT_DICT].keys()) if self[PORT_DICT] is not None else None
        port_values = tuple(self[PORT_DICT][k] for k in port_keys) if port_keys is not None else None
        
        feat_keys = sorted(self[FEATURE_NAMES].keys()) if self[FEATURE_NAMES] is not None else None
        feat_values = tuple(self[FEATURE_NAMES][k] for k in feat_keys) if feat_keys is not None else None

        children = (self[FEATURE_ARRAY], self[NON_FICTITIOUS], port_values, feat_values)
        aux = (port_keys, feat_keys, self._backend)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Sequence[Any]) -> JaxHyperEdgeSet:
        """
        Unflattens a PyTree, required for JAX compatibility.
        """
        feature_array, non_fictitious, port_values, feat_values = children
        port_keys, feat_keys, backend = aux_data
        
        port_dict = dict(zip(port_keys, port_values)) if port_keys is not None else None
        feature_names = dict(zip(feat_keys, feat_values)) if feat_keys is not None else None
        
        return cls(
            port_dict=port_dict,
            feature_array=feature_array,
            feature_names=feature_names,
            non_fictitious=non_fictitious,
        )

    @classmethod
    def from_numpy_hyper_edge_set(
        cls, hyper_edge_set: HyperEdgeSet, device: Any | None = None, dtype: str = "float32"
    ) -> JaxHyperEdgeSet:
        from energnn.graph.utils import np_to_jnp
        port_dict = np_to_jnp(hyper_edge_set.port_dict, device=device, dtype=dtype)
        feature_array = np_to_jnp(hyper_edge_set.feature_array, device=device, dtype=dtype)
        feature_names = np_to_jnp(hyper_edge_set.feature_names, device=device, dtype=dtype)
        non_fictitious = np_to_jnp(hyper_edge_set.non_fictitious, device=device, dtype=dtype)
        return cls(
            port_dict=port_dict, feature_array=feature_array, feature_names=feature_names, non_fictitious=non_fictitious
        )

    def to_numpy_hyper_edge_set(self) -> HyperEdgeSet:
        from energnn.graph.utils import jnp_to_np
        port_dict = jnp_to_np(self.port_dict)
        feature_array = jnp_to_np(self.feature_array)
        feature_names = jnp_to_np(self.feature_names)
        non_fictitious = jnp_to_np(self.non_fictitious)
        return HyperEdgeSet(
            port_dict=port_dict, feature_array=feature_array, feature_names=feature_names, non_fictitious=non_fictitious
        )


def collate_hyper_edge_sets(hyper_edge_set_list: list[HyperEdgeSet]) -> HyperEdgeSet:
    """
    Collate a list of HyperEdgeSet into a single batched HyperEdgeSet.

    Each HyperEdgeSet in the input list is assumed to have the same feature and port schema.
    This function stacks the per-edge attributes along the 0-th axis.

    :param hyper_edge_set_list: Sequence of HyperEdgeSet objects to batch together. Must be non-empty.
    :return: A single batched HyperEdgeSet.

    :raises IndexError: Raised if `hyper_edge_set_list` is empty.
    :raises ValueError: Raised if not all HyperEdgeSet share the same keys in port_names or feature_names.
    """
    if not hyper_edge_set_list:
        raise IndexError("collate_edges requires at least one Edge to collate.")

    first_hyper_edge_set = hyper_edge_set_list[0]
    backend = first_hyper_edge_set._backend

    # Check the consistency of keys
    for e in hyper_edge_set_list[1:]:
        _check_keys_consistency(first_hyper_edge_set, e)

    # Collate feature arrays
    if first_hyper_edge_set.feature_array is not None:
        feature_array = backend.stack([e.feature_array for e in hyper_edge_set_list], axis=0)
    else:
        feature_array = None

    # Collate feature names
    if first_hyper_edge_set.feature_names is not None:
        feature_names = {
            k: backend.stack([e.feature_names[k] for e in hyper_edge_set_list], axis=0)
            for k in first_hyper_edge_set.feature_names
        }
    else:
        feature_names = None

    # Collate port dicts
    if first_hyper_edge_set.port_dict is not None:
        port_dict = {
            k: backend.stack([e.port_dict[k] for e in hyper_edge_set_list], axis=0)
            for k in first_hyper_edge_set.port_dict
        }
    else:
        port_dict = None

    # Collate non-fictitious masks
    if first_hyper_edge_set.non_fictitious is not None:
        non_fictitious = backend.stack([e.non_fictitious for e in hyper_edge_set_list], axis=0)
    else:
        non_fictitious = None

    if isinstance(first_hyper_edge_set, JaxHyperEdgeSet):
        return JaxHyperEdgeSet(
            port_dict=port_dict,
            feature_array=feature_array,
            feature_names=feature_names,
            non_fictitious=non_fictitious,
        )
    return HyperEdgeSet(
        port_dict=port_dict,
        feature_array=feature_array,
        feature_names=feature_names,
        non_fictitious=non_fictitious,
        backend=backend,
    )


def separate_hyper_edge_sets(hyper_edge_set_batch: HyperEdgeSet) -> list[HyperEdgeSet]:
    """
    Separate a batched HyperEdgeSet into its constituent HyperEdgeSet instances.

    The input HyperEdgeSet must have been created by :py:func:`collate_hyper_edge_sets` or otherwise
    its property "array" must return a 3D array.

    :param hyper_edge_set_batch: The batched HyperEdgeSet to unstack.
    :return: List of HyperEdgeSet instances, each corresponding to one batch element.

    :raises ValueError: If `hyper_edge_set_batch.is_batch` is False.
    """
    if not hyper_edge_set_batch.is_batch:
        raise ValueError("Input is not a batch, impossible to separate.")

    backend = hyper_edge_set_batch._backend

    if hyper_edge_set_batch.feature_array is not None:
        feature_array_list = backend.unstack(hyper_edge_set_batch.feature_array, axis=0)
    else:
        feature_array_list = [None] * hyper_edge_set_batch.n_batch

    if hyper_edge_set_batch.feature_names is not None:
        a = {k: backend.unstack(hyper_edge_set_batch.feature_names[k], axis=0) for k in hyper_edge_set_batch.feature_names}
        keys = list(a.keys())
        feature_names_list = [{k: a[k][i] for k in keys} for i in range(hyper_edge_set_batch.n_batch)]
    else:
        feature_names_list = [None] * hyper_edge_set_batch.n_batch

    if hyper_edge_set_batch.port_dict is not None:
        a = {k: backend.unstack(hyper_edge_set_batch.port_dict[k], axis=0) for k in hyper_edge_set_batch.port_dict}
        keys = list(a.keys())
        port_dict_list = [{k: a[k][i] for k in keys} for i in range(hyper_edge_set_batch.n_batch)]
    else:
        port_dict_list = [None] * hyper_edge_set_batch.n_batch

    if hyper_edge_set_batch.non_fictitious is not None:
        non_fictitious_list = backend.unstack(hyper_edge_set_batch.non_fictitious, axis=0)
    else:
        non_fictitious_list = [None] * hyper_edge_set_batch.n_batch

    hyper_edge_set_list = []
    for fa, fn, ad, nf in zip(feature_array_list, feature_names_list, port_dict_list, non_fictitious_list):
        if isinstance(hyper_edge_set_batch, JaxHyperEdgeSet):
            hyper_edge_set = JaxHyperEdgeSet(port_dict=ad, feature_array=fa, feature_names=fn, non_fictitious=nf)
        else:
            hyper_edge_set = HyperEdgeSet(
                port_dict=ad, feature_array=fa, feature_names=fn, non_fictitious=nf, backend=backend
            )
        hyper_edge_set_list.append(hyper_edge_set)
    return hyper_edge_set_list


def concatenate_hyper_edge_sets(hyper_edge_set_list: list[HyperEdgeSet]) -> HyperEdgeSet:
    """
    Concatenate several single HyperEdgeSet into one single HyperEdgeSet.

    Unlike :py:func:`collate_hyper_edge_sets`, this does *not* create a batch dimension,
    but simply stacks objects end-to-end.

    :param hyper_edge_set_list: List of single (non-batched) HyperEdgeSet
    :returns: One HyperEdgeSet with n_obj = sum of all inputs’ n_obj
    """
    first_hes = hyper_edge_set_list[0]
    backend = first_hes._backend
    
    port_dict = {
        k: backend.concatenate([hes.port_dict[k] for hes in hyper_edge_set_list], axis=0)
        for k in first_hes.port_dict
    }
    feature_array = backend.concatenate([hes.feature_array for hes in hyper_edge_set_list], axis=0)
    feature_names = first_hes.feature_names
    non_fictitious = backend.concatenate([hes.non_fictitious for hes in hyper_edge_set_list], axis=0)
    
    if isinstance(first_hes, JaxHyperEdgeSet):
        return JaxHyperEdgeSet(
            port_dict=port_dict, feature_array=feature_array, feature_names=feature_names, non_fictitious=non_fictitious
        )
    return HyperEdgeSet(
        port_dict=port_dict,
        feature_array=feature_array,
        feature_names=feature_names,
        non_fictitious=non_fictitious,
        backend=backend,
    )


def check_dict_shape(*, d: dict[str, Any] | None, n_objects: int | None, backend: Backend | None = None) -> int | None:
    """
    Ensure all arrays in a dictionary have the same size on their last axis.
    """
    backend = backend or NumpyBackend()
    if d is not None:
        if n_objects is None:
            item = next(iter(d.values()))
            n_objects = backend.shape(item)[-1]
        for name, arr in d.items():
            if backend.shape(arr)[-1] != n_objects:
                raise ValueError(f"Array for key '{name}' has last dimension {backend.shape(arr)[-1]}, expected {n_objects}.")
    return n_objects


def build_hyper_edge_set_shape(
    *,
    port_dict: dict[str, Any] | None,
    feature_dict: dict[str, Any] | None,
    backend: Backend | None = None,
) -> Any:
    """
    Builds an array representing the number of hyper-edges.
    """
    backend = backend or NumpyBackend()
    if port_dict is None and feature_dict is None:
        raise ValueError("At least one of port_dict or feature_dict must be provided.")

    n_objects = check_dict_shape(d=port_dict, n_objects=None, backend=backend)
    n_objects = check_dict_shape(d=feature_dict, n_objects=n_objects, backend=backend)
    return backend.array(n_objects, dtype="float32")


def dict2array(features_dict: dict[str, Any] | None, backend: Backend | None = None) -> Any | None:
    """
    Stack a dictionary of arrays into a single array along the last axis.
    """
    backend = backend or NumpyBackend()
    if features_dict is None:
        return None
    return backend.stack([features_dict[k] for k in sorted(features_dict)], axis=-1)


def check_dict_or_none(_input: dict | Any | None) -> dict | None:
    """
    Validate that the input is either a dict or None.
    """
    if isinstance(_input, dict):
        return _input
    if _input is None:
        return None
    raise ValueError(f"Expected dict or None, got {type(_input)}")


def check_no_nan(
    *,
    port_dict: dict[str, Any] | None,
    feature_dict: dict[str, Any] | None,
    backend: Backend | None = None,
) -> None:
    """
    Ensure there are no NaN values in port or feature arrays.
    """
    backend = backend or NumpyBackend()
    for name, arr in (port_dict or {}).items():
        if backend.any(backend.isnan(arr)):
            raise ValueError(f"NaN detected in port array for key '{name}'.")
    for name, arr in (feature_dict or {}).items():
        if backend.any(backend.isnan(arr)):
            raise ValueError(f"NaN detected in feature array for key '{name}'.")


def check_valid_ports(port_dict: dict[str, Any] | None, backend: Backend | None = None) -> None:
    """
    Ensure that ports map only to integer-valued addresses.
    """
    backend = backend or NumpyBackend()
    for name, arr in (port_dict or {}).items():
        # Using np.allclose might be tricky for JAX backend if we don't have it in Backend
        # but most backends support basic comparison
        if not backend.all(backend.np.isclose(arr, backend.array(arr, dtype="int32"))):
             raise ValueError(f"Non-integer values detected in port array for key '{name}'.")


def _check_keys_consistency(hes_1, hes_2):
    if (hes_1.port_names is None) != (hes_2.port_names is None):
        raise ValueError("Mismatch in presence of port_names among hyper-edge sets.")
    if (hes_1.feature_names is None) != (hes_2.feature_names is None):
        raise ValueError("Mismatch in presence of feature_names among hyper-edge sets.")
    if hes_1.port_names and hes_1.port_names.keys() != hes_2.port_names.keys():
        raise ValueError("Inconsistent port_names keys among hyper-edge sets.")
    if hes_1.feature_names and hes_1.feature_names.keys() != hes_2.feature_names.keys():
        raise ValueError("Inconsistent feature_names keys among hyper-edge sets.")

# Backward compatibility aliases
collate_hyper_edge_sets_jax = collate_hyper_edge_sets
separate_hyper_edge_sets_jax = separate_hyper_edge_sets
concatenate_hyper_edge_sets_jax = concatenate_hyper_edge_sets
check_dict_shape_jax = check_dict_shape
build_hyper_edge_set_shape_jax = build_hyper_edge_set_shape
dict2array_jax = dict2array
check_dict_or_none_jax = check_dict_or_none
check_no_nan_jax = check_no_nan
check_valid_ports_jax = check_valid_ports
_check_keys_consistency_jax = _check_keys_consistency
