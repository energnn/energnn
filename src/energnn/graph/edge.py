# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from energnn.graph.utils import to_numpy

FEATURE_ARRAY = "feature_array"
FEATURE_NAMES = "feature_names"
ADDRESS_DICT = "address_dict"
NON_FICTITIOUS = "non_fictitious"


class Edge(dict):
    """
    A collection of hyper-edges of the same class, optionally batched.

    Internally this is just a dict storing four entries.

    :param address_dict: Mapping from port name to array of shape (n_edges,) or (batch, n_edges).
        Dictionary that contains all hyper-edge addresses.
    :param feature_array: Array that contains all hyper-edge features.
    :param feature_names: Dictionary from feature names to  index in `feature_array`.
    :param non_fictitious: Mask array set to 1 for non fictitious objects and to 0 for fictitious objects.
    """

    def __init__(
        self,
        *,
        address_dict: dict[str, np.ndarray] | None,
        feature_array: np.ndarray | None,
        feature_names: dict[str, int] | None,
        non_fictitious: np.ndarray,
    ) -> None:
        super().__init__()
        self[ADDRESS_DICT] = address_dict
        self[FEATURE_ARRAY] = feature_array
        self[FEATURE_NAMES] = feature_names
        self[NON_FICTITIOUS] = non_fictitious

    @classmethod
    def from_dict(
        cls,
        *,
        address_dict: dict[str, Any] | None = None,
        feature_dict: dict[str, Any] | None = None,
    ) -> Edge:
        """
        Build an Edge from raw dicts of addresses and features.

        Both inputs may be None, in which case the corresponding properties
        are set to None and only `non_fictitious` of length zero is created.

        :param address_dict: Dictionary of addresses, each key correspond to a port name and the values are the
        corresponding addresses for each object stored into an array.
        :param feature_dict: Dictionary of features, each key correspond to a feature name and the values are the
        corresponding features for each object stored into an array.
        :returns: A properly structured `Edge` instance.
        :raises ValueError: If addresses or features contain NaNs or if shapes mismatch.
        """
        # Convert inputs to pure numpy arrays / dicts
        address_dict = check_dict_or_none(to_numpy(address_dict))
        feature_dict = check_dict_or_none(to_numpy(feature_dict))

        check_valid_addresses(address_dict)
        check_no_nan(address_dict=address_dict, feature_dict=feature_dict)

        # Build feature_names and feature_array
        if feature_dict is not None:
            feature_names = {name: idx for idx, name in enumerate(sorted(feature_dict))}
            feature_array = dict2array(feature_dict)
        else:
            feature_names, feature_array = None, None

        # Build non_fictitious mask
        shape = build_edge_shape(address_dict=address_dict, feature_dict=feature_dict)
        non_fictitious = np.ones(int(shape))

        return cls(
            address_dict=address_dict,
            feature_array=feature_array,
            feature_names=feature_names,
            non_fictitious=non_fictitious,
        )

    def __str__(self) -> str:
        """
        Render the Edge as a pandas DataFrame string.

        If `is_single`, uses a single-level index:
            object_id
        If `is_batch`, uses two-level index:
            batch_id, object_id

        :returns:
            String representation of a pandas.DataFrame.
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
            raise ValueError("Edge is neither a single nor a batched instance.")

        d = {}
        if self.address_names is not None:
            for k, v in sorted(self.address_dict.items()):
                d[("addresses", k)] = v.reshape([-1])
        if self.feature_names is not None:
            for k, v in sorted(self.feature_dict.items()):
                d[("features", k)] = v.reshape([-1])

        return pd.DataFrame(d, index=index).__str__()

    @property
    def array(self) -> np.ndarray:
        """
        Concatenate (features, addresses) along the last axis.

        :returns:
            Combined array of shape
            - single: (n_obj, n_feats + n_ports)
            - batch:  (batch, n_obj, n_feats + n_ports)
        """
        array = []
        if self.feature_array is not None:
            array.append(self.feature_array)
        if self.address_array is not None:
            array.append(self.address_array)
        return np.concatenate(array, axis=-1)

    @property
    def is_batch(self) -> bool:
        """
        True if `array` is 3-D: (batch, n_obj, features+ports).
        """
        return len(self.array.shape) == 3

    @property
    def is_single(self) -> bool:
        """
        True if `array` is 2-D: (n_obj, features+ports).
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
            raise ValueError("Edge is neither a single edge nor a batched edge.")

    @property
    def n_batch(self) -> int:
        """
        Number of batches. Only valid if `is_batch` is True.
        :raises ValueError: if not a batch.
        """
        if self.is_batch:
            return int(self.array.shape[0])
        else:
            raise ValueError("Edge is not batched.")

    @property
    def feature_array(self) -> np.ndarray | None:
        return self[FEATURE_ARRAY]

    @feature_array.setter
    def feature_array(self, value: np.ndarray) -> None:
        self[FEATURE_ARRAY] = value

    @property
    def feature_names(self) -> dict[str, np.ndarray] | None:
        return self[FEATURE_NAMES]

    @property
    def address_array(self) -> np.ndarray | None:
        """
        Return stacked addresses array of shape
        (n_obj, n_ports) or (batch, n_obj, n_ports).
        """
        if self.address_dict is None:
            return None
        return dict2array(self.address_dict)

    @property
    def address_names(self) -> dict[str, np.ndarray] | None:
        """
        Map port name to column index in `address_array`.
        """
        if self.address_dict is None:
            return None
        return {k: np.array(idx) for idx, k in enumerate(sorted(self.address_dict.keys()))}

    @property
    def address_dict(self) -> dict[str, np.ndarray] | None:
        return self[ADDRESS_DICT]

    @address_dict.setter
    def address_dict(self, value: dict[str, np.ndarray] | None) -> None:
        self[ADDRESS_DICT] = value

    @property
    def non_fictitious(self) -> np.ndarray:
        """
        Mask of shape (n_obj,) or (batch, n_obj).
        1 = real edge, 0 = padded/fictitious.
        """
        return self[NON_FICTITIOUS]

    @non_fictitious.setter
    def non_fictitious(self, value: np.ndarray) -> None:
        self[NON_FICTITIOUS] = value

    @property
    def feature_dict(self) -> dict[str, np.ndarray] | None:
        """
        Un-stack `feature_array` into a dict: feature_name --> array.

        :returns: dict of shape (n_obj,) or (batch, n_obj) per feature.
        """
        if not self.feature_names:
            return None

        result = dict()
        for k, v in self.feature_names.items():
            # last axis holds features
            if self.is_batch:
                result[k] = self.feature_array[..., int(v[0])]
            else:
                result[k] = self.feature_array[..., int(v)]
        return result

    @property
    def feature_flat_array(self) -> np.ndarray | None:
        """
        Flatten all features into one long vector per (batch,) by Fortran ordering.

        :returns:
            Single instance: 1D array of length n_obj * n_feats.
            Batched instance: 2D array of shape (batch, n_obj * n_feats).
        """
        if self.feature_array is None:
            return None

        shape = [self.n_batch, -1] if self.is_batch else -1
        return self.feature_array.reshape(shape, order="F")

    @feature_flat_array.setter
    def feature_flat_array(self, array: np.ndarray) -> None:
        """
        Update the feature array from a flat Fortran-ordered array.

        :param array: Must match the shape of current `.feature_flat_array`.
        :raises ValueError: if shapes mismatch.
        """
        flat = self.feature_flat_array
        if flat is None or flat.shape != array.shape:
            raise ValueError("Shape mismatch for feature_flat_array setter.")
        if self.feature_names is not None:
            if self.is_single:
                self.feature_array = array.reshape([self.n_obj, -1], order="F")
            elif self.is_batch:
                self.feature_array = array.reshape([self.n_batch, self.n_obj, -1], order="F")

    def pad(self, target_shape: np.ndarray | int) -> None:
        """
        Pad a *single* Edge with a series of zeros for features and max-int for addresses
         so that shapes match the `target_shape`.

        :param target_shape: desired n_obj after padding; must be ≥ current n_obj.
        :raises ValueError: If called on a batch or if target_shape < current n_obj.
        """
        if not self.is_single:
            raise ValueError("Edge is batched, impossible to pad.")

        old_n_obj = self.n_obj

        if old_n_obj > target_shape:
            raise ValueError("Provided target_shape is smaller than current shape, padding is impossible! ")

        # Pad features
        if self.feature_array is not None:
            self.feature_array = np.pad(self.feature_array, [(0, int(target_shape) - old_n_obj), (0, 0)])

        # Pad addresses
        if self.address_dict is not None:
            for k, v in self.address_dict.items():
                self.address_dict[k] = np.pad(v, [0, int(target_shape) - old_n_obj])

        # Pad fictitious mask
        if self.non_fictitious is not None:
            self.non_fictitious = np.pad(self.non_fictitious, [0, int(target_shape) - old_n_obj])

    def unpad(self, target_shape: np.ndarray | int) -> None:
        """
        Remove all objects beyond index `target` in a *single* Edge.

        :param target_shape: new n_obj; must be ≤ current n_obj.
        :raises ValueError: If called on a batch or if target_shape > current n_obj.
        """

        """Removes fictitious objects."""
        if not self.is_single:
            raise ValueError("Edge is batched, impossible to unpad.")

        if self.n_obj < target_shape:
            raise ValueError("Provided target_shape is higher than current shape, unpadding is impossible! ")

        # Unpad features
        if self.feature_array is not None:
            self.feature_array = self.feature_array[: int(target_shape)]

        # Unpad addresses
        if self.address_dict is not None:
            for k, v in self.address_dict.items():
                self.address_dict[k] = v[: int(target_shape)]

        # Unpad fictitious mask
        if self.non_fictitious is not None:
            self.non_fictitious = self.non_fictitious[: int(target_shape)]

    def offset_addresses(self, offset: np.ndarray | int) -> None:
        """Adds an offset on all addresses. Should only be used before graph concatenation.

        :param offset: scalar or array to add to each address array.
        """
        self.address_dict = {k: a + np.array(offset) for k, a in self.address_dict.items()}


def collate_edges(edge_list: list[Edge]) -> Edge:
    """
    Collate a list of Hyper Edges into a single batched Hyper Edge.

    Each Edge in the input list is assumed to have the same feature and address schema.
    This function stacks the per-edge attributes along the 0-th axis.

    :param edge_list: sequence of Edge objects to batch together. Must be non-empty.
    :return: a single batched Hyper Edge

    :raises IndexError: if `edge_list` is empty.
    :raises ValueError: if not all edges share the same keys in address_names or feature_names.
    """
    if not edge_list:
        raise IndexError("collate_edges requires at least one Edge to collate.")

    first_edge = edge_list[0]

    # Check consistency of keys
    for e in edge_list[1:]:
        _check_keys_consistency(first_edge, e)

    # Collate feature arrays
    if first_edge.feature_array is not None:
        feature_array = np.stack([e.feature_array for e in edge_list], axis=0)
    else:
        feature_array = None

    # Collate feature names
    if first_edge.feature_names is not None:
        feature_names = {k: np.stack([e.feature_names[k] for e in edge_list]) for k in first_edge.feature_names}
    else:
        feature_names = None

    # Collate address dicts
    if first_edge.address_dict is not None:
        address_dict = {k: np.stack([e.address_dict[k] for e in edge_list]) for k in first_edge.address_dict}
    else:
        address_dict = None

    # Collate non fictitious masks
    if first_edge.non_fictitious is not None:
        non_fictitious = np.stack([e.non_fictitious for e in edge_list])
    else:
        non_fictitious = None

    return Edge(
        address_dict=address_dict, feature_array=feature_array, feature_names=feature_names, non_fictitious=non_fictitious
    )


def separate_edges(edge_batch: Edge) -> list[Edge]:
    """
    Separate a batched Hyper Edge into its constituent Hyper Edges instances.

    The input Edge must have been created by :py:func:`collate_edges` or otherwise
    its property "array" must return a 3D array.

    :param edge_batch: the batched Edge to unstack.
    :return: list of Edge instances, each corresponding to one batch element.

    :raises ValueError: if `edge_batch.is_batch` is False.
    """
    if not edge_batch.is_batch:
        raise ValueError("Input is not a batch, impossible to separate.")

    if edge_batch.feature_array is not None:
        feature_array_list = np.unstack(edge_batch.feature_array, axis=0)
    else:
        feature_array_list = [None] * edge_batch.n_batch

    if edge_batch.feature_names is not None:
        a = {k: np.unstack(edge_batch.feature_names[k]) for k in edge_batch.feature_names}
        feature_names_list = [dict(zip(a, t)) for t in zip(*a.values())]  # TODO : vérifier que ça fonctionne comme on veut.
        # feature_names_list = np.unstack(edge_batch.feature_names, axis=0)
    else:
        feature_names_list = [None] * edge_batch.n_batch

    if edge_batch.address_dict is not None:
        a = {k: np.unstack(edge_batch.address_dict[k]) for k in edge_batch.address_dict}
        address_dict_list = [dict(zip(a, t)) for t in zip(*a.values())]  # TODO : vérifier que ça fonctionne comme on veut.
        # address_dict_list = np.unstack(edge_batch.address_dict, axis=0)
    else:
        address_dict_list = [None] * edge_batch.n_batch

    if edge_batch.non_fictitious is not None:
        non_fictitious_list = np.unstack(edge_batch.non_fictitious, axis=0)
    else:
        non_fictitious_list = [None] * edge_batch.n_batch

    edge_list = []
    for fa, fn, ad, nf in zip(feature_array_list, feature_names_list, address_dict_list, non_fictitious_list):
        edge = Edge(address_dict=ad, feature_array=fa, feature_names=fn, non_fictitious=nf)
        edge_list.append(edge)
    return edge_list


def concatenate_edges(edge_list: list[Edge]) -> Edge:
    """
    Concatenate several single Edges into one single Edge.

    Unlike :py:func:`collate_edges`, this does *not* create a batch dimension,
    but simply stacks objects end-to-end.

    :param edge_list: list of single (non-batched) Edge
    :returns: One Edge with n_obj = sum of all inputs’ n_obj
    """
    address_dict = {k: np.concatenate([edge.address_dict[k] for edge in edge_list]) for k in edge_list[0].address_dict}
    feature_array = np.concatenate([edge.feature_array for edge in edge_list], axis=0)
    feature_names = edge_list[0].feature_names
    non_fictitious = np.concatenate([edge.non_fictitious for edge in edge_list])
    return Edge(
        address_dict=address_dict, feature_array=feature_array, feature_names=feature_names, non_fictitious=non_fictitious
    )


def check_dict_shape(*, d: dict[str, np.ndarray] | None, n_objects: int | None) -> int | None:
    """
    Ensure all arrays in a dictionary have the same size on their last axis.

    If `n_objects` is not provided, it is inferred from the first array’s last dimension.
    Otherwise, every array’s last dimension must match the given `n_objects`.

    :param d: mapping from feature/address name to numpy array
                   where each array’s last axis is object-indexed.
    :param n_objects: optional expected size of the last axis; if None, will be inferred.
    :return: the validated or inferred `n_objects`.

    :raises ValueError: if any array’s last dimension does not match `n_objects`.
    """
    if d is not None:
        if n_objects is None:
            item: np.ndarray = next(iter(d.values()))
            n_objects = item.shape[-1]
        for name, arr in d.items():
            if arr.shape[-1] != n_objects:
                raise ValueError(f"Array for key '{name}' has last dimension {arr.shape[-1]}, expected {n_objects}.")
    return n_objects


def build_edge_shape(
    *,
    address_dict: dict[str, np.ndarray] | None,
    feature_dict: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """
    Build a numpy array representing the number of edges.

    Validates that `address_dict` and `feature_dict` have consistent sizes
    on their last dimensions, and returns a scalar numpy array containing that count.

    :param address_dict: mapping of address names to numpy arrays, or None.
    :param feature_dict: mapping of feature names to numpy arrays, or None.
    :return: a scalar numpy array of dtype float32 with the number of objects.
    :raises ValueError: if both inputs are None, or if their shapes conflict.
    """
    if address_dict is None and feature_dict is None:
        raise ValueError("At least one of address_dict or feature_dict must be provided.")

    n_objects = check_dict_shape(d=address_dict, n_objects=None)
    n_objects = check_dict_shape(d=feature_dict, n_objects=n_objects)
    return np.array(n_objects, dtype=np.dtype("float32"))


def dict2array(features_dict: dict[str, np.ndarray] | None) -> np.ndarray | None:
    """
    Stack a dictionary of arrays into a single array along the last axis.

    The arrays are stacked in alphabetical order of their dictionary keys.

    :param features_dict: mapping from feature name to numpy array, or None.
    :return: a stacked array with an added last dimension for features, or None.
    """
    if features_dict is None:
        return None
    return np.stack([features_dict[k] for k in sorted(features_dict)], axis=-1)


def check_dict_or_none(_input: dict | np.ndarray | None) -> dict | None:
    """
    Validate that the input is either a dict or None.

    :param _input: object to validate
    :return: the input if it was a dict or None
    :raises ValueError: if `_input` is neither dict nor None
    """
    if isinstance(_input, dict):
        return _input
    if _input is None:
        return None
    raise ValueError(f"Expected dict or None, got {type(_input)}")


def check_no_nan(
    *,
    address_dict: dict[str, np.ndarray] | None,
    feature_dict: dict[str, np.ndarray] | None,
) -> None:
    """
    Ensure there are no NaN values in address or feature arrays.

    :param address_dict: mapping of address names to arrays, or None.
    :param feature_dict: mapping of feature names to arrays, or None.
    :raises ValueError: if any array contains NaN.
    """
    for name, arr in (address_dict or {}).items():
        if np.any(np.isnan(arr)):
            raise ValueError(f"NaN detected in address array for key '{name}'.")
    for name, arr in (feature_dict or {}).items():
        if np.any(np.isnan(arr)):
            raise ValueError(f"NaN detected in feature array for key '{name}'.")


def check_valid_addresses(address_dict: dict[str, np.ndarray] | None) -> None:
    """
    Ensure that address arrays only contain integer-valued entries.

    :param address_dict: mapping of address names to arrays, or None.
    :raises ValueError: if any address array has non-integer entries.
    """
    for name, arr in (address_dict or {}).items():
        if not np.allclose(arr, np.int32(arr)):
            raise ValueError(f"Non-integer values detected in address array for key '{name}'.")


def _check_keys_consistency(edge_1, edge_2):
    if (edge_1.address_names is None) != (edge_2.address_names is None):
        raise ValueError("Mismatch in presence of address_names among edges.")
    if (edge_1.feature_names is None) != (edge_2.feature_names is None):
        raise ValueError("Mismatch in presence of feature_names among edges.")
    if edge_1.address_names and edge_1.address_names.keys() != edge_2.address_names.keys():
        raise ValueError("Inconsistent address_names keys among edges.")
    if edge_1.feature_names and edge_1.feature_names.keys() != edge_2.feature_names.keys():
        raise ValueError("Inconsistent feature_names keys among edges.")
