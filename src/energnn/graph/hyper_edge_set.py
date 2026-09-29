# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from jax.tree_util import register_pytree_node_class

from energnn.graph.backend import PRESERVE_DTYPE, Backend, NumpyBackend
from energnn.graph.formatting import format_hyper_edge_set
from energnn.graph.utils import to_numpy

FEATURE_ARRAY = "feature_array"
FEATURE_NAMES = "feature_names"
PORT_DICT = "port_dict"
NON_FICTITIOUS = "non_fictitious"


@register_pytree_node_class
class HyperEdgeSet(dict):
    """
    A collection of hyper-edges of the same class, optionally batched.

    Internally this is a dict storing four entries.  All array operations are
    delegated to the provided *backend*, making instances transparent to both
    NumPy and JAX pipelines.

    :param backend: Array backend (:class:`NumpyBackend` or :class:`JaxBackend`).
    :param port_dict: Mapping from a port name to an integer address array of shape
                      ``(n_edges,)`` or ``(batch, n_edges)``.
    :param feature_array: Array that contains all hyper-edge features.
    :param feature_names: Dictionary from feature names to index in ``feature_array``.
    :param non_fictitious: Mask array set to 1 for real objects and 0 for fictitious ones.
    """

    def __init__(
        self,
        *,
        backend: Backend | None = None,
        port_dict: dict | None,
        feature_array,
        feature_names: dict | None,
        non_fictitious,
    ) -> None:
        super().__init__()
        self._backend: Backend = backend if backend is not None else NumpyBackend()
        self[PORT_DICT] = port_dict
        self[FEATURE_ARRAY] = feature_array
        self[FEATURE_NAMES] = feature_names
        self[NON_FICTITIOUS] = non_fictitious

    # ------------------------------------------------------------------
    # JAX PyTree protocol
    # ------------------------------------------------------------------

    def tree_flatten(self):
        children = list(self.values())
        aux = (tuple(self.keys()), self._backend)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: Sequence[Any]) -> HyperEdgeSet:
        keys, backend = aux_data
        d = dict(zip(keys, children))
        return cls(
            backend=backend,
            port_dict=d[PORT_DICT],
            feature_array=d[FEATURE_ARRAY],
            feature_names=d[FEATURE_NAMES],
            non_fictitious=d[NON_FICTITIOUS],
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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

        Ports are stored as ``int32`` (integer-valued float inputs are accepted and cast);
        features are stored as ``float32``.

        :param port_dict: Port-name → address array mapping.
        :param feature_dict: Feature-name → feature array mapping.
        :param backend: Array backend to use.  Defaults to :class:`NumpyBackend`.
        :returns: A properly structured ``HyperEdgeSet`` instance.
        :raises ValueError: If ports or features contain NaNs, if ports have fractional
            values, or if shapes mismatch.
        """
        if backend is None:
            backend = NumpyBackend()
        xp = backend.xp

        # Validate on numpy (safe for both input types)
        port_dict_np = check_dict_or_none(to_numpy(port_dict))
        feature_dict_np = check_dict_or_none(to_numpy(feature_dict))

        check_valid_ports(port_dict_np)
        check_no_nan(port_dict=port_dict_np, feature_dict=feature_dict_np)

        # Ports are indices: int32. Features are values: float32.
        if port_dict_np is not None:
            port_dict_np = {k: v.astype(np.int32) for k, v in port_dict_np.items()}
        if feature_dict_np is not None:
            feature_dict_np = {k: v.astype(np.float32) for k, v in feature_dict_np.items()}

        # Convert to backend arrays
        port_dict_b = {k: xp.array(v) for k, v in port_dict_np.items()} if port_dict_np is not None else None

        if feature_dict_np is not None:
            feature_names: dict | None = {name: idx for idx, name in enumerate(sorted(feature_dict_np))}
            feature_array = xp.stack([xp.array(feature_dict_np[k]) for k in sorted(feature_dict_np)], axis=-1)
        else:
            feature_names, feature_array = None, None

        n_objects = _compute_n_objects(port_dict_np, feature_dict_np)
        non_fictitious = xp.ones(n_objects)

        return cls(
            backend=backend,
            port_dict=port_dict_b,
            feature_array=feature_array,
            feature_names=feature_names,
            non_fictitious=non_fictitious,
        )

    # ------------------------------------------------------------------
    # Backend conversion
    # ------------------------------------------------------------------

    def to_backend(self, new_backend: Backend) -> HyperEdgeSet:
        """Return a copy of this ``HyperEdgeSet`` with arrays converted to ``new_backend``.

        The backend's floating dtype only applies to ``feature_array`` and ``non_fictitious``;
        ports and feature-name indices are integers and keep their dtype.
        """
        port_dict_b = (
            {k: new_backend.from_numpy(np.asarray(v), dtype=PRESERVE_DTYPE) for k, v in self.port_dict.items()}
            if self.port_dict is not None
            else None
        )
        feature_array_b = new_backend.from_numpy(np.array(self.feature_array)) if self.feature_array is not None else None
        feature_names_b = (
            {k: new_backend.from_numpy(np.asarray(v), dtype=PRESERVE_DTYPE) for k, v in self.feature_names.items()}
            if self.feature_names is not None
            else None
        )
        non_fictitious_b = new_backend.from_numpy(np.array(self.non_fictitious))
        return type(self)(
            backend=new_backend,
            port_dict=port_dict_b,
            feature_array=feature_array_b,
            feature_names=feature_names_b,
            non_fictitious=non_fictitious_b,
        )

    @classmethod
    def from_numpy_hyper_edge_set(cls, hyper_edge_set: HyperEdgeSet, device=None, dtype: str = "float32") -> HyperEdgeSet:
        """Convert a NumPy-backed ``HyperEdgeSet`` to a JAX-backed one."""
        from energnn.graph.backend import JaxBackend

        return hyper_edge_set.to_backend(JaxBackend(device=device, dtype=dtype))

    def to_numpy_hyper_edge_set(self) -> HyperEdgeSet:
        """Convert this ``HyperEdgeSet`` to a NumPy-backed one."""
        return self.to_backend(NumpyBackend())

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return format_hyper_edge_set(self)

    def _repr_pretty_(self, p, cycle: bool) -> None:
        """Display the formatted text in IPython/Jupyter instead of the raw dict ``repr``."""
        p.text("..." if cycle else str(self))

    # ------------------------------------------------------------------
    # Core array properties
    # ------------------------------------------------------------------

    @property
    def array(self):
        """Concatenate (features, ports) along the last axis."""
        xp = self._backend.xp
        parts = []
        if self.feature_array is not None:
            parts.append(self.feature_array)
        if self.port_array is not None:
            parts.append(self.port_array)
        return xp.concatenate(parts, axis=-1)

    @property
    def is_batch(self) -> bool:
        """True if ``array`` is 3-D: ``(batch, n_obj, features+ports)``."""
        return len(self.array.shape) == 3

    @property
    def is_single(self) -> bool:
        """True if ``array`` is 2-D: ``(n_obj, features+ports)``."""
        return len(self.array.shape) == 2

    @property
    def n_obj(self) -> int:
        """Number of hyper-edges per instance."""
        if self.is_single:
            return int(self.array.shape[0])
        elif self.is_batch:
            return int(self.array.shape[1])
        else:
            raise ValueError("HyperEdgeSet is neither single nor batched.")

    @property
    def n_batch(self) -> int:
        """Number of batches; valid only when ``is_batch`` is True."""
        if self.is_batch:
            return int(self.array.shape[0])
        raise ValueError("HyperEdgeSet is not batched.")

    # ------------------------------------------------------------------
    # Dict-entry accessors
    # ------------------------------------------------------------------

    @property
    def feature_array(self):
        return self[FEATURE_ARRAY]

    @feature_array.setter
    def feature_array(self, value) -> None:
        self[FEATURE_ARRAY] = value

    @property
    def feature_names(self) -> dict | None:
        return self[FEATURE_NAMES]

    @property
    def port_array(self):
        """Stacked port array of shape ``(n_obj, n_ports)`` or ``(batch, n_obj, n_ports)``."""
        if self.port_dict is None:
            return None
        xp = self._backend.xp
        return xp.stack([self.port_dict[k] for k in sorted(self.port_dict)], axis=-1)

    @property
    def port_names(self) -> dict | None:
        """Maps port name to column index in ``port_array``."""
        if self.port_dict is None:
            return None
        xp = self._backend.xp
        return {k: xp.array(idx) for idx, k in enumerate(sorted(self.port_dict.keys()))}

    @property
    def port_dict(self) -> dict | None:
        return self[PORT_DICT]

    @port_dict.setter
    def port_dict(self, value) -> None:
        self[PORT_DICT] = value

    @property
    def non_fictitious(self):
        return self[NON_FICTITIOUS]

    @non_fictitious.setter
    def non_fictitious(self, value) -> None:
        self[NON_FICTITIOUS] = value

    @property
    def feature_dict(self) -> dict | None:
        """Unstack ``feature_array`` into a dict: feature_name → array slice."""
        if not self.feature_names:
            return None
        xp = self._backend.xp
        result = {}
        for k, v in self.feature_names.items():
            if self.is_batch:
                result[k] = self.feature_array[..., xp.array(v[0], int)]
            else:
                result[k] = self.feature_array[..., xp.array(v, int)]
        return result

    @property
    def feature_flat_array(self):
        """Flatten all features into one long vector per ``(batch,)`` in Fortran order."""
        if self.feature_array is None:
            return None
        shape = [self.n_batch, -1] if self.is_batch else -1
        return self.feature_array.reshape(shape, order="F")

    @feature_flat_array.setter
    def feature_flat_array(self, array) -> None:
        flat = self.feature_flat_array
        if flat is None or flat.shape != array.shape:
            raise ValueError("Shape mismatch for feature_flat_array setter.")
        if self.feature_names is not None:
            if self.is_single:
                self.feature_array = array.reshape([self.n_obj, -1], order="F")
            elif self.is_batch:
                self.feature_array = array.reshape([self.n_batch, self.n_obj, -1], order="F")

    # ------------------------------------------------------------------
    # Feature edition
    # ------------------------------------------------------------------

    def set_feature(self, name: str, value) -> None:
        """
        Add a feature column, or replace an existing one, in place.

        Also available as attribute assignment: ``hes.error = array`` is ``hes.set_feature("error", array)``,
        and ``hes.error`` reads the column back.

        :param name: Feature name.
        :param value: Array of shape ``(n_obj,)`` for a single set, ``(n_batch, n_obj)`` for a batch. It is cast
            to the dtype of the existing features (``float32`` when there is none).
        :raises ValueError: If the shape does not match the objects of this set.
        """
        xp = self._backend.xp
        expected = (self.n_batch, self.n_obj) if self.is_batch else (self.n_obj,)
        dtype = self.feature_array.dtype if self.feature_array is not None else xp.float32
        column = xp.asarray(value, dtype=dtype)
        if tuple(column.shape) != expected:
            raise ValueError(f"Feature '{name}' has shape {tuple(column.shape)}, expected {expected}.")
        names = dict(self.feature_names or {})
        columns = [self.feature_array[..., i] for i in range(len(names))]
        if name in names:
            columns[_feature_index(names[name])] = column
        else:
            names[name] = _stored_index(self, len(columns))
            columns.append(column)
        self.feature_array = xp.stack(columns, axis=-1)
        self[FEATURE_NAMES] = names

    def __getattr__(self, name: str):
        """Read a feature column by name: ``hes.error``. Only called when no regular attribute matches."""
        if name.startswith("_"):
            raise AttributeError(name)
        names = self.get(FEATURE_NAMES)
        if names and name in names:
            return self.feature_array[..., _feature_index(names[name])]
        raise AttributeError(f"{type(self).__name__} has neither an attribute nor a feature named '{name}'.")

    def __setattr__(self, name: str, value) -> None:
        """Regular attributes and properties behave as usual; any other name becomes a feature column."""
        if name.startswith("_") or hasattr(type(self), name):
            super().__setattr__(name, value)
        else:
            self.set_feature(name, value)

    # ------------------------------------------------------------------
    # Padding / un-padding
    # ------------------------------------------------------------------

    def pad(self, target_shape) -> None:
        """Pad a *single* HyperEdgeSet to ``target_shape`` objects with zeros/zeros."""
        if not self.is_single:
            raise ValueError("HyperEdgeSet is batched, impossible to pad.")

        old_n_obj = self.n_obj
        if old_n_obj > target_shape:
            raise ValueError("Provided target_shape is smaller than current shape, padding is impossible!")

        xp = self._backend.xp

        if self.feature_array is not None:
            self.feature_array = xp.pad(self.feature_array, [(0, int(target_shape) - old_n_obj), (0, 0)])

        if self.port_dict is not None:
            for k, v in self.port_dict.items():
                self.port_dict[k] = xp.pad(v, [0, int(target_shape) - old_n_obj])

        if self.non_fictitious is not None:
            self.non_fictitious = xp.pad(self.non_fictitious, [0, int(target_shape) - old_n_obj])

    def unpad(self, target_shape) -> None:
        """Remove padding to restore ``target_shape`` objects in a *single* HyperEdgeSet."""
        if not self.is_single:
            raise ValueError("HyperEdgeSet is batched, impossible to unpad.")
        if self.n_obj < target_shape:
            raise ValueError("Provided target_shape is higher than current shape, unpadding is impossible!")

        if self.feature_array is not None:
            self.feature_array = self.feature_array[: int(target_shape)]

        if self.port_dict is not None:
            for k, v in self.port_dict.items():
                self.port_dict[k] = v[: int(target_shape)]

        if self.non_fictitious is not None:
            self.non_fictitious = self.non_fictitious[: int(target_shape)]

    def offset_addresses(self, offset) -> None:
        """Add ``offset`` to every port address; used before graph concatenation.

        The offset is cast to each port array's dtype so that int32 ports stay int32.
        """
        xp = self._backend.xp
        if self.port_dict is not None:
            self.port_dict = {k: a + xp.asarray(offset, dtype=a.dtype) for k, a in self.port_dict.items()}


# ---------------------------------------------------------------------------
# Module-level batch/collate functions
# ---------------------------------------------------------------------------


def collate_hyper_edge_sets(hyper_edge_set_list: list[HyperEdgeSet]) -> HyperEdgeSet:
    """
    Collate a list of HyperEdgeSet into a single batched HyperEdgeSet.

    :param hyper_edge_set_list: Non-empty sequence of HyperEdgeSet objects with the same schema.
    :return: A single batched HyperEdgeSet.
    :raises IndexError: If ``hyper_edge_set_list`` is empty.
    :raises ValueError: If key schemas are inconsistent.
    """
    if not hyper_edge_set_list:
        raise IndexError("collate_hyper_edge_sets requires at least one HyperEdgeSet to collate.")

    cls = type(hyper_edge_set_list[0])
    backend = hyper_edge_set_list[0]._backend
    xp = backend.xp
    first = hyper_edge_set_list[0]

    for e in hyper_edge_set_list[1:]:
        _check_keys_consistency(first, e)

    feature_array = (
        xp.stack([e.feature_array for e in hyper_edge_set_list], axis=0) if first.feature_array is not None else None
    )
    feature_names = (
        {
            k: xp.stack([e.feature_names[k] for e in hyper_edge_set_list if e.feature_names is not None])
            for k in first.feature_names
        }
        if first.feature_names is not None
        else None
    )
    port_dict = (
        {k: xp.stack([e.port_dict[k] for e in hyper_edge_set_list if e.port_dict is not None]) for k in first.port_dict}
        if first.port_dict is not None
        else None
    )
    non_fictitious = xp.stack([e.non_fictitious for e in hyper_edge_set_list]) if first.non_fictitious is not None else None

    return cls(
        backend=backend,
        port_dict=port_dict,
        feature_array=feature_array,
        feature_names=feature_names,
        non_fictitious=non_fictitious,
    )


def separate_hyper_edge_sets(hyper_edge_set_batch: HyperEdgeSet) -> list[HyperEdgeSet]:
    """
    Split a batched HyperEdgeSet into its constituent HyperEdgeSet instances.

    :param hyper_edge_set_batch: A batched HyperEdgeSet (3-D array).
    :return: List of single HyperEdgeSet instances.
    :raises ValueError: If the input is not batched.
    """
    if not hyper_edge_set_batch.is_batch:
        raise ValueError("Input is not a batch, impossible to separate.")

    cls = type(hyper_edge_set_batch)
    backend = hyper_edge_set_batch._backend
    xp = backend.xp
    n_batch = hyper_edge_set_batch.n_batch

    feature_array_list = (
        xp.unstack(hyper_edge_set_batch.feature_array, axis=0)
        if hyper_edge_set_batch.feature_array is not None
        else [None] * n_batch
    )

    if hyper_edge_set_batch.feature_names is not None:
        a = {k: xp.unstack(hyper_edge_set_batch.feature_names[k]) for k in hyper_edge_set_batch.feature_names}
        feature_names_list: list[dict | None] = [dict(zip(a, t)) for t in zip(*a.values())]
    else:
        feature_names_list = [None] * n_batch

    if hyper_edge_set_batch.port_dict is not None:
        a = {k: xp.unstack(hyper_edge_set_batch.port_dict[k]) for k in hyper_edge_set_batch.port_dict}
        port_dict_list: list[dict | None] = [dict(zip(a, t)) for t in zip(*a.values())]
    else:
        port_dict_list = [None] * n_batch

    non_fictitious_list = (
        xp.unstack(hyper_edge_set_batch.non_fictitious, axis=0)
        if hyper_edge_set_batch.non_fictitious is not None
        else [None] * n_batch
    )

    return [
        cls(backend=backend, port_dict=ad, feature_array=fa, feature_names=fn, non_fictitious=nf)
        for fa, fn, ad, nf in zip(feature_array_list, feature_names_list, port_dict_list, non_fictitious_list)
    ]


def concatenate_hyper_edge_sets(hyper_edge_set_list: list[HyperEdgeSet]) -> HyperEdgeSet:
    """
    Concatenate several single HyperEdgeSet into one single HyperEdgeSet (no new batch dim).

    :param hyper_edge_set_list: List of single (non-batched) HyperEdgeSet objects.
    :returns: One HyperEdgeSet with n_obj = sum of all inputs' n_obj.
    """
    cls = type(hyper_edge_set_list[0])
    backend = hyper_edge_set_list[0]._backend
    xp = backend.xp
    first = hyper_edge_set_list[0]

    port_dict = (
        {
            k: xp.concatenate([hes.port_dict[k] for hes in hyper_edge_set_list if hes.port_dict is not None])
            for k in first.port_dict
        }
        if first.port_dict is not None
        else None
    )
    feature_array = xp.concatenate([hes.feature_array for hes in hyper_edge_set_list], axis=0)
    feature_names = first.feature_names
    non_fictitious = xp.concatenate([hes.non_fictitious for hes in hyper_edge_set_list])

    return cls(
        backend=backend,
        port_dict=port_dict,
        feature_array=feature_array,
        feature_names=feature_names,
        non_fictitious=non_fictitious,
    )


def merge_hyper_edge_sets(left: HyperEdgeSet, right: HyperEdgeSet, *, suffixes: tuple[str, str] | None = None) -> HyperEdgeSet:
    """
    Join two hyper-edge sets describing the same objects: union of ports and features.

    :func:`concatenate_hyper_edge_sets` stacks objects; this joins attributes object by object, the key
    being the object index, like a pandas ``merge`` on an implicit key.

    :param left: Hyper-edge set whose ports and features come first.
    :param right: Hyper-edge set describing the same objects, converted to ``left``'s backend if needed.
    :param suffixes: Suffixes appended to every feature name of ``left`` and of ``right`` respectively, e.g.
        ``("", "_pred")``. Without them, a feature name present on both sides is an error. Ports present on
        both sides must be identical.
    :return: A new hyper-edge set with the same objects and fictitious mask.
    :raises ValueError: On a different number of objects or batch size, different fictitious masks,
        a port present on both sides with different addresses, or a feature-name collision.
    """
    backend = left._backend
    xp = backend.xp
    if type(right._backend) is not type(backend):
        right = right.to_backend(backend)
    if left.is_batch != right.is_batch or left.n_obj != right.n_obj or (left.is_batch and left.n_batch != right.n_batch):
        detail = f", batches of {left.n_batch} vs {right.n_batch}" if left.is_batch and right.is_batch else ""
        raise ValueError(
            f"Cannot merge hyper-edge sets describing different objects: {left.n_obj} vs {right.n_obj} objects{detail}."
        )
    if not _same_arrays(left.non_fictitious, right.non_fictitious):
        raise ValueError("Cannot merge hyper-edge sets whose fictitious masks differ.")

    port_dict: dict[str, Any] | None = None
    if left.port_dict is not None or right.port_dict is not None:
        port_dict = dict(left.port_dict or {})
        for name, addresses in (right.port_dict or {}).items():
            if name in port_dict and not _same_arrays(port_dict[name], addresses):
                raise ValueError(f"Port '{name}' exists on both sides with different addresses.")
            port_dict.setdefault(name, addresses)

    left_features: dict[str, Any] = left.feature_names or {}
    right_features: dict[str, Any] = right.feature_names or {}
    left_names = sorted(left_features, key=lambda k: _feature_index(left_features[k]))
    right_names = sorted(right_features, key=lambda k: _feature_index(right_features[k]))
    left_map, right_map = _resolve_names(left_names, right_names, suffixes, "Features")

    feature_names: dict[str, Any] | None = None
    feature_array = None
    if left_names or right_names:
        columns = [left.feature_array[..., _feature_index(left_features[name])] for name in left_names]
        columns += [right.feature_array[..., _feature_index(right_features[name])] for name in right_names]
        names = [left_map[name] for name in left_names] + [right_map[name] for name in right_names]
        feature_array = xp.stack(columns, axis=-1)
        feature_names = {name: _stored_index(left, index) for index, name in enumerate(names)}

    return type(left)(
        backend=backend,
        port_dict=port_dict,
        feature_array=feature_array,
        feature_names=feature_names,
        non_fictitious=left.non_fictitious,
    )


# ---------------------------------------------------------------------------
# Validation helpers (always operate on NumPy arrays)
# ---------------------------------------------------------------------------


def check_dict_shape(*, d: dict | None, n_objects: int | None) -> int | None:
    """
    Ensure all arrays in a dict share the same last-axis size.

    :param d: Dict of arrays, or None.
    :param n_objects: Expected last-axis size; inferred from the first array when None.
    :return: The validated or inferred ``n_objects``.
    :raises ValueError: If any array's last dimension differs from ``n_objects``.
    """
    if d is not None:
        if n_objects is None:
            item = next(iter(d.values()))
            n_objects = item.shape[-1]
        for name, arr in d.items():
            if arr.shape[-1] != n_objects:
                raise ValueError(f"Array for key '{name}' has last dimension {arr.shape[-1]}, expected {n_objects}.")
    return n_objects


def build_hyper_edge_set_shape(
    *,
    port_dict: dict | None,
    feature_dict: dict | None,
) -> np.ndarray:
    """
    Return a scalar int32 NumPy array with the number of hyper-edges.

    :param port_dict: Port arrays, or None.
    :param feature_dict: Feature arrays, or None.
    :return: ``np.array(n_objects, dtype=int32)``.
    :raises ValueError: If both inputs are None, or if their sizes conflict.
    """
    if port_dict is None and feature_dict is None:
        raise ValueError("At least one of port_dict or feature_dict must be provided.")
    n_objects = check_dict_shape(d=port_dict, n_objects=None)
    n_objects = check_dict_shape(d=feature_dict, n_objects=n_objects)
    return np.array(n_objects, dtype=np.dtype("int32"))


def dict2array(features_dict: dict | None) -> np.ndarray | None:
    """
    Stack a sorted dict of NumPy arrays into a single array along the last axis.

    :param features_dict: Dict of NumPy arrays, or None.
    :return: Stacked NumPy array, or None.
    """
    if features_dict is None:
        return None
    return np.stack([features_dict[k] for k in sorted(features_dict)], axis=-1)


def check_dict_or_none(_input) -> dict | None:
    """
    Validate that the input is a dict or None.

    :raises ValueError: If it is neither.
    """
    if isinstance(_input, dict):
        return _input
    if _input is None:
        return None
    raise ValueError(f"Expected dict or None, got {type(_input)}")


def check_no_nan(*, port_dict: dict | None, feature_dict: dict | None) -> None:
    """
    Ensure there are no NaN values in port or feature arrays.

    Integer arrays cannot hold NaN and are skipped.

    :raises ValueError: If any array contains NaN.
    """
    for name, arr in (port_dict or {}).items():
        if np.issubdtype(arr.dtype, np.floating) and np.any(np.isnan(arr)):
            raise ValueError(f"NaN detected in port array for key '{name}'.")
    for name, arr in (feature_dict or {}).items():
        if np.issubdtype(arr.dtype, np.floating) and np.any(np.isnan(arr)):
            raise ValueError(f"NaN detected in feature array for key '{name}'.")


def check_valid_ports(port_dict: dict | None) -> None:
    """
    Ensure all port arrays contain only integer-valued entries.

    Integer-typed arrays are valid by construction; floating arrays (e.g. from legacy graphs)
    are accepted as long as every value is exactly representable as an int32.

    :raises ValueError: If any port array has non-integer values.
    """
    for name, arr in (port_dict or {}).items():
        if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
            continue
        if np.any(np.isnan(arr)) or not np.allclose(arr, np.int32(arr)):
            raise ValueError(f"Non-integer values detected in port array for key '{name}'.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_n_objects(port_dict: dict | None, feature_dict: dict | None) -> int:
    """Return the number of objects (last-axis size) shared by all arrays."""
    n = check_dict_shape(d=port_dict, n_objects=None)
    n = check_dict_shape(d=feature_dict, n_objects=n)
    if n is None:
        raise ValueError("At least one of port_dict or feature_dict must be provided.")
    return int(n)


def _check_keys_consistency(hes_1: HyperEdgeSet, hes_2: HyperEdgeSet) -> None:
    if (hes_1.port_names is None) != (hes_2.port_names is None):
        raise ValueError("Mismatch in presence of port_names among hyper-edge sets.")
    if (hes_1.feature_names is None) != (hes_2.feature_names is None):
        raise ValueError("Mismatch in presence of feature_names among hyper-edge sets.")
    if hes_1.port_names and hes_2.port_names and hes_1.port_names.keys() != hes_2.port_names.keys():
        raise ValueError("Inconsistent port_names keys among hyper-edge sets.")
    if hes_1.feature_names and hes_2.feature_names and hes_1.feature_names.keys() != hes_2.feature_names.keys():
        raise ValueError("Inconsistent feature_names keys among hyper-edge sets.")


def _feature_index(value) -> int:
    """Feature index as an int, whatever the stored form (int, 0-d array, or one index per batch element)."""
    array = np.asarray(value)
    return int(array.reshape(-1)[0]) if array.ndim else int(array)


def _stored_index(hes: HyperEdgeSet, index: int):
    """Feature index in the form the hyper-edge set uses: an int for a single set, one per batch element."""
    xp = hes._backend.xp
    if hes.is_batch:
        return xp.full((hes.n_batch,), index, dtype=xp.int32)
    return index


def _same_arrays(a, b) -> bool:
    a_np, b_np = np.asarray(a), np.asarray(b)
    return a_np.shape == b_np.shape and bool(np.array_equal(a_np, b_np))


def _resolve_names(
    left: list[str], right: list[str], suffixes: tuple[str, str] | None, what: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Append the suffixes to every name of each side; error on a collision without suffixes or after suffixing."""
    common = sorted(set(left) & set(right))
    if common and suffixes is None:
        raise ValueError(f"{what} {common} exist on both sides; pass suffixes=('...', '...') to disambiguate them.")
    left_suffix, right_suffix = suffixes if suffixes is not None else ("", "")
    left_map = {name: name + left_suffix for name in left}
    right_map = {name: name + right_suffix for name in right}
    merged = list(left_map.values()) + list(right_map.values())
    duplicates = sorted({name for name in merged if merged.count(name) > 1})
    if duplicates:
        raise ValueError(f"{what} {duplicates} collide after applying suffixes {suffixes}.")
    return left_map, right_map
