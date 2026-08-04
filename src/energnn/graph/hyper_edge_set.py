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
from energnn.graph.formatting import format_float_column, format_int_column, render_grouped_table, select_rows
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

    def _summary(self) -> str:
        """One-line description: object, port and feature counts."""
        parts = []
        if self.is_batch:
            parts.append(f"batch of {self.n_batch}")
        n_obj = self.n_obj
        obj_part = f"{n_obj} object{'s' if n_obj != 1 else ''}"
        if self.non_fictitious is not None:
            n_real = int(np.asarray(self.non_fictitious).sum())
            n_total = int(np.asarray(self.non_fictitious).size)
            if n_real != n_total and not self.is_batch:
                obj_part += f" ({n_real} real)"
        parts.append(obj_part)
        n_ports = len(self.port_dict) if self.port_dict is not None else 0
        parts.append(f"{n_ports} port{'s' if n_ports != 1 else ''}")
        n_feat = len(self.feature_names) if self.feature_names is not None else 0
        parts.append(f"{n_feat} feature{'s' if n_feat != 1 else ''}")
        return " · ".join(parts)

    def _table_lines(self) -> list[str]:
        """Render ports and features as an aligned text table (no pandas)."""
        if self.is_single:
            n_rows = self.n_obj
            index_values = [("", [str(i) for i in range(n_rows)])]
        elif self.is_batch:
            n_rows = self.n_batch * self.n_obj
            index_values = [
                ("batch", [str(b) for b in range(self.n_batch) for _ in range(self.n_obj)]),
                ("obj", [str(i) for _ in range(self.n_batch) for i in range(self.n_obj)]),
            ]
        else:
            raise ValueError("HyperEdgeSet is neither single nor batched.")

        rows, ellipsis_at = select_rows(n_rows)

        def take(cells: list[str]) -> list[str]:
            return [cells[r] for r in rows]

        index_columns = [(name, take(cells)) for name, cells in index_values]

        groups = []
        if self.port_dict is not None:
            columns = [(k, take(format_int_column(np.asarray(v).reshape(-1)))) for k, v in sorted(self.port_dict.items())]
            groups.append(("ports", columns))
        if self.feature_names is not None:
            columns = [(k, take(format_float_column(np.asarray(v).reshape(-1)))) for k, v in sorted(self.feature_dict.items())]
            groups.append(("features", columns))

        return render_grouped_table(index_columns, groups, ellipsis_at=ellipsis_at)

    def __str__(self) -> str:
        return "\n".join([f"HyperEdgeSet · {self._summary()}", *self._table_lines()])

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
        {k: xp.stack([e.feature_names[k] for e in hyper_edge_set_list]) for k in first.feature_names}
        if first.feature_names is not None
        else None
    )
    port_dict = (
        {k: xp.stack([e.port_dict[k] for e in hyper_edge_set_list]) for k in first.port_dict}
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
        feature_names_list = [dict(zip(a, t)) for t in zip(*a.values())]
    else:
        feature_names_list = [None] * n_batch

    if hyper_edge_set_batch.port_dict is not None:
        a = {k: xp.unstack(hyper_edge_set_batch.port_dict[k]) for k in hyper_edge_set_batch.port_dict}
        port_dict_list = [dict(zip(a, t)) for t in zip(*a.values())]
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
        {k: xp.concatenate([hes.port_dict[k] for hes in hyper_edge_set_list]) for k in first.port_dict}
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
    if hes_1.port_names and hes_1.port_names.keys() != hes_2.port_names.keys():
        raise ValueError("Inconsistent port_names keys among hyper-edge sets.")
    if hes_1.feature_names and hes_1.feature_names.keys() != hes_2.feature_names.keys():
        raise ValueError("Inconsistent feature_names keys among hyper-edge sets.")
