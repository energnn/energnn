# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import copy

import numpy as np


PRESERVE_DTYPE = "preserve"
"""Sentinel for :meth:`Backend.from_numpy`: keep the input dtype instead of casting to the
backend's floating dtype. Used for integer data (port addresses, shape counts, name indices),
which must never go through a floating representation. The backend may still narrow 64-bit
types to their 32-bit counterparts (e.g. JAX with x64 disabled)."""


class Backend:
    """Abstract array backend. Subclasses wrap NumPy or JAX and expose a uniform interface."""

    @property
    def xp(self):
        """The underlying array module (``numpy`` or ``jax.numpy``)."""
        raise NotImplementedError

    def scatter_max(self, array, indices, values):
        """Scatter-max update: writes ``max(array[i], values[i])`` at each index in ``indices``."""
        raise NotImplementedError

    def copy(self, array):
        """Return an independent copy of ``array`` (deep copy for NumPy; identity for JAX)."""
        raise NotImplementedError

    def from_numpy(self, x, dtype: str = "float32"):
        """Convert a NumPy array or dict of NumPy arrays to this backend's native array type.

        ``dtype`` is the target dtype; pass :data:`PRESERVE_DTYPE` to keep the input dtype
        (up to 64→32-bit narrowing on backends that require it).
        """
        raise NotImplementedError

    def to_numpy(self, x):
        """Convert this backend's array or dict of arrays back to NumPy."""
        raise NotImplementedError


class NumpyBackend(Backend):
    """Backend that delegates linear algebra to NumPy."""

    @property
    def xp(self):
        return np

    def scatter_max(self, array, indices, values):
        np.maximum.at(array, indices, values)
        return array

    def copy(self, array):
        return copy.deepcopy(array)

    def from_numpy(self, x, dtype: str = "float32"):
        if x is None:
            return None
        if isinstance(x, dict):
            return {k: self.from_numpy(v, dtype) for k, v in x.items()}
        if dtype == PRESERVE_DTYPE:
            return np.array(x)
        return np.array(x, dtype=np.dtype(dtype))

    def to_numpy(self, x):
        if x is None:
            return None
        if isinstance(x, dict):
            return {k: np.array(v) for k, v in x.items()}
        return np.array(x)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NumpyBackend)

    def __hash__(self) -> int:
        return hash(type(self))

    def __repr__(self) -> str:
        return "NumpyBackend()"


class JaxBackend(Backend):
    """Backend that delegates linear algebra to JAX, with optional device placement and dtype."""

    def __init__(self, device=None, dtype: str = "float32") -> None:
        self.device = device
        self.dtype = dtype

    @property
    def xp(self):
        import jax.numpy as jnp

        return jnp

    def scatter_max(self, array, indices, values):
        return array.at[indices].max(values)

    def copy(self, array):
        return array  # JAX arrays are immutable

    def from_numpy(self, x, dtype: str | None = None):
        import jax
        import jax.numpy as jnp

        actual_dtype = dtype if dtype is not None else self.dtype
        if x is None:
            return None
        if isinstance(x, dict):
            return {k: self.from_numpy(v, dtype) for k, v in x.items()}
        # PRESERVE_DTYPE: let jnp.array keep the input dtype (with jax's default 64→32 narrowing).
        arr = jnp.array(x) if actual_dtype == PRESERVE_DTYPE else jnp.array(x, dtype=actual_dtype)
        if self.device is not None:
            arr = jax.device_put(arr, self.device)
        return arr

    def to_numpy(self, x):
        if x is None:
            return None
        if isinstance(x, dict):
            return {k: np.array(v) for k, v in x.items()}
        return np.array(x)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, JaxBackend) and self.device == other.device and self.dtype == other.dtype

    def __hash__(self) -> int:
        return hash((type(self), str(self.device), self.dtype))

    def __repr__(self) -> str:
        return f"JaxBackend(device={self.device!r}, dtype={self.dtype!r})"
