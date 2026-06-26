# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import copy

import numpy as np


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
        """Convert a NumPy array or dict of NumPy arrays to this backend's native array type."""
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
            return {k: np.array(v, dtype=np.dtype(dtype)) for k, v in x.items()}
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
            result = {k: jnp.array(v, dtype=actual_dtype) for k, v in x.items()}
            if self.device is not None:
                result = {k: jax.device_put(v, self.device) for k, v in result.items()}
            return result
        arr = jnp.array(x, dtype=actual_dtype)
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
