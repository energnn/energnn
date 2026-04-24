# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def np_to_jnp(
    x: np.ndarray | dict[str, np.ndarray] | None, device: jax.Device | None = None, dtype: str = "float32"
) -> jax.Array | dict[str, jax.Array] | None:
    """
    Convert NumPy arrays or dictionary of NumPy arrays to JAX arrays.

    This function handles both individual NumPy arrays and dictionaries
    mapping string keys to NumPy arrays. It converts each array to a JAX array
    with the specified data type and places it on the given device if provided.

    :param x: NumPy array or dict of NumPy arrays to convert. If None, returns None.
    :param device: JAX device to place the arrays on. If None, the default JAX device is used.
    :param dtype: Data type for the JAX arrays (e.g., 'float32').
    :return: JAX array or dict of JAX arrays matching the structure of the input,
             or None if the input is None.
    """
    if x is None:
        return None
    elif isinstance(x, dict):
        return {k: jnp.array(v, dtype=dtype) for k, v in x.items()}
    else:
        return jnp.array(x, dtype=dtype)


def jnp_to_np(x: jax.Array | dict[str, jax.Array] | None) -> np.ndarray | dict[str, np.ndarray] | None:
    """
    Convert JAX arrays or mappings of JAX arrays back to NumPy arrays.

    This function handles both individual JAX arrays and dictionaries mapping
    string keys to JAX arrays. It converts each array to a NumPy array.

    :param x: JAX array or dict of JAX arrays to convert. If None, returns None.
    :return: NumPy array or dict of NumPy arrays matching the input structure,
             or None if the input is None.
    """
    if x is None:
        return None
    elif isinstance(x, dict):
        return {k: np.array(v) for k, v in x.items()}
    else:
        return np.array(x)


def to_numpy(a: dict | np.ndarray | jax.Array | tuple | None) -> dict | np.ndarray | None:
    """
    Converts a NumPy array, JAX array, or tuple of values into a NumPy array (dtype float32),
    or converts the values in a dictionary accordingly.

    - If `a` is None, returns None.
    - If `a` is a np.ndarray, jax.Array, jnp.ndarray, or tuple, it is converted to a np.ndarray (float32).
    - If `a` is a dict with some values being arrays or tuples, only those values are converted;
      others remain unchanged.
    - In all other cases, a TypeError is raised.

    :param a: A np.ndarray, jax.Array, tuple, dict, or None.
    :returns: Either None, a np.ndarray, or a dict with the same keys and converted np.ndarray values.
    :raises TypeError: If `a` is not of an expected or supported type.
    """

    if a is None:
        return None

    def _to_np(x: Any) -> Any:
        # On traite np.ndarray, jax.Array et tuple
        if isinstance(x, (np.ndarray, jax.Array, np.ndarray, tuple)):
            return np.array(x, dtype=np.dtype("float32"))
        else:
            return x

    if isinstance(a, dict):
        output: dict[Any, np.ndarray] = {}
        for key, value in a.items():
            output[key] = _to_np(value)  # seules les values “ArrayLike” seront converties
        return output

    # Cas array-like, tuple et object
    if isinstance(a, (np.ndarray, jax.Array, np.ndarray, tuple, object)):
        return _to_np(a)

    raise TypeError(f"Type {type(a)} non pris en charge par to_numpy")
