#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
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
    # TODO : update jax to specify device.
    if x is None:
        return None
    elif isinstance(x, dict):
        return {k: jnp.array(v, dtype=dtype) for k, v in x.items()}
        # return {k: jnp.array(v, device=device, dtype=dtype) for k, v in x.items()}
    else:
        return jnp.array(x, dtype=dtype)
        # return jnp.array(x, device=device, dtype=dtype)
    # elif isinstance(x, np.ndarray):
    #     return jnp.array(x, device=device, dtype=dtype)
    # else:
    #     raise TypeError(f"Unsupported type {type(x)}.")


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
    # else:
    #     raise TypeError(f"Unsupported type {type(x)}.")
