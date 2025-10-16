#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import jax

# import jax.numpy as jnp
import numpy as np
from typing import Any


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
        output: dict[Any, np.array] = {}
        for key, value in a.items():
            output[key] = _to_np(value)  # seules les values “ArrayLike” seront converties
        return output

    # Cas array-like, tuple et object
    if isinstance(a, (np.ndarray, jax.Array, np.ndarray, tuple, object)):
        return _to_np(a)

    # elif isinstance(a, object): # A supprimer, car cette condition est toujours vraie
    #    return a

    raise TypeError(f"Type {type(a)} non pris en charge par to_numpy")


# def to_jax_numpy(a: dict | np.ndarray | jax.Array | tuple | None) -> dict | jax.Array | None:
#     """Converts a dictionary of numpy values into a dictionary of jax.numpy objects."""
#     if a is None:
#         return None
#     elif isinstance(a, np.ndarray) or isinstance(a, jax.Array) or isinstance(a, tuple):
#         return jnp.array(a, dtype=jnp.dtype("float32"))
#     elif isinstance(a, dict):
#         output_dict = dict()
#         for key, value in a.items():
#             if isinstance(value, np.ndarray) or isinstance(value, jax.Array) or isinstance(value, tuple):
#                 output_dict[key] = jnp.array(value, dtype=jnp.dtype("float32"))
#             else:
#                 output_dict[key] = value
#         return output_dict
#     elif isinstance(a, object):
#         return a
#     else:
#         raise TypeError()
