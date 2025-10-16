#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from typing import Any

import jax

from energnn.normalizer.normalization_function.normalization_function import (
    NormalizationFunction,
)


class IdentityFunction(NormalizationFunction):
    """Identity function."""

    def __init__(self):
        pass

    def init_aux(self, array: jax.Array) -> Any:
        """Does nothing."""
        return None

    def update_aux(self, array: jax.Array, aux: Any) -> Any:
        """Does nothing."""
        return None

    def compute_params(self, array: jax.Array, aux: Any) -> jax.Array:
        """Does nothing."""
        return jax.numpy.array([])

    def apply(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """Returns input `array`."""
        return array * non_fictitious

    def apply_inverse(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """Returns input `array`."""
        return array * non_fictitious

    def gradient_inverse(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """Constant gradient equal to one."""
        return (array * 0.0 + 1.0) * non_fictitious
