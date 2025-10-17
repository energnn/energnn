# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod
from typing import Any

import jax


class NormalizationFunction(ABC):
    """
    Abstract base class defining the interface for normalization transformations on arrays.

    Subclasses must implement methods to initialize, update, compute normalization parameters, apply the normalization,
    its inverse, and compute gradients of the normalization function.
    """

    @abstractmethod
    def init_aux(self, array: jax.Array) -> Any:
        """
        Initializes the `aux` variable used to then create `params`.

        :param array: Input jax array.
        :return: Initialized `aux`.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def update_aux(self, array: jax.Array, aux: Any) -> Any:
        """
        Updates the `aux` variable used to then create `params`.

        :param array: New input array to update `aux`.
        :param aux: Existing `aux`.
        :return: Updated `aux` value.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_params(self, array: jax.Array, aux: Any) -> jax.Array:
        """
        Computes `params` based on the information gathered in `aux`.

        :param array: Input jax array.
        :param aux: `aux` value
        :return: Array of normalization parameters.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """
        Applies the normalization function over `array`.

        :param params: Parameters defining the normalization.
        :param array: Input array to normalize.
        :param non_fictitious: Mask array indicating valid entries.
        :return: Normalized array of the same shape as `array`.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_inverse(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """
        Applies the inverse of the normalization function over `array`.

        :param params: Normalization parameters used in the apply method.
        :param array: Normalized array to invert.
        :param non_fictitious: Mask array indicating valid entries.
        :return: Denormalized array matching the original input scale.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def gradient_inverse(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """
        Computes the gradient of the inverse of the normalization function over `array`.

        :param params: Normalization function parameters.
        :param array: Input array for which the gradient is computed.
        :param non_fictitious: Mask array indicating valid entries.
        :return: Gradient of the inverse mapping.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError
