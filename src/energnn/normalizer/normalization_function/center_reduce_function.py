# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import jax
import jax.numpy as jnp

from energnn.normalizer.normalization_function.normalization_function import (
    NormalizationFunction,
)


class CenterReduceFunction(NormalizationFunction):
    r"""
    Affine function that centers and reduces values of a certain class and of a certain feature.

    .. math::
        f(x) = \frac{x - \mu}{\sigma + \epsilon},

    where :math:`\mu` and :math:`\sigma` respectively are the empirical mean
    and standard deviations of the values observed on the problem loader,
    for a given class and a given feature.

    :param epsilon: Small positive offset :math:`\epsilon` to avoid a division by zero.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Create a CenterReduceFunction instance.

        :param epsilon: Stability offset added to denominator.
        """
        self.epsilon = epsilon

    def init_aux(self, array: jax.Array) -> list:
        """
        Init an empty list which will serve to store all the feature values.

        :param array: Feature array. (argument unused here)
        :return: Empty list to collect feature values.

        """
        return []

    def update_aux(self, array: jax.Array, aux: list[jax.Array]) -> list[jax.Array]:
        """
        Stores feature arrays in the `aux` list.

        :param array: New feature array.
        :param aux: Current list of stored feature arrays.
        :return: Updated list including the newest feature array.
        """
        dataset_list = aux
        dataset_list.append(array)
        return dataset_list

    def compute_params(self, array: jax.Array, aux: list[jax.Array]) -> jax.Array:
        """
        Returns the mean and standard deviation of values stored in `aux`.

        Concatenates all stored feature arrays and calculates per-feature statistics.

        :param array: Feature jax array. (argument unused here)
        :param aux: List of stored feature arrays.
        :return: Array stacking [mean, std].
        """
        dataset_list = aux
        dataset_array = jnp.concatenate(dataset_list, axis=0)
        mean = jnp.nanmean(dataset_array, axis=0)
        std = jnp.nanstd(dataset_array, axis=0)
        return jnp.stack([mean, std], axis=0)

    def apply(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """
        Centers and reduces based on the mean and std stored in `params`.

        It performs forward row-wise normalization on an input array and masks invalid entries.

        :param params: Parameters [mean, std] of normalization.
        :param array: Input feature array.
        :param non_fictitious: Mask indicating valid entries.
        :return: Normalized and masked array.
        """
        mean, std = params[0], params[1]
        return (
            jax.vmap(
                jax.vmap(forward, in_axes=(0, None, None, None)),
                in_axes=(1, 0, 0, None),
                out_axes=1,
            )(array, mean, std, self.epsilon)
            * non_fictitious
        )

    def apply_inverse(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """
        Invert normalization to restore original values.

        Applies `inverse` row-wise and masks invalid entries.

        :param params: Parameters [mean, std] of normalization.
        :param array: Normalized data.
        :param non_fictitious: Mask indicating valid entries.
        :return: Denormalized and masked array.
        """
        mean, std = params[0], params[1]
        return (
            jax.vmap(
                jax.vmap(inverse, in_axes=(0, None, None, None)),
                in_axes=(1, 0, 0, None),
                out_axes=1,
            )(array, mean, std, self.epsilon)
            * non_fictitious
        )

    def gradient_inverse(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """
        Computes the gradient of the inverse of the normalization.

        :param params: Parameters [mean, std] of normalization.
        :param array: Normalized input data.
        :param non_fictitious: Mask indicating valid entries.
        :return: Gradient array of same shape as input.
        """
        mean, std = params[0], params[1]
        return (
            jax.vmap(
                jax.vmap(jax.grad(inverse), in_axes=(0, None, None, None)),
                in_axes=(1, 0, 0, None),
                out_axes=1,
            )(array, mean, std, self.epsilon)
            * non_fictitious
        )


def forward(array: jax.Array, mean: jax.Array, std: jax.Array, epsilon: float = 1e-8) -> jax.Array:
    """
    Center and scale an array using provided mean and standard deviation.

    Implements:
    ``(array - mean) / (std + epsilon)``

    :param array: Input array to normalize.
    :param mean: Per-feature mean.
    :param std: Per-feature standard deviation.
    :param epsilon: Small offset to avoid division by zero.
    :return: Normalized array.
    """
    return (array - mean) / (std + epsilon)


def inverse(array: jax.Array, mean: jax.Array, std: jax.Array, epsilon: float = 1e-8) -> jax.Array:
    """
    Inverse transformation of `forward` to recover original scale.

    Implements:
    ``array * (std + epsilon) + mean``

    :param array: Normalized input array.
    :param mean: Per-feature mean used in forward.
    :param std: Per-feature standard deviation used in forward.
    :param epsilon: Stability offset matching forward.
    :return: Denormalized array.
    """
    return array * (std + epsilon) + mean
