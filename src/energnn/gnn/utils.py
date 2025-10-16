#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from typing import Callable

import flax
import jax
from flax import linen as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network module using Flax's Linen API.

    The MLP consists of a sequence of Dense layers with an optional activation,
    followed by a final output layer with configurable initialization and activation.

    :param hidden_size: Sizes of each hidden layer.
    :param activation: Activation function applied after each hidden layer.
                       If None, no activation is applied between hidden layers.
    :param out_size: Number of units in the output layer.
    :param name: Module name.
    :param final_kernel_zero_init: If True, the final layer is initialized with zeros.
                                   Otherwise, use LeCun normal initialization.
    :param final_activation: Activation function applied after the final layer.
                             If None, no activation is applied to the output.

    :return: Flax Linen module representing the MLP.
    """

    hidden_size: list[int]
    activation: Callable | None
    out_size: int
    name: str
    final_kernel_zero_init: bool = False
    final_activation: Callable | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass for the MLP.

        :param x: Input array.
        :returns: Output array of shape (..., out_size).
        """
        for d in self.hidden_size:
            x = nn.Dense(d)(x)
            if self.activation is not None:
                x = self.activation(x)

        if self.final_kernel_zero_init:
            kernel_init = flax.linen.initializers.zeros_init()
        else:
            kernel_init = flax.linen.initializers.lecun_normal()

        x = nn.Dense(features=self.out_size, kernel_init=kernel_init)(x)

        if self.final_activation is not None:
            x = self.final_activation(x)
        return x

        # if self.final_kernel_zero_init:
        #     return nn.Dense(
        #         features=self.out_size,
        #         use_bias=self.use_bias,
        #         kernel_init=flax.linen.initializers.zeros_init(),
        #     )(x)
        # y = nn.Dense(features=self.out_size, use_bias=self.use_bias)(x)
        # if self.final_activation is not None:
        #     y = self.final_activation(y)
        # return y


def gather(*, coordinates: jax.Array, addresses: jax.Array) -> jax.Array:
    """
    Gather elements from a coordinate array at specified indices.

    Uses JAX's `at` indexing with 'drop' mode and zero fill for out-of-bounds.

    :param coordinates: Array from which to gather values.
    :param addresses: Integer indices specifying which elements to gather.
    :returns: Gathered elements of the same shape as `addresses`.
    """
    return coordinates.at[addresses.astype(int)].get(mode="drop", fill_value=0.0)


def scatter_add(*, accumulator: jax.Array, increment: jax.Array, addresses: jax.Array) -> jax.Array:
    """
    Scatter_add increments into an accumulator array at specified indices.

    :param accumulator: Array to which increments are added.
    :param increment: Values to add at the specified indices.
    :param addresses: Integer indices where increments should be added.
    :returns: Updated accumulator array after adding increments.
    """
    return accumulator.at[addresses.astype(int)].add(increment, mode="drop")
