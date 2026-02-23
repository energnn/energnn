# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations

from typing import Callable

import jax
from flax import nnx
from flax.nnx import initializers
from flax.typing import Initializer

Activation = Callable[[jax.Array], jax.Array]


class MLP(nnx.Module):
    """
    Multi-Layer Perceptron (MLP) neural network module using Flax's NNX API.

    :param in_size: Size of the input vector.
    :param hidden_sizes: Sizes of each hidden layer.
    :param activation: Activation function applied after each hidden layer.
    :param out_size: Number of units in the output vector.
    :param use_bias: If True, bias terms are added to the output of each layer.
    :param kernel_init: Initializer function for the weights of each layer.
    :param bias_init: Initializer function for the bias terms of each layer.
    :param final_activation: Activation function applied after the final layer.
    :param rngs: ``nnx.Rngs`` used to initialize sub-layers.
    :param seed: Optional seed for RNG streams used to initialize sub-layers.
    :return: Flax NNX module representing the MLP.
    """

    def __init__(
        self,
        *,
        in_size: int,
        hidden_sizes: list[int],
        activation: Activation = jax.nn.relu,
        out_size: int = 1,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        rngs: nnx.Rngs | None = None,
        seed: int | None = None,
    ) -> None:

        if in_size <= 0:
            raise ValueError(f"in_size must be positive, got {in_size}")
        if out_size <= 0:
            raise ValueError(f"out_size must be positive, got {out_size}")
        if any(h <= 0 for h in hidden_sizes):
            raise ValueError(f"All hidden sizes must be positive, got {hidden_sizes}")

        self.in_size = int(in_size)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.activation = activation
        self.out_size = int(out_size)
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.final_activation = final_activation

        if seed is not None:
            rngs = nnx.Rngs(seed)
        elif rngs is None:
            raise ValueError("Either 'rngs' or 'seed' must be provided to initialize MLP.")

        self.sequential = self._build_sequential(rngs=rngs)

    def _build_sequential(self, rngs: nnx.Rngs) -> nnx.Sequential:
        all_sizes = [self.in_size, *self.hidden_sizes, self.out_size]
        all_activations = [self.activation] * len(self.hidden_sizes) + [self.final_activation]
        layers: list = []
        for i in range(len(all_sizes) - 1):
            layers.append(
                nnx.Linear(
                    in_features=all_sizes[i],
                    out_features=all_sizes[i + 1],
                    use_bias=self.use_bias,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    rngs=rngs,
                )
            )
            if all_activations[i] is not None:
                layers.append(all_activations[i])
        return nnx.Sequential(*layers)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Forward pass through the MLP.

        :param inputs: Input array with feature size on the last axis.
        :returns: Output array with the last axis equal to ``out_size``.
        """
        return self.sequential(inputs)


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
