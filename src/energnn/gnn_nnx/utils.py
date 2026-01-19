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


class LazyLinear(nnx.Module):
    """
    A lazily-initialized linear layer for Flax NNX.

    The inner ``nnx.Linear`` is created on the first call using the input's
    last dimension as ``in_features``. This preserves the Linen-style lazy
    convenience for heterogeneous graph batches while using the NNX module
    system.

    :param out_features: Number of output features.
    :param use_bias: Whether to include a bias term. Defaults to True.
    :param kernel_init: Initializer for the kernel. Defaults to ``lecun_normal``.
    :param bias_init: Initializer for the bias. Defaults to zeros.
    :param rngs: An ``nnx.Rngs`` object or integer seed used for parameter
                 initialization. Required for eager initialization of the
                 inner ``nnx.Linear``.

    :return: An NNX Module that creates its internal Linear on first call.
    """

    def __init__(self, out_features: int, *, use_bias: bool = True, kernel_init: Initializer = initializers.lecun_normal(),
            bias_init: Initializer = initializers.zeros_init(), rngs: nnx.Rngs | int | None = None) -> None:
        if rngs is None:
            raise ValueError("LazyLinear expects an `nnx.Rngs` or int seed for rngs")
        if isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        self.out_features = int(out_features)
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.rngs = rngs

        # placeholder for the inner layer
        self._linear = None

    def create_inner(self, in_features: int) -> None:
        """Create the inner ``nnx.Linear`` with the inferred input dimension.

        The inner layer is eagerly initialized by passing ``rngs`` so that its
        parameters exist immediately after creation.
        """
        self._linear = nnx.Linear(
            in_features=in_features,
            out_features=self.out_features,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=self.rngs,
        )

    def build_from_sample(self, sample: jax.Array) -> None:
        """
        Create the inner ``nnx.Linear`` using a sample data.

        :param sample: Array whose final axis is the feature dimension.
        """
        if sample.ndim == 0:
            raise ValueError("Input to LazyLinear must have at least one axis")
        in_features = int(sample.shape[-1])
        self.create_inner(in_features)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Forward pass: create inner ``nnx.Linear`` on demand and apply it.

        :param inputs: Input array whose final axis is the feature dimension.
        :returns: Projected array with last axis ``out_features``.
        """
        if self._linear is None:
            self.build_from_sample(inputs)

        return self._linear(inputs)


class MLP(nnx.Module):
    """
    Multi-Layer Perceptron (MLP) neural network module using Flax's NNX API.

    The MLP consists of a sequence of Dense layers with an optional activation,
    followed by a final output layer with configurable initialization and activation.

    :param hidden_size: Sizes of each hidden layer.
    :param activation: Activation function applied after each hidden layer.
                       If None, no activation is applied between hidden layers.
    :param out_size: Number of units in the output layer.
    :param rngs: ``nnx.Rngs`` or integer seed used to initialize sub-layers.
    :param name: Optional module name.
    :param final_kernel_zero_init: If True, the final layer is initialized with zeros.
                                   Otherwise, use LeCun normal initialization.
    :param final_activation: Activation function applied after the final layer.
                             If None, no activation is applied to the output.

    :return: Flax NNX module representing the MLP.
    """

    def __init__(self, hidden_size: list[int], *, activation: Activation | None = jax.nn.relu, out_size: int = 1,
            rngs: nnx.Rngs | int | None = None, final_kernel_zero_init: bool = False, final_activation: Activation | None = None,
            name: str | None = None ) -> None:

        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        self.hidden_size = [int(h) for h in hidden_size]
        self.activation = activation
        self.out_size = int(out_size)
        self.rngs = rngs
        self.final_kernel_zero_init = final_kernel_zero_init
        self.final_activation = final_activation
        self.name = name

        layers: list = []
        if self.activation is not None:
            for d in self.hidden_size:
                layers.extend([LazyLinear(d, rngs=self.rngs), self.activation])
        else:
            for d in self.hidden_size:
                layers.append(LazyLinear(d, rngs=self.rngs))

        final_kernel_init = (
            initializers.zeros_init() if final_kernel_zero_init else initializers.lecun_normal()
        )
        layers.append(
            LazyLinear(
                self.out_size,
                rngs=self.rngs,
                kernel_init=final_kernel_init,
                bias_init=initializers.zeros_init(),
            )
        )
        if self.final_activation is not None:
            layers.append(self.final_activation)

        self.sequential = nnx.Sequential(*layers)

    def build_from_sample(self, sample: jax.Array) -> None:

        if sample.ndim == 0:
            raise ValueError("Input to MLP must have at least one axis")

        in_features = int(sample.shape[-1])
        for i in range(len(self.sequential.layers)):
            layer = self.sequential.layers[i]
            if isinstance(layer, LazyLinear):
                layer.create_inner(in_features)
                # update in_features to the output size of the created layer
                in_features = layer.out_features
            else:
                # static callable (activation) — no in_features change
                continue

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Forward pass through the MLP.

        The input's last axis is used as the feature axis for the first LazyLinear.

        :param inputs: Input array with feature dimension on the last axis.
        :returns: Output array with last axis equal to ``out_size``.
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
