#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from typing import Any

import jax
import jax.numpy as jnp

from energnn.normalizer.normalization_function.normalization_function import (
    NormalizationFunction,
)

EPS = 1e-8


class CDFPWLinearFunction(NormalizationFunction):
    r"""
    Piecewise Linear approximation of the empirical Cumulative Distribution Function.

    It interpolates :math:`x` based on a series of breakpoint coordinates :math:`(xp, fp)`.
    The first slope is extended to :math:`-\infty` and the last slope is extended to
    :math:`+\infty`, so that the function is a bijective mapping :math:`\mathbb{R} \rightarrow \mathbb{R}`.

    .. math::

        \begin{align}
        f(x) &= \text{interpolate}(x; xp, fp),\\
        \end{align}

    Breakpoints coordinates are computed by estimating the empirical quantiles
    for a given class and a given feature, over a problem loader.

    .. math::
        p &= [0, 1/N, \dots, 1] \\
        q &= [\hat{q}_0, \hat{q}_{1/N}, \dots, \hat{q}_1],

    where :math:`N` corresponds to ``n_breakpoints``.
    Some quantiles may be equal, so conflicts are resolved as follows.

    .. math::
        xp &= q \times (1-s) + p \times s \\
        s &= \max(r \times (\overline{q} - \underline{q}), a)

    where :math:`r` refers to the ``relative_min_slope``
    and :math:`a` refers to the ``absolute_min_slope``.
    Finally, probabilities are scaled and translated to fit a uniform distribution between
    -1 and +1.

    .. math::
        fp = -1 + 2 \times p.

    :param n_breakpoints: Number of evenly spaced quantiles that should be computed on the
        empirical distribution. Corresponds to the amount of breakpoints in the piecewise linear
        approximation of the CDF.
    """

    def __init__(
        self,
        n_breakpoints: int = 20,
    ):
        """
        Initialize CDFPWLinearFunction.

        :param n_breakpoints: Number of quantile breakpoints (>=1).
        """
        self.n_breakpoints = n_breakpoints

    def init_aux(self, array: jax.Array) -> list:
        """
        Init an empty list which will serve to store all the feature values.

        :param array: Feature array. (argument unused here)
        :return: Empty list to accumulate feature batches.
        """
        return []

    def update_aux(self, array: jax.Array, aux: list) -> list:
        """
        Stores feature arrays in the `aux` list.

        :param array: New feature array.
        :param aux: Current list of stored feature arrays.
        :return: Updated list including the newest feature array.
        """
        dataset_list = aux
        dataset_list.append(array)
        return dataset_list

    def compute_params(self, array: jax.Array, aux: Any) -> jax.Array:
        """
        Return piecewise-linear CDF parameters.

        Compute quantiles from array, and resolves conflicts to produce a bijective mapping.

        :param array: Feature jax array. (argument unused here)
        :param aux: List of stored feature arrays.
        :return: Parameter array.
        """
        dataset_list = aux
        dataset_array = jnp.concatenate(dataset_list, axis=0)
        if dataset_array.size > 0:
            p, q = get_proba_quantiles(dataset_array, self.n_breakpoints)
        else:
            p = jnp.arange(start=0, stop=1 + 1.0 / self.n_breakpoints, step=1.0 / self.n_breakpoints)
            q = jnp.zeros(shape=[self.n_breakpoints + 1, dataset_array.shape[1]])
            p = jnp.expand_dims(p, axis=1) + 0.0 * q
        p_merged, q_merged = merge_equal_quantiles(p, q)
        xp, fp = q_merged, -1 + 2 * p_merged
        return jnp.stack([xp, fp], axis=0)

    def apply(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """
        Apply the inverse of the PWL approximation of the CDF.

        :param params: Parameters of normalization.
        :param array: Input feature array.
        :param non_fictitious: Mask indicating valid entries.
        :return: Normalized and masked array.
        """
        xp, fp = params[0], params[1]
        return jax.vmap(forward, in_axes=[1, 1, 1], out_axes=1)(array, xp, fp) * non_fictitious

    def apply_inverse(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """
        Invert normalization to restore original values.

        Applies the PWL approximation of the CDF.

        :param params: Parameters of normalization.
        :param array: Normalized data.
        :param non_fictitious: Mask indicating valid entries.
        :return: Denormalized and masked array.
        """
        xp, fp = params[0], params[1]
        return jax.vmap(inverse, in_axes=[1, 1, 1], out_axes=1)(array, fp, xp) * non_fictitious

    def gradient_inverse(self, params: jax.Array, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """
        Computes the derivative of the PWL approximation of the CDF.

        :param params: Parameters of normalization.
        :param array: Normalized input data.
        :param non_fictitious: Mask indicating valid entries.
        :return: Gradient array of same shape as input.
        """
        xp, fp = params[0], params[1]
        vvmap_inverse = jax.vmap(jax.vmap(jax.grad(inverse), in_axes=(0, None, None)), in_axes=(1, 1, 1), out_axes=1)
        return vvmap_inverse(array, fp, xp) * non_fictitious


def get_proba_quantiles(x: jax.Array, n_breakpoints: int) -> tuple[jax.Array, jax.Array]:
    """
    Compute probability-quantile pairs for piecewise-linear CDF approximation.

    Generates evenly spaced probabilities p in [0,1] of length n_breakpoints+1,
    and their corresponding empirical quantiles q from the data array.

    :param x: Data array to compute quantiles over.
    :param n_breakpoints: Number of segments for piecewise-linear CDF.
    :return: Tuple (p, q) where:
             - p: Array of probabilities.
             - q: Array of quantile values.
    """
    p = jnp.arange(start=0, stop=1 + 1.0 / n_breakpoints, step=1.0 / n_breakpoints)
    q = jnp.nanquantile(x, p, axis=0)
    p = jnp.expand_dims(p, axis=1) + 0.0 * q
    return p, q


def merge_equal_quantiles(p: jax.Array, q: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Resolve conflicts for zero slope by adding a minimal slope.

    For each feature dimension, if adjacent q values are equal, replaces
    corresponding p entries by their averaged probability to ensure strict
    piecewise-linearity.

    :param p: Probability array.
    :param q: Quantile array
    :return: Tuple of merged probabilities and quantiles
    """

    def merge(p, q):
        val, index, inverse, count = jnp.unique(q, return_index=True, return_inverse=True, return_counts=True)
        unique_p = 0.0 * val
        unique_p = unique_p.at[inverse].add(p) / count
        p = unique_p.at[inverse].get()
        return p, q

    p_list, q_list = [], []
    for i in range(p.shape[1]):
        p_i, q_i = merge(p[:, i], q[:, i])
        p_list.append(p_i)
        q_list.append(q_i)
    p, q = jnp.stack(p_list, axis=1), jnp.stack(q_list, axis=1)

    return p, q


def forward(x: jax.Array, xp: jax.Array, fp: jax.Array):
    r"""
    Piecewise-linear CDF mapping of input values.

    Interpolates x given breakpoints (xp, fp), with constant slopes extended
    beyond the first and last segments to enforce bijectivity on :math:`\mathbb{R}`.

    :param x: Input values.
    :param xp: Array of quantile breakpoints.
    :param fp: Array of mapped probabilities, scaled to [-1,1].
    :return: Transformed values of same shape as x.
    """
    EPS = 1e-6
    interp_term = jnp.interp(x, xp, fp)
    left_term = jnp.minimum(x - xp[0], 0) * (fp[1] - fp[0] + EPS) / (xp[1] - xp[0] + EPS)
    right_term = jnp.maximum(x - xp[-1], 0) * (fp[-1] - fp[-2] + EPS) / (xp[-1] - xp[-2] + EPS)
    return interp_term + left_term + right_term


def inverse(f: jax.Array, fp: jax.Array, xp: jax.Array):
    """
    Interpolates for f based on pair (fp, xp), inverse of :py:func:`forward`.

    :param f: Mapped values in [-1,1].
    :param fp: Probabilities breakpoints used in forward.
    :param xp: Quantile breakpoints used in forward.
    :return: Original-scale values corresponding to f.
    """
    return forward(f, fp, xp)
