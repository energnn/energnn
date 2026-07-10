# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union
import jax
import jax.numpy as jnp

ArrayLike = Union[float, Sequence[float], jnp.ndarray]
FloatArray = jnp.ndarray


def _as_1d_float32(x: ArrayLike) -> FloatArray:
    """
    Converts input to a 1D float32 JAX array.
    """
    arr = jnp.asarray(x, dtype=jnp.float32)
    return jnp.ravel(arr)


def _split_weights(x: FloatArray, w: ArrayLike | None) -> FloatArray:
    """
    Prepares the weights array to match the input data shape.
    """
    if w is None:
        return jnp.ones_like(x, dtype=jnp.float32)

    w_arr = jnp.asarray(w, dtype=jnp.float32)
    if w_arr.ndim == 0:
        return jnp.full_like(x, float(w_arr), dtype=jnp.float32)

    w_arr = jnp.ravel(w_arr)
    if w_arr.shape[0] != x.shape[0]:
        raise ValueError("w must be either a scalar or have the same length as x.")
    return w_arr


def _compress_sorted(
    centroids: FloatArray,
    max_centroids: int,
    internal_capacity: int,
) -> FloatArray:
    """
    Compresses a set of centroids by merging those that are close to each other.

    This function implements the core T-Digest compression logic using the k1 scale function.
    It sorts the input centroids by mean and then uses a scan to decide which centroids
    to merge based on the cumulative weight and the maximum allowed weight for a centroid
    at its current quantile.

    :param centroids: Array of shape (N, 2) where each row is [mean, weight].
    :param max_centroids: The target number of centroids (compression parameter K).
    :param internal_capacity: The fixed size of the output centroid array (for JAX compatibility).

    :return: A compressed centroid array of shape (internal_capacity, 2).
    """
    # centroids: (N, 2) [mean, weight]
    # returns: (internal_capacity, 2)

    K = jnp.array(max_centroids, dtype=jnp.float32)
    K_int = internal_capacity

    values = centroids[:, 0]
    weights = centroids[:, 1]

    # Sort centroids by mean to allow sequential merging
    order = jnp.argsort(values, stable=True)
    # Sorted means (m for mean)
    sorted_m = values[order]
    # Sorted weights/counts (c for count)
    sorted_c = weights[order]

    total_w = jnp.maximum(jnp.sum(sorted_c), jnp.array(1e-7, dtype=jnp.float32))

    # Constants for optimized k-function logic (k1 scale function)
    # The k1 scale function is: k(q) = (K/pi) * arcsin(2q - 1)
    # The merging criteria is: k(q_after) - k(q_before) <= 1
    # This ensures smaller centroids at the edges (q near 0 or 1).
    delta = jnp.array(jnp.pi, dtype=jnp.float32) / K
    cos_delta = jnp.cos(delta)
    sin_delta = jnp.sin(delta)
    # Pre-calculated term for the optimized arcsin difference check
    one_minus_cos_delta_div_2 = (jnp.array(1.0, dtype=jnp.float32) - cos_delta) / jnp.array(2.0, dtype=jnp.float32)

    def compute_cluster_ids(c_arr, tw):
        def body(state, w):
            q_base, curr_w, k_idx = state

            # Current quantile at the start of the candidate cluster
            q_b = q_base / tw
            # Potential quantile at the end if we include the current point
            q_p = (q_base + curr_w + w) / tw

            # Optimized check for k(q_p) - k(q_b) <= 1
            # Equivalent to: q_p <= q_b*cos(delta) + sqrt(q_b*(1-q_b))*sin(delta) + (1-cos(delta))/2
            # derived from sin(arcsin(2q_b-1) + delta)
            term_sqrt = jnp.sqrt(jnp.maximum(q_b * (jnp.array(1.0, dtype=jnp.float32) - q_b), 0.0))
            q_limit = one_minus_cos_delta_div_2 + q_b * cos_delta + term_sqrt * sin_delta

            # Merge if:
            # 1. Candidate cluster is empty (curr_w == 0)
            # 2. It respects the scale function limit
            # 3. We haven't exceeded the fixed internal capacity
            should_merge = (curr_w == 0) | (q_p <= q_limit + jnp.array(1e-7, dtype=jnp.float32)) | (k_idx >= K_int - 1)

            cond = (w > 0) & (~should_merge)
            new_state = (
                jnp.where(cond, q_base + curr_w, q_base),  # Update q_base if we start a new cluster
                jnp.where(cond, w, curr_w + w),  # Reset or increment current cluster weight
                jnp.where(cond, k_idx + 1, k_idx),  # Increment cluster ID if new cluster
            )
            return new_state, new_state[2]

        init_state = (jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32), 0)
        _, ids = jax.lax.scan(body, init_state, c_arr)
        return ids

    cluster_ids = compute_cluster_ids(sorted_c, total_w)

    # Sum weights and weighted means for each cluster
    # Aggregate weights for each identified cluster (segment)
    new_c = jax.ops.segment_sum(sorted_c, cluster_ids, num_segments=K_int)
    # Aggregate weighted means (sum of mean * weight) for each cluster
    new_m_weighted = jax.ops.segment_sum(sorted_m * sorted_c, cluster_ids, num_segments=K_int)

    # Compute new means
    # If a cluster has weight > 0, calculate the weighted average.
    # Otherwise, use a placeholder value for empty segments.
    new_m = jnp.where(
        new_c > 0, new_m_weighted / jnp.maximum(new_c, jnp.array(1e-7, dtype=jnp.float32)), jnp.array(-1e30, dtype=jnp.float32)
    )
    # Ensure means are strictly non-decreasing (handling potential numerical noise)
    new_m = jnp.maximum.accumulate(new_m)

    return jnp.stack([new_m, new_c], axis=-1)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxTDigest:
    """
    JAX-compatible T-Digest structure.

    T-Digest is a data structure for estimating quantiles and cumulative distribution functions (CDF)
    from data streams or large datasets with high accuracy, especially at the tails.

    The algorithm works by clustering data points into 'centroids'. Each centroid represents
    a group of points with a mean value and a total weight (number of points).
    The number of centroids is kept bounded by a 'compression' process that merges
    centroids while respecting a scale function (usually k1). This scale function
    ensures that centroids near the tails are smaller (more precise) than those in the middle.

    Attributes:
        max_centroids: Compression parameter (K). Higher values mean more precision and more centroids.
        centroids: Array of shape (capacity, 2) storing [mean, weight] for each cluster.
        stats: Array of shape (3,) storing [total_mass, min_value, max_value].

    References:
        - Ted Dunning and Otmar Ertl, "Computing Extremely Accurate Quantiles Using t-Digests"
        - https://github.com/tdunning/t-digest
    """

    max_centroids: int
    centroids: FloatArray  # (capacity, 2) -> [mean, weight]
    stats: FloatArray  # (3,) -> [mass, min_value, max_value]

    @classmethod
    def empty(cls, max_centroids: int = 1000) -> "JaxTDigest":
        """
        Initializes an empty T-Digest.
        """
        capacity = int(1.5 * max_centroids)
        return cls(
            max_centroids=max_centroids,
            centroids=jnp.stack(
                [jnp.full((capacity,), jnp.array(-1e30, dtype=jnp.float32)), jnp.zeros((capacity,), dtype=jnp.float32)],
                axis=-1,
            ),
            stats=jnp.array([0.0, jnp.inf, -jnp.inf], dtype=jnp.float32),
        )

    def tree_flatten(self):
        children = (self.centroids, self.stats)
        aux_data = (self.max_centroids,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)

    def is_empty(self) -> bool:
        return self.mass == 0

    @property
    def internal_capacity(self) -> int:
        return self.centroids.shape[0]

    @property
    def mass(self) -> float:
        return self.stats[..., 0]

    @property
    def min_value(self) -> float:
        return self.stats[..., 1]

    @property
    def max_value(self) -> float:
        return self.stats[..., 2]

    def _merge_unsorted(self, x: FloatArray, w: FloatArray) -> "JaxTDigest":
        """
        Merges new observations (unsorted) into the existing T-Digest.
        """
        new_data = jnp.stack([x, w], axis=-1)
        combined = jnp.concatenate([self.centroids, new_data], axis=0)

        new_centroids = _compress_sorted(combined, self.max_centroids, self.internal_capacity)

        new_mass = jnp.sum(w) + self.mass
        new_min = jnp.minimum(self.min_value, jnp.min(jnp.where(w > 0, x, jnp.array(jnp.inf, dtype=jnp.float32))))
        new_max = jnp.maximum(self.max_value, jnp.max(jnp.where(w > 0, x, jnp.array(-jnp.inf, dtype=jnp.float32))))

        return JaxTDigest(
            max_centroids=self.max_centroids,
            centroids=new_centroids,
            stats=jnp.stack([new_mass, new_min, new_max]).astype(jnp.float32),
        )

    def batch_update(self, x: ArrayLike, w: ArrayLike | None = None) -> "JaxTDigest":
        """
        Updates the T-Digest with a batch of new values.

        :param x: Array of values to add.
        :param w: Optional array of weights for each value. Defaults to 1.0 for each value.
        """
        x_arr = _as_1d_float32(x)
        if x_arr.size == 0:
            return self

        w_arr = _split_weights(x_arr, w)
        return self._merge_unsorted(x_arr, w_arr)

    def quantile_vec(self, q: ArrayLike) -> FloatArray:
        """
        Estimates quantiles for a given set of probabilities.

        Quantile estimation is done by linear interpolation between centroid means.
        The 'quantile' of a centroid is defined as the midpoint of its weight range
        in the sorted order of centroids.

        :param q: Array of probabilities in [0, 1].

        :return: Array of estimated quantiles.
        """
        q_arr = jnp.asarray(q, dtype=jnp.float32)
        if q_arr.size == 0:
            return q_arr

        means = self.centroids[:, 0]
        weights = self.centroids[:, 1]

        # Cumulative weight at the end of each centroid
        cum = jnp.cumsum(weights)
        # Probabilities at the center of each centroid
        mid_q = (cum - jnp.array(0.5, dtype=jnp.float32) * weights) / jnp.maximum(
            self.mass, jnp.array(1e-7, dtype=jnp.float32)
        )

        # Interpolate between centroid centers
        res = jnp.interp(q_arr, mid_q, means, left=self.min_value, right=self.max_value)

        # Boundary conditions and empty digest handling
        res = jnp.where(self.mass == 0, jnp.nan, res)
        res = jnp.where(q_arr <= 0.0, self.min_value, res)
        res = jnp.where(q_arr >= 1.0, self.max_value, res)

        return res

    def cdf_vec(self, x: ArrayLike) -> FloatArray:
        """
        Estimates the Cumulative Distribution Function (CDF) at given values.

        :param x: Array of values at which to estimate the CDF.

        :return: Array of estimated CDF values in [0, 1].
        """
        x_arr = _as_1d_float32(x)
        if x_arr.size == 0:
            return x_arr

        means = self.centroids[:, 0]
        weights = self.centroids[:, 1]

        cum = jnp.cumsum(weights)
        # Probabilities at the center of each centroid
        mid_q = (cum - jnp.array(0.5, dtype=jnp.float32) * weights) / jnp.maximum(
            self.mass, jnp.array(1e-7, dtype=jnp.float32)
        )

        # Interpolate probabilities for given values
        res = jnp.interp(x_arr, means, mid_q, left=0.0, right=1.0)
        res = jnp.where(self.mass == 0, jnp.nan, res)
        return res
