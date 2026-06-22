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

jax.config.update("jax_enable_x64", True)

ArrayLike = Union[float, Sequence[float], jnp.ndarray]
FloatArray = jnp.ndarray


def _as_1d_float64(x: ArrayLike) -> FloatArray:
    arr = jnp.asarray(x, dtype=jnp.float64)
    return jnp.ravel(arr)


def _split_weights(x: FloatArray, w: ArrayLike | None) -> FloatArray:
    if w is None:
        return jnp.ones_like(x, dtype=jnp.float64)

    w_arr = jnp.asarray(w, dtype=jnp.float64)
    if w_arr.ndim == 0:
        return jnp.full_like(x, float(w_arr), dtype=jnp.float64)

    w_arr = jnp.ravel(w_arr)
    if w_arr.shape[0] != x.shape[0]:
        raise ValueError("w must be either a scalar or have the same length as x.")
    return w_arr


def _compress_sorted(
    centroids: FloatArray,
    max_centroids: int,
    internal_capacity: int,
) -> FloatArray:
    # centroids: (N, 2) [mean, weight]
    # returns: (internal_capacity, 2)

    K = float(max_centroids)
    K_int = internal_capacity

    values = centroids[:, 0]
    weights = centroids[:, 1]

    order = jnp.argsort(values, stable=True)
    sorted_m = values[order]
    sorted_c = weights[order]

    total_w = jnp.maximum(jnp.sum(sorted_c), 1e-12)

    # Constants for optimized k-function logic (k1 scale function)
    delta = jnp.pi / K
    cos_delta = jnp.cos(delta)
    sin_delta = jnp.sin(delta)
    one_minus_cos_delta_div_2 = (1.0 - cos_delta) / 2.0

    def compute_cluster_ids(c_arr, tw):
        def body(state, w):
            q_base, curr_w, k_idx = state

            q_b = q_base / tw
            q_p = (q_base + curr_w + w) / tw

            # Safe sqrt for q near 0 or 1
            term_sqrt = jnp.sqrt(jnp.maximum(q_b * (1.0 - q_b), 0.0))
            q_limit = one_minus_cos_delta_div_2 + q_b * cos_delta + term_sqrt * sin_delta

            should_merge = (q_p <= q_limit + 1e-15) | (k_idx >= K_int - 1)

            cond = (w > 0) & (~should_merge)
            new_state = (
                jnp.where(cond, q_base + curr_w, q_base),
                jnp.where(cond, w, curr_w + w),
                jnp.where(cond, k_idx + 1, k_idx),
            )
            return new_state, new_state[2]

        init_state = (0.0, 0.0, 0)
        _, ids = jax.lax.scan(body, init_state, c_arr)
        return ids

    cluster_ids = compute_cluster_ids(sorted_c, total_w)

    new_c = jax.ops.segment_sum(sorted_c, cluster_ids, num_segments=K_int)
    new_m_weighted = jax.ops.segment_sum(sorted_m * sorted_c, cluster_ids, num_segments=K_int)

    new_m = jnp.where(new_c > 0, new_m_weighted / jnp.maximum(new_c, 1e-12), -1e40)
    new_m = jnp.maximum.accumulate(new_m)

    return jnp.stack([new_m, new_c], axis=-1)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JaxTDigest:
    max_centroids: int
    centroids: FloatArray  # (capacity, 2) -> [mean, weight]
    stats: FloatArray  # (3,) -> [mass, min_value, max_value]

    @classmethod
    def empty(cls, max_centroids: int = 1000) -> "JaxTDigest":
        capacity = int(1.5 * max_centroids)
        return cls(
            max_centroids=max_centroids,
            centroids=jnp.stack([jnp.full((capacity,), -1e40), jnp.zeros((capacity,))], axis=-1),
            stats=jnp.array([0.0, jnp.inf, -jnp.inf], dtype=jnp.float64),
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
        new_data = jnp.stack([x, w], axis=-1)
        combined = jnp.concatenate([self.centroids, new_data], axis=0)

        new_centroids = _compress_sorted(combined, self.max_centroids, self.internal_capacity)

        new_mass = jnp.sum(w) + self.mass
        new_min = jnp.minimum(self.min_value, jnp.min(jnp.where(w > 0, x, jnp.inf)))
        new_max = jnp.maximum(self.max_value, jnp.max(jnp.where(w > 0, x, -jnp.inf)))

        return JaxTDigest(
            max_centroids=self.max_centroids,
            centroids=new_centroids,
            stats=jnp.stack([new_mass, new_min, new_max]),
        )

    def batch_update(self, x: ArrayLike, w: ArrayLike | None = None) -> "JaxTDigest":
        x_arr = _as_1d_float64(x)
        if x_arr.size == 0:
            return self

        w_arr = _split_weights(x_arr, w)
        return self._merge_unsorted(x_arr, w_arr)

    def quantile_vec(self, q: ArrayLike) -> FloatArray:
        q_arr = jnp.asarray(q, dtype=jnp.float64)
        if q_arr.size == 0:
            return q_arr

        means = self.centroids[:, 0]
        weights = self.centroids[:, 1]

        cum = jnp.cumsum(weights)
        mid_q = (cum - 0.5 * weights) / jnp.maximum(self.mass, 1e-12)

        res = jnp.interp(q_arr, mid_q, means, left=self.min_value, right=self.max_value)

        res = jnp.where(self.mass == 0, jnp.nan, res)
        res = jnp.where(q_arr <= 0.0, self.min_value, res)
        res = jnp.where(q_arr >= 1.0, self.max_value, res)

        return res

    def cdf_vec(self, x: ArrayLike) -> FloatArray:
        x_arr = _as_1d_float64(x)
        if x_arr.size == 0:
            return x_arr

        means = self.centroids[:, 0]
        weights = self.centroids[:, 1]

        cum = jnp.cumsum(weights)
        mid_q = (cum - 0.5 * weights) / jnp.maximum(self.mass, 1e-12)

        res = jnp.interp(x_arr, means, mid_q, left=0.0, right=1.0)
        res = jnp.where(self.mass == 0, jnp.nan, res)
        return res
