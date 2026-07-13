# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union
import logging
import jax
import jax.numpy as jnp
from flax import nnx
from energnn.graph import GraphStructure, Graph, HyperEdgeSet
from .normalizer import Normalizer

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


class TDigestModule(nnx.Module):
    """
    Maintains and applies T-Digest normalization for a set of features using pure JAX.

    This module uses the T-Digest algorithm JAX-implemented to estimate quantiles and map input
    features to a target distribution (piecewise linear interpolation).
    """

    def __init__(
        self,
        in_size: int,
        update_limit: int,
        n_breakpoints: int,
        max_centroids: int,
        use_running_average: bool,
        saturation_strategy: str | None = None,
        clip_min: float | None = None,
        clip_max: float | None = None,
        update_frequency: int = 1,
    ):
        """
        Initializes the TDigestModule.

        :param in_size: Number of features to normalize.
        :param update_limit: Maximum number of update steps allowed.
        :param n_breakpoints: Number of points for the interpolation grid.
        :param max_centroids: Maximum number of centroids for the T-Digest.
        :param use_running_average: If True, skips updates and uses current state (inference mode).
        :param saturation_strategy: Strategy for saturation, either "hard" (clipping to [clip_min, clip_max]),
            "soft" (applying tanh function), or None (no saturation). Defaults to None.
        :param clip_min: Minimum value for hard saturation. Required if saturation_strategy is "hard".
        :param clip_max: Maximum value for hard saturation. Required if saturation_strategy is "hard".
        :param update_frequency: Frequency of update steps. Defaults to 1 (update at every step).
            Updates are always performed at the first step (step 0).
        """
        if saturation_strategy not in [None, "hard", "soft"]:
            raise ValueError(f"saturation_strategy must be None, 'hard' or 'soft', got {saturation_strategy}")
        if saturation_strategy == "hard":
            if clip_min is None or clip_max is None:
                raise ValueError("clip_min and clip_max must be provided when saturation_strategy is 'hard'")
            if clip_min >= clip_max:
                raise ValueError(f"clip_min must be strictly less than clip_max, got {clip_min} >= {clip_max}")

        self.in_size = in_size
        self.update_limit = update_limit
        self.update_frequency = update_frequency
        self.n_breakpoints = n_breakpoints
        self.max_centroids = max_centroids
        self.use_running_average = use_running_average
        self.saturation_strategy = saturation_strategy
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.updates = nnx.Variable(jnp.array([0], dtype=jnp.int32))
        self.train_steps = nnx.Variable(jnp.array([0], dtype=jnp.int32))

        # Initialize vmapped JaxTDigest
        def init_digest(_):
            return JaxTDigest.empty(self.max_centroids)

        self.digest = nnx.Variable(jax.vmap(init_digest)(jnp.arange(self.in_size)))

        # Grid for interpolation
        self.p_grid = jnp.linspace(0.0, 1.0, self.n_breakpoints, dtype=jnp.float32)
        self.fp = self.p_grid * 2.0 - 1.0
        self.dfp_left = self.fp[1] - self.fp[0]
        self.dfp_right = self.fp[-1] - self.fp[-2]
        self.xp_var = nnx.Variable(
            jnp.tile(jnp.linspace(-1.0, 1.0, self.n_breakpoints, dtype=jnp.float32)[:, None], (1, self.in_size))
        )

        # Slopes for extrapolation [left, right]
        self.slopes_var = nnx.Variable(jnp.ones((2, self.in_size), dtype=jnp.float32))

    @property
    def min_var(self):
        return self.digest.get_value().min_value

    @property
    def max_var(self):
        return self.digest.get_value().max_value

    def _update_digest(self, array: jax.Array, mask: jax.Array):
        """Pure JAX implementation of T-Digest update."""
        array = array.astype(jnp.float32)
        mask = mask.astype(jnp.float32)

        # Update Digest and Grid in a single vmap
        def update_and_get_xp(d, x, w, p):
            new_d = d.batch_update(x, w)
            new_xp = new_d.quantile_vec(p)
            return new_d, new_xp

        new_digest, new_xp = jax.vmap(update_and_get_xp, in_axes=(0, 1, None, None), out_axes=(0, 1))(
            self.digest.get_value(), array, mask.squeeze(-1), self.p_grid
        )

        # Compute new slopes
        EPS = 1e-6
        new_left_slope = self.dfp_left / (new_xp[1, :] - new_xp[0, :] + EPS)
        new_right_slope = self.dfp_right / (new_xp[-1, :] - new_xp[-2, :] + EPS)
        new_slopes = jnp.stack([new_left_slope, new_right_slope], axis=0)

        return new_digest, new_xp, new_slopes

    def __call__(self, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        is_training = not self.use_running_average
        should_update = (
            is_training & (self.updates[...] < self.update_limit)[0] & (self.train_steps[...] % self.update_frequency == 0)[0]
        )

        if array.ndim == 3:
            B, N, F = array.shape
            flat_array = array.reshape(B * N, F)
            flat_mask = non_fictitious.reshape(B * N, 1)
        else:
            flat_array = array
            flat_mask = non_fictitious

        # Update state (Synchronous update)
        def update_fn(a, m):
            new_digest, new_xp, new_sl = self._update_digest(a, m)
            return new_digest, new_xp, new_sl, jnp.array(True)

        def no_update_fn(a, m):
            return (
                self.digest.get_value(),
                self.xp_var[...],
                self.slopes_var[...],
                jnp.array(False),
            )

        new_digest, new_xp, new_sl, did_update = jax.lax.cond(should_update, update_fn, no_update_fn, flat_array, flat_mask)

        if is_training:
            self.updates[...] += jnp.where(did_update, 1, 0)
            self.train_steps[...] += 1
            self.digest.set_value(jax.tree.map(jax.lax.stop_gradient, new_digest))
            self.xp_var[...] = jax.lax.stop_gradient(new_xp)
            self.slopes_var[...] = jax.lax.stop_gradient(new_sl)

        # Normalize with current (potentially updated) state
        xp = self.xp_var[...]
        slopes = self.slopes_var[...]

        def forward_local(x_feat, xp_feat, slopes_feat):
            interp_term = jnp.interp(x_feat, xp_feat, self.fp)
            # Extrapolation
            left_term = jnp.minimum(x_feat - xp_feat[0], 0.0) * slopes_feat[0]
            right_term = jnp.maximum(x_feat - xp_feat[-1], 0.0) * slopes_feat[1]
            return interp_term + left_term + right_term

        # Apply normalization
        if array.ndim == 3:
            out = jax.vmap(forward_local, in_axes=(1, 1, 0), out_axes=1)(flat_array, xp, slopes.T)
            out = out.reshape(B, N, F)
        else:
            out = jax.vmap(forward_local, in_axes=(1, 1, 0), out_axes=1)(array, xp, slopes.T)

        if self.saturation_strategy is not None:
            if self.saturation_strategy == "hard":
                # Warning mechanism only for hard clipping
                def log_clipping(has_clipped):
                    if has_clipped:
                        logging.warning(
                            f"Normalization saturation occurred: some values were outside [{self.clip_min}, {self.clip_max}]"
                        )

                has_clipped = jnp.any((out < self.clip_min) | (out > self.clip_max))
                jax.debug.callback(log_clipping, has_clipped)
                out = jnp.clip(out, self.clip_min, self.clip_max)
            elif self.saturation_strategy == "soft":
                out = jnp.tanh(out)
            else:
                raise ValueError(f"Unknown saturation_strategy: {self.saturation_strategy}")

        return out * non_fictitious


class TDigestNormalizer(Normalizer):
    """
    Graph-level normalizer that maintains a TDigestModule for each hyper-edge set type.

    This normalizer uses T-Digests to map feature distributions to a target grid
    (usually [-1, 1]), providing a non-parametric alternative to standard normalization.
    """

    def __init__(
        self,
        in_structure: GraphStructure,
        update_limit: int,
        n_breakpoints: int = 20,
        max_centroids: int = 1000,
        use_running_average: bool = False,
        saturation_strategy: str | None = None,
        clip_min: float | None = None,
        clip_max: float | None = None,
        update_frequency: int = 1,
    ):
        """
        Initializes the TDigestNormalizer.

        :param in_structure: Structure of the input graph.
        :param update_limit: Maximum number of updates allowed for the T-Digests.
        :param n_breakpoints: Number of breakpoints for the interpolation grid.
        :param max_centroids: Maximum number of centroids for each T-Digest.
        :param use_running_average: Initial state for the running average flag.
        :param saturation_strategy: Strategy for saturation, either "hard" (clipping to [clip_min, clip_max]),
            "soft" (applying tanh function), or None (no saturation). Defaults to None.
        :param clip_min: Minimum value for hard saturation. Required if saturation_strategy is "hard".
        :param clip_max: Maximum value for hard saturation. Required if saturation_strategy is "hard".
        :param update_frequency: Frequency of update steps for each T-Digest. Defaults to 1 (update at every step).
            Updates are always performed at the first step (step 0).
        """
        if saturation_strategy not in [None, "hard", "soft"]:
            raise ValueError(f"saturation_strategy must be None, 'hard' or 'soft', got {saturation_strategy}")
        if saturation_strategy == "hard":
            if clip_min is None or clip_max is None:
                raise ValueError("clip_min and clip_max must be provided when saturation_strategy is 'hard'")
            if clip_min >= clip_max:
                raise ValueError(f"clip_min must be strictly less than clip_max, got {clip_min} >= {clip_max}")

        self.in_structure = in_structure
        self.update_limit = update_limit
        self.update_frequency = update_frequency
        self.n_breakpoints = n_breakpoints
        self.max_centroids = max_centroids
        self.use_running_average = use_running_average
        self.saturation_strategy = saturation_strategy
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.module_dict = self._build_module_dict()

    def _build_module_dict(self) -> dict[str, dict[str, TDigestModule]]:
        """Creates a TDigest module for each hyper-edge set key in the graph structure."""
        module_dict = {}
        for key, hyper_edge_set_structure in self.in_structure.hyper_edge_sets.items():
            if hyper_edge_set_structure.feature_list is not None:
                in_size = len(hyper_edge_set_structure.feature_list)
                module_dict[key] = TDigestModule(
                    in_size=in_size,
                    update_limit=self.update_limit,
                    n_breakpoints=self.n_breakpoints,
                    max_centroids=self.max_centroids,
                    use_running_average=self.use_running_average,
                    saturation_strategy=self.saturation_strategy,
                    clip_min=self.clip_min,
                    clip_max=self.clip_max,
                    update_frequency=self.update_frequency,
                )
            else:
                module_dict[key] = None
        return nnx.data(module_dict)

    def set_running_average(self, use: bool):
        """
        Sets the running average flag for the normalizer and all its sub-modules.

        :param use: If True, enables inference mode (no updates).
        """
        self.use_running_average = use
        for module in self.module_dict.values():
            if module is not None:
                module.use_running_average = use

    def __call__(self, *, graph: Graph, get_info: bool = False) -> tuple[Graph, dict]:
        """
        Apply normalization to hyper-edge sets within a Graph context using TDigest modules. This method normalizes the
        hyper-edge sets' feature arrays and updates the associated context graph accordingly.

        :param graph: Graph representing the graph structure containing hyper-edge sets with feature arrays
                      to be normalized.
        :param get_info: Boolean flag that indicates whether to return additional information about input and output graphs.
        :return: A tuple containing the normalized Graph and an optional dictionary holding quantile information
                 about the input and output graphs.
        """

        hyper_edge_set_norm_dict = {
            k: (hyper_edge_set, self.module_dict[k])
            for k, hyper_edge_set in graph.hyper_edge_sets.items()
            if k in self.module_dict.keys() and self.module_dict[k] is not None
        }

        def apply_norm(edge_norm: tuple[HyperEdgeSet, TDigestModule]) -> HyperEdgeSet:
            hyper_edge_set, normalizer = edge_norm
            array = hyper_edge_set.feature_array
            if array is not None:
                if array.shape[-2] > 0:
                    array = normalizer(array, jnp.expand_dims(hyper_edge_set.non_fictitious, -1))
            return HyperEdgeSet(
                backend=hyper_edge_set._backend,
                feature_array=array,
                feature_names=hyper_edge_set.feature_names,
                non_fictitious=hyper_edge_set.non_fictitious,
                port_dict=hyper_edge_set.port_dict,
            )

        normalized_hyper_edge_sets = jax.tree.map(
            apply_norm, hyper_edge_set_norm_dict, is_leaf=(lambda x: isinstance(x, tuple))
        )

        # Merge with non-normalized sets
        final_sets = dict(graph.hyper_edge_sets)
        final_sets.update(normalized_hyper_edge_sets)

        normalized_context = Graph(
            backend=graph._backend,
            hyper_edge_sets=final_sets,
            non_fictitious_addresses=graph.non_fictitious_addresses,
            true_shape=graph.true_shape,
            current_shape=graph.current_shape,
        )

        if get_info:
            info = {"input_graph": graph.quantiles(), "output_graph": normalized_context.quantiles()}
        else:
            info = {}

        return normalized_context, info
