# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import jax
import jax.numpy as jnp
from flax import nnx
from energnn.graph import GraphStructure, Graph, HyperEdgeSet
from .normalizer import Normalizer
from .jax_tdigest import JaxTDigest


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
