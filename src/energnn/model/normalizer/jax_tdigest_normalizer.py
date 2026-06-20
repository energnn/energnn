# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import jax
import jax.numpy as jnp
from flax import nnx
from energnn.graph import GraphStructure
from energnn.graph.jax import JaxGraph, JaxHyperEdgeSet
from .normalizer import Normalizer


class JAXTDigestModule(nnx.Module):
    """
    Maintains and applies T-Digest normalization for a set of features using pure JAX.

    This module implements a 'Merging Digest' approach entirely in JAX, avoiding
    host-device synchronization overhead.
    """

    def __init__(
        self,
        in_size: int,
        update_limit: int,
        n_breakpoints: int,
        max_centroids: int,
        use_running_average: bool,
    ):
        self.in_size = in_size
        self.update_limit = update_limit
        self.n_breakpoints = n_breakpoints
        self.max_centroids = max_centroids
        self.use_running_average = use_running_average

        self.updates = nnx.Variable(jnp.array([0], dtype=jnp.int32))

        # State variables
        self.min_var = nnx.Variable(jnp.full((self.in_size,), jnp.nan, dtype=jnp.float32))
        self.max_var = nnx.Variable(jnp.full((self.in_size,), jnp.nan, dtype=jnp.float32))

        # Centroids: mean (m) and weight (c)
        self.centroids_m_var = nnx.Variable(jnp.zeros((self.max_centroids, self.in_size), dtype=jnp.float32))
        self.centroids_c_var = nnx.Variable(jnp.zeros((self.max_centroids, self.in_size), dtype=jnp.float32))

        # Grid for interpolation
        p_grid = jnp.linspace(0.0, 1.0, self.n_breakpoints)
        self.p_grid_var = nnx.Variable(p_grid)
        self.fp_var = nnx.Variable(jnp.tile((2.0 * p_grid - 1.0)[:, None], (1, self.in_size)))
        self.xp_var = nnx.Variable(jnp.tile(jnp.linspace(-1.0, 1.0, self.n_breakpoints)[:, None], (1, self.in_size)))

        # Slopes for extrapolation
        self.left_slope_var = nnx.Variable(jnp.ones((self.in_size,), dtype=jnp.float32))
        self.right_slope_var = nnx.Variable(jnp.ones((self.in_size,), dtype=jnp.float32))

        # Target quantiles for re-sampling (k1 scale function)
        target_p, target_w = self._get_target_quantiles(self.max_centroids)
        self.target_p_var = nnx.Variable(target_p)
        self.target_w_var = nnx.Variable(target_w)

    def _get_target_quantiles(self, n: int):
        """Generates target quantile positions and weights using k1 scale function."""
        k = jnp.linspace(-1.0, 1.0, n + 1)
        p_boundaries = (jnp.sin(k * jnp.pi / 2.0) + 1.0) / 2.0
        p_centers = (p_boundaries[:-1] + p_boundaries[1:]) / 2.0
        weights = jnp.diff(p_boundaries)
        return p_centers, weights

    def _update_digest(self, array: jax.Array, mask: jax.Array):
        """Pure JAX implementation of T-Digest update (Merging Digest)."""
        # array: (N, F), mask: (N, 1)
        F = self.in_size
        K = self.max_centroids
        B = array.shape[0]

        # 1. Update Min/Max
        batch_min = jnp.min(jnp.where(mask, array, jnp.inf), axis=0)
        batch_max = jnp.max(jnp.where(mask, array, -jnp.inf), axis=0)

        new_min = jnp.where(jnp.isnan(self.min_var[...]), batch_min, jnp.minimum(self.min_var[...], batch_min))
        new_max = jnp.where(jnp.isnan(self.max_var[...]), batch_max, jnp.maximum(self.max_var[...], batch_max))

        # 2. Merge Centroids
        # Concat current centroids and new data points
        all_m = jnp.concatenate([self.centroids_m_var[...], array], axis=0)  # (K+B, F)
        all_c = jnp.concatenate([self.centroids_c_var[...], jnp.broadcast_to(mask, (B, F))], axis=0)  # (K+B, F)

        # Sort along feature axis
        idx = jnp.argsort(all_m, axis=0)
        sorted_m = jnp.take_along_axis(all_m, idx, axis=0)
        sorted_c = jnp.take_along_axis(all_c, idx, axis=0)

        # Cumulative weights and total count
        cum_c = jnp.cumsum(sorted_c, axis=0)
        total_c = cum_c[-1:]  # (1, F)

        # Centroid positions (mid-point of weight increments)
        pos = cum_c - sorted_c / 2.0

        # Target quantiles for re-sampling (k1 scale function)
        target_p = self.target_p_var[...]
        target_w = self.target_w_var[...]
        p_grid = self.p_grid_var[...]

        # Combine interpolation points to do it in one pass
        combined_p = jnp.concatenate([target_p, p_grid])

        def interp_feat(p, m, tot, t_p):
            # p, m are (K+B,), tot is scalar, t_p is (combined_size,)
            return jnp.interp(t_p * tot, p, m)

        combined_interp = jax.vmap(interp_feat, in_axes=(1, 1, 1, None), out_axes=1)(pos, sorted_m, total_c, combined_p)
        new_centroids_m = combined_interp[:K]
        new_centroids_c = target_w[:, None] * total_c
        new_interp_interp = combined_interp[K:]

        # Compute new slopes
        EPS = 1e-6
        fp = self.fp_var[...]
        new_left_slope = (fp[1, :] - fp[0, :]) / (new_interp_interp[1, :] - new_interp_interp[0, :] + EPS)
        new_right_slope = (fp[-1, :] - fp[-2, :]) / (new_interp_interp[-1, :] - new_interp_interp[-2, :] + EPS)

        return new_min, new_max, new_centroids_m, new_centroids_c, new_interp_interp, new_left_slope, new_right_slope

    def __call__(self, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        is_training = not self.use_running_average
        should_update = is_training & (self.updates[...] < self.update_limit)[0]

        if array.ndim == 3:
            B, N, F = array.shape
            flat_array = array.reshape(B * N, F)
            flat_mask = non_fictitious.reshape(B * N, 1)
        else:
            flat_array = array
            flat_mask = non_fictitious

        def update_fn(a, m):
            return (*self._update_digest(a, m), jnp.array(True))

        def no_update_fn(a, m):
            return (
                self.min_var[...],
                self.max_var[...],
                self.centroids_m_var[...],
                self.centroids_c_var[...],
                self.xp_var[...],
                self.left_slope_var[...],
                self.right_slope_var[...],
                jnp.array(False),
            )

        # Update state if needed
        res = jax.lax.cond(should_update, update_fn, no_update_fn, flat_array, flat_mask)

        new_min, new_max, new_cm, new_cc, new_xp, new_ls, new_rs, did_update = res

        if is_training:
            self.updates[...] += jnp.where(did_update, 1, 0)
            self.min_var[...] = new_min
            self.max_var[...] = new_max
            self.centroids_m_var[...] = new_cm
            self.centroids_c_var[...] = new_cc
            self.xp_var[...] = new_xp
            self.left_slope_var[...] = new_ls
            self.right_slope_var[...] = new_rs

        # Apply normalization
        xp = self.xp_var[...]
        fp = self.fp_var[...]
        ls = self.left_slope_var[...]
        rs = self.right_slope_var[...]

        def forward_local(x_feat, xp_feat, fp_feat, ls_feat, rs_feat):
            interp_term = jnp.interp(x_feat, xp_feat, fp_feat)
            # Extrapolation
            left_term = jnp.minimum(x_feat - xp_feat[0], 0.0) * ls_feat
            right_term = jnp.maximum(x_feat - xp_feat[-1], 0.0) * rs_feat
            return interp_term + left_term + right_term

        # Always flatten for consistent vmap application
        if array.ndim == 3:
            B, N, F = array.shape
            out = jax.vmap(forward_local, in_axes=(1, 1, 1, 0, 0), out_axes=1)(flat_array, xp, fp, ls, rs)
            out = out.reshape(B, N, F)
        else:
            out = jax.vmap(forward_local, in_axes=(1, 1, 1, 0, 0), out_axes=1)(array, xp, fp, ls, rs)

        return out * non_fictitious


class JAXTDigestNormalizer(Normalizer):
    """
    Graph-level normalizer that maintains a JAXTDigestModule for each hyper-edge set type.
    Pure JAX version of TDigestNormalizer.
    """

    def __init__(
        self,
        in_structure: GraphStructure,
        update_limit: int,
        n_breakpoints: int = 20,
        max_centroids: int = 1000,
        use_running_average: bool = False,
    ):
        self.in_structure = in_structure
        self.update_limit = update_limit
        self.n_breakpoints = n_breakpoints
        self.max_centroids = max_centroids
        self.use_running_average = use_running_average

        self.module_dict = self._build_module_dict()

    def _build_module_dict(self) -> dict[str, dict[str, JAXTDigestModule]]:
        module_dict = {}
        for key, hyper_edge_set_structure in self.in_structure.hyper_edge_sets.items():
            if hyper_edge_set_structure.feature_list is not None:
                in_size = len(hyper_edge_set_structure.feature_list)
                module_dict[key] = JAXTDigestModule(
                    in_size=in_size,
                    update_limit=self.update_limit,
                    n_breakpoints=self.n_breakpoints,
                    max_centroids=self.max_centroids,
                    use_running_average=self.use_running_average,
                )
            else:
                module_dict[key] = None
        return nnx.data(module_dict)

    def set_running_average(self, use: bool):
        self.use_running_average = use
        for module in self.module_dict.values():
            if module is not None:
                module.use_running_average = use

    def __call__(self, *, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        hyper_edge_set_norm_dict = {
            k: (hyper_edge_set, self.module_dict[k])
            for k, hyper_edge_set in graph.hyper_edge_sets.items()
            if k in self.module_dict.keys() and self.module_dict[k] is not None
        }

        def apply_norm(edge_norm: tuple[JaxHyperEdgeSet, JAXTDigestModule]) -> JaxHyperEdgeSet:
            hyper_edge_set, normalizer = edge_norm
            array = hyper_edge_set.feature_array
            if array is not None:
                if array.shape[-2] > 0:
                    array = normalizer(array, jnp.expand_dims(hyper_edge_set.non_fictitious, -1))
            return JaxHyperEdgeSet(
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

        normalized_context = JaxGraph(
            hyper_edge_sets=final_sets,
            non_fictitious_addresses=graph.non_fictitious_addresses,
            true_shape=graph.true_shape,
            current_shape=graph.current_shape,
        )

        info = {"input_graph": graph.quantiles(), "output_graph": normalized_context.quantiles()} if get_info else {}
        return normalized_context, info
