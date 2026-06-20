# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import concurrent.futures
from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from fastdigest import TDigest
from flax import nnx
from jax import ShapeDtypeStruct
from jax.experimental import io_callback

from energnn.graph import Graph, GraphStructure, HyperEdgeSet
from .normalizer import Normalizer

# Global pool to avoid overhead of creating it on each call
_POOL = concurrent.futures.ThreadPoolExecutor()


def _merge_single_feature_quantiles(pf: np.ndarray, qf: np.ndarray) -> np.ndarray:
    """Helper to merge quantiles for a single feature."""
    vals, inv, counts = np.unique(qf, return_inverse=True, return_counts=True)
    sum_p_per_unique = np.zeros_like(vals, dtype=np.float64)
    np.add.at(sum_p_per_unique, inv, pf)
    avg_p_per_unique = sum_p_per_unique / counts
    return avg_p_per_unique[inv].astype(np.float32)


def _merge_equal_quantiles_host(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolves equal-quantile conflicts by averaging probabilities for identical quantile values.
    Parallelized version using the global ThreadPoolExecutor.
    """
    K, F = q.shape
    q_out = q.astype(np.float32)

    # Parallelize over features if F is large enough to justify the overhead
    if F > 16:
        futures = [_POOL.submit(_merge_single_feature_quantiles, p[:, f], q_out[:, f]) for f in range(F)]
        results = [f.result() for f in futures]
    else:
        results = [_merge_single_feature_quantiles(p[:, f], q_out[:, f]) for f in range(F)]

    p_out = np.stack(results, axis=1)
    return p_out, q_out


def _update_single_feature(
    feature_array: np.ndarray,
    _max_centroids: int,
    _min: float,
    _max: float,
    _c_m: np.ndarray,
    _c_c: np.ndarray,
) -> tuple[int, float, float, list[tuple[float, float]]]:
    """Update TDigest for a single feature."""
    valid_mask = _c_c > 0
    ms = _c_m[valid_mask].tolist()
    cs = _c_c[valid_mask].tolist()

    tdigest_dict = {
        "max_centroids": _max_centroids,
        "min": 0.0 if np.isnan(_min) else _min,
        "max": 0.0 if np.isnan(_max) else _max,
        "centroids": [{"m": m, "c": c} for m, c in zip(ms, cs)],
    }

    tdigest = TDigest.from_dict(tdigest_dict)

    if feature_array.size > 0:
        tdigest.batch_update(feature_array)

    return tdigest.max_centroids, tdigest.min(), tdigest.max(), tdigest.centroids


def _quantiles_single_feature(
    _max_centroids: int,
    _min: float,
    _max: float,
    _c_m: np.ndarray,
    _c_c: np.ndarray,
    p_list: np.ndarray,
) -> np.ndarray:
    """Extract quantiles for a single feature from its state."""
    valid_mask = _c_c > 0
    ms = _c_m[valid_mask].tolist()
    cs = _c_c[valid_mask].tolist()

    tdigest_dict = {
        "max_centroids": _max_centroids,
        "min": 0.0 if np.isnan(_min) else _min,
        "max": 0.0 if np.isnan(_max) else _max,
        "centroids": [{"m": m, "c": c} for m, c in zip(ms, cs)],
    }
    tdigest = TDigest.from_dict(tdigest_dict)
    return np.array([tdigest.quantile(p) for p in p_list], dtype=np.float32)


def _update_tdigest_host(
    max_centroids: Sequence[int],
    min_val: Sequence[float],
    max_val: Sequence[float],
    centroids_m: np.ndarray,
    centroids_c: np.ndarray,
    array: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host-side callback to update T-Digest state only."""
    if array.ndim == 3:
        B, N, F = array.shape
        array = array.reshape(B * N, F)
        mask = mask.reshape(B * N, 1)
    else:
        N, F = array.shape

    mask = mask.flatten().astype(bool)
    array = array[mask]

    n_features = array.shape[-1]
    max_c_limit = centroids_m.shape[0]

    if n_features > 16:
        futures = [
            _POOL.submit(
                _update_single_feature,
                array[:, i],
                int(max_centroids[i]),
                float(min_val[i]),
                float(max_val[i]),
                centroids_m[:, i],
                centroids_c[:, i],
            )
            for i in range(n_features)
        ]
        results = [f.result() for f in futures]
    else:
        results = [
            _update_single_feature(
                array[:, i],
                int(max_centroids[i]),
                float(min_val[i]),
                float(max_val[i]),
                centroids_m[:, i],
                centroids_c[:, i],
            )
            for i in range(n_features)
        ]

    new_max_centroids = np.zeros(n_features, dtype=np.int32)
    new_min_array = np.zeros(n_features, dtype=np.float32)
    new_max_array = np.zeros(n_features, dtype=np.float32)
    new_c_m_matrix = np.zeros((max_c_limit, n_features), dtype=np.float32)
    new_c_c_matrix = np.zeros((max_c_limit, n_features), dtype=np.float32)

    for i, (m_c, mi, ma, cents) in enumerate(results):
        new_max_centroids[i] = m_c
        new_min_array[i] = mi
        new_max_array[i] = ma

        num_cents = len(cents)
        if num_cents > 0:
            ms, cs = zip(*cents)
            actual_num = min(num_cents, max_c_limit)
            new_c_m_matrix[:actual_num, i] = ms[:actual_num]
            new_c_c_matrix[:actual_num, i] = cs[:actual_num]

    return new_max_centroids, new_min_array, new_max_array, new_c_m_matrix, new_c_c_matrix


def _compute_quantiles_host(
    max_centroids: np.ndarray,
    min_val: np.ndarray,
    max_val: np.ndarray,
    centroids_m: np.ndarray,
    centroids_c: np.ndarray,
    p_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure host-side callback to compute xp and fp from state."""
    # p_grid is (F, K) or just K (if same for all)
    # We assume p_grid is (K,) or (F, K)
    K = p_grid.shape[-1]
    n_features = centroids_m.shape[1]
    if p_grid.ndim == 1:
        p_list = p_grid
        p_matrix = np.tile(p_list, (n_features, 1))
    else:
        p_list = p_grid[0]  # Just for internal single feature use if needed
        p_matrix = p_grid

    if n_features > 16:
        futures = [
            _POOL.submit(
                _quantiles_single_feature,
                int(max_centroids[i]),
                float(min_val[i]),
                float(max_val[i]),
                centroids_m[:, i],
                centroids_c[:, i],
                p_matrix[i],
            )
            for i in range(n_features)
        ]
        new_q_matrix = np.stack([f.result() for f in futures], axis=0)  # (F, K)
    else:
        new_q_matrix = np.stack(
            [
                _quantiles_single_feature(
                    int(max_centroids[i]),
                    float(min_val[i]),
                    float(max_val[i]),
                    centroids_m[:, i],
                    centroids_c[:, i],
                    p_matrix[i],
                )
                for i in range(n_features)
            ],
            axis=0,
        )

    p_merged, q_merged = _merge_equal_quantiles_host(p_matrix.T, new_q_matrix.T)
    new_xp = q_merged.astype(np.float32)
    new_fp = (-1.0 + 2.0 * p_merged).astype(np.float32)

    return new_fp, new_xp


def _ingest_new_data(
    max_centroids: Sequence[int],
    min_val: Sequence[float],
    max_val: Sequence[float],
    centroids_m: np.ndarray,
    centroids_c: np.ndarray,
    fp: np.ndarray,
    xp: np.ndarray,
    array: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legacy wrapper for tests and backward compatibility."""
    new_max_c, new_min, new_max, new_c_m, new_c_c = _update_tdigest_host(
        max_centroids, min_val, max_val, centroids_m, centroids_c, array, mask
    )
    K = fp.shape[0]
    p_grid = np.linspace(0, 1, K)
    new_fp, new_xp = _compute_quantiles_host(new_max_c, new_min, new_max, new_c_m, new_c_c, p_grid)
    return new_max_c, new_min, new_max, new_c_m, new_c_c, new_fp, new_xp


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _tdigest_update(
    array: jax.Array,
    non_fictitious: jax.Array,
    module_state: tuple[jax.Array, ...],
    in_size: int,
    max_centroids: int,
) -> tuple[jax.Array, ...]:
    """Updates T-Digest state using IO callback."""
    (max_centroids_val, min_val, max_val, centroids_m, centroids_c, _, _) = module_state

    result_shapes = (
        ShapeDtypeStruct((in_size,), jnp.int32),  # max_centroids
        ShapeDtypeStruct((in_size,), jnp.float32),  # min
        ShapeDtypeStruct((in_size,), jnp.float32),  # max
        ShapeDtypeStruct((max_centroids, in_size), jnp.float32),  # centroids_m
        ShapeDtypeStruct((max_centroids, in_size), jnp.float32),  # centroids_c
    )

    return io_callback(
        _update_tdigest_host,
        result_shapes,
        max_centroids_val,
        min_val,
        max_val,
        centroids_m,
        centroids_c,
        array,
        non_fictitious,
    )


def _tdigest_update_fwd(array, non_fictitious, module_state, in_size, max_centroids):
    new_state = _tdigest_update(array, non_fictitious, module_state, in_size, max_centroids)
    return new_state, (array, non_fictitious)


def _tdigest_update_bwd(in_size, max_centroids, res, grads):
    array, non_fictitious = res
    return 0 * array, None, None


_tdigest_update.defvjp(_tdigest_update_fwd, _tdigest_update_bwd)


def _tdigest_get_quantiles(
    state: tuple[jax.Array, ...],
    p_grid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Retrieves xp and fp from state using a pure callback."""
    (max_centroids_val, min_val, max_val, centroids_m, centroids_c) = state
    n_breakpoints = p_grid.shape[-1]
    in_size = max_centroids_val.shape[0]

    result_shapes = (
        ShapeDtypeStruct((n_breakpoints, in_size), jnp.float32),  # fp
        ShapeDtypeStruct((n_breakpoints, in_size), jnp.float32),  # xp
    )

    return jax.pure_callback(
        _compute_quantiles_host,
        result_shapes,
        max_centroids_val,
        min_val,
        max_val,
        centroids_m,
        centroids_c,
        p_grid,
    )


class TDigestModule(nnx.Module):
    """
    Maintains and applies T-Digest normalization for a set of features.

    This module uses the T-Digest algorithm to estimate quantiles and map input
    features to a target distribution (piecewise linear interpolation).
    It supports batch updates via an IO callback and provides a fast inference path.
    """

    def __init__(
        self,
        in_size: int,
        update_limit: int,
        n_breakpoints: int,
        max_centroids: int,
        use_running_average: bool,
    ):
        """
        Initializes the TDigestModule.

        :param in_size: Number of features to normalize.
        :param update_limit: Maximum number of update steps allowed.
        :param n_breakpoints: Number of points for the interpolation grid.
        :param max_centroids: Maximum number of centroids for the T-Digest.
        :param use_running_average: If True, skips updates and uses current state (inference mode).
        """
        self.in_size = in_size
        self.update_limit = update_limit
        self.n_breakpoints = n_breakpoints
        self.max_centroids = max_centroids
        self.use_running_average = use_running_average

        self.updates = nnx.Variable(jnp.array([0], dtype=jnp.int32))

        self.max_centroids_var = nnx.Variable(jnp.array([self.max_centroids] * self.in_size, dtype=jnp.int32))
        self.min_var = nnx.Variable(jnp.array([jnp.nan] * self.in_size, dtype=jnp.float32))
        self.max_var = nnx.Variable(jnp.array([jnp.nan] * self.in_size, dtype=jnp.float32))
        self.centroids_m_var = nnx.Variable(jnp.zeros([self.max_centroids, self.in_size], dtype=jnp.float32))
        self.centroids_c_var = nnx.Variable(jnp.zeros([self.max_centroids, self.in_size], dtype=jnp.float32))
        self.fp_var = nnx.Variable(jnp.linspace(-1, 1, self.n_breakpoints)[:, None] + jnp.zeros([1, self.in_size]))
        self.xp_var = nnx.Variable(jnp.linspace(-1, 1, self.n_breakpoints)[:, None] + jnp.zeros([1, self.in_size]))

    def __call__(self, array: jax.Array, non_fictitious: jax.Array) -> jax.Array:
        """
        Normalizes the input array using the current T-Digest state.

        Asynchronous update: Normalization uses the state from the PREVIOUS call,
        while the current batch is used to update the state for the NEXT call.
        """
        is_training = not self.use_running_average
        should_update = is_training & (self.updates[...] < self.update_limit)[0]

        # 1. Use CURRENT xp/fp for normalization (potentially from previous step)
        xp = self.xp_var[...]
        fp = self.fp_var[...]

        if is_training:
            # 2. Trigger ASYNCHRONOUS update with current data
            module_state = (
                self.max_centroids_var[...],
                self.min_var[...],
                self.max_var[...],
                self.centroids_m_var[...],
                self.centroids_c_var[...],
                self.fp_var[...],
                self.xp_var[...],
            )

            # We split update and quantile extraction
            new_state = jax.lax.cond(
                should_update,
                lambda a, m: _tdigest_update(a, m, module_state, self.in_size, self.max_centroids),
                lambda a, m: module_state[:5],
                array,
                non_fictitious,
            )

            # Extract new quantiles from new state (pure callback)
            # This can happen in parallel with the forward pass calculation below
            p_grid = jnp.linspace(0, 1, self.n_breakpoints)
            new_fp, new_xp = _tdigest_get_quantiles(new_state, p_grid)

            # 3. Update state variables for NEXT call
            self.updates[...] = jnp.where(should_update, self.updates[...] + 1, self.updates[...])
            self.max_centroids_var[...] = jax.lax.stop_gradient(new_state[0])
            self.min_var[...] = jax.lax.stop_gradient(new_state[1])
            self.max_var[...] = jax.lax.stop_gradient(new_state[2])
            self.centroids_m_var[...] = jax.lax.stop_gradient(new_state[3])
            self.centroids_c_var[...] = jax.lax.stop_gradient(new_state[4])
            self.fp_var[...] = jax.lax.stop_gradient(new_fp)
            self.xp_var[...] = jax.lax.stop_gradient(new_xp)

        # 4. Perform normalization with "old" xp/fp
        def forward_local(x_feat, xp_feat, fp_feat):
            EPS = 1e-6
            interp_term = jnp.interp(x_feat, xp_feat, fp_feat)
            left_term = (
                jnp.minimum(x_feat - xp_feat[0], 0.0) * (fp_feat[1] - fp_feat[0] + EPS) / (xp_feat[1] - xp_feat[0] + EPS)
            )
            right_term = (
                jnp.maximum(x_feat - xp_feat[-1], 0.0) * (fp_feat[-1] - fp_feat[-2] + EPS) / (xp_feat[-1] - xp_feat[-2] + EPS)
            )
            return interp_term + left_term + right_term

        if array.ndim == 3:
            out = jax.vmap(
                lambda a: jax.vmap(forward_local, in_axes=(1, 1, 1), out_axes=1)(a, xp, fp),
                in_axes=0,
                out_axes=0,
            )(array)
        else:
            out = jax.vmap(forward_local, in_axes=(1, 1, 1), out_axes=1)(array, xp, fp)

        out = out * non_fictitious
        return out


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
    ):
        """
        Initializes the TDigestNormalizer.

        :param in_structure: Structure of the input graph.
        :param update_limit: Maximum number of updates allowed for the T-Digests.
        :param n_breakpoints: Number of breakpoints for the interpolation grid.
        :param max_centroids: Maximum number of centroids for each T-Digest.
        :param use_running_average: Initial state for the running average flag.
        """
        self.in_structure = in_structure
        self.update_limit = update_limit
        self.n_breakpoints = n_breakpoints
        self.max_centroids = max_centroids
        self.use_running_average = use_running_average

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
        # module_dict is wrapped in nnx.data
        for module in self.module_dict.values():
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
            if k in self.module_dict.keys()
        }

        def apply_norm(edge_norm: tuple[HyperEdgeSet, TDigestModule]) -> HyperEdgeSet:
            hyper_edge_set, normalizer = edge_norm
            array = hyper_edge_set.feature_array
            if hyper_edge_set.feature_array is not None:
                if hyper_edge_set.feature_array.shape[-2] > 0:
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

        normalized_context = Graph(
            backend=graph._backend,
            hyper_edge_sets=normalized_hyper_edge_sets,
            non_fictitious_addresses=graph.non_fictitious_addresses,
            true_shape=graph.true_shape,
            current_shape=graph.current_shape,
        )

        if get_info:
            info = {"input_graph": graph.quantiles(), "output_graph": normalized_context.quantiles()}
        else:
            info = {}

        return normalized_context, info
