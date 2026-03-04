# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from fastdigest import TDigest
from flax import nnx
from jax import ShapeDtypeStruct
from jax.experimental import io_callback

from energnn.graph import GraphStructure
from energnn.graph.jax import JaxEdge, JaxGraph
from .normalizer import Normalizer


def _merge_equal_quantiles_host(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolves equal-quantile conflicts by averaging probabilities for identical quantile values.

    When some adjacent quantiles in `q` are equal (zero slope), this function
    computes a merged probability vector per feature so the piecewise-linear CDF
    remains bijective (strictly monotone in the probability coordinate). The
    algorithm groups identical quantile values per column and averages the
    corresponding probabilities.

    :param p: Probability grid as a 2-D array of shape (K, F) (values in [0,1]).
    :param q: Quantiles matrix of shape (K, F).
        Each column q[:, f] contains quantiles for feature f.
    :return: Tuple (p_merged, q_merged) where both have shape (K, F) and p_merged
        contains the merged/averaged probabilities per unique quantile value
        per feature, and q_merged is equal to `q` (cast to float32).
    """
    K, F = q.shape
    p_out = np.zeros((K, F), dtype=np.float32)
    q_out = q.astype(np.float32)
    for f in range(F):
        qf = q_out[:, f]
        pf = p[:, f]
        vals, inv, counts = np.unique(qf, return_inverse=True, return_counts=True)
        sum_p_per_unique = np.zeros_like(vals, dtype=np.float64)
        np.add.at(sum_p_per_unique, inv, pf)
        avg_p_per_unique = sum_p_per_unique / counts
        p_out[:, f] = avg_p_per_unique[inv].astype(np.float32)
    return p_out, q_out


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
    """
    Host-side callback to update T-Digest statistics using the fastdigest library.

    This function processes a batch of data, updates the underlying T-Digest structures,
    and extracts new interpolation points (xp, fp) for the normalization.

    :param max_centroids: Maximum number of centroids allowed for each feature.
    :param min_val: Current minimum value for each feature.
    :param max_val: Current maximum value for each feature.
    :param centroids_m: Centroid means for each feature.
    :param centroids_c: Centroid counts for each feature.
    :param fp: Interpolation probabilities (mapped to [-1, 1]).
    :param xp: Interpolation quantiles.
    :param array: New data batch of shape (N, F) or (B, N, F).
    :param mask: Mask for valid data.
    :return: Updated state variables for the TDigest modules.
    """
    # Variables are individual numpy arrays
    # array has shape (N, F) or (B, N, F)
    # mask has shape (N, 1) or (B, N, 1)

    if array.ndim == 3:
        # Batched: (B, N, F)
        B, N, F = array.shape
        array = array.reshape(B * N, F)
        mask = mask.reshape(B * N, 1)
    else:
        N, F = array.shape

    # Apply mask
    mask = mask.flatten().astype(bool)
    array = array[mask]

    n_features = array.shape[-1]
    K = fp.shape[0]

    new_max_centroids = []
    new_min_list = []
    new_max_list = []
    new_c_c_list = []
    new_c_m_list = []
    new_p_matrix = []
    new_q_matrix = []

    for i in range(n_features):
        feature_array = array[:, i]
        _max_centroids = int(max_centroids[i])
        _min = float(min_val[i])
        _max = float(max_val[i])
        _c_m = centroids_m[:, i]
        _c_c = centroids_c[:, i]

        tdigest_dict = {
            "max_centroids": _max_centroids,
            "min": _min,
            "max": _max,
            "centroids": [{"m": float(m), "c": float(c)} for m, c in zip(_c_m, _c_c) if c > 0.0],
        }

        # Handle NaNs from initialization
        if np.isnan(tdigest_dict["min"]):
            tdigest_dict["min"] = 0.0
        if np.isnan(tdigest_dict["max"]):
            tdigest_dict["max"] = 0.0

        tdigest = TDigest.from_dict(tdigest_dict)
        if len(feature_array) > 0:
            tdigest.batch_update(np.array(feature_array))

        new_tdigest_dict = tdigest.to_dict()

        new_max_centroids.append(new_tdigest_dict["max_centroids"])
        new_min_list.append(new_tdigest_dict["min"])
        new_max_list.append(new_tdigest_dict["max"])

        c_c = np.array([centroid["c"] for centroid in new_tdigest_dict["centroids"]])
        c_c = np.pad(c_c, (0, _max_centroids - len(c_c)), mode="constant", constant_values=0.0)
        new_c_c_list.append(c_c)
        c_m = np.array([centroid["m"] for centroid in new_tdigest_dict["centroids"]])
        c_m = np.pad(c_m, (0, _max_centroids - len(c_m)), mode="constant", constant_values=0.0)
        new_c_m_list.append(c_m)

        p_list = np.linspace(0, 1, K)
        q_list = [tdigest.quantile(float(p)) for p in p_list]
        new_p_matrix.append(p_list)
        new_q_matrix.append(np.asarray(q_list))

    new_p_matrix = np.stack(new_p_matrix, axis=0)  # (F, K)
    new_q_matrix = np.stack(new_q_matrix, axis=0)  # (F, K)

    p_merged, q_merged = _merge_equal_quantiles_host(new_p_matrix.T, new_q_matrix.T)
    new_xp = q_merged.astype(np.float32)
    new_fp = (-1.0 + 2.0 * p_merged).astype(np.float32)

    return (
        np.array(new_max_centroids, dtype=np.int32),
        np.array(new_min_list, dtype=np.float32),
        np.array(new_max_list, dtype=np.float32),
        np.stack(new_c_m_list, axis=1).astype(np.float32),
        np.stack(new_c_c_list, axis=1).astype(np.float32),
        new_fp,
        new_xp,
    )


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _tdigest_apply(
    array: jax.Array,
    non_fictitious: jax.Array,
    should_update: jax.Array,
    module_state: tuple[jax.Array, ...],
    in_size: int,
    max_centroids: int,
    n_breakpoints: int,
) -> tuple[jax.Array, ...]:
    """
    Applies normalization and optionally updates T-Digest statistics.
    This function is wrapped in custom_vjp to handle the non-differentiable io_callback.
    """
    (
        max_centroids_val,
        min_val,
        max_val,
        centroids_m,
        centroids_c,
        fp,
        xp,
    ) = module_state

    def update_fn(array: jax.Array, non_fictitious: jax.Array) -> tuple[jax.Array, ...]:
        result_shapes = (
            ShapeDtypeStruct((in_size,), jnp.int32),  # max_centroids
            ShapeDtypeStruct((in_size,), jnp.float32),  # min
            ShapeDtypeStruct((in_size,), jnp.float32),  # max
            ShapeDtypeStruct((max_centroids, in_size), jnp.float32),  # centroids_m
            ShapeDtypeStruct((max_centroids, in_size), jnp.float32),  # centroids_c
            ShapeDtypeStruct((n_breakpoints, in_size), jnp.float32),  # fp
            ShapeDtypeStruct((n_breakpoints, in_size), jnp.float32),  # xp
        )

        return io_callback(
            _ingest_new_data,
            result_shapes,
            max_centroids_val,
            min_val,
            max_val,
            centroids_m,
            centroids_c,
            fp,
            xp,
            array,
            non_fictitious,
        )

    new_vars = jax.lax.cond(
        should_update,
        lambda a, m: update_fn(a, m),
        lambda a, m: (max_centroids_val, min_val, max_val, centroids_m, centroids_c, fp, xp),
        array,
        non_fictitious,
    )
    return new_vars


def _tdigest_apply_fwd(
    array: jax.Array,
    non_fictitious: jax.Array,
    should_update: jax.Array,
    module_state: tuple[jax.Array, ...],
    in_size: int,
    max_centroids: int,
    n_breakpoints: int,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    new_vars = _tdigest_apply(array, non_fictitious, should_update, module_state, in_size, max_centroids, n_breakpoints)
    return (new_vars), (array, non_fictitious, new_vars[6], new_vars[5])


def _tdigest_apply_bwd(
    in_size: int, max_centroids: int, n_breakpoints: int, res: tuple[jax.Array, ...], grads: Any
) -> tuple[jax.Array, ...]:
    array, non_fictitious, xp, fp = res
    return 0 * array, None, None, None


_tdigest_apply.defvjp(_tdigest_apply_fwd, _tdigest_apply_bwd)


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

        If in training mode and under the update limit, it also triggers a state update.

        :param array: Input array of shape (..., in_size).
        :param non_fictitious: Mask for valid (non-fictitious) items.
        :return: Normalized array of the same shape as input.
        """
        is_training = not self.use_running_average
        should_update = is_training & (self.updates[...] < self.update_limit)[0]

        if is_training:
            module_state = (
                self.max_centroids_var[...],
                self.min_var[...],
                self.max_var[...],
                self.centroids_m_var[...],
                self.centroids_c_var[...],
                self.fp_var[...],
                self.xp_var[...],
            )

            new_vars = _tdigest_apply(
                array,
                non_fictitious,
                should_update,
                module_state,
                self.in_size,
                self.max_centroids,
                self.n_breakpoints,
            )

            # Update state variables (side effects)
            self.updates[...] = jnp.where(should_update, self.updates[...] + 1, self.updates[...])
            self.max_centroids_var[...] = jax.lax.stop_gradient(new_vars[0])
            self.min_var[...] = jax.lax.stop_gradient(new_vars[1])
            self.max_var[...] = jax.lax.stop_gradient(new_vars[2])
            self.centroids_m_var[...] = jax.lax.stop_gradient(new_vars[3])
            self.centroids_c_var[...] = jax.lax.stop_gradient(new_vars[4])
            self.fp_var[...] = jax.lax.stop_gradient(new_vars[5])
            self.xp_var[...] = jax.lax.stop_gradient(new_vars[6])

        xp = self.xp_var[...]
        fp = self.fp_var[...]

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
    Graph-level normalizer that maintains a TDigestModule for each edge type.

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
        """Creates a TDigest module for each edge key in the graph structure."""
        module_dict = {}
        for edge_key, edge_structure in self.in_structure.edges.items():
            if edge_structure.feature_list is not None:  # and len(edge_structure.feature_list) > 0:
                in_size = len(edge_structure.feature_list)
                module_dict[edge_key] = TDigestModule(
                    in_size=in_size,
                    update_limit=self.update_limit,
                    n_breakpoints=self.n_breakpoints,
                    max_centroids=self.max_centroids,
                    use_running_average=self.use_running_average,
                )
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

    def __call__(self, *, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Apply normalization to edges within a JaxGraph context using TDigest modules. This method normalizes the
        edges' feature arrays and updates the associated context graph accordingly.

        :param graph: JaxGraph representing the graph structure containing edges with feature arrays to be normalized.
        :param get_info: Boolean flag that indicates whether to return additional information about input and output graphs.
        :return: A tuple containing the normalized JaxGraph and an optional dictionary holding quantile information
                 about the input and output graphs.
        """

        edge_norm_dict = {k: (edge, self.module_dict[k]) for k, edge in graph.edges.items() if k in self.module_dict.keys()}

        def apply_norm(edge_norm: tuple[JaxEdge, TDigestModule]) -> JaxEdge:
            edge, normalizer = edge_norm
            array = edge.feature_array
            if edge.feature_array is not None:
                if edge.feature_array.shape[-2] > 0:
                    array = normalizer(array, jnp.expand_dims(edge.non_fictitious, -1))
            return JaxEdge(
                feature_array=array,
                feature_names=edge.feature_names,
                non_fictitious=edge.non_fictitious,
                address_dict=edge.address_dict,
            )

        normalized_edge_dict = jax.tree.map(apply_norm, edge_norm_dict, is_leaf=(lambda x: isinstance(x, tuple)))

        normalized_context = JaxGraph(
            edges=normalized_edge_dict,
            non_fictitious_addresses=graph.non_fictitious_addresses,
            true_shape=graph.true_shape,
            current_shape=graph.current_shape,
        )

        if get_info:
            info = {"input_graph": graph.quantiles(), "output_graph": normalized_context.quantiles()}
        else:
            info = {}

        return normalized_context, info
