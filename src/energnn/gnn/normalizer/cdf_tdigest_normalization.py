import threading
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import io_callback
from jax import ShapeDtypeStruct
from fastdigest import TDigest
from energnn.graph.jax import JaxEdge, JaxGraph

# Public registries used by host callbacks to store TDigest instances and per-digest locks.
GLOBAL_DIGEST_REGISTRY: dict[int, TDigest] = {}
GLOBAL_DIGEST_LOCKS: dict[int, threading.Lock] = {}
# Global lock protecting creation in the registries.
_GLOBAL_REGISTRY_LOCK = threading.Lock()


def _ensure_digest(key: int, max_centroids: int):
    """
    Ensure a TDigest instance and its per-digest lock exist for the given `key`.

    This function is thread-safe: it acquires a global registry lock for the
    creation step to avoid a race where multiple threads create/overwrite the
    same registry entry concurrently.

    :param key: Integer key identifying the TDigest instance in the global registry.
    :param max_centroids: Maximum number of centroids to allocate when constructing
        a new TDigest.
    :return: None
    """
    with _GLOBAL_REGISTRY_LOCK:
        if key not in GLOBAL_DIGEST_REGISTRY:
            GLOBAL_DIGEST_REGISTRY[key] = TDigest(max_centroids=max_centroids)
            GLOBAL_DIGEST_LOCKS[key] = threading.Lock()


def _merge_equal_quantiles_host(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve equal-quantile conflicts by averaging probabilities for identical quantile values.

    When some adjacent quantiles in `q` are equal (zero slope), this function
    computes a merged probability vector per feature so the piecewise-linear CDF
    remains bijective (strictly monotone in the probability coordinate). The
    algorithm groups identical quantile values per column and averages the
    corresponding probabilities.

    :param p: Probability grid as a 1-D array of length K (values in [0,1]).
    :param q: Quantiles matrix of shape (K, F), where K = len(p) and F is the
        number of features. Each column q[:, f] contains quantiles for feature f.
    :return: Tuple (p_merged, q_merged) where both have shape (K, F) and p_merged
        contains the merged/averaged probabilities per unique quantile value
        per feature, and q_merged is equal to `q` (cast to float32).
    """
    K, F = q.shape
    p_vec = np.asarray(p).reshape(K)
    p_out = np.zeros((K, F), dtype=np.float32)
    q_out = q.astype(np.float32)
    for f in range(F):
        qf = q_out[:, f]
        vals, inv, counts = np.unique(qf, return_inverse=True, return_counts=True)
        sum_p_per_unique = np.zeros_like(vals, dtype=np.float64)
        np.add.at(sum_p_per_unique, inv, p_vec)
        avg_p_per_unique = sum_p_per_unique / counts
        p_out[:, f] = avg_p_per_unique[inv].astype(np.float32)
    return p_out, q_out


def _host_update_and_extract_multi(
    batch_np: np.ndarray, digest_keys_np: np.ndarray, p_grid_np: np.ndarray, max_centroids_np: np.ndarray
):
    """
    Host-side callback: update TDigests with provided samples and extract quantiles.

    This function is intended to be called via `jax.experimental.io_callback`. It:
      - ensures a TDigest exists for each feature key,
      - updates each TDigest with the samples for that feature (column),
      - extracts quantiles at the requested probability grid,
      - merges equal quantiles (resolve zero-slope segments) per feature,
      - returns xp and fp arrays suitable to be stored as device-side BatchStat fields.

    The expected shapes and types:
      - `batch_np`: 2-D array of shape (N, F) containing samples (N = B * n_items, F features).
      - `digest_keys_np`: 1-D integer array of shape (F,) giving the registry key per feature.
      - `p_grid_np`: 1-D float array of shape (K,) giving the probability grid in [0,1].
      - `max_centroids_np`: scalar int specifying max centroids for creation (if needed).

    The returned values are `(xp, fp)` where:
      - `xp` has shape (K, F) and contains quantile values for each p in the grid,
      - `fp` has shape (K, F) and contains mapped probabilities scaled to [-1, 1]
        (calculated as `-1 + 2 * p_merged`).

    :param batch_np: Numpy array shape (N, F) with samples flattened across batch & items.
    :param digest_keys_np: Numpy array shape (F,) of integer digest registry keys.
    :param p_grid_np: Numpy array shape (K,) of probability points in ascending order.
    :param max_centroids_np: Scalar (or length-1 array) specifying maximum centroids.
    :return: Tuple (xp, fp) both numpy arrays with shape (K, F) and dtype float32.
    """
    batch = np.asarray(batch_np)
    digest_keys = np.asarray(digest_keys_np).astype(int)
    p_grid = np.asarray(p_grid_np).astype(np.float32)
    max_centroids = int(np.asarray(max_centroids_np).item())
    K = p_grid.shape[0]
    F = batch.shape[1]
    q_matrix = np.zeros((K, F), dtype=np.float32)

    # Ensure all digests exist
    for k in digest_keys.tolist():
        _ensure_digest(int(k), max_centroids)

    # Update & extract per-feature using the per-feature digest (host-side)
    for f in range(F):
        key = int(digest_keys[f])
        d = GLOBAL_DIGEST_REGISTRY[key]
        lock = GLOBAL_DIGEST_LOCKS[key]
        col = batch[:, f].reshape(-1)
        with lock:
            d.batch_update(col)
            q_list = [d.quantile(float(p)) for p in p_grid]
            q_matrix[:, f] = np.asarray(q_list).reshape(K)

    # merge equal quantiles column-wise and compute fp
    p_merged, q_merged = _merge_equal_quantiles_host(p_grid, q_matrix)
    xp = q_merged.astype(np.float32)
    fp = (-1.0 + 2.0 * p_merged).astype(np.float32)
    return xp, fp


class MultiFeatureTDigestNorm(nnx.Module):
    """
    Module that maintains one TDigest per feature and provides a piecewise-linear
    mapping of values to the interval [-1, 1] based on estimated quantiles.

    The module:
      - keeps host-side TDigest instances per feature (stored in a global registry),
      - updates these digests via `io_callback` during the forward pass (batched),
      - stores a device-side summary (xp/fp, centroids, mins/maxs) in BatchStat fields
        so that the state can be checkpointed and reconstructed later,
      - applies a piecewise-linear forward mapping using xp/fp.

    Note:
      - On the first calls the TDigest objects will be created and updated.
    """

    def __init__(self, features: int, update_limit: int, n_breakpoints: int = 20, digest_base_key: int = 1000,
                 max_centroids: int = 1000, use_running_average: bool = False):
        """
        Initialize the MultiFeatureTDigestNorm module.

        :param features: Number of features (F, number of columns).
        :param update_limit: Number of forward calls that will cause digest updates. After
            `update_limit` updates, if `use_running_average` is True, the module will stop
            updating TDigest and reuse running xp/fp values.
        :param n_breakpoints: Number of evenly spaced quantile breakpoints minus one
            (so K = n_breakpoints + 1 quantile points in total).
        :param digest_base_key: Base integer key used to compute per-feature digest keys:
            digest_key[f] = digest_base_key + f.
        :param max_centroids: Maximum centroids used when constructing TDigest instances.
        :param use_running_average: If True, stop updating host digests and reuse the last
            xp/fp. Automatically set to True in `eval` mode and to `False` in `train` mode.
        """
        self.features = int(features)
        self.n_breakpoints = int(n_breakpoints)
        self.K = self.n_breakpoints + 1
        self.max_centroids = int(max_centroids)
        self.digest_keys = np.arange(digest_base_key, digest_base_key + self.features, dtype=np.int32)
        self.update_limit = int(update_limit)
        self.updates = 0
        self.use_running_average = use_running_average

        # xp/fp BatchStat device-side shape (K, F)
        self.xp = nnx.BatchStat(jnp.zeros((self.K, self.features), dtype=jnp.float32))
        p_grid = np.linspace(0.0, 1.0, self.K, dtype=np.float32)
        fp_init = (-1.0 + 2.0 * p_grid)[:, None] * np.ones((1, self.features), dtype=np.float32)
        self.fp = nnx.BatchStat(jnp.array(fp_init, dtype=jnp.float32))

        # TDigest state stored as BatchStat for checkpointing
        # scalars per-feature: min and max
        self.digest_min = nnx.BatchStat(jnp.zeros((self.features,), dtype=jnp.float32))
        self.digest_max = nnx.BatchStat(jnp.zeros((self.features,), dtype=jnp.float32))
        # number of centroids currently stored per feature
        self.digest_n_centroids = nnx.BatchStat(jnp.zeros((self.features,), dtype=jnp.int32))
        # centroids arrays fixed-size: shape (F, max_centroids)
        self.centroids_m = nnx.BatchStat(jnp.zeros((self.features, self.max_centroids), dtype=jnp.float32))
        self.centroids_c = nnx.BatchStat(jnp.zeros((self.features, self.max_centroids), dtype=jnp.float32))
        # store max_centroids
        self.digest_max_centroids = nnx.BatchStat(jnp.array(self.max_centroids, dtype=jnp.int32))

        # register all digests in registry
        for k in self.digest_keys.tolist():
            _ensure_digest(int(k), self.max_centroids)

    def _sync_state_from_registry(self):
        """
        Synchronize device-side BatchStat fields from host-side TDigest registry.

        This method should be called after host digests have been updated (for example after an `io_callback`).
        """
        F = self.features
        M = self.max_centroids

        mins = np.zeros((F,), dtype=np.float32)
        maxs = np.zeros((F,), dtype=np.float32)
        n_centroids = np.zeros((F,), dtype=np.int32)
        m_arr = np.zeros((F, M), dtype=np.float32)
        c_arr = np.zeros((F, M), dtype=np.float32)

        for f, key in enumerate(self.digest_keys.tolist()):
            key = int(key)
            if key not in GLOBAL_DIGEST_REGISTRY:
                # keep zeros if digest missing
                continue
            d = GLOBAL_DIGEST_REGISTRY[key]
            td = d.to_dict()

            mins[f] = float(td.get("min", 0.0))
            maxs[f] = float(td.get("max", 0.0))
            centroids = td.get("centroids", []) or []
            n = min(len(centroids), M)
            n_centroids[f] = int(n)
            for i in range(n):
                cent_i = centroids[i]
                m_arr[f, i] = float(cent_i.get("m", 0.0))
                c_arr[f, i] = float(cent_i.get("c", 0.0))

        # assign to BatchStat values (convert to jnp arrays)
        self.digest_min.value = jnp.array(mins, dtype=jnp.float32)
        self.digest_max.value = jnp.array(maxs, dtype=jnp.float32)
        self.digest_n_centroids.value = jnp.array(n_centroids, dtype=jnp.int32)
        self.centroids_m.value = jnp.array(m_arr, dtype=jnp.float32)
        self.centroids_c.value = jnp.array(c_arr, dtype=jnp.float32)
        self.digest_max_centroids.value = jnp.array(self.max_centroids, dtype=jnp.int32)

    def _reconstruct_digests_from_state(self):
        """
        Recreate host-side TDigest objects from checkpointed BatchStat fields.

        This method reconstructs a TDigest-like dictionary per feature and uses
        the TDigest's `from_dict` to restore host digest state. Call this on the
        host after loading variables from a checkpoint to ensure `GLOBAL_DIGEST_REGISTRY`
        contains TDigest instances consistent with the saved BatchStat values.
        """

        mins = np.asarray(self.digest_min.value).astype(np.float32)
        maxs = np.asarray(self.digest_max.value).astype(np.float32)
        n_centroids = np.asarray(self.digest_n_centroids.value).astype(np.int32)
        m_arr = np.asarray(self.centroids_m.value).astype(np.float32)
        c_arr = np.asarray(self.centroids_c.value).astype(np.float32)
        max_centroids_saved = int(np.asarray(self.digest_max_centroids.value).item())

        for f, key in enumerate(self.digest_keys.tolist()):
            key = int(key)
            td_dict = {
                "max_centroids": max_centroids_saved,
                "min": float(mins[f]) if mins is not None else 0.0,
                "max": float(maxs[f]) if maxs is not None else 0.0,
                "centroids": [{"m": float(m_arr[f, i]), "c": float(c_arr[f, i])} for i in range(int(n_centroids[f]))],
            }

            _ensure_digest(key, max_centroids_saved)
            GLOBAL_DIGEST_REGISTRY[key].from_dict(td_dict)

    def __call__(self, x: jax.Array):
        """
        Apply normalization to input values and optionally update host TDigest state.

        The input `x` can be either:
          - single array with shape (n_items, F), or
          - batched array with shape (B, n_items, F).

        On each forward call (while `updates < update_limit` or when `use_running_average`
        is False), the module will:
          - update host TDigest objects and extract xp/fp arrays,
          - store xp/fp to device BatchStat fields and synchronize digest centroids into
            BatchStat for checkpointing.

        After obtaining xp/fp, the module applies the piecewise-linear mapping to
        all features and returns an array of the same shape as input with values
        mapped to approximately [-1, 1].

        :param x: JAX array with shape (n_items, F) or (B, n_items, F).
        :return: JAX array with same shape as `x` containing normalized values.
        """
        # detect batch dim
        if x.ndim == 2:
            # single-graph: make into batch of size 1 for uniform handling
            is_batched = False
            x_batch = x[None, ...]  # shape (1, n_items, F)
        elif x.ndim == 3:
            is_batched = True
            x_batch = x  # shape (B, n_items, F)
        else:
            raise ValueError("Input x must be shape (n_items,F) or (B,n_items,F)")

        F = x_batch.shape[2]
        assert F == self.features

        if (self.updates < self.update_limit) or (not self.use_running_average):
            self.updates += 1

            B = x_batch.shape[0]
            n_items = x_batch.shape[1]

            flattened = jnp.reshape(x_batch, (B * n_items, F))

            p_grid = jnp.linspace(0.0, 1.0, self.K, dtype=jnp.float32)
            result_shapes = (
                ShapeDtypeStruct((self.K, F), jnp.float32),
                ShapeDtypeStruct((self.K, F), jnp.float32),
            )

            xp_jax, fp_jax = io_callback(
                _host_update_and_extract_multi,
                result_shapes,
                flattened,  # (N, F) on host
                jnp.array(self.digest_keys, dtype=jnp.int32),
                p_grid,
                jnp.array(self.max_centroids, dtype=jnp.int32),
                ordered=True,
            )

            self.xp.value = xp_jax
            self.fp.value = fp_jax

            # Sync host digest states into BatchStat so checkpoints include centroids
            self._sync_state_from_registry()

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

        def per_graph_fn(graph_x):
            return jax.vmap(forward_local, in_axes=(1, 1, 1), out_axes=1)(graph_x, self.xp.value, self.fp.value)

        out_batch = jax.vmap(per_graph_fn)(x_batch)  # shape (B, n_items, F)

        if not is_batched:
            return out_batch[0]
        return out_batch


class GraphTDigestNorm(nnx.Module):
    """
    Graph-level wrapper that maintains a MultiFeatureTDigestNorm for each edge key.

    This module dynamically instantiates, on host, a `MultiFeatureTDigestNorm` per
    graph edge type encountered in the provided `JaxGraph`.
    """
    def __init__(self, update_limit: int, n_breakpoints: int = 20, digest_base_key: int = 1000, max_centroids: int = 1000,
                 max_per_edge_features: int = 10000, use_running_average: bool = False
                 ):
        """
        Initialize GraphTDigestNorm.

        :param update_limit: Number of host-update steps allowed per MultiFeatureTDigestNorm.
        :param n_breakpoints: Number of breakpoints for quantile estimation.
        :param digest_base_key: Base integer used to compute per-edge digest base keys.
        :param max_centroids: Maximum centroids for each TDigest.
        :param max_per_edge_features: Maximum number of features for an edge.
        :param use_running_average: If True, stop updating host digests and reuse the last xp/fp.
            Automatically set to True in `eval` mode and to `False` in `train` mode.
        """
        # store general hyperparams
        self.n_breakpoints = int(n_breakpoints)
        self.digest_base_key = int(digest_base_key)
        self.max_centroids = int(max_centroids)
        self.edge_keys = tuple()
        self.update_limit = update_limit
        self.use_running_average = use_running_average
        self.max_per_edge_features = max(max_per_edge_features, 10000)

    def _make_module_for_edge(self, edge_key: str, n_features: int, edge_index: int):
        """
        Create and attach a MultiFeatureTDigestNorm submodule for the given edge key.

        This helper attaches the created submodule as an attribute named
        `norm_{edge_key}` so that Flax's module discovery will include it in
        the variables tree and checkpoints.

        :param edge_key: String key identifying the graph edge type.
        :param n_features: Number of features (columns) for this edge.
        :param edge_index: Integer index used to deterministically compute the digest base key.
        :return: The created MultiFeatureTDigestNorm instance.
        """
        base = self.digest_base_key + edge_index * self.max_per_edge_features
        mod = MultiFeatureTDigestNorm(
            update_limit=self.update_limit, features=n_features, n_breakpoints=self.n_breakpoints, digest_base_key=base,
            max_centroids=self.max_centroids, use_running_average=self.use_running_average
        )
        setattr(self, f"norm_{edge_key}", mod)
        return mod

    def initialize_from_example(self, context_example: JaxGraph):
        """
        Initialize and attach per-edge normalizers based on a JaxGraph example.

        This method inspects `context_example.edges` and for every edge with a non-empty
        feature array instantiates a corresponding `MultiFeatureTDigestNorm` with
        the appropriate number of features.

        :param context_example: Example `JaxGraph` used to infer edge keys and feature counts.
        :return: None
        """
        keys = []
        for i, (edge_key, edge) in enumerate(context_example.edges.items()):
            if context_example.edges[edge_key].feature_array is not None:
                if context_example.edges[edge_key].feature_array.shape[-2] > 0:
                    n_features = int(edge.feature_array.shape[-1])
                    if not hasattr(self, f"norm_{edge_key}"):
                        self._make_module_for_edge(edge_key, n_features, edge_index=i)
                    keys.append(edge_key)

        self.edge_keys = tuple(keys)

    def __call__(self, *, context: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Apply per-edge normalization to a `JaxGraph` instance.

        For every edge found in `context.edges`, this method will locate the
        corresponding `MultiFeatureTDigestNorm` submodule (created previously)
        and apply it to the edge's `feature_array`. If a submodule does not exist
        yet and the edge contains features, the submodule is lazily instantiated.

        The function returns a new `JaxGraph` with normalized edge features and
        (optionally) a small info dictionary containing input/output quantiles.

        :param context: `JaxGraph` instance containing edges to normalize.
        :param get_info: If True, return quantile info for input and output graphs.
        :return: Tuple (normalized_context, info_dict). `info_dict` is empty if `get_info` is False.
        """

        info: dict = {}

        def apply_norm(edge, normalizer):
            """
            Apply the MultiFeatureTDigestNorm to a single JaxEdge.

            :param edge: JaxEdge instance.
            :param normalizer: Callable or nnx.Module implementing normalization for the edge.
            :return: JaxEdge with normalized feature_array (or identical JaxEdge if no features).
            """

            non_fictitious = jnp.expand_dims(edge.non_fictitious, -1)
            feature_names = None

            if edge.feature_array is not None:
                feature_names = {f"norm_{k}": v for k, v in edge.feature_names.items()}
                if edge.feature_array.shape[-2] > 0:
                    feature_names = {f"norm_{k}": v for k, v in edge.feature_names.items()}
                    normalized_array = normalizer(edge.feature_array) * non_fictitious
                    edge = JaxEdge(
                        feature_array=normalized_array,
                        feature_names=feature_names,
                        non_fictitious=edge.non_fictitious,
                        address_dict=edge.address_dict,
                    )
                else:
                    edge = JaxEdge(
                        feature_array=edge.feature_array, feature_names=feature_names,
                        non_fictitious=edge.non_fictitious, address_dict=edge.address_dict
                    )
            else:
                edge = JaxEdge(
                    feature_array=edge.feature_array, feature_names=feature_names, non_fictitious=edge.non_fictitious,
                    address_dict=edge.address_dict
                )

            return edge

        incoming_keys = list(context.edges.keys())
        norm_dict = {}
        for edge_key in incoming_keys:
            attr_name = f"norm_{edge_key}"
            if context.edges[edge_key].feature_array is not None:
                if context.edges[edge_key].feature_array.shape[-2] > 0:
                    if not hasattr(self, attr_name):
                        edge_index = len(self.edge_keys)
                        n_features = int(context.edges[edge_key].feature_array.shape[-1])
                        self._make_module_for_edge(edge_key, n_features, edge_index=edge_index)
                        self.edge_keys = tuple(list(self.edge_keys) + [edge_key])

            norm_dict[edge_key] = getattr(self, attr_name, None)

        normalized_edge_dict = jax.tree.map(
            apply_norm,
            context.edges,
            norm_dict,
            is_leaf=(lambda x: isinstance(x, JaxEdge)),
        )

        normalized_context = JaxGraph(
            edges=normalized_edge_dict,
            non_fictitious_addresses=context.non_fictitious_addresses,
            true_shape=context.true_shape,
            current_shape=context.current_shape,
        )

        if get_info:
            info = {"input_graph": context.quantiles(), "output_graph": normalized_context.quantiles()}

        return normalized_context, info
