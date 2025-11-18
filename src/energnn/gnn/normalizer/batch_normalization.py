import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import BatchNorm
from energnn.graph.jax import JaxEdge, JaxGraph

class GraphBatchNorm(nnx.Module):
    """
    Graph-level wrapper that maintains a BatchNorm for each edge key.

    This module dynamically instantiates a `BatchNorm` per
    graph edge type encountered in the provided `JaxGraph`.
    """
    def __init__(self, use_running_average: bool = False, axis: int = -1, momentum: float = 0.99, epsilon: float = 1e-5,
                 use_bias: bool = True, use_scale: bool = True, use_fast_variance: bool = True):

        """
        Initialize GraphBatchNorm.

        :param use_running_average: if True, the stored batch statistics will be used instead
            of computing the batch statistics on the input.
        :param axis: the feature or non-batch axis of the input.
        :param momentum: decay rate for the exponential moving average of the batch statistics.
        :param epsilon: a small float added to variance to avoid dividing by zero.
        :param use_bias:  if True, bias (beta) is added.
        :param use_scale: if True, multiply by scale (gamma).
        :param use_fast_variance: If true, use a faster, but less numerically stable,
            calculation for the variance.
        """
        self.use_running_average = use_running_average
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.use_fast_variance = use_fast_variance
        self.edge_keys = tuple()

    def _make_module_for_edge(self, edge_key: str, n_features: int):
        """
        Create and attach a BatchNorm submodule for the given edge key.

        This helper attaches the created submodule as an attribute named
        `norm_{edge_key}` so that Flax's module discovery will include it in
        the variables tree and checkpoints.

        :param edge_key: String key identifying the graph edge type.
        :param n_features: Number of features (columns) for this edge.
        :return: The created BatchNorm instance.
        """
        mod = BatchNorm(
            num_features = n_features, axis=self.axis, momentum= self.momentum, epsilon=self.epsilon, use_bias=self.use_bias,
            digest_base_key=self.use_scale, use_fast_variance=self.use_fast_variance, use_running_average = self.use_running_average
        )
        setattr(self, f"norm_{edge_key}", mod)
        return mod

    def initialize_from_example(self, context_example: JaxGraph):
        """
        Initialize and attach per-edge normalizers based on a JaxGraph example.

        This method inspects `context_example.edges` and for every edge with a non-empty
        feature array instantiates a corresponding `BatchNorm` with
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
                        self._make_module_for_edge(edge_key, n_features)
                    keys.append(edge_key)

        self.edge_keys = tuple(keys)

    def __call__(self, *, context: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Apply per-edge normalization to a `JaxGraph` instance.

        For every edge found in `context.edges`, this method will locate the
        corresponding `BatchNorm` submodule (created previously)
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
            Apply the BatchNorm to a single JaxEdge.

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
                    feature_array=edge.feature_array, feature_names=feature_names, non_fictitious=edge.non_fictitious, address_dict=edge.address_dict
                )

            return edge

        incoming_keys = list(context.edges.keys())
        norm_dict = {}
        for edge_key in incoming_keys:
            attr_name = f"norm_{edge_key}"
            if context.edges[edge_key].feature_array is not None:
                if context.edges[edge_key].feature_array.shape[-2] > 0:
                    if not hasattr(self, attr_name):
                        n_features = int(context.edges[edge_key].feature_array.shape[-1])
                        self._make_module_for_edge(edge_key, n_features)
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
