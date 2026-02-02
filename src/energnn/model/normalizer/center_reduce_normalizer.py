import jax
import jax.numpy as jnp
from flax import nnx

from energnn.graph import JaxGraph
from energnn.graph.jax import JaxEdge
from .normalizer import Normalizer


class CenterReduceNormalizer(Normalizer):
    """Graph-level wrapper that maintains an EdgeCenterReduceNormalizer for each edge key."""

    def __init__(
        self,
        update_limit: int,
        beta_1: float = 0.9,
        beta_2: float = 0.9,
        epsilon: float = 1e-6,
        use_running_average: bool = False,
    ):
        """
        Initializes a GraphCenterReduceNorm instance.

        :param update_limit: Threshold for the maximum updates to be performed.
        :param beta_1: Exponential decay rate for the first moment estimates. Defaults to 0.9.
        :param beta_2: Exponential decay rate for the second moment estimates. Defaults to 0.999.
        :param epsilon: Small constant added to improve numerical stability. Defaults to 1e-6.
        :param use_running_average: Flag indicating whether or not to use a running average. Defaults to False.
            Automatically set to True in `eval` mode and to `False` in `train` mode.
        """

        self.update_limit = update_limit
        self.use_running_average = use_running_average
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.updates = 0
        self.edge_keys = tuple()

    def _make_module_for_edge(self, edge_key: str, n_features: int) -> None:
        """
        Create and attach an EdgeCenterReduceNormalizer submodule for the given edge key.

        :param edge_key: String key identifying the graph edge type.
        :param n_features: Number of features (columns) for this edge.
        """
        mod = EdgeCenterReduceNormalizer(
            n_features,
            update_limit=self.update_limit,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            use_running_average=self.use_running_average,
        )
        setattr(self, f"norm_{edge_key}", mod)

    def initialize_from_example(self, context_example: JaxGraph) -> None:
        """
        Initializes normalization modules for each edge class based on the provided context example.
        Only edges with non-empty feature arrays are considered for module initialization.

        :param context_example: An instance of JaxGraph that provides edge definitions and associated
            feature arrays to initialize normalization modules.
        """
        keys = []
        for edge_key, edge in context_example.edges.items():
            if edge.feature_array is not None:
                if edge.feature_array.shape[-2] > 0:
                    n_features = int(edge.feature_array.shape[-1])
                    if not hasattr(self, f"norm_{edge_key}"):
                        self._make_module_for_edge(edge_key, n_features)
                    keys.append(edge_key)
        self.edge_keys = tuple(keys)

    def __call__(self, *, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Apply normalization to edges within a JaxGraph context using EdgeCenterReduceNormalizer. This method normalizes the
        edges' feature arrays and updates the associated context graph accordingly.

        :param graph: JaxGraph representing the graph structure containing edges with feature arrays to be normalized.
        :param get_info: Boolean flag indicating whether to return additional information about input and output graphs.
        :return: A tuple containing the normalized JaxGraph and an optional dictionary holding quantile information
                 about the input and output graphs.
        """
        info: dict = {}

        def apply_norm(edge: JaxEdge, normalizer: EdgeCenterReduceNormalizer) -> JaxEdge:
            """
            Apply the EdgeCenterReduceNormalizer to a single JaxEdge.

            :param edge: JaxEdge instance.
            :param normalizer: Callable or nnx.Module implementing normalization for the edge.
            :return: JaxEdge with normalized feature_array (or identical JaxEdge if no features).
            """
            array = edge.feature_array
            if edge.feature_array is not None:
                if edge.feature_array.shape[-2] > 0:
                    # array = normalizer(array, edge.non_fictitious)  # * jnp.expand_dims(edge.non_fictitious, -1)
                    array = normalizer(array, jnp.expand_dims(edge.non_fictitious, -1))
            return JaxEdge(
                feature_array=array,
                feature_names=edge.feature_names,
                non_fictitious=edge.non_fictitious,
                address_dict=edge.address_dict,
            )

        incoming_keys = list(graph.edges.keys())
        norm_dict = {}
        for edge_key in incoming_keys:
            attr_name = f"norm_{edge_key}"

            if graph.edges[edge_key].feature_array is not None:
                if graph.edges[edge_key].feature_array.shape[-2] > 0:
                    if not hasattr(self, attr_name):
                        n_features = int(graph.edges[edge_key].feature_array.shape[-1])
                        self._make_module_for_edge(edge_key, n_features)
                        self.edge_keys = tuple(list(self.edge_keys) + [edge_key])

            norm_dict[edge_key] = getattr(self, attr_name, None)

        normalized_edge_dict = jax.tree.map(
            apply_norm,
            graph.edges,
            norm_dict,
            is_leaf=(lambda x: isinstance(x, JaxEdge)),
        )

        normalized_context = JaxGraph(
            edges=normalized_edge_dict,
            non_fictitious_addresses=graph.non_fictitious_addresses,
            true_shape=graph.true_shape,
            current_shape=graph.current_shape,
        )

        if get_info:
            info = {"input_graph": graph.quantiles(), "output_graph": normalized_context.quantiles()}

        return normalized_context, info


class EdgeCenterReduceNormalizer(nnx.Module):
    """
    EdgeCenterReduceNormalizer normalizes Edge data using a feature-wise mean and variance
    calculation while supporting running averages and bias correction.
    """

    def __init__(
        self,
        n_features: int,
        update_limit: int,
        beta_1: float = 0.9,
        beta_2: float = 0.9,
        epsilon: float = 1e-6,
        use_running_average: bool = False,
    ):
        """
        Initializes the instance with the necessary configurations and state variables for
        adaptive moment estimation and related operations.

        :param n_features: Specifies the number of features to be handled by the class.
        :param update_limit: Indicates the maximum number of updates allowed for this instance.
        :param beta_1: The exponential decay rate for the first moment estimation. Defaults to 0.9.
        :param beta_2: The exponential decay rate for the second moment estimation. Defaults to 0.999.
        :param epsilon: A small value added to prevent division by zero during calculations. Defaults to 1e-6.
        :param use_running_average: Determines whether to use a running average for parameter updates. Defaults to False.
            Automatically set to True in `eval` mode and to `False` in `train` mode.
        """
        self.n_features = n_features
        self.update_limit = update_limit
        self.use_running_average = use_running_average
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.updates = 0

        self.mean = nnx.Variable(jnp.zeros(n_features))
        self.var = nnx.Variable(jnp.ones(n_features))

    def __call__(self, x: jax.Array, mask: jax.Array = None):

        # Check input.
        if x.ndim == 2:
            is_batched = False
        elif x.ndim == 3:
            is_batched = True
        else:
            raise ValueError("Input x must be shape (n_items,F) or (B,n_items,F)")
        assert x.shape[-1] == self.n_features

        # If rolling mean and variance should be updated.
        if (self.updates < self.update_limit) or (not self.use_running_average):

            if is_batched:
                current_mean = x.mean(axis=(0, 1), where=(mask != 0.0))
                current_var = x.var(axis=(0, 1), where=(mask != 0.0))
            else:
                current_mean = x.mean(axis=0, where=(mask != 0.0))
                current_var = x.var(axis=0, where=(mask != 0.0))

            if self.updates == 0:
                self.mean = current_mean
                self.var = current_var
            else:
                self.mean = self.beta_1 * self.mean + (1 - self.beta_1) * current_mean
                self.var = self.beta_2 * self.var + (1 - self.beta_2) * current_var

            self.updates += 1

        # Correct bias
        mean_hat = self.mean / (1 - self.beta_1**self.updates)
        var_hat = self.var / (1 - self.beta_2**self.updates)

        return (x - mean_hat) / (jnp.sqrt(var_hat) + self.epsilon) * mask
