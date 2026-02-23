import jax
import jax.numpy as jnp
from flax import nnx

from energnn.graph import GraphStructure, JaxGraph
from energnn.graph.jax import JaxEdge
from .normalizer import Normalizer


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
        self.update_limit = nnx.Variable(jnp.array([update_limit]))
        self.use_running_average = use_running_average
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.updates = nnx.Variable(jnp.array([0]))

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
        is_training = not self.use_running_average
        # We use jnp.where to handle the updates even if jitted, to avoid TracerBoolConversionError.
        # However, the assignment itself must happen.

        if is_batched:
            current_mean = x.mean(axis=(0, 1), where=(mask != 0.0))
            current_var = x.var(axis=(0, 1), where=(mask != 0.0))
        else:
            current_mean = x.mean(axis=0, where=(mask != 0.0))
            current_var = x.var(axis=0, where=(mask != 0.0))

        if self.mean._can_update or self.var._can_update:
            stop_gradient = jax.lax.stop_gradient
        else:

            def stop_gradient(_x):
                return _x

        should_update = is_training & (self.updates[...] < self.update_limit[...])[0]

        new_mean = jnp.where(
            self.updates[...] == 0,
            current_mean,
            self.beta_1 * self.mean[...] + (1 - self.beta_1) * current_mean,
        )
        new_var = jnp.where(
            self.updates[...] == 0,
            current_var,
            self.beta_2 * self.var[...] + (1 - self.beta_2) * current_var,
        )

        self.mean[...] = stop_gradient(jnp.where(should_update, new_mean, self.mean[...]))
        self.var[...] = stop_gradient(jnp.where(should_update, new_var, self.var[...]))
        self.updates[...] = jnp.where(should_update, self.updates[...] + 1, self.updates[...])

        # Correct bias
        # We add epsilon to denominator to avoid division by zero when updates is 0
        mean_hat = self.mean / (1 - self.beta_1**self.updates + self.epsilon)
        var_hat = self.var / (1 - self.beta_2**self.updates + self.epsilon)

        return (x - mean_hat) / (jnp.sqrt(var_hat) + self.epsilon) * mask


class CenterReduceNormalizer(Normalizer):
    r"""
    Graph-level wrapper that maintains an EdgeCenterReduceNormalizer for each edge key.

    For a given feature of a given edge class, the output is defined as follows.

    .. math::
        x' = \frac{x - \mu}{\sqrt{\sigma^2} + \epsilon}

    where :math:`\mu` (resp. :math:`\sigma^2`) is the exponential moving average of the empirical mean (resp. variance)
    with decay rate `beta_1` (resp. `beta_2`).

    :param in_structure: GraphStructure of the input graph.
    :param update_limit: Threshold for the maximum updates to be performed.
    :param beta_1: Exponential decay rate for the first moment estimates. Defaults to 0.9.
    :param beta_2: Exponential decay rate for the second moment estimates. Defaults to 0.999.
    :param epsilon: Small constant added to improve numerical stability. Defaults to 1e-6.
    :param use_running_average: Flag that indicates whether to use a running average or not. Defaults to False.
        Automatically set to True in `eval` mode and to `False` in `train` mode.
    """

    def __init__(
        self,
        in_structure: GraphStructure,
        update_limit: int,
        beta_1: float = 0.9,
        beta_2: float = 0.9,
        epsilon: float = 1e-6,
        use_running_average: bool = False,
    ):
        self.in_structure = in_structure
        self.update_limit = update_limit
        self.use_running_average = use_running_average
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.module_dict = self._build_module_dict()

    def _build_module_dict(self) -> dict[str, EdgeCenterReduceNormalizer]:
        """Creates a Center Reduce Normalizer module for each edge key in the graph structure."""
        module_dict = {}
        for edge_key, edge_structure in self.in_structure.edges.items():
            if edge_structure.feature_list is not None and len(edge_structure.feature_list) > 0:
                in_size = len(edge_structure.feature_list)
                module_dict[edge_key] = EdgeCenterReduceNormalizer(
                    in_size,
                    update_limit=self.update_limit,
                    beta_1=self.beta_1,
                    beta_2=self.beta_2,
                    epsilon=self.epsilon,
                    use_running_average=self.use_running_average,
                )
        return nnx.data(module_dict)

    def __call__(self, *, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Apply normalization to edges within a JaxGraph context using EdgeCenterReduceNormalizer. This method normalizes the
        edges' feature arrays and updates the associated context graph accordingly.

        :param graph: JaxGraph representing the graph structure containing edges with feature arrays to be normalized.
        :param get_info: Boolean flag that indicates whether to return additional information about input and output graphs.
        :return: A tuple containing the normalized JaxGraph and an optional dictionary holding quantile information
                 about the input and output graphs.
        """

        edge_norm_dict = {k: (edge, self.module_dict[k]) for k, edge in graph.edges.items() if k in self.module_dict.keys()}

        def apply_norm(edge_norm: tuple[JaxEdge, EdgeCenterReduceNormalizer]) -> JaxEdge:
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
