# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from flax.typing import Initializer

from energnn.graph import GraphStructure, JaxGraph
from energnn.model.utils import Activation, MLP, gather, scatter_add, scatter_max


class MessagePassingFunction(nnx.Module, ABC):
    r"""Interface for a message function :math:`\xi_\theta` in a GNN message passing scheme."""

    @abstractmethod
    def __call__(self, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """Should take as input a tuple (graph, coordinates) and return new coordinates."""
        raise NotImplementedError


class LocalSumMessagePassingFunction(MessagePassingFunction):
    r"""
    Local sum-based message function module for GNN message passing.

    This module aggregates messages from each node's local neighborhood by applying
    a class- and port-specific MLP :math:`\xi^{c,o}_\theta` to hyper-edge features and neighbor coordinates,
    summing the results across all incoming ports, and applying a final activation :math:`\sigma`.

    For each address :math:`a`, the output is defined as:

    .. math::
        \psi_\theta(h,x)_a = \sigma \left( \sum_{(c,e,o)\in \mathcal{N}_x(a)} \xi^{c,o}_\theta(h_e, x_e)\right),

    where :math:`\xi^{c,o}_\theta` is a class-specific and port-specific MLP, :math:`\sigma` is an
    element-wise activation function, and :math:`h_e := (h_{o(e)})_{o \in {\mathcal{O}^c}}` is the concatenation of
    port coordinates of hyper-edge :math:`e`.

    :param in_graph_structure: Input graph structure.
    :param in_array_size: Size of the input coordinate arrays.
    :param hidden_sizes: Hidden sizes of the MLPs :math:`\xi^{c,o}_\theta`.
    :param activation: Activation function for the MLPs :math:`\xi^{c,o}_\theta`.
    :param out_size: Output size of the MLPs :math:`\xi^{c,o}_\theta`.
    :param use_bias: Whether to use bias in the MLPs :math:`\xi^{c,o}_\theta`.
    :param kernel_init: Kernel initializer for the MLPs :math:`\xi^{c,o}_\theta`.
    :param bias_init: Bias initializer for the MLPs :math:`\xi^{c,o}_\theta`.
    :param final_activation: Final activation function for the MLPs :math:`\xi^{c,o}_\theta`.
    :param outer_activation: Activation function :math:`\sigma` applied over the output.
    :param encoded_feature_size: None if the input data has not been encoded, otherwise the size of the encoded features.
    :param port_scatter_blacklist: Dictionary mapping hyper-edge set keys to lists of port keys to be excluded from the sum.
    :param seed: Seed for RNG streams for weight initialization.
    """

    def __init__(
        self,
        in_graph_structure: GraphStructure,
        in_array_size: int,
        hidden_sizes: list[int],
        activation: Activation = nnx.relu,
        out_size: int = 1,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        outer_activation: Activation = nnx.tanh,
        encoded_feature_size: int | None = None,
        port_scatter_blacklist: dict[str, list[str]] | None = None,
        seed: int | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        self.in_graph_structure = in_graph_structure
        self.in_array_size = in_array_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.out_size = out_size
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.final_activation = final_activation
        self.outer_activation = outer_activation
        self.encoded_feature_size = encoded_feature_size
        if port_scatter_blacklist is None:
            self.port_scatter_blacklist = {}
        else:
            self.port_scatter_blacklist = port_scatter_blacklist

        self.mlp_tree = self._build_mlp_tree(seed=seed, rngs=rngs)

    def _build_mlp_tree(self, seed: int = 0, rngs: nnx.Rngs | None = None) -> dict[str, dict[str, MLP]]:
        if rngs is None:
            rngs = nnx.Rngs(seed)
        elif seed is not None:
            raise ValueError("Seed must be None when rngs are provided.")
        mlp_tree = {}

        for key, hyper_edge_set_structure in self.in_graph_structure.hyper_edge_sets.items():
            if hyper_edge_set_structure.port_list is not None and len(hyper_edge_set_structure.port_list) > 0:
                n_ports = len(hyper_edge_set_structure.port_list)
                in_size = self.in_array_size * n_ports
                if hyper_edge_set_structure.feature_list is not None and len(hyper_edge_set_structure.feature_list) > 0:
                    if self.encoded_feature_size is not None:
                        in_size += self.encoded_feature_size
                    else:
                        in_size += len(hyper_edge_set_structure.feature_list)

                if key not in mlp_tree.keys():
                    mlp_tree[key] = {}

                for port_key in hyper_edge_set_structure.port_list:
                    if port_key not in self.port_scatter_blacklist.get(key, []):
                        mlp_tree[key][port_key] = MLP(
                            in_size=in_size,
                            hidden_sizes=self.hidden_sizes,
                            activation=self.activation,
                            out_size=self.out_size,
                            use_bias=self.use_bias,
                            kernel_init=self.kernel_init,
                            bias_init=self.bias_init,
                            final_activation=self.final_activation,
                            rngs=rngs,
                        )
        return nnx.data(mlp_tree)

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:

        def sum_over_edges(_accumulator, edge_mlp_tuple):
            """Sums the output of class and port specific MLPs through ports of all hyper-edge sets in the graph."""
            hyper_edge_set, mlp_dict = edge_mlp_tuple

            input_array = []
            if hyper_edge_set.feature_names is not None:
                input_array.append(hyper_edge_set.feature_array)
            for port_name, port_array in hyper_edge_set.port_dict.items():
                input_array.append(gather(coordinates=coordinates, addresses=port_array))
            input_array = jnp.concatenate(input_array, axis=-1)
            non_fictitious_mask = jnp.expand_dims(hyper_edge_set.non_fictitious, -1)

            def sum_over_ports(__accumulator: jax.Array, mlp_port: tuple[MLP, jax.Array]) -> jax.Array:
                """Sums the outputs of port-specific MLPs through ports of a given hyper-edge set."""
                mlp, _port_array = mlp_port
                increment = mlp(input_array * non_fictitious_mask) * non_fictitious_mask
                return scatter_add(accumulator=__accumulator, increment=increment, addresses=_port_array)

            mlp_port_dict = {port_name: (mlp, hyper_edge_set.port_dict[port_name]) for port_name, mlp in mlp_dict.items()}
            return jax.tree.reduce(
                sum_over_ports, mlp_port_dict, initializer=_accumulator, is_leaf=lambda x: isinstance(x, tuple)
            )

        initializer = jnp.zeros((coordinates.shape[0], self.out_size))
        edge_mlp_dict = {key: (hyper_edge_set, self.mlp_tree[key]) for key, hyper_edge_set in graph.hyper_edge_sets.items()}
        accumulator = jax.tree.reduce(
            sum_over_edges,
            edge_mlp_dict,
            initializer=initializer,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        return self.outer_activation(accumulator), {}


class IdentityMessagePassingFunction(MessagePassingFunction):
    r"""
    Identity local message function module for GNN message passing.

    This module returns the node features unchanged as the local message.
    It implements the identity mapping on node features:

    .. math::
        h^\rightarrow_a = h_a
    """

    def __init__(self):
        pass

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        return coordinates, {}


class GATv2MessagePassingFunction(MessagePassingFunction):
    r"""
    GATv2-style attention message passing for H2MG (Item 1 of
    attention-backlog, section 3.1).

    For each address :math:`a`, the output is defined as:

    .. math::
        s_{c, e, o} = s_\theta^{c, o}(h_e, x_e) \in \mathbb{R}

    .. math::
        \alpha_{c, e, o} = \frac{\exp(s_{c, e, o})}{\sum_{(c',e',o') \in \mathcal{N}_x(a)} \exp(s_{c', e', o'})}

    .. math::
        \psi_\theta(h, x)_a = \sum_{(c, e, o) \in \mathcal{N}_x(a)} \alpha_{c, e, o} \, \xi^{c, o}_\theta(h_e, x_e)

    where :math:`s_\theta^{c, o}` is a per-(class, port) scoring MLP outputting a
    scalar logit, :math:`\xi^{c, o}_\theta` is a per-(class, port) value MLP
    outputting a vector message, and :math:`h_e := (h_{o(e)})_{o \in \mathcal{O}^c}`
    is the concatenation of port coordinates of hyper-edge :math:`e`.

    Implementation uses the equivalent numerator/denominator form, which avoids
    materialising :math:`\alpha` explicitly and supports gradient flow through
    both score and value branches. For numerical stability under unbounded
    scores, the standard segment-softmax max-subtraction is applied: a
    per-receiver maximum :math:`m_a = \max_{(c, e, o) \in \mathcal{N}_x(a)} s_{c, e, o}`
    is computed via ``scatter_max``, gathered back per source, subtracted from
    the score before exponentiating, and cancels exactly between numerator and
    denominator:

    .. math::
        \psi_\theta(h, x)_a = \frac{N_a}{\varepsilon + D_a}, \qquad
        N_a = \sum_{(c, e, o) \in \mathcal{N}_x(a)}
              \exp(s_{c, e, o} - m_a) \, \xi^{c, o}_\theta(h_e, x_e), \qquad
        D_a = \sum_{(c', e', o') \in \mathcal{N}_x(a)} \exp(s_{c', e', o'} - m_a)

    The small constant :math:`\varepsilon` guards against division by zero on
    addresses with no real (non-fictitious) incoming neighbours.

    Permutation equivariance holds: ``scatter_add`` is commutative on the source
    axis, and the per-receiver normalisation preserves equivariance on the
    receiver axis. Output of address :math:`a` permutes consistently with any
    permutation of addresses or of objects within a class.

    Per-(class, port) factoring mirrors :class:`LocalSumMessagePassingFunction`;
    the receiver perspective is encoded by which port's MLP is applied (the
    port whose address equals :math:`a`).

    :param in_graph_structure: Input graph structure (defines classes and ports).
    :param in_array_size: Size of coordinate vectors per address.
    :param hidden_sizes: Hidden layer sizes shared by score and value MLPs.
    :param activation: Inner activation of the MLPs.
    :param out_size: Output feature size per address.
    :param use_bias: Whether to use bias in MLPs.
    :param kernel_init: Kernel initializer for MLPs.
    :param bias_init: Bias initializer for MLPs.
    :param final_activation: Activation applied at the final MLP layer.
    :param outer_activation: Activation applied to the final aggregated output.
    :param encoded_feature_size: None if input edge features are raw; otherwise
        the size of the encoded features (passed through Encoder).
    :param port_scatter_blacklist: Optional mapping from hyper-edge class to a
        list of port names to exclude from aggregation. Mirrors the same flag
        in :class:`LocalSumMessagePassingFunction`.
    :param eps: Numerical stability term in the softmax denominator.
    :param score_uses_receiver: If True (default, matching the GATv2 paper),
        the score MLP also takes the receiver coordinate ``h_a`` as
        a dedicated input feature (the coord of the address that this port
        lands on), mirroring Brody et al.'s ``[h_a || h_e]`` concatenation.
        When False, the receiver perspective is encoded only implicitly via
        the per-(class, port) MLP factoring -- the receiver coord is already
        present in the concatenated port-coordinate vector and the per-port
        weights learn to attend to it. Setting this False trades a small
        amount of score-MLP input width for a less explicit signal pathway
        and makes the receiver perspective implicit rather than explicit.
    :param seed: Seed for RNG (mutually exclusive with ``rngs``).
    :param rngs: ``nnx.Rngs`` for initialization (mutually exclusive with
        ``seed``).

    References:
        Brody, Alon, Yahav. "How Attentive are Graph Attention Networks?"
        ICLR 2022. The paper specifies the score as
        :math:`s = a^\top \mathrm{LeakyReLU}(W [h_a \| h_e])`,
        i.e. one linear layer + LeakyReLU + a final linear projection to
        a scalar. The score MLP here is a strict superset: setting
        ``hidden_sizes=[d]`` with ``activation=nnx.leaky_relu`` and
        ``final_activation=None`` recovers the paper exactly. Deeper or
        wider hidden stacks generalise beyond it.

        The attention-backlog specification, section 3.1 (Item 1).
    """

    def __init__(
        self,
        in_graph_structure: GraphStructure,
        in_array_size: int,
        hidden_sizes: list[int],
        activation: Activation = nnx.relu,
        out_size: int = 1,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        outer_activation: Activation = nnx.tanh,
        encoded_feature_size: int | None = None,
        port_scatter_blacklist: dict[str, list[str]] | None = None,
        eps: float = 1e-9,
        score_uses_receiver: bool = True,
        seed: int | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        self.in_graph_structure = in_graph_structure
        self.in_array_size = in_array_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.out_size = out_size
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.final_activation = final_activation
        self.outer_activation = outer_activation
        self.encoded_feature_size = encoded_feature_size
        if port_scatter_blacklist is None:
            self.port_scatter_blacklist = {}
        else:
            self.port_scatter_blacklist = port_scatter_blacklist
        self.eps = eps
        self.score_uses_receiver = score_uses_receiver

        self.score_mlp_tree, self.value_mlp_tree = self._build_mlp_trees(seed=seed, rngs=rngs)

    def _build_mlp_trees(
        self, seed: int | None = 0, rngs: nnx.Rngs | None = None
    ) -> tuple[dict[str, dict[str, MLP]], dict[str, dict[str, MLP]]]:
        """Build the per-(class, port) score and value MLP trees.

        Both trees share the same input structure as
        :class:`LocalSumMessagePassingFunction`: concatenation of edge features
        (if any) and gathered coordinates for every port of the hyper-edge.
        The score MLP outputs a scalar logit; the value MLP outputs a vector
        of size ``out_size``.
        """
        if rngs is None:
            rngs = nnx.Rngs(seed)
        elif seed is not None:
            raise ValueError("Seed must be None when rngs are provided.")

        score_tree: dict[str, dict[str, MLP]] = {}
        value_tree: dict[str, dict[str, MLP]] = {}

        for key, hyper_edge_set_structure in self.in_graph_structure.hyper_edge_sets.items():
            if hyper_edge_set_structure.port_list is not None and len(hyper_edge_set_structure.port_list) > 0:
                n_ports = len(hyper_edge_set_structure.port_list)
                in_size = self.in_array_size * n_ports
                if hyper_edge_set_structure.feature_list is not None and len(hyper_edge_set_structure.feature_list) > 0:
                    if self.encoded_feature_size is not None:
                        in_size += self.encoded_feature_size
                    else:
                        in_size += len(hyper_edge_set_structure.feature_list)

                # When score_uses_receiver=True, the score MLP gets an
                # extra `in_array_size` slot at the end for the explicit
                # receiver coordinate. Value MLP is unchanged either way.
                score_in_size = in_size + (self.in_array_size if self.score_uses_receiver else 0)

                score_tree[key] = {}
                value_tree[key] = {}
                for port_key in hyper_edge_set_structure.port_list:
                    if port_key in self.port_scatter_blacklist.get(key, []):
                        continue
                    score_tree[key][port_key] = MLP(
                        in_size=score_in_size,
                        hidden_sizes=self.hidden_sizes,
                        activation=self.activation,
                        out_size=1,
                        use_bias=self.use_bias,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init,
                        final_activation=self.final_activation,
                        rngs=rngs,
                    )
                    value_tree[key][port_key] = MLP(
                        in_size=in_size,
                        hidden_sizes=self.hidden_sizes,
                        activation=self.activation,
                        out_size=self.out_size,
                        use_bias=self.use_bias,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init,
                        final_activation=self.final_activation,
                        rngs=rngs,
                    )
        return nnx.data(score_tree), nnx.data(value_tree)

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """Run one GATv2 attention message-passing step.

        :param graph: Input H2MG graph (single instance shape; coupler vmaps batch).
        :param coordinates: Latent coordinates per address, shape ``(n_addr, in_array_size)``.
        :param get_info: If True, return additional info for tracking.
        :return: Aggregated coordinates per address, shape ``(n_addr, out_size)``.
        """
        n_addr = coordinates.shape[0]
        neg_inf = jnp.float32(-1.0e30)

        # Pass 1: compute per-receiver max of the score, scattering into a
        # shared accumulator across all (class, port) pairs. Fictitious sources
        # contribute -inf so they never raise the receiver max above a real
        # neighbour's logit. Cached per-port tensors avoid re-running the
        # score / value MLPs in pass 2.
        max_acc = jnp.full((n_addr, 1), neg_inf, dtype=jnp.float32)
        cached: list[tuple[jax.Array, jax.Array, jax.Array, jax.Array, MLP]] = []

        for key, hyper_edge_set in graph.hyper_edge_sets.items():
            if key not in self.score_mlp_tree:
                continue
            score_dict = self.score_mlp_tree[key]
            value_dict = self.value_mlp_tree[key]

            # Build per-edge input (features concat with port-gathered coords),
            # following the LocalSumMessagePassingFunction layout exactly.
            input_array_parts: list[jax.Array] = []
            if hyper_edge_set.feature_names is not None:
                input_array_parts.append(hyper_edge_set.feature_array)
            for port_name, port_array in hyper_edge_set.port_dict.items():
                input_array_parts.append(gather(coordinates=coordinates, addresses=port_array))
            input_array = jnp.concatenate(input_array_parts, axis=-1)
            non_fictitious_mask = jnp.expand_dims(hyper_edge_set.non_fictitious, -1)
            masked_input = input_array * non_fictitious_mask

            for port_name in score_dict:
                score_mlp = score_dict[port_name]
                value_mlp = value_dict[port_name]
                port_array = hyper_edge_set.port_dict[port_name]

                if self.score_uses_receiver:
                    receiver_coord = gather(coordinates=coordinates, addresses=port_array)
                    score_input = jnp.concatenate([masked_input, receiver_coord * non_fictitious_mask], axis=-1)
                else:
                    score_input = masked_input
                score = score_mlp(score_input)
                # Fictitious sources must not influence the per-receiver max.
                score_for_max = jnp.where(non_fictitious_mask, score, neg_inf)
                max_acc = scatter_max(accumulator=max_acc, increment=score_for_max, addresses=port_array)
                cached.append((port_array, score, masked_input, non_fictitious_mask, value_mlp))

        # Pass 2: subtract the per-receiver max, exponentiate, and accumulate
        # both the value-weighted numerator and the softmax denominator.
        num_acc = jnp.zeros((n_addr, self.out_size))
        den_acc = jnp.zeros((n_addr, 1))
        for port_array, score, masked_input, non_fictitious_mask, value_mlp in cached:
            max_at_source = gather(coordinates=max_acc, addresses=port_array)
            # Branch on the mask before the subtract so fictitious entries
            # never see neg_inf (avoids 0 * inf = NaN on degenerate receivers).
            score_shifted = jnp.where(non_fictitious_mask, score - max_at_source, jnp.zeros_like(score))
            exp_score = jnp.exp(score_shifted) * non_fictitious_mask
            value = value_mlp(masked_input) * non_fictitious_mask
            num_acc = scatter_add(accumulator=num_acc, increment=exp_score * value, addresses=port_array)
            den_acc = scatter_add(accumulator=den_acc, increment=exp_score, addresses=port_array)

        # Normalise; the eps term avoids 0/0 on addresses with no incoming
        # real neighbours (output is 0 in that case, which is what we want).
        output = num_acc / (den_acc + self.eps)
        return self.outer_activation(output), {}


class MultiHeadQKVMessagePassingFunction(MessagePassingFunction):
    r"""
    Q/K/V dot-product attention message passing for H2MG (Item 3 of
    attention-backlog).

    For each address :math:`a`, the v1 single-head output is defined as:

    .. math::
        Q_a = Q_\theta(h_a) \in \mathbb{R}^{d_{QK}},

    .. math::
        K_{c, e, o} = K_\theta^{c, o}(h_e, x_e) \in \mathbb{R}^{d_{QK}},
        \quad
        V_{c, e, o} = V_\theta^{c, o}(h_e, x_e) \in \mathbb{R}^{d_V},

    .. math::
        \psi_\theta(h, x)_a = \sigma\!\left(
            \sum_{(c, e, o) \in \mathcal{N}_x(a)}
            \frac{K_{c, e, o}^\top Q_a}{\sqrt{d_{QK}}}
            \; V_{c, e, o}
        \right),

    where :math:`Q_\theta` is a single per-address query MLP,
    :math:`K_\theta^{c, o}` and :math:`V_\theta^{c, o}` are per-(class, port)
    key and value MLPs operating on :math:`(h_e, x_e)` (the concatenation of
    port-gathered coordinates and hyper-edge features), and :math:`\sigma`
    is the ``outer_activation`` applied to the aggregated output.

    Per the specification rephrasing, this is equivalent to

    .. math::
        \psi_\theta(h, x)_a = \sigma\!\left(
            \Big(\sum_{(c, e, o) \in \mathcal{N}_x(a)}
                 V_{c, e, o} \, K_{c, e, o}^\top \Big) Q_a / \sqrt{d_{QK}}
        \right),

    making explicit that :math:`Q_a` is factored out of the sum. This is the
    "linear attention" form of dot-product attention (Katharopoulos et al.
    2020): no softmax, raw bilinear scores, single pass over edges.

    Implementation uses one forward pass: query MLP applied once per address,
    key + value MLPs applied per edge, score is the per-edge dot product
    :math:`K_{c, e, o}^\top Q_{\mathrm{rcv}(c, e, o)}` (Q gathered at the
    receiver port), weighted value :math:`(\mathrm{score}) \, V_{c, e, o}` is
    ``scatter_add``-aggregated per receiver.

    Permutation equivariance holds: ``scatter_add`` is commutative on the
    source axis, the bilinear product factors through each edge
    independently, and a permutation of addresses permutes both Q and the
    K, V edge inputs consistently.

    Per-(class, port) factoring on K, V mirrors
    :class:`LocalSumMessagePassingFunction` and
    :class:`GATv2MessagePassingFunction`. Q is per-address with a single
    shared MLP because the query depends only on the receiver coordinate
    :math:`h_a`, not on the edge structure -- consistent with the backlog
    spec :math:`Q_a = Q_\theta(h_a)`.

    Differences from :class:`GATv2MessagePassingFunction`:

    - **No softmax**: scores are raw dot products, not exponentiated and
      normalised per receiver. This is the explicit "linear attention" form
      from the backlog (sec 3.3); softmax variant is left for future work.
    - **Single pass instead of two**: GATv2 needs a per-receiver max for
      softmax stability; the QKV form does not.
    - **Bilinear score** instead of MLP-scalar score: in GATv2 the score is
      the scalar output of an MLP applied to :math:`(h_a, h_e, x_e)`;
      here it is :math:`K^\top Q`, where K and Q are themselves
      :math:`d_{QK}`-dimensional MLP outputs. This is the canonical Q/K
      construction of "Attention Is All You Need" (Vaswani et al. 2017).

    **Score scaling (``scale_scores``).** Backlog sec 3.3 writes the raw
    :math:`K^\top Q`; standard practice in Vaswani et al. 2017 scales by
    :math:`1/\sqrt{d_{QK}}` so the variance of the score stays
    :math:`\mathcal{O}(1)` when :math:`K, Q` are i.i.d. unit-variance.
    Without scaling, scores grow with :math:`d_{QK}`, the weighted-value
    accumulator can dominate ``outer_activation``'s linear range
    (saturating tanh), and gradients vanish. We adopt scaling as the
    default (a deviation from the literal specification, analogous to the
    corrected denominator in
    :class:`GlobalAggregationMessagePassingFunction`) and expose
    ``scale_scores=False`` to disable it.

    **Multi-head deferred for v1** (v1 ships single-head; multi-head
    extension stacks
    :math:`d_{QK}` -> :math:`H \times d_{QK}/H` and looping the equation
    above per head).

    :param in_graph_structure: Input graph structure (defines classes and
        ports).
    :param in_array_size: Size of coordinate vectors per address.
    :param hidden_sizes: Hidden layer sizes shared by Q, K, V MLPs.
    :param d_qk: Dimension of the query / key projection.
    :param activation: Inner activation of the MLPs.
    :param out_size: Output feature size per address (also the value
        dimension :math:`d_V`).
    :param use_bias: Whether to use bias in MLPs.
    :param kernel_init: Kernel initializer for MLPs.
    :param bias_init: Bias initializer for MLPs.
    :param final_activation: Activation applied at the final MLP layer.
    :param outer_activation: Activation applied to the final aggregated
        output.
    :param encoded_feature_size: None if input edge features are raw;
        otherwise the size of the encoded features (passed through Encoder).
    :param port_scatter_blacklist: Optional mapping from hyper-edge class
        to a list of port names to exclude from aggregation. Mirrors the
        same flag in :class:`LocalSumMessagePassingFunction`.
    :param scale_scores: If True (default), divide each per-edge score
        :math:`K^\top Q` by :math:`\sqrt{d_{QK}}` (Vaswani et al. 2017).
        Set False to follow the unscaled bilinear form of the specification.
    :param eps: Reserved for future normaliser variants; currently unused
        in the no-softmax v1 forward.
    :param seed: Seed for RNG (mutually exclusive with ``rngs``).
    :param rngs: ``nnx.Rngs`` for initialization (mutually exclusive with
        ``seed``).

    References:
        Vaswani et al. "Attention Is All You Need." NeurIPS 2017
        (canonical scaled dot-product attention).
        Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive
        Transformers with Linear Attention." ICML 2020 (no-softmax form).
    """

    def __init__(
        self,
        in_graph_structure: GraphStructure,
        in_array_size: int,
        hidden_sizes: list[int],
        d_qk: int = 8,
        activation: Activation = nnx.relu,
        out_size: int = 1,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        outer_activation: Activation = nnx.tanh,
        encoded_feature_size: int | None = None,
        port_scatter_blacklist: dict[str, list[str]] | None = None,
        scale_scores: bool = True,
        eps: float = 1e-9,
        seed: int | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        self.in_graph_structure = in_graph_structure
        self.in_array_size = in_array_size
        self.hidden_sizes = hidden_sizes
        self.d_qk = d_qk
        self.activation = activation
        self.out_size = out_size
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.final_activation = final_activation
        self.outer_activation = outer_activation
        self.encoded_feature_size = encoded_feature_size
        if port_scatter_blacklist is None:
            self.port_scatter_blacklist = {}
        else:
            self.port_scatter_blacklist = port_scatter_blacklist
        self.scale_scores = scale_scores
        self.eps = eps

        if rngs is None:
            rngs = nnx.Rngs(seed if seed is not None else 0)
        elif seed is not None:
            raise ValueError("Seed must be None when rngs are provided.")

        # Q MLP: single per-address, in_size = in_array_size, out_size = d_qk.
        self.query_mlp = MLP(
            in_size=in_array_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            out_size=d_qk,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            final_activation=final_activation,
            rngs=rngs,
        )

        # K and V per-(class, port) trees, mirror LocalSum input layout.
        self.key_mlp_tree, self.value_mlp_tree = self._build_kv_trees(rngs=rngs)

    def _build_kv_trees(
        self,
        rngs: nnx.Rngs,
    ) -> tuple[dict[str, dict[str, MLP]], dict[str, dict[str, MLP]]]:
        """Build the per-(class, port) key and value MLP trees.

        Both trees share the same input structure as
        :class:`LocalSumMessagePassingFunction`: concatenation of edge
        features (if any) and gathered coordinates for every port of the
        hyper-edge. Key outputs :math:`d_{QK}`; value outputs ``out_size``.
        """
        key_tree: dict[str, dict[str, MLP]] = {}
        value_tree: dict[str, dict[str, MLP]] = {}

        for key, hyper_edge_set_structure in self.in_graph_structure.hyper_edge_sets.items():
            if hyper_edge_set_structure.port_list is not None and len(hyper_edge_set_structure.port_list) > 0:
                n_ports = len(hyper_edge_set_structure.port_list)
                in_size = self.in_array_size * n_ports
                if hyper_edge_set_structure.feature_list is not None and len(hyper_edge_set_structure.feature_list) > 0:
                    if self.encoded_feature_size is not None:
                        in_size += self.encoded_feature_size
                    else:
                        in_size += len(hyper_edge_set_structure.feature_list)

                key_tree[key] = {}
                value_tree[key] = {}
                for port_key in hyper_edge_set_structure.port_list:
                    if port_key in self.port_scatter_blacklist.get(key, []):
                        continue
                    key_tree[key][port_key] = MLP(
                        in_size=in_size,
                        hidden_sizes=self.hidden_sizes,
                        activation=self.activation,
                        out_size=self.d_qk,
                        use_bias=self.use_bias,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init,
                        final_activation=self.final_activation,
                        rngs=rngs,
                    )
                    value_tree[key][port_key] = MLP(
                        in_size=in_size,
                        hidden_sizes=self.hidden_sizes,
                        activation=self.activation,
                        out_size=self.out_size,
                        use_bias=self.use_bias,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init,
                        final_activation=self.final_activation,
                        rngs=rngs,
                    )
        return nnx.data(key_tree), nnx.data(value_tree)

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """Run one Q/K/V single-head attention message-passing step.

        :param graph: Input H2MG graph (single instance shape; coupler vmaps batch).
        :param coordinates: Latent coordinates per address, shape ``(n_addr, in_array_size)``.
        :param get_info: Unused; kept for interface compatibility.
        :return: Aggregated coordinates per address, shape ``(n_addr, out_size)``.
        """
        n_addr = coordinates.shape[0]

        # Compute Q once per address.
        q_per_address = self.query_mlp(coordinates)  # (n_addr, d_qk)

        # Scaling factor for the bilinear score.
        scale = jnp.float32(1.0 / jnp.sqrt(jnp.float32(self.d_qk))) if self.scale_scores else jnp.float32(1.0)

        accumulator = jnp.zeros((n_addr, self.out_size))

        for key, hyper_edge_set in graph.hyper_edge_sets.items():
            if key not in self.key_mlp_tree:
                continue
            key_dict = self.key_mlp_tree[key]
            value_dict = self.value_mlp_tree[key]

            # Build per-edge input (features concat with port-gathered coords),
            # following the LocalSum / GATv2 layout exactly.
            input_array_parts: list[jax.Array] = []
            if hyper_edge_set.feature_names is not None:
                input_array_parts.append(hyper_edge_set.feature_array)
            for port_name, port_array in hyper_edge_set.port_dict.items():
                input_array_parts.append(gather(coordinates=coordinates, addresses=port_array))
            input_array = jnp.concatenate(input_array_parts, axis=-1)
            non_fictitious_mask = jnp.expand_dims(hyper_edge_set.non_fictitious, -1)
            masked_input = input_array * non_fictitious_mask

            for port_name in key_dict:
                key_mlp = key_dict[port_name]
                value_mlp = value_dict[port_name]
                port_array = hyper_edge_set.port_dict[port_name]

                # K and V per edge. Masked so fictitious edges contribute 0.
                k_per_edge = key_mlp(masked_input) * non_fictitious_mask  # (n_edge, d_qk)
                v_per_edge = value_mlp(masked_input) * non_fictitious_mask  # (n_edge, out_size)

                # Gather Q at the receiver port of every edge.
                q_at_receiver = gather(coordinates=q_per_address, addresses=port_array)  # (n_edge, d_qk)

                # Bilinear score per edge.
                score = jnp.sum(k_per_edge * q_at_receiver, axis=-1, keepdims=True) * scale  # (n_edge, 1)

                # Weighted value, masked so fictitious edges contribute 0 even
                # if their numeric K or V were non-zero (defensive).
                weighted_v = v_per_edge * score * non_fictitious_mask  # (n_edge, out_size)

                accumulator = scatter_add(accumulator=accumulator, increment=weighted_v, addresses=port_array)

        return self.outer_activation(accumulator), {}


class GlobalAggregationMessagePassingFunction(MessagePassingFunction):
    r"""
    Global aggregation message passing for H2MG (Item 2 of attention-backlog).

    Every receiving address gets the same global summary of the per-address
    coordinates. This is a context-vector layer: when wrapped in
    :class:`RecurrentCoupler` alongside per-address messages, every address
    obtains a window onto the whole graph. Used as a context channel
    in compositions and in :class:`VirtualAddressRecurrentCoupler` (Item 5).

    For each address :math:`a`, the output is defined as:

    .. math::
        \psi_\theta(h, x)_a = \sigma\!\left(
            \frac{1}{|\mathcal{A}_x^{\mathrm{real}}| + \varepsilon}
            \sum_{a' \in \mathcal{A}_x^{\mathrm{real}}}
            \xi_\theta(h_{a'})
        \right),

    where :math:`\xi_\theta` is a single per-address value MLP applied to
    the coordinate vector, :math:`\mathcal{A}_x^{\mathrm{real}}` is the
    set of non-fictitious addresses (padding masked out via
    ``non_fictitious_addresses``), and :math:`\sigma` is the
    ``outer_activation`` applied to the aggregated output.

    The denominator uses the count of real addresses plus a small
    :math:`\varepsilon` guard, deviating from the specification denominator
    :math:`|\mathcal{A}_x|` to avoid dilution by padding from fictitious
    addresses.

    Output broadcasts the same vector to every real receiver and zeros the
    fictitious ones. Permutation INVARIANCE (stronger than equivariance)
    holds because the mean is symmetric in its arguments — the result does
    not depend on the order of addresses.

    Differences from :class:`LocalSumMessagePassingFunction`:

    - No per-(class, port) factoring: the formulation in backlog sec 3.2 is
      pure address-level. A single ``value_mlp`` operates on the coordinate
      vector directly, not on per-port concatenations.
    - No hyper-edge features consumed: messages depend only on coordinates,
      so ``encoded_feature_size`` and ``port_scatter_blacklist`` are not
      part of this constructor surface.
    - Reducer fixed to ``mean`` for v1. The spec also mentions ``min`` /
      ``max`` / ``sum`` and an attention-weighted form; those are deferred
      (backlog sec 3.2 proposed resolution; multi-head and alternative reducers deferred).
    - Multi-head form deferred for v1 likewise.

    :param in_graph_structure: Input graph structure (stored for interface
        consistency with sibling message functions; unused in the forward
        body since aggregation is address-level, not class-level).
    :param in_array_size: Size of the coordinate vector per address.
    :param hidden_sizes: Hidden layer sizes of the per-address value MLP.
    :param activation: Inner activation of the MLP.
    :param out_size: Output feature size per address.
    :param use_bias: Whether to use bias in the MLP.
    :param kernel_init: Kernel initializer.
    :param bias_init: Bias initializer.
    :param final_activation: Activation applied at the MLP's last layer.
    :param outer_activation: Activation applied to the aggregated output
        before broadcasting back to every receiver.
    :param eps: Numerical guard in the mean denominator (default 1e-9). The
        denominator is the count of non-fictitious addresses; ``eps`` only
        matters in the degenerate case of zero real addresses.
    :param seed: Seed for RNG (mutually exclusive with ``rngs``).
    :param rngs: ``nnx.Rngs`` for initialization (mutually exclusive with
        ``seed``).

    References:
        The attention-backlog specification, section 3.2 (Item 2).
    """

    def __init__(
        self,
        in_graph_structure: GraphStructure,
        in_array_size: int,
        hidden_sizes: list[int],
        activation: Activation = nnx.relu,
        out_size: int = 1,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        outer_activation: Activation = nnx.tanh,
        eps: float = 1e-9,
        seed: int | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        self.in_graph_structure = in_graph_structure
        self.in_array_size = in_array_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.out_size = out_size
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.final_activation = final_activation
        self.outer_activation = outer_activation
        self.eps = eps

        if rngs is None:
            rngs = nnx.Rngs(seed if seed is not None else 0)
        elif seed is not None:
            raise ValueError("Seed must be None when rngs are provided.")
        self.value_mlp = MLP(
            in_size=in_array_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            out_size=out_size,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            final_activation=final_activation,
            rngs=rngs,
        )

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """Run one global-aggregation message-passing step.

        :param graph: Input H2MG graph (single instance shape; coupler vmaps batch).
        :param coordinates: Latent coordinates per address, shape ``(n_addr, in_array_size)``.
        :param get_info: Unused; kept for interface compatibility.
        :return: Aggregated coordinates per address, shape ``(n_addr, out_size)``. The
            same global summary is broadcast to every real receiver and zeroed on
            fictitious receivers.
        """
        n_addr = coordinates.shape[0]
        mask = jnp.expand_dims(graph.non_fictitious_addresses, -1)

        # Per-address value MLP; Flax NNX Linear handles leading batch axes,
        # so applying value_mlp to (n_addr, in_array_size) yields (n_addr, out_size).
        # Mask BEFORE summation so fictitious addresses contribute 0.
        values = self.value_mlp(coordinates) * mask

        # Mean with corrected denominator: count of real addresses + eps.
        total = jnp.sum(values, axis=0)
        n_real = jnp.sum(graph.non_fictitious_addresses)
        mean_summary = total / (n_real + self.eps)

        # Broadcast the same summary to every receiver, zero on fictitious.
        broadcast = jnp.broadcast_to(mean_summary, (n_addr, self.out_size))
        output = broadcast * mask
        return self.outer_activation(output), {}


class PerformerMessagePassingFunction(MessagePassingFunction):
    r"""
    Linear-attention message passing for H2MG (Item 4 of attention-backlog,
    section 3.4).

    All-to-all linear attention over addresses. Unlike GATv2 / MultiHeadQKV
    which aggregate over neighbours of the receiver via the graph
    connectivity, this layer aggregates over **every** address. The graph
    topology (port_dict of hyper-edges) is not used here.

    For each address :math:`a`, the v1 output is defined as:

    .. math::
        Q_a = Q_\theta(h_a) \in \mathbb{R}^{d_{QK}},
        \quad K_{a'} = K_\theta(h_{a'}) \in \mathbb{R}^{d_{QK}},
        \quad V_{a'} = V_\theta(h_{a'}) \in \mathbb{R}^{d_V},

    .. math::
        \psi_\theta(h, x)_a = \sigma\!\left(
            \frac{1}{\sqrt{d_{QK}}}
            \sum_{a' \in \mathcal{A}_x^{\mathrm{real}}}
            (K_{a'}^\top Q_a)\, V_{a'}
        \right),

    where :math:`Q_\theta, K_\theta, V_\theta` are three per-address MLPs
    applied to the latent coordinates, :math:`\mathcal{A}_x^{\mathrm{real}}`
    is the set of non-fictitious addresses (padding masked out), and
    :math:`\sigma` is the ``outer_activation`` applied after aggregation.

    The kernel-trick reformulation,

    .. math::
        \psi_\theta(h, x)_a = \sigma\!\left(
            \frac{1}{\sqrt{d_{QK}}}
            \Bigl(\sum_{a'} V_{a'} K_{a'}^\top\Bigr) Q_a
        \right),

    makes the linear-time computation explicit: the outer product
    :math:`M = \sum_{a'} V_{a'} K_{a'}^\top \in \mathbb{R}^{d_V \times d_{QK}}`
    is accumulated once over all addresses, then multiplied by :math:`Q_a`
    per receiver. Total cost is :math:`O(n_{\mathrm{addr}} \cdot d_V \cdot d_{QK})`
    rather than the :math:`O(n_{\mathrm{addr}}^2 \cdot d_V)` of a naive
    pairwise attention.

    **Scaling.** The specification writes the raw bilinear form
    :math:`K^\top Q`. Implementation follows the convention of Vaswani et
    al. 2017 and divides the score by :math:`\sqrt{d_{QK}}` to keep the
    variance of the score :math:`\mathcal{O}(1)` when :math:`Q, K` are
    i.i.d. unit-variance. Flag ``scale_scores=True`` by default,
    configurable via the ``scale_scores`` flag.

    **No softmax.** The spec is the no-softmax linear-attention form
    (Katharopoulos et al. 2020). The kernel trick above only holds because
    the softmax is absent. The class name "Performer" is retained but
    the random-feature kernel approximation of Choromanski et al. 2021 is
    deferred to v2; here only the plain linear-attention variant is
    implemented.

    **Multi-head deferred for v1** (single-head shipped first).

    **Fictitious-address masking.** Q, K and V are multiplied by
    ``graph.non_fictitious_addresses`` before the outer-product
    accumulation so that padded addresses do not contribute to the sum.

    Differences from sibling classes:

    - :class:`LocalSumMessagePassingFunction`,
      :class:`GATv2MessagePassingFunction`,
      :class:`MultiHeadQKVMessagePassingFunction`: those aggregate over
      neighbours of the receiver via per-(class, port) MLPs; this one
      aggregates over **all addresses** with a single shared MLP for each
      of :math:`Q, K, V`.
    - :class:`GlobalAggregationMessagePassingFunction`: that one
      computes a plain mean over all addresses (no Q-dependent weighting);
      this one weights the sum by :math:`K^\top Q` so the output depends
      on the receiver via :math:`Q_a`.

    :param in_graph_structure: Input graph structure. Stored for interface
        consistency with sibling message functions; not used in the
        forward body since aggregation is address-level.
    :param in_array_size: Size of coordinate vectors per address.
    :param hidden_sizes: Hidden layer sizes shared by Q, K, V MLPs.
    :param d_qk: Dimension of the query / key projection.
    :param activation: Inner activation of the MLPs.
    :param out_size: Output feature size per address (also the value
        dimension :math:`d_V`).
    :param use_bias: Whether to use bias in MLPs.
    :param kernel_init: Kernel initializer for MLPs.
    :param bias_init: Bias initializer for MLPs.
    :param final_activation: Activation applied at the final MLP layer.
    :param outer_activation: Activation applied to the aggregated output.
    :param scale_scores: If True (default), divide the score
        :math:`K^\top Q` by :math:`\sqrt{d_{QK}}` (Vaswani et al. 2017).
        Set False to follow the unscaled bilinear form of the specification.
    :param eps: Reserved for future normalised variants; unused in v1.
    :param seed: Seed for RNG (mutually exclusive with ``rngs``).
    :param rngs: ``nnx.Rngs`` for initialization (mutually exclusive with
        ``seed``).

    References:
        Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive
        Transformers with Linear Attention." ICML 2020 (no-softmax linear
        attention form; the kernel-trick rephrasing used here).
        Vaswani et al. "Attention Is All You Need." NeurIPS 2017
        (scaled dot-product attention).

        The attention-backlog specification, section 3.4 (Item 4).
    """

    def __init__(
        self,
        in_graph_structure: GraphStructure,
        in_array_size: int,
        hidden_sizes: list[int],
        d_qk: int = 8,
        activation: Activation = nnx.relu,
        out_size: int = 1,
        use_bias: bool = True,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        final_activation: Activation | None = None,
        outer_activation: Activation = nnx.tanh,
        scale_scores: bool = True,
        eps: float = 1e-9,
        seed: int | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        self.in_graph_structure = in_graph_structure
        self.in_array_size = in_array_size
        self.hidden_sizes = hidden_sizes
        self.d_qk = d_qk
        self.activation = activation
        self.out_size = out_size
        self.use_bias = use_bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.final_activation = final_activation
        self.outer_activation = outer_activation
        self.scale_scores = scale_scores
        self.eps = eps

        if rngs is None:
            rngs = nnx.Rngs(seed if seed is not None else 0)
        elif seed is not None:
            raise ValueError("Seed must be None when rngs are provided.")

        self.query_mlp = MLP(
            in_size=in_array_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            out_size=d_qk,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            final_activation=final_activation,
            rngs=rngs,
        )
        self.key_mlp = MLP(
            in_size=in_array_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            out_size=d_qk,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            final_activation=final_activation,
            rngs=rngs,
        )
        self.value_mlp = MLP(
            in_size=in_array_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            out_size=out_size,
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            final_activation=final_activation,
            rngs=rngs,
        )

    def __call__(self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False) -> tuple[jax.Array, dict]:
        """Run one linear-attention all-to-all message-passing step.

        :param graph: Input H2MG graph (single instance shape; coupler vmaps batch).
        :param coordinates: Latent coordinates per address, shape ``(n_addr, in_array_size)``.
        :param get_info: Unused; kept for interface compatibility.
        :return: Aggregated coordinates per address, shape ``(n_addr, out_size)``.
        """
        # Per-address Q, K, V from coordinates.
        q = self.query_mlp(coordinates)  # (n_addr, d_qk)
        k = self.key_mlp(coordinates)  # (n_addr, d_qk)
        v = self.value_mlp(coordinates)  # (n_addr, out_size)

        # Mask fictitious addresses: their contributions to K and V must be
        # zero so that the outer-product sum below excludes padding.
        mask = jnp.expand_dims(graph.non_fictitious_addresses, -1)  # (n_addr, 1)
        k_masked = k * mask
        v_masked = v * mask

        # Kernel-trick form: accumulate outer product M = sum_a V_a K_a^T once,
        # then multiply by Q per receiver. Cost O(n_addr * d_V * d_qk).
        # M shape: (out_size, d_qk).
        m = jnp.einsum("ad,af->df", v_masked, k_masked)

        # Apply scaling, then per-receiver multiplication output_a = M @ Q_a.
        scale = jnp.float32(1.0 / jnp.sqrt(jnp.float32(self.d_qk))) if self.scale_scores else jnp.float32(1.0)
        # output shape: (n_addr, out_size)
        output = jnp.einsum("df,af->ad", m, q) * scale

        return self.outer_activation(output), {}
