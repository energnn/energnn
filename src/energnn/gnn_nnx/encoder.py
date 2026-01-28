# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable

from flax import nnx
from flax.nnx import initializers
from flax.typing import Array, Initializer
import jax
import jax.numpy as jnp
import jax.random

from energnn.gnn_nnx.utils import MLP
from energnn.graph.jax import JaxEdge, JaxGraph

MAX_INTEGER = 2147483647
Activation = Callable[[Array], Array]


class Encoder(ABC):
    r"""
    Interface for the graph encoder :math:`E_\theta`.

    Subclasses must implement methods to initialize parameters and apply the encoder
    to a JaxGraph object.
    """

    @abstractmethod
    def __init__(self):
        """
        Abstract constructor.
        Implementations may define module parameters or internal state.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def init(self, *, rngs: jax.Array, context: JaxGraph) -> dict:
        """
        Should return initialized encoder weights.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph.
        :return: Initialized parameters.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def init_with_output(self, *, rngs: jax.Array, context: JaxGraph) -> tuple[tuple[JaxGraph, dict], dict]:
        """
        Initialize encoder parameters and return encoded graph and parameters.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph.
        :return: Tuple ((encoded graph, encoder parameters), other info dict).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, params: dict, context: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Apply encoder to input graph and return encoded `context`.

        :param params: Parameters.
        :param context: Input graph to encode.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Tuple (encoded graph, info).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError


class IdentityEncoder(Encoder):
    r"""
    Identity encoder that returns the input graph unchanged.

    .. math::
        \tilde{x} = x
    """

    def __init__(self):
        pass

    def init(self, *, rngs: jax.Array, context: JaxGraph) -> dict:
        """Return empty parameters (no learnable weights)."""
        return {}

    def init_with_output(self, *, rngs: jax.Array, context: JaxGraph) -> tuple[tuple[JaxGraph, dict], dict]:
        """
        Initialize the encoder and returns the output graph (unmodified graph) and empty parameter dicts.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Input graph.
        :return: ((input graph, empty dict), empty dict)
        """
        return (context, {}), {}

    def apply(self, params: dict, context: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """Apply the identity encoder and return the input graph without changes.

        :param params: Parameters.
        :param context: Input graph to encode.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: Input graph and empty info dict.
        """
        return context, {}


class MLPEncoder(nnx.Module, Encoder):
    r"""
    Encoder that applies class-specific Multi Layer Perceptrons.

    .. math::
        \begin{align}
        &\forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, & \tilde{x}_e = \phi_\theta^c(x_e),
        \end{align}

    where :math:`({\phi}_{\theta}^c)_{c\in C}` are a set of class-specific MLPs.

    :param hidden_size: Hidden sizes for each MLP.
    :param out_size: Output size for each MLP.
    :param activation: Activation function to use inside MLPs.
    :param rngs: nnx.Rngs or integer seed used to derive RNG streams for per-type MLPs.
    :return: NNX Module that encodes edges using class-specific MLPs.
    """

    def __init__(
        self,
        hidden_size: list[int],
        *,
        out_size: int,
        activation: Activation | None = jax.nn.relu,
        rngs: nnx.Rngs | int | None = None,
        built: bool = False,
    ) -> None:

        if rngs is None:
            rngs = nnx.Rngs(0)
        elif isinstance(rngs, int):
            rngs = nnx.Rngs(rngs)

        self.hidden_size = [int(h) for h in hidden_size]
        self.out_size = int(out_size)
        self.activation = activation
        self.rngs = rngs
        self.mlps: nnx.Dict = nnx.Dict()

    def _build_mlps_for_context(self, context: JaxGraph) -> nnx.Dict:
        """
        Ensure that an MLP exists for each edge class in `context.edges`.

        For each key `k` in `context.edges`:
          - if an MLP is not already present in self.mlps, instantiate it;
          - once instantiated, if the corresponding edge has a non-None `feature_array`, call
            `mlp.build_from_sample(edge.feature_array)` to lazily initialize
            internal Linear layers.

        :param context: JaxGraph used to infer input feature dimensions when available.
        :returns: nnx.Dict mapping edge-class keys to MLP instances (created/initialized).
        """
        for k, edge in context.edges.items():
            # create MLP for key if missing
            if k not in self.mlps:
                self.mlps[k] = MLP(
                    hidden_size=self.hidden_size,
                    out_size=self.out_size,
                    activation=self.activation,
                    rngs=self.rngs,
                    name=str(k),
                )

                # if feature_array is present, lazily initialize sub-layers from sample
                if edge.feature_array is not None:
                    self.mlps[k].build_from_sample(edge.feature_array)

        return self.mlps

    def __call__(self, *, context: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Apply the Multi Layer Perceptron neural network to edges of an input graph and return the corresponding graph.

        Each edge type (key in `context.edges`) gets its own MLP.

        :param context: Input graph with edges to encode.
        :param get_info: Flag to return additional information for tracking purpose.
        :return: Encoded graph and additional info dictionary.
        """
        info: dict = {}

        feature_names = {f"lat_{i}": jnp.array(i) for i in range(self.out_size)}

        mlp_dict = self._build_mlps_for_context(context)

        plain_mlps = {k: mlp_dict[k] for k in context.edges.keys()}

        def apply_mlp(edge: JaxEdge, mlp: MLP) -> JaxEdge:
            """Apply the MLP to an edge"""
            if edge.feature_array is not None:
                encoded_array = mlp(edge.feature_array)
                return JaxEdge(
                    feature_array=encoded_array,
                    feature_names=feature_names,
                    non_fictitious=edge.non_fictitious,
                    address_dict=edge.address_dict,
                )
            else:
                return JaxEdge(
                    feature_array=None,
                    feature_names=None,
                    non_fictitious=edge.non_fictitious,
                    address_dict=edge.address_dict,
                )

        encoded_edge_dict = jax.tree.map(
            apply_mlp,
            context.edges,
            plain_mlps,
            is_leaf=(lambda x: isinstance(x, JaxEdge)),
        )

        encoded_context = JaxGraph(
            edges=encoded_edge_dict,
            non_fictitious_addresses=context.non_fictitious_addresses,
            true_shape=context.true_shape,
            current_shape=context.current_shape,
        )

        return encoded_context, info
