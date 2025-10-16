#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import jax.numpy as jnp
import jax.random

from energnn.gnn.utils import MLP
from energnn.graph.jax import JaxGraph
from .local_message_function import IdentityLocalMessageFunction, LocalMessageFunction
from .remote_message_function import IdentityRemoteMessageFunction, RemoteMessageFunction
from .self_message_function import IdentitySelfMessageFunction, SelfMessageFunction

SELF = "self"
LOCAL = "local"
REMOTE = "remote"
PHI = "phi"


class CouplingFunction:
    r"""
    Endomorphism of the space of address coordinates, parameterized by the context graph :math:`x`.

    This coupling function composes self, local, and remote message transformations
    with an outer MLP :math:`\phi_\theta` to update node coordinates.

    :param MLP phi: Multi Layer Perceptron :math:`\phi_\theta`.
    :param SelfMessageFunction self_message_function: Self message function :math:`\psi_\theta^\circlearrowleft`.
    :param LocalMessageFunction local_message_function: Local message function :math:`\psi_\theta^\rightarrow`.
    :param RemoteMessageFunction remote_message_function: Remote message function :math:`\psi_\theta^\leadsto`.
    """

    phi: MLP
    self_message_function: SelfMessageFunction = IdentitySelfMessageFunction()
    local_message_function: LocalMessageFunction = IdentityLocalMessageFunction()
    remote_message_function: RemoteMessageFunction = IdentityRemoteMessageFunction()

    def __init__(
        self,
        phi: MLP,
        self_message_function: SelfMessageFunction,
        local_message_function: LocalMessageFunction,
        remote_message_function: RemoteMessageFunction,
    ) -> None:
        """
        Initialize the coupling components.

        :param phi: MLP to combine message contributions.
        :param self_message_function: Module for self-messages.
        :param local_message_function: Module for local messages.
        :param remote_message_function: Module for remote messages.
        """
        self.phi = phi
        self.self_message_function = self_message_function
        self.local_message_function = local_message_function
        self.remote_message_function = remote_message_function

    def init(self, *, rngs: jax.Array, context: JaxGraph, coordinates: jax.Array) -> dict:
        r"""
        Initialize parameters for all submodules with separate PRNG keys.

        Sets the output dimension of :math:`\phi_\theta` to the coordinate width.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Graph context for message functions.
        :param coordinates: Node coordinate array.
        :return: Mapping from component labels to their initialized parameters.
        """
        self.phi.out_size = coordinates.shape[1]
        params = {}
        rng_1, rng_2, rng_3, rng_4 = jax.random.split(rngs, 4)
        (self_m, _), params[SELF] = self.self_message_function.init_with_output(
            rngs=rng_1, context=context, coordinates=coordinates
        )
        (local, _), params[LOCAL] = self.local_message_function.init_with_output(
            rngs=rng_2, context=context, coordinates=coordinates
        )
        (remote, _), params[REMOTE] = self.remote_message_function.init_with_output(
            rngs=rng_3, context=context, coordinates=coordinates
        )
        params[PHI] = self.phi.init(rng_4, jnp.concatenate([self_m, local, remote], axis=1))
        return params

    def init_with_output(self, *, rngs: jax.Array, context: JaxGraph, coordinates: jax.Array) -> tuple[jax.Array, dict]:
        """
        Initialize submodule parameters and compute the initial update.

        Similar to `init`, but also returns the transformed coordinates, it masks out fictitious nodes using
        `context.non_fictitious_addresses`.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Graph context.
        :param coordinates: Node coordinate array.
        :return: Tuple of (updated coordinates, parameter dict).
        """
        self.phi.out_size = coordinates.shape[1]
        params = {}
        rng_1, rng_2, rng_3, rng_4 = jax.random.split(rngs, 4)
        (self_m, _), params[SELF] = self.self_message_function.init_with_output(
            rngs=rng_1, context=context, coordinates=coordinates
        )
        (local, _), params[LOCAL] = self.local_message_function.init_with_output(
            rngs=rng_2, context=context, coordinates=coordinates
        )
        (remote, _), params[REMOTE] = self.remote_message_function.init_with_output(
            rngs=rng_3, context=context, coordinates=coordinates
        )
        output, params[PHI] = self.phi.init_with_output(rng_4, jnp.concatenate([self_m, local, remote], axis=1))
        output = output * jnp.expand_dims(context.non_fictitious_addresses, -1)
        return output, params

    def apply(
        self, params: dict, *, context: JaxGraph, coordinates: jax.Array, get_info: bool = False
    ) -> tuple[jax.Array, dict]:
        r"""
        Apply the coupling function to input coordinates.

        Steps:
          1. Compute self, local, and remote messages via submodules.
          2. Concatenate the three message tensors.
          3. Apply the coupling MLP :math:`\phi_\theta`.
          4. Mask out fictitious nodes.

        :param params: Initialized parameters for each component.
        :param context: Graph context containing masks and adjacency.
        :param coordinates: Input node coordinates array.
        :param get_info: If True, collect auxiliary outputs from submodules.
        :return: Tuple of (new coordinates, info dict keyed by "self", "local", "remote").
        """
        infos = {}
        self_m, infos["self"] = self.self_message_function.apply(
            params[SELF], context=context, coordinates=coordinates, get_info=get_info
        )
        local, infos["local"] = self.local_message_function.apply(
            params[LOCAL], context=context, coordinates=coordinates, get_info=get_info
        )
        remote, infos["remote"] = self.remote_message_function.apply(
            params[REMOTE], context=context, coordinates=coordinates, get_info=get_info
        )
        output = self.phi.apply(params[PHI], jnp.concatenate([self_m, local, remote], axis=1))
        output = output * jnp.expand_dims(context.non_fictitious_addresses, -1)
        return output, infos
