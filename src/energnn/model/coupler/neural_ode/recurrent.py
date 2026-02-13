# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import logging

import jax
import jax.numpy as jnp
from flax import nnx

from energnn.graph import JaxGraph
from energnn.model.utils import MLP
from .message_function import MessageFunction
from ..coupler import Coupler

logger = logging.getLogger(__name__)


class RecurrentCoupler(Coupler):
    r"""
    Simplified version of the Neural Ordinary Differential Equation solver.

    The following recurrent system is used.:

    .. math::
        \forall a \in \mathcal{A}_x, h_a(t+\delta t) = h_a(t+\delta t) + \delta t \phi_\theta(\psi^1_\theta(h;x)_a, \dots, \psi^n_\theta(h;x)_a),

    with the following initial condition:

    .. math::
        \forall a \in \mathcal{A}_x, h_a(t=0) = [0, \dots, 0].

    :param phi: Outer MLP :math:`\phi_\theta`.
    :param message_functions: List of message functions :math:`(\psi^i_\theta)_i`.
    :param n_steps: Number of message passing steps.
    """

    def __init__(
        self,
        phi: MLP,
        message_functions: list[MessageFunction],
        n_steps: int,
    ):
        super().__init__()
        self.phi = phi
        self.message_functions = nnx.List(message_functions)
        self.n_steps = n_steps

        self.dt = 1 / self.n_steps

    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[jax.Array, dict]:

        def F(t, coordinates, graph):
            """Residual function."""
            messages = []
            for m in self.message_functions:
                message, info = m(graph=graph, coordinates=coordinates)
                messages.append(message)
            messages = jnp.concatenate(messages, axis=-1)
            return self.phi(messages)

        h = jnp.zeros([jnp.shape(graph.non_fictitious_addresses)[0], self.phi.out_size])

        dt = 1 / self.n_steps
        for _ in range(self.n_steps):
            h = h + dt * F(0, h, graph)

        return h, {}

    @staticmethod
    def log_solved():
        """Log a message indicating successful ODE solve."""
        logger.info("ODE solved")
