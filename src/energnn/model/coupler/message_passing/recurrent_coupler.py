# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from energnn.graph import Graph
from energnn.model.utils import MLP
from .message_passing_function import MessagePassingFunction
from ..coupler import Coupler

logger = logging.getLogger(__name__)


class RecurrentCoupler(Coupler):
    r"""
    Simplified version of the Neural Ordinary Differential Equation solver.

    The following recurrent system is used.:

    .. math::
        \forall a \in \mathcal{A}_x, h_a(t+\delta t) = h_a(t+\delta t) +
        \delta t \times \phi_\theta(\psi^1_\theta(h;x)_a, \dots, \psi^n_\theta(h;x)_a),

    with the following initial condition:

    .. math::
        \forall a \in \mathcal{A}_x, h_a(t=0) = [0, \dots, 0].

    :param phi: Outer MLP :math:`\phi_\theta`.
    :param message_functions: List of message functions :math:`(\psi^i_\theta)_i`.
    :param n_steps: Number of message passing steps.
    :param use_scan: If True, iterate with :func:`jax.lax.scan` instead of an unrolled Python loop.
        This keeps the compiled program size independent of ``n_steps``.
    :param use_remat: If True (and ``use_scan``), apply :func:`jax.checkpoint` to each scan step,
        so the backward pass only stores the per-step carries instead of all intermediate activations.
    """

    def __init__(
        self,
        phi: MLP,
        message_functions: list[MessagePassingFunction],
        n_steps: int,
        use_scan: bool = True,
        use_remat: bool = True,
    ):
        super().__init__()
        self.phi = phi
        self.message_functions = nnx.List(message_functions)
        self.n_steps = n_steps
        self.use_scan = use_scan
        self.use_remat = use_remat

        self.dt = 1 / self.n_steps

    def __call__(self, graph: Graph, get_info: bool = False) -> tuple[jax.Array, dict]:

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
        if self.use_scan:

            def step(carry, _):
                return carry + dt * F(0, carry, graph), None

            step_fn = jax.checkpoint(step) if self.use_remat else step
            h, _ = jax.lax.scan(step_fn, h, None, length=self.n_steps)
        else:
            for _ in range(self.n_steps):
                h = h + dt * F(0, h, graph)

        return h, {}

    @staticmethod
    def log_solved():
        """Log a message indicating successful ODE solve."""
        logger.info("ODE solved")
