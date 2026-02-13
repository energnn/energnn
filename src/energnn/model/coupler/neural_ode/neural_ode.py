# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import logging

import diffrax
import jax
import jax.numpy as jnp
from flax import nnx

from energnn.graph import JaxGraph
from energnn.model.utils import MLP
from .message_function import MessageFunction
from ..coupler import Coupler

logger = logging.getLogger(__name__)


class NeuralODECoupler(Coupler):
    r"""
    Output coordinates are computed by solving a Neural Ordinary Differential Equation.

    The following ordinary differential equation is integrated between 0 and 1:

    .. math::
        \forall a \in \mathcal{A}_x, \frac{dh_a}{dt} = \phi_\theta(\psi^1_\theta(h;x)_a, \dots, \psi^n_\theta(h;x)_a),

    with the following initial condition:

    .. math::
        \forall a \in \mathcal{A}_x, h_a(t=0) = [0, \dots, 0].

    Implementation relies on Patrick Kidger's `Diffrax <https://docs.kidger.site/diffrax/>`_.

    :param phi: Outer MLP :math:`\phi_\theta`.
    :param message_functions: List of message functions :math:`(\psi^i_\theta)_i`.
    :param latent_dimension: Dimension of address latent coordinates.
    :param dt: Initial step size value.
    :param stepsize_controller: Controller for adaptive step size methods.
    :param adjoint: Method used for backpropagation.
    :param solver: Numerical solver for the ODE.
    :param max_steps: Maximum number of steps allowed for the solving of the ODE.
    """

    def __init__(
        self,
        phi: MLP,
        message_functions: list[MessageFunction],
        dt: float,
        stepsize_controller: diffrax.AbstractStepSizeController,
        adjoint: diffrax.AbstractAdjoint,
        solver: diffrax.AbstractSolver,
        max_steps: int,
    ):
        super().__init__()
        self.phi = phi
        self.message_functions = nnx.List(message_functions)
        self.dt = dt
        self.stepsize_controller = stepsize_controller
        self.solver = solver
        self.adjoint = adjoint
        self.max_steps = max_steps

    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[jax.Array, dict]:

        def F(t, coordinates, graph):
            """Second member of the Neural ODE."""
            messages = []
            for m in self.message_functions:
                message, info = m(graph=graph, coordinates=coordinates)
                messages.append(message)
            messages = jnp.concatenate(messages, axis=-1)
            return self.phi(messages)

        h_0 = jnp.zeros([jnp.shape(graph.non_fictitious_addresses)[0], self.phi.out_size])

        solution = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(F),
            solver=self.solver,
            t0=0,
            t1=1,
            dt0=self.dt,
            y0=h_0,
            saveat=diffrax.SaveAt(t1=True),
            args=graph,
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            max_steps=self.max_steps,
        )
        return solution.ys[-1], {}

    @staticmethod
    def log_solved():
        """Log a message indicating successful ODE solve."""
        logger.info("ODE solved")
