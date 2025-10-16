#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import logging
from abc import ABC, abstractmethod

import diffrax
import jax.numpy as jnp
import jax.random

from energnn.gnn.coupler.coupling_function import CouplingFunction
from energnn.graph.jax import JaxGraph

logger = logging.getLogger(__name__)


class SolvingMethod(ABC):
    """
    Interface for solving methods.

    Subclasses implement strategies to initialize and iteratively solve for address
    latent coordinates based on a coupling function and the graph context.
    """

    @abstractmethod
    def __init__(self):
        """
        Constructor to initialize solving method.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_coordinates(self, *, context: JaxGraph) -> jax.Array:
        """
        Provides initial address coordinates before solving.

        :param context: Graph context.
        :return: Default coordinates for addresses.
        """
        raise NotImplementedError

    @abstractmethod
    def solve(
        self,
        *,
        params: dict,
        function: CouplingFunction,
        coordinates_init: jax.Array,
        context: JaxGraph,
        get_info: bool = False,
    ) -> tuple[jax.Array, dict]:
        """
        Should associate addresses with coordinates that reflect their coupling through the context graph.

        :param params: Parameters for the coupling function.
        :param function: CouplingFunction instance defining address updates.
        :param coordinates_init: Initial coordinate estimates.
        :param context: Graph context for message functions and masking.
        :param get_info: Whether to return auxiliary solver information.
        :return: Tuple of (solved coordinates, info dictionary).
        """
        raise NotImplementedError


class ZeroSolvingMethod(SolvingMethod):
    """
    Trivial solver that returns zero coordinates.

    :param latent_dimension: Dimension of address latent coordinates.
    """

    def __init__(self, latent_dimension: int):
        self.latent_dimension = latent_dimension

    def initialize_coordinates(self, *, context: JaxGraph) -> jax.Array:
        """
        Initialize coordinates to zeros for all non-fictitious nodes.

        :param context: Graph context with non-fictitious mask.
        :return: Zero array of shape (N, latent_dimension).
        """
        return jnp.zeros([jnp.shape(context.non_fictitious_addresses)[0], self.latent_dimension])

    def solve(
        self,
        *,
        params: dict,
        function: CouplingFunction,
        coordinates_init: jax.Array,
        context: JaxGraph,
        get_info: bool = False,
    ) -> tuple[jax.Array, dict]:
        """
        Return the initial coordinates without modification.

        :param params: Parameters for the coupling function.
        :param function: CouplingFunction instance defining address updates.
        :param coordinates_init: Initial coordinate estimates.
        :param context: Graph context for message functions and masking.
        :param get_info: Whether to return auxiliary solver information.
        :return: Tuple of (coordinates_init, empty dictionary).
        """
        return coordinates_init, {}


class NeuralODESolvingMethod(SolvingMethod):
    r"""
    Address coordinates are computed by solving a Neural Ordinary Differential Equation.

    The following ordinary differential equation is integrated between 0 and 1:

    .. math::
        \frac{dh}{dt}=F_{\theta}(h;x).

    Implementation relies on Patrick Kidger's `Diffrax <https://docs.kidger.site/diffrax/>`_.

    :param latent_dimension: Dimension of address latent coordinates.
    :param dt: Initial step size value.
    :param stepsize_controller: Controller for adaptive step size methods.
    :param adjoint: Method used for backpropagation.
    :param solver: Numerical solver for the ODE.
    :param max_steps: Maximum number of steps allowed for the solving of the ODE.
    """

    latent_dimension: int
    dt: float = 0.1
    stepsize_controller: diffrax.AbstractStepSizeController
    adjoint: diffrax.AbstractAdjoint
    solver: diffrax.AbstractSolver
    max_steps: int = 1000

    def __init__(
        self,
        latent_dimension: int,
        dt: float,
        stepsize_controller: diffrax.AbstractStepSizeController,
        adjoint: diffrax.AbstractAdjoint,
        solver: diffrax.AbstractSolver,
        max_steps: int,
    ):
        self.latent_dimension = latent_dimension
        self.dt = dt
        self.stepsize_controller = stepsize_controller
        self.solver = solver
        self.adjoint = adjoint
        self.max_steps = max_steps

    def initialize_coordinates(self, *, context: JaxGraph) -> jax.Array:
        """
        Initialize the ODE state to zeros for each address.

        :param context: Graph context with non-fictitious mask.
        :return: Zero array of shape (N, latent_dimension).
        """
        return jnp.zeros([jnp.shape(context.non_fictitious_addresses)[0], self.latent_dimension])

    @staticmethod
    def log_solved():
        """Log a message indicating successful ODE solve."""
        logger.info("ODE solved")

    def solve(
        self,
        *,
        params: dict,
        function: CouplingFunction,
        coordinates_init: jax.Array,
        context: JaxGraph,
        get_info: bool = False,
    ) -> tuple[jax.Array, dict]:
        """
        Solve the Neural ODE from t=0 to t=1 using diffrax.

        :param params: Parameters for the coupling function.
        :param function: CouplingFunction providing the ODE vector field.
        :param coordinates_init: Initial state array.
        :param context: Graph context.
        :param get_info: If True, collect per-step diagnostics.
        :return: Tuple of (final coordinates, info dict with ODE diagnostics).
        """

        info = {}

        def apply(t, coordinates, context):
            latent_coordinates_update, _ = function.apply(
                params=params,
                context=context,
                coordinates=coordinates,
            )
            return latent_coordinates_update

        ode_coupling_term = diffrax.ODETerm(apply)

        if get_info:
            # Metrics are computed two times this way, check if an efficient method it is introcued in diffrax
            # https://github.com/patrick-kidger/diffrax/issues/462

            def save_at_fn(t, y, context):
                return (
                    y,
                    jax.lax.stop_gradient(
                        function.apply(
                            params=params,
                            context=context,
                            coordinates=y,
                        )
                    )[1],
                )

            save_at = diffrax.SaveAt(ts=jnp.arange(0, 1 + self.dt, self.dt), fn=save_at_fn)
        else:
            save_at = diffrax.SaveAt(t1=True)

        solution = diffrax.diffeqsolve(
            terms=ode_coupling_term,
            solver=self.solver,
            t0=0,
            t1=1,
            dt0=self.dt,
            y0=coordinates_init,
            saveat=save_at,
            args=context,
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            max_steps=self.max_steps,
        )

        if get_info:
            final_latent_coordinates, ode_info = solution.ys[0][-1], solution.ys[1]
            info["ode_info"] = ode_info
        else:
            final_latent_coordinates = solution.ys[-1]

        jax.debug.callback(NeuralODESolvingMethod.log_solved)

        return final_latent_coordinates, info
