# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import flax
import jax.random

from energnn.graph.jax import JaxGraph
from .coupling_function.coupling_function import CouplingFunction
from .solving_method import SolvingMethod

METHOD = "solving_method"
FUNCTION = "coupling_function"


class Coupler(flax.struct.PyTreeNode):
    r"""
    Coupler :math:`C_{\theta}` that associates addresses of a context :math:`x` with latent coordinates.

    The coupler orchestrates the generation of initial latent coordinates via a solving method,
    followed by refinement through a learnable coupling function.

    :param coupling_function: Endomorphism :math:`F_\theta` that induces a coupling between addresses.
    :param solving_method: Functional :math:`S` that processes the coupling function into address coordinates.
    """

    coupling_function: CouplingFunction
    solving_method: SolvingMethod

    def init(self, *, rngs: jax.Array, context: JaxGraph) -> dict:
        """
        Initialize the coupling function parameters for the given graph context.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Graph context for initialization.
        :return: Dictionary containing initialized parameters under key "FUNCTION".
        :raises ValueError: If coordinate initialization fails.
        """
        params = {}
        rng_s, rng_c = jax.random.split(rngs, num=2)
        coordinates = self.solving_method.initialize_coordinates(context=context)
        params[FUNCTION] = self.coupling_function.init(rngs=rng_c, context=context, coordinates=coordinates)
        return params

    def init_with_output(self, *, rngs: jax.Array, context: JaxGraph) -> tuple[tuple[jax.Array, dict], dict]:
        """
        Initialize parameters and compute initial latent coordinates output.

        :param rngs: JAX Pseudo-Random Number Generator (PRNG) array.
        :param context: Graph context for initialization.
        :return: A tuple ((coordinates, empty_info), params_dict) where:
                 - coordinates (jax.Array): Initial latent coordinates.
                 - params_dict (dict): Parameters of the coupling function under key "FUNCTION".
        """
        params = {}
        rng_s, rng_c = jax.random.split(rngs, num=2)
        coordinates = self.solving_method.initialize_coordinates(context=context)
        _, params[FUNCTION] = self.coupling_function.init_with_output(rngs=rng_c, context=context, coordinates=coordinates)
        return (coordinates, {}), params

    def apply(self, params: dict, *, context: JaxGraph, get_info: bool = False) -> tuple[jax.Array, dict]:
        """
        Compute refined latent coordinates for the graph using the coupling function.

        The workflow is:

        1. Re-generate initial latent coordinates with the solving method.
        2. Invoke the solving method's `solve` routine, passing the coupling function,
           context, and initial coordinates to iteratively refine the coordinates.

        :param params: Dictionary containing the coupling function parameters under "FUNCTION".
        :param context: A given graph context.
        :param get_info: If True, return diagnostic information from the solving method.
        :return: A tuple `(coordinates, info)` where:
                 - coordinates (jax.Array): Refined latent address coordinates.
                 - info (dict): Solver metadata and diagnostics.
        """
        # TODO : mettre des args verbose et info, et vérifier que ça jit bien.
        # TODO : Réfléchir à des visualisation.
        coordinates = self.solving_method.initialize_coordinates(context=context)
        coordinates, info = self.solving_method.solve(
            params=params[FUNCTION], function=self.coupling_function, context=context, coordinates_init=coordinates
        )

        return coordinates, info
