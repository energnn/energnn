#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import diffrax

from energnn.gnn.coupler import Coupler
from energnn.gnn.coupler.coupler import METHOD, FUNCTION
from energnn.gnn.coupler.coupling_function import CouplingFunction
from energnn.gnn.coupler.coupling_function import (
    IdentityLocalMessageFunction,
    IdentityRemoteMessageFunction,
    IdentitySelfMessageFunction,
)
from energnn.gnn.coupler.solving_method import NeuralODESolvingMethod, ZeroSolvingMethod
from energnn.gnn.utils import MLP
from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph
from tests.utils import TestProblemLoader
from tests.gnn.utils import make_dummy_coupling_mock, assert_single, assert_batch


np.random.seed(0)

n = 6
pb_loader = TestProblemLoader(
    dataset_size=4,
    n_batch=2,
    context_edge_params={
        "node": {"n_obj": n, "feature_list": ["a", "b"], "address_list": ["0"]},
        "edge": {"n_obj": n, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
    },
    oracle_edge_params={
        "node": {"n_obj": n, "feature_list": ["e"]},
        "edge": {"n_obj": n, "feature_list": ["f"]},
    },
    n_addr=n,
    shuffle=False,
)
pb_batch = next(iter(pb_loader))
context_batch, _ = pb_batch.get_context()
jax_context_batch = JaxGraph.from_numpy_graph(context_batch)
context = separate_graphs(context_batch)[0]
jax_context = JaxGraph.from_numpy_graph(context)

def test_zero_solving_method():
    coupling_function = CouplingFunction(
        phi=MLP(hidden_size=[8], activation=nn.relu, out_size=1),
        self_message_function=IdentitySelfMessageFunction(),
        local_message_function=IdentityLocalMessageFunction(),
        remote_message_function=IdentityRemoteMessageFunction(),
    )
    solving_method = ZeroSolvingMethod(latent_dimension=16)
    coupler = Coupler(coupling_function=coupling_function, solving_method=solving_method)
    params, output, infos = assert_single(coupler=coupler, seed=0, context=jax_context)
    output, infos = assert_batch(params=params, coupler=coupler, context=jax_context_batch)


def test_node_solving_method():
    coupling_function = CouplingFunction(
        phi=MLP(hidden_size=[8], activation=nn.relu, out_size=1),
        self_message_function=IdentitySelfMessageFunction(),
        local_message_function=IdentityLocalMessageFunction(),
        remote_message_function=IdentityRemoteMessageFunction(),
    )

    solving_method = NeuralODESolvingMethod(
        latent_dimension=16,
        dt=0.1,
        stepsize_controller=diffrax._step_size_controller.ConstantStepSize(),
        adjoint=diffrax._adjoint.RecursiveCheckpointAdjoint(),
        solver=diffrax._solver.Euler(),
        max_steps=1000,
    )
    coupler = Coupler(coupling_function=coupling_function, solving_method=solving_method)
    params, output, infos = assert_single(coupler=coupler, seed=0, context=jax_context)
    output, infos = assert_batch(params=params, coupler=coupler, context=jax_context_batch)


def test_zero_solving_method_end_to_end_returns_initial_zeros_when_used_in_coupler():
    """Integration test: ZeroSolvingMethod should initialize zeros and solve returns the same zeros."""
    coupling_function = make_dummy_coupling_mock()
    solving_method = ZeroSolvingMethod(latent_dimension=4)
    coupler = Coupler(coupling_function=coupling_function, solving_method=solving_method)

    rng = jax.random.PRNGKey(2)
    params = coupler.init(rngs=rng, context=jax_context)

    coords, info = coupler.apply(params=params, context=jax_context)
    # coordinates should be zeros of shape (n_addresses, latent_dimension)
    assert coords.shape == (n, 4)
    assert np.allclose(np.array(coords), np.zeros((n, 4)))
    # info returned by ZeroSolvingMethod.solve is {} per implementation
    assert info == {}


def test_neuralode_solving_method_runs_and_returns_shapes_and_info():
    """
    Run NeuralODESolvingMethod with a coupling that returns zero vector (so ODE RHS=0).
    This ensures the diffrax integration path is exercised but numerically stable and deterministic.
    """
    class ZeroCoupling:
        def init(self, *, rngs, context, coordinates):
            return {"z": 0}

        def init_with_output(self, *, rngs, context, coordinates):
            return ((jnp.zeros_like(coordinates), {}), {"z": 0})

        def apply(self, params, context, coordinates, get_info=False):
            # return zero derivative and an *array-like* info for diffrax compatibility
            # derivative:
            deriv = jnp.zeros_like(coordinates)
            # info: choose a JAX array (scalar) so diffrax can allocate/save it across time steps
            info = {"zero": jnp.array(1.0)}
            return deriv, info

    latent_dim = 3

    # Use an adaptive solver that supports error-estimates (Tsit5) and a PID controller for step-size
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    adjoint = diffrax.RecursiveCheckpointAdjoint()

    solving_method = NeuralODESolvingMethod(
        latent_dimension=latent_dim,
        dt=0.1,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        solver=solver,
        max_steps=500,
    )

    # sanity: initialize coordinates
    initial_coords = solving_method.initialize_coordinates(context=jax_context)
    assert initial_coords.shape == (n, latent_dim)
    assert np.allclose(np.array(initial_coords), np.zeros((n, latent_dim)))

    # Solve with RHS=0: expect final coordinates == initial (zeros)
    final_coords, info = solving_method.solve(
        params={"z": 0}, function=ZeroCoupling(), coordinates_init=initial_coords, context=jax_context, get_info=False
    )
    assert final_coords.shape == (n, latent_dim)
    assert np.allclose(np.array(final_coords), np.zeros((n, latent_dim)), atol=1e-6)

    # Now ask for info: should return dict containing "ode_info"
    final_coords2, info2 = solving_method.solve(
        params={"z": 0}, function=ZeroCoupling(), coordinates_init=initial_coords, context=jax_context, get_info=True
    )
    # Vérifier la présence et la nature numérique
    assert "ode_info" in info2
    ode_info = info2["ode_info"]
    leaves = jax.tree_util.tree_leaves(ode_info)
    assert all(isinstance(x, (jnp.ndarray, np.ndarray)) for x in leaves)

    for x in leaves:
        np.testing.assert_array_equal(np.array(x), np.ones(x.shape))
    np.testing.assert_allclose(np.array(final_coords2), np.zeros((n, latent_dim)), atol=1e-6)
