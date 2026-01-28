#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import diffrax
from unittest.mock import MagicMock

from energnn.gnn.coupler.solving_method import ZeroSolvingMethod, NeuralODESolvingMethod
from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph
from tests.utils import TestProblemLoader
from energnn.gnn.coupler.coupling_function import CouplingFunction

# deterministic RNG for reproducibility in tests
np.random.seed(0)
jax.random.PRNGKey(0)

# Build a small test ProblemLoader and graphs
n = 6
pb_loader = TestProblemLoader(
    dataset_size=4,
    n_batch=2,
    context_edge_params={
        "node": {"n_obj": n, "feature_list": ["a", "b"], "address_list": ["0"]},
        "edge": {"n_obj": n, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
    },
    oracle_edge_params={"node": {"n_obj": n, "feature_list": ["e"]}, "edge": {"n_obj": n, "feature_list": ["f"]}},
    n_addr=n,
    shuffle=False,
)
pb_batch = next(iter(pb_loader))
context_batch, _ = pb_batch.get_context()
jax_context_batch = JaxGraph.from_numpy_graph(context_batch)
context_single = separate_graphs(context_batch)[0]
jax_context = JaxGraph.from_numpy_graph(context_single)


def make_constant_coupling_mock(C):
    """
    Return a MagicMock that mimics a CouplingFunction with an `apply` method.
    Always returns a constant field `C` broadcast to the shape of `coordinates`.
    """
    C_jnp = jnp.array(C).astype(jnp.float32)

    def apply_fn(params, context, coordinates, get_info=False):
        # Broadcast C to coordinates shape if possible
        target_shape = coordinates.shape
        # If C_jnp is scalar -> broadcast
        if C_jnp.ndim == 0:
            out = jnp.ones(target_shape, dtype=jnp.float32) * C_jnp
        elif C_jnp.ndim == 1 and C_jnp.shape[0] == target_shape[1]:
            # broadcast across addresses
            out = jnp.tile(C_jnp[None, :], (target_shape[0], 1))
        else:
            # attempt broadcast directly
            out = jnp.broadcast_to(C_jnp, target_shape)
        return out, {}

    m = MagicMock(spec=CouplingFunction)
    m.apply = apply_fn
    return m


# Tests for ZeroSolvingMethod
def test_zero_solver_initialize_coordinates_shape_and_dtype():
    latent_dim = 3
    solver = ZeroSolvingMethod(latent_dimension=latent_dim)
    coords0 = solver.initialize_coordinates(context=jax_context)
    # shape: (N, latent_dim) where N is number of addresses
    expected_n = int(np.array(jax_context.non_fictitious_addresses).shape[0])
    assert coords0.shape == (expected_n, latent_dim)
    # all zeros
    assert np.all(np.array(coords0) == 0.0)
    # dtype is numeric (float)
    assert np.issubdtype(np.array(coords0).dtype, np.floating)


def test_zero_solver_solve_returns_same_array_and_empty_info():
    latent_dim = 2
    solver = ZeroSolvingMethod(latent_dimension=latent_dim)
    # create a non-zero initial coordinate array
    n_addr = int(np.array(jax_context.non_fictitious_addresses).shape[0])
    coords_init = jnp.arange(n_addr * latent_dim, dtype=jnp.float32).reshape((n_addr, latent_dim))
    # ZeroSolvingMethod.solve simply returns coordinates_init unchanged
    out, info = solver.solve(params={}, function=None, coordinates_init=coords_init, context=jax_context, get_info=True)
    np.testing.assert_allclose(np.array(out), np.array(coords_init))
    assert info == {}


def test_zero_solver_with_zero_addresses():
    latent_dim = 4
    solver = ZeroSolvingMethod(latent_dimension=latent_dim)
    # Build a context with zero addresses
    ctx_zero = JaxGraph(
        edges=jax_context.edges,
        non_fictitious_addresses=jnp.array([], dtype=jnp.float32),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )
    coords0 = solver.initialize_coordinates(context=ctx_zero)
    assert coords0.shape == (0, latent_dim)


# Tests for NeuralODESolvingMethod
def make_neuralode(latent_dim=2, dt=0.1, max_steps=500, rtol=1e-6, atol=1e-6):
    """
    Small helper returning a NeuralODESolvingMethod with Tsit5 + PID controller and checkpoint adjoint.
    """
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    adjoint = diffrax.RecursiveCheckpointAdjoint()
    return NeuralODESolvingMethod(
        latent_dimension=latent_dim,
        dt=dt,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        solver=solver,
        max_steps=max_steps,
    )


def test_neuralode_initialize_coordinates_shape_and_dtype():
    latent_dim = 3
    method = make_neuralode(latent_dim=latent_dim)
    coords0 = method.initialize_coordinates(context=jax_context)
    expected_n = int(np.array(jax_context.non_fictitious_addresses).shape[0])
    assert coords0.shape == (expected_n, latent_dim)
    assert np.all(np.array(coords0) == 0.0)


def test_neuralode_solve_constant_field_returns_h0_plus_c():
    """
    For dh/dt = C constant, solution at t=1 is h(1)=h0 + C.
    Use the ConstantFunction to return C and check final result.
    """
    latent_dim = 2
    method = make_neuralode(latent_dim=latent_dim, dt=0.1, max_steps=500)
    n_addr = int(np.array(jax_context.non_fictitious_addresses).shape[0])

    # constant vector C (per-dimension)
    C = jnp.array([0.5, -0.3], dtype=jnp.float32)
    const_f = make_constant_coupling_mock(C)

    # initial coordinates h0 (non-zero)
    h0 = jnp.ones((n_addr, latent_dim), dtype=jnp.float32) * 2.0

    final, info = method.solve(params={}, function=const_f, coordinates_init=h0, context=jax_context, get_info=False)

    # expected h1 = h0 + C * (t1 - t0) = h0 + C
    expected = np.array(h0) + np.array(C)[None, :]
    np.testing.assert_allclose(np.array(final), expected, rtol=1e-5, atol=1e-5)


def test_neuralode_solve_with_zero_initial_and_constant_field():
    """
    For dh/dt = C with h0 = 0, final should equal C.
    """
    latent_dim = 2
    method = make_neuralode(latent_dim=latent_dim, dt=0.05, max_steps=1000)
    n_addr = int(np.array(jax_context.non_fictitious_addresses).shape[0])
    C = 0.25  # scalar -> broadcast
    const_f = make_constant_coupling_mock(C)

    h0 = jnp.zeros((n_addr, latent_dim), dtype=jnp.float32)
    final, _ = method.solve(params={}, function=const_f, coordinates_init=h0, context=jax_context, get_info=False)

    expected = np.ones_like(np.array(h0)) * C
    np.testing.assert_allclose(np.array(final), expected, rtol=1e-5, atol=1e-5)


def test_neuralode_get_info_returns_ode_info():
    """
    When get_info=True, solve should populate 'ode_info' key in returned info.
    """
    latent_dim = 2
    method = make_neuralode(latent_dim=latent_dim, dt=0.2, max_steps=500)
    n_addr = int(np.array(jax_context.non_fictitious_addresses).shape[0])
    C = jnp.array([0.1, 0.2], dtype=jnp.float32)
    const_f = make_constant_coupling_mock(C)
    h0 = jnp.zeros((n_addr, latent_dim), dtype=jnp.float32)

    final, info = method.solve(params={}, function=const_f, coordinates_init=h0, context=jax_context, get_info=True)
    # info must contain 'ode_info'
    assert "ode_info" in info
    # ode_info should be array-like (diffrax returns diagnostic arrays), ensure it's not empty
    ode_info = info["ode_info"]
    assert ode_info is not None


def test_neuralode_respects_max_steps_raises_on_tight_max_steps():
    """
    If max_steps is too small to march to t=1, diffrax should raise an exception;
    ensure solve surfaces an exception in that case.
    """
    latent_dim = 2
    # choose dt large and max_steps small to provoke failure
    method = make_neuralode(latent_dim=latent_dim, dt=1.0, max_steps=0)
    n_addr = int(np.array(jax_context.non_fictitious_addresses).shape[0])
    const_f = make_constant_coupling_mock(0.1)
    h0 = jnp.zeros((n_addr, latent_dim), dtype=jnp.float32)

    with pytest.raises(Exception):
        _ = method.solve(params={}, function=const_f, coordinates_init=h0, context=jax_context, get_info=False)


def test_neuralode_with_different_solvers_consistency():
    """
    Run solve with two adaptive solvers (Tsit5 and Dopri5) on a simple constant field
    and check outputs are similar. Both solvers accept PIDController so no runtime error
    about adaptive controllers should occur.
    """
    latent_dim = 2
    n_addr = int(np.array(jax_context.non_fictitious_addresses).shape[0])
    C = jnp.array([0.3, -0.2], dtype=jnp.float32)
    const_f = make_constant_coupling_mock(C)
    h0 = jnp.zeros((n_addr, latent_dim), dtype=jnp.float32)

    # Tsit5 with PID controller
    method_tsit = NeuralODESolvingMethod(
        latent_dimension=latent_dim,
        dt=0.05,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        solver=diffrax.Tsit5(),
        max_steps=1000,
    )
    final_tsit, _ = method_tsit.solve(params={}, function=const_f, coordinates_init=h0, context=jax_context, get_info=False)

    # Dopri5 with PID controller
    # If your diffrax version does not expose Dopri5, replace with an adaptive solver available (e.g. Heun if supported).
    method_dopri = NeuralODESolvingMethod(
        latent_dimension=latent_dim,
        dt=0.05,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        solver=diffrax.Dopri5(),
        max_steps=1000,
    )
    final_dopri, _ = method_dopri.solve(params={}, function=const_f, coordinates_init=h0, context=jax_context, get_info=False)

    # They should be close to each other and close to C broadcast to (n_addr, latent_dim)
    np.testing.assert_allclose(np.array(final_tsit), np.array(final_dopri), rtol=1e-6, atol=1e-6)

    # Build expected array with same shape as final_tsit
    expected = np.broadcast_to(np.array(C)[None, :], np.array(final_tsit).shape)
    np.testing.assert_allclose(np.array(final_tsit), expected, rtol=1e-6, atol=1e-6)
