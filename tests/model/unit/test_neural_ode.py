#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import pytest
from flax import nnx

from energnn.model.coupler.neural_ode.neural_ode import NeuralODECoupler
from energnn.model.utils import MLP
from tests.utils import TestProblemLoader

# deterministic RNG for reproducibility in tests
np.random.seed(0)
jax.random.PRNGKey(0)

# Build a small test ProblemLoader and graphs (same structure as in the original tests)
n_max = 10
pb_loader = TestProblemLoader(seed=0, batch_size=4, n_max=n_max)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)


class ConstantMessage:
    """A simple callable message function that returns a constant vector (per-address)."""

    def __init__(self, vec):
        self.vec = jnp.asarray(vec, dtype=jnp.float32)

    def __call__(self, *, graph, coordinates):
        # broadcast vec to (n_addresses, vec_dim)
        n = coordinates.shape[0]
        return jnp.tile(self.vec[None, :], (n, 1)), {}


class IdentityPhi(nnx.Module):
    """Simple identity phi for testing (to bypass MLP complexity)."""

    def __init__(self, out_size: int):
        self.out_size = out_size

    def __call__(self, x):
        return jnp.asarray(x, dtype=jnp.float32)


def make_coupler(
    *,
    phi=None,
    message_functions,
    dt: float = 0.1,
    solver=None,
    stepsize_controller=None,
    adjoint=None,
    max_steps: int = 1000,
):
    """
    Convenience factory for NeuralODECoupler.
    """
    if phi is None:
        # Default tiny MLP if none provided
        phi = MLP(in_size=1, hidden_sizes=[], out_size=1, activation=None, seed=0)

    coupler = NeuralODECoupler(
        phi=phi,
        message_functions=message_functions,
        dt=dt,
        stepsize_controller=stepsize_controller or diffrax.ConstantStepSize(),
        adjoint=adjoint or diffrax.RecursiveCheckpointAdjoint(),
        solver=solver or diffrax.Tsit5(),
        max_steps=max_steps,
    )
    return coupler


def test_neuralodecoupler_numeric_integration_basic():
    """
    If phi is identity and message function returns a constant vector C,
    with Euler solver and dt=1.0, then y(1) = y(0) + F(0, y(0)) * 1 = C (since y(0)=0).
    Verifies shapes, types and numeric precision of the ODE integration.
    """
    C = jnp.array([0.5, -0.3], dtype=jnp.float32)
    latent_dim = int(C.shape[0])
    mf = [ConstantMessage(C)]
    coupler = make_coupler(
        phi=IdentityPhi(out_size=latent_dim),
        message_functions=mf,
        dt=1.0,
        solver=diffrax.Euler(),
        max_steps=10,
    )

    out, info = coupler(graph=jax_context, get_info=False)

    # Check shapes and types
    n_addr = int(jax_context.non_fictitious_addresses.shape[0])
    assert out.shape == (n_addr, latent_dim)
    assert isinstance(out, jnp.ndarray)
    assert info == {}

    # Check numeric value
    expected = jnp.tile(C[None, :], (n_addr, 1))
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-6, atol=1e-6)


def test_neuralodecoupler_solver_respects_max_steps_and_raises():
    """
    If max_steps is too small (e.g. 0), the solver should raise an error.
    """
    C = jnp.array([0.1, 0.2], dtype=jnp.float32)
    latent_dim = int(C.shape[0])
    mf = [ConstantMessage(C)]
    coupler = make_coupler(
        phi=IdentityPhi(out_size=latent_dim),
        message_functions=mf,
        dt=1.0,
        solver=diffrax.Euler(),
        max_steps=0,
    )

    with pytest.raises(Exception):
        _ = coupler(graph=jax_context, get_info=False)


def test_neuralodecoupler_adaptive_solvers_consistency():
    """
    Run the same simple constant-message test with two adaptive solvers (Tsit5 and Dopri5)
    to check for consistency and interop with PIDController.
    """
    C = jnp.array([0.3, -0.2], dtype=jnp.float32)
    latent_dim = int(C.shape[0])
    mf = [ConstantMessage(C)]

    common_kwargs = {
        "phi": IdentityPhi(out_size=latent_dim),
        "message_functions": mf,
        "dt": 0.05,
        "stepsize_controller": diffrax.PIDController(rtol=1e-6, atol=1e-6),
        "max_steps": 1000,
    }

    # Tsit5
    coupler_tsit = make_coupler(solver=diffrax.Tsit5(), **common_kwargs)
    out_tsit, _ = coupler_tsit(graph=jax_context, get_info=False)

    # Dopri5
    coupler_dopri = make_coupler(solver=diffrax.Dopri5(), **common_kwargs)
    out_dopri, _ = coupler_dopri(graph=jax_context, get_info=False)

    # They should be numerically close
    np.testing.assert_allclose(np.array(out_tsit), np.array(out_dopri), rtol=1e-5, atol=1e-5)

    # And close to analytical expectation
    n_addr = int(jax_context.non_fictitious_addresses.shape[0])
    expected = jnp.tile(C[None, :], (n_addr, 1))
    np.testing.assert_allclose(np.array(out_tsit), np.array(expected), rtol=1e-4, atol=1e-4)


def test_neuralodecoupler_multiple_message_functions_and_concatenation_shape():
    """
    Ensure multiple message functions are correctly concatenated.
    Concatenated dimension (2+1=3) is passed to phi.
    """
    m1 = ConstantMessage(jnp.array([1.0, 2.0], dtype=jnp.float32))
    m2 = ConstantMessage(jnp.array([3.0], dtype=jnp.float32))
    mf_list = [m1, m2]

    coupler = make_coupler(
        phi=IdentityPhi(out_size=3),
        message_functions=mf_list,
        dt=1.0,
        solver=diffrax.Euler(),
        max_steps=20,
    )
    out, _ = coupler(graph=jax_context, get_info=False)

    n_addr = int(jax_context.non_fictitious_addresses.shape[0])
    assert out.shape == (n_addr, 3)


def test_neuralodecoupler_vmap_jit_compatibility():
    """
    Ensure the coupler is compatible with JAX transformations (vmap, jit).
    """
    C = jnp.array([0.4, -0.1], dtype=jnp.float32)
    latent_dim = int(C.shape[0])
    mf = [ConstantMessage(C)]
    coupler = make_coupler(
        phi=IdentityPhi(out_size=latent_dim),
        message_functions=mf,
        dt=1.0,
        solver=diffrax.Euler(),
        max_steps=50,
    )

    # Vectorize and JIT
    apply_vmap = jax.vmap(lambda g, gi: coupler(graph=g, get_info=gi), in_axes=(0, None), out_axes=0)
    apply_vmap_jit = jax.jit(apply_vmap)

    # Call once to initialize any internal state (though identity phi has none)
    _ = coupler(graph=jax_context, get_info=False)

    out1, info1 = apply_vmap(jax_context_batch, False)
    out2, info2 = apply_vmap_jit(jax_context_batch, False)

    # Compare results
    np.testing.assert_allclose(np.array(out1), np.array(out2), rtol=1e-6, atol=1e-6)
    assert info1 == {}
    assert info2 == {}
