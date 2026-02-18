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
from flax import nnx

from energnn.model.coupler.neural_ode.recurrent import RecurrentCoupler
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


def make_coupler(*, phi=None, message_functions, n_steps: int = 10):
    """Convenience factory for RecurrentCoupler."""
    if phi is None:
        # default tiny MLP if none provided
        phi = MLP(in_size=1, hidden_sizes=[], out_size=1, activation=None, seed=0)
    return RecurrentCoupler(phi=phi, message_functions=message_functions, n_steps=n_steps)


@pytest.mark.parametrize("n_steps", [1, 2, 5, 10, 50])
def test_recurrentcoupler_numeric_constant_message_basic(n_steps):
    """
    If phi is identity and message function returns a constant vector C,
    Euler updates with dt=1/n_steps repeated n_steps times yield h = C.
    """
    C = jnp.array([0.5, -0.3], dtype=jnp.float32)
    latent_dim = int(C.shape[0])
    mf = [ConstantMessage(C)]
    coupler = make_coupler(phi=IdentityPhi(out_size=latent_dim), message_functions=mf, n_steps=n_steps)

    out, info = coupler(graph=jax_context, get_info=False)

    # Check shapes and types
    n_addr = int(jax_context.non_fictitious_addresses.shape[0])
    assert out.shape == (n_addr, latent_dim)
    assert isinstance(out, jnp.ndarray)
    assert info == {}

    # Expected: every row equals C
    expected = jnp.tile(C[None, :], (n_addr, 1))
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-6, atol=1e-6)


def test_recurrentcoupler_multiple_message_functions_and_concatenation_shape():
    """
    Ensure multiple message functions are correctly concatenated before phi.
    Concatenated dimension (2+1=3) must match phi.in_size and out has shape (n_addr, 3) for identity phi.
    """
    m1 = ConstantMessage(jnp.array([1.0, 2.0], dtype=jnp.float32))
    m2 = ConstantMessage(jnp.array([3.0], dtype=jnp.float32))
    mf_list = [m1, m2]

    coupler = make_coupler(phi=IdentityPhi(out_size=3), message_functions=mf_list, n_steps=5)
    out, _ = coupler(graph=jax_context, get_info=False)

    n_addr = int(jax_context.non_fictitious_addresses.shape[0])
    assert out.shape == (n_addr, 3)


def test_recurrentcoupler_vmap_jit_compatibility():
    """
    Ensure the coupler is compatible with JAX transformations (vmap, jit).
    """
    C = jnp.array([0.4, -0.1], dtype=jnp.float32)
    latent_dim = int(C.shape[0])
    mf = [ConstantMessage(C)]
    coupler = make_coupler(phi=IdentityPhi(out_size=latent_dim), message_functions=mf, n_steps=20)

    # Vectorize and JIT
    apply_vmap = jax.vmap(lambda g, gi: coupler(graph=g, get_info=gi), in_axes=(0, None), out_axes=0)
    apply_vmap_jit = jax.jit(apply_vmap)

    # Call once to initialize any internal state (identity phi has none)
    _ = coupler(graph=jax_context, get_info=False)

    out1, info1 = apply_vmap(jax_context_batch, False)
    out2, info2 = apply_vmap_jit(jax_context_batch, False)

    # Compare results
    np.testing.assert_allclose(np.array(out1), np.array(out2), rtol=1e-6, atol=1e-6)
    assert info1 == {}
    assert info2 == {}


def test_recurrentcoupler_zero_steps_raises():
    """n_steps=0 leads to division by zero in initialization or call -> should raise."""
    C = jnp.array([0.1], dtype=jnp.float32)
    mf = [ConstantMessage(C)]
    with pytest.raises(Exception):
        _ = make_coupler(phi=IdentityPhi(out_size=1), message_functions=mf, n_steps=0)


def test_recurrentcoupler_init_deterministic_with_same_seed():
    """
    Two couplers with phi MLPs initialized using the same seed must produce identical outputs.
    """
    C = jnp.array([0.2, 0.3, -0.1], dtype=jnp.float32)
    in_dim = int(C.shape[0])
    # Same-seed MLPs
    phi1 = MLP(in_size=in_dim, hidden_sizes=[4], out_size=3, activation=None, seed=7)
    phi2 = MLP(in_size=in_dim, hidden_sizes=[4], out_size=3, activation=None, seed=7)
    mf = [ConstantMessage(C)]

    rc1 = make_coupler(phi=phi1, message_functions=mf, n_steps=5)
    rc2 = make_coupler(phi=phi2, message_functions=mf, n_steps=5)

    out1, info1 = rc1(graph=jax_context, get_info=False)
    out2, info2 = rc2(graph=jax_context, get_info=False)

    np.testing.assert_allclose(np.array(out1), np.array(out2), rtol=1e-6, atol=1e-6)
    assert info1 == {}
    assert info2 == {}
