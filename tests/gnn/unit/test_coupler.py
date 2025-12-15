#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import jax
import jax.numpy as jnp
import chex
import diffrax
import pytest

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


class DummyCoupling:
    """Simple coupling stub that returns a fixed param on init and whose apply returns zeros of matching shape."""
    def init(self, *, rngs, context, coordinates):
        return {"dummy": 1}

    def init_with_output(self, *, rngs, context, coordinates):
        # return ((output, info), params)
        zeros = jnp.zeros_like(coordinates)
        return ((zeros, {}), {"dummy": 1})

    def apply(self, params, context, coordinates, get_info=False):
        # returns zero update (same shape as coordinates) and info
        return jnp.zeros_like(coordinates), {"stub": jnp.array(1.0)}


class StubSolver:
    """Stub solver that records that it was called and returns a pre-defined coordinate/result."""
    def __init__(self, coords_out):
        self.coords_out = coords_out
        self.called = False

    def initialize_coordinates(self, *, context):
        # return zeros shaped array based on context
        return jnp.zeros([jnp.shape(context.non_fictitious_addresses)[0], self.coords_out.shape[1]])

    def solve(self, *, params, function, context, coordinates_init, get_info=False):
        self.called = True
        return self.coords_out, {"stub_solve": jnp.array(1.0)}

def assert_single(*, coupler: Coupler, seed: int, context: JaxGraph):
    rngs = jax.random.PRNGKey(seed)
    params_1 = coupler.init(context=context, rngs=rngs)
    output_3, infos_3 = coupler.apply(params_1, context=context)
    output_4, infos_4 = coupler.apply(params_1, context=context, get_info=True)

    chex.assert_trees_all_equal(output_3, output_4)

    return params_1, output_4, infos_4


def assert_batch(*, params: dict, coupler: Coupler, context: JaxGraph):

    def apply(params, context, get_info):
        return coupler.apply(params, context=context, get_info=get_info)

    apply_vmap = jax.vmap(apply, in_axes=[None, 0, None], out_axes=0)
    output_batch_1, infos_1 = apply_vmap(params, context, False)
    output_batch_2, infos_2 = apply_vmap(params, context, True)

    apply_vmap_jit = jax.jit(apply_vmap)
    output_batch_3, infos_3 = apply_vmap_jit(params, context, False)
    output_batch_4, infos_4 = apply_vmap_jit(params, context, True)

    chex.assert_trees_all_close(output_batch_1, output_batch_2, output_batch_3, output_batch_4, rtol=1e-4)
    chex.assert_trees_all_equal(infos_2, infos_4)
    return output_batch_1, infos_1


def test_coupler_init_returns_function_key_and_value():
    """Coupler.init must return a dict containing key 'FUNCTION' with the value returned by coupling.init."""
    dummy_coupling = DummyCoupling()
    dummy_solver = StubSolver(coords_out=jnp.ones((n, 4)) * 2.0)
    coupler = Coupler(coupling_function=dummy_coupling, solving_method=dummy_solver)

    rng = jax.random.PRNGKey(0)
    params = coupler.init(rngs=rng, context=jax_context)

    assert isinstance(params, dict)
    assert FUNCTION in params
    # our DummyCoupling.init returns {"dummy":1}
    assert params[FUNCTION] == {"dummy": 1}


def test_coupler_init_with_output_returns_coordinates_and_params():
    """init_with_output should produce tuple ((coords, {}), params) and coords match initialize_coordinates shape."""
    dummy_coupling = DummyCoupling()
    dummy_solver = StubSolver(coords_out=jnp.ones((n, 3)))
    coupler = Coupler(coupling_function=dummy_coupling, solving_method=dummy_solver)

    rng = jax.random.PRNGKey(1)
    (coords_out, info), params = coupler.init_with_output(rngs=rng, context=jax_context)

    # Structure checks
    assert isinstance(params, dict)
    assert FUNCTION in params
    assert isinstance(coords_out, jnp.ndarray)
    assert info == {}

    # coords_out shape equals solver.initialize_coordinates() shape (n addresses x latent dim)
    init_coords = dummy_solver.initialize_coordinates(context=jax_context)
    assert coords_out.shape == init_coords.shape


def test_coupler_apply_delegates_to_solver_and_returns_values():
    """Coupler.apply must call solving_method.solve and return its (coords, info)."""
    expected_coords = jnp.arange(n * 5).reshape((n, 5)).astype(jnp.float32)
    stub_solver = StubSolver(coords_out=expected_coords)
    dummy_coupling = DummyCoupling()
    coupler = Coupler(coupling_function=dummy_coupling, solving_method=stub_solver)

    params = {FUNCTION: {"dummy": 1}}
    coords, info = coupler.apply(params=params, context=jax_context)

    assert stub_solver.called is True
    # coords must equal expected
    assert coords.shape == expected_coords.shape
    np.testing.assert_allclose(np.array(coords), np.array(expected_coords), rtol=1e-6)
    assert info == {"stub_solve": jnp.array(1.0)}


def test_coupler_apply_raises_if_function_key_missing():
    """If params does not contain the FUNCTION key, a KeyError should occur when Coupler.apply tries to find it."""
    dummy_coupling = DummyCoupling()
    dummy_solver = StubSolver(coords_out=jnp.zeros((n, 2)))
    coupler = Coupler(coupling_function=dummy_coupling, solving_method=dummy_solver)

    with pytest.raises(KeyError):
        # pass params without "FUNCTION"
        coupler.apply(params={}, context=jax_context)
