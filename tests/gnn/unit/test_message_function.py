#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import unfreeze, freeze

from energnn.gnn.coupler.coupling_function import (
    EmptyLocalMessageFunction,
    EmptyRemoteMessageFunction,
    EmptySelfMessageFunction,
    IdentityLocalMessageFunction,
    IdentityRemoteMessageFunction,
    IdentitySelfMessageFunction,
    LocalMessageFunction,
    MLPSelfMessageFunction,
    RemoteMessageFunction,
    SelfMessageFunction,
    SumLocalMessageFunction,
)
from energnn.gnn.utils import gather, scatter_add
from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph, JaxEdge
from tests.utils import TestProblemLoader
from tests.gnn.utils import set_dense_layers_to_identity_or_zero

# deterministic
np.random.seed(0)

# Small fixture graphs from TestProblemLoader (reused)
n = 10
pb_loader = TestProblemLoader(
    dataset_size=8,
    n_batch=4,
    context_edge_params={
        "node": {"n_obj": n, "feature_list": ["a", "b"], "address_list": ["0"]},
        "edge": {"n_obj": n, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
    },
    oracle_edge_params={
        "node": {"n_obj": n, "feature_list": ["e"]},
        "edge": {"n_obj": n, "feature_list": ["f"]},
    },
    n_addr=n,
    shuffle=True,
)
pb_batch = next(iter(pb_loader))
context_batch, _ = pb_batch.get_context()
jax_context_batch = JaxGraph.from_numpy_graph(context_batch)
context = separate_graphs(context_batch)[0]
jax_context = JaxGraph.from_numpy_graph(context)
coordinates = jnp.array(np.random.uniform(size=(10, 7)))
coordinates_batch = jnp.array(np.random.uniform(size=(4, 10, 7)))


# Generic assert wrappers
def assert_function_output(*, function, seed: int, context: JaxGraph, coordinates: jax.Array):
    rngs = jax.random.PRNGKey(seed)
    params = function.init(rngs=rngs, context=context, coordinates=coordinates)
    out1, infos1 = function.apply(params, context=context, coordinates=coordinates, get_info=False)
    out2, infos2 = function.apply(params, context=context, coordinates=coordinates, get_info=True)
    chex.assert_trees_all_equal(out1, out2)
    assert infos1 == {}
    return params, out2, infos2


def assert_function_vmap_jit_output(*, params: dict, function, context: JaxGraph, coordinates: jax.Array):
    def apply_fn(params, ctx, coords, get_info):
        return function.apply(params, context=ctx, coordinates=coords, get_info=get_info)

    apply_vmap = jax.vmap(apply_fn, in_axes=[None, 0, 0, None], out_axes=0)
    out1, info1 = apply_vmap(params, context, coordinates, False)
    out2, info2 = apply_vmap(params, context, coordinates, True)
    apply_vmap_jit = jax.jit(apply_vmap)
    out3, info3 = apply_vmap_jit(params, context, coordinates, False)
    out4, info4 = apply_vmap_jit(params, context, coordinates, True)

    chex.assert_trees_all_equal(out1, out2, out3, out4)
    chex.assert_trees_all_equal(info2, info4)
    assert info1 == {}
    assert info3 == {}
    return out1, info1


def test_empty_self_message_function():
    fn = EmptySelfMessageFunction()
    params, out, infos = assert_function_output(function=fn, seed=0, context=jax_context, coordinates=coordinates)
    assert out.shape == (coordinates.shape[0], 0)
    out_b, info_b = assert_function_vmap_jit_output(params=params, function=fn, context=jax_context_batch, coordinates=coordinates_batch)


def test_identity_self_message_function():
    fn = IdentitySelfMessageFunction()
    params, out, infos = assert_function_output(function=fn, seed=0, context=jax_context, coordinates=coordinates)
    chex.assert_trees_all_equal(out, coordinates)
    out_b, info_b = assert_function_vmap_jit_output(params=params, function=fn, context=jax_context_batch, coordinates=coordinates_batch)
    chex.assert_trees_all_equal(out_b, coordinates_batch)


def test_empty_local_message_function():
    fn = EmptyLocalMessageFunction()
    params, out, infos = assert_function_output(function=fn, seed=0, context=jax_context, coordinates=coordinates)
    assert out.shape == (coordinates.shape[0], 0)
    out_b, info_b = assert_function_vmap_jit_output(params=params, function=fn, context=jax_context_batch, coordinates=coordinates_batch)


def test_identity_local_message_function():
    fn = IdentityLocalMessageFunction()
    params, out, infos = assert_function_output(function=fn, seed=0, context=jax_context, coordinates=coordinates)
    chex.assert_trees_all_equal(out, coordinates)
    out_b, info_b = assert_function_vmap_jit_output(params=params, function=fn, context=jax_context_batch, coordinates=coordinates_batch)
    chex.assert_trees_all_equal(out_b, coordinates_batch)


def test_empty_remote_message_function():
    fn = EmptyRemoteMessageFunction()
    params, out, infos = assert_function_output(function=fn, seed=0, context=jax_context, coordinates=coordinates)
    assert out.shape == (coordinates.shape[0], 0)
    out_b, info_b = assert_function_vmap_jit_output(params=params, function=fn, context=jax_context_batch, coordinates=coordinates_batch)


def test_identity_remote_message_function():
    fn = IdentityRemoteMessageFunction()
    params, out, infos = assert_function_output(function=fn, seed=0, context=jax_context, coordinates=coordinates)
    chex.assert_trees_all_equal(out, coordinates)
    out_b, info_b = assert_function_vmap_jit_output(params=params, function=fn, context=jax_context_batch, coordinates=coordinates_batch)
    chex.assert_trees_all_equal(out_b, coordinates_batch)


def test_mlp_self_message_numeric_identity_and_masking():
    """
    Make the self_mlp act as identity, and verify output == coordinates * mask
    """
    # choose out_size = coordinate dim to allow identity mapping
    d = coordinates.shape[1]
    fn = MLPSelfMessageFunction(hidden_size=[], out_size=d, activation=None, final_layer_activation=(lambda x: x))
    rng = jax.random.PRNGKey(111)
    params = fn.init(rngs=rng, context=jax_context, coordinates=coordinates)
    # patch self_mlp Dense to identity
    params_patched = set_dense_layers_to_identity_or_zero(params, "self_mlp", set_identity=True)
    out, infos = fn.apply(params_patched, context=jax_context, coordinates=coordinates, get_info=False)
    # mask
    mask = np.array(jax_context.non_fictitious_addresses)
    expected = np.array(coordinates) * mask[:, None]
    np.testing.assert_allclose(np.array(out), expected, rtol=0.0, atol=1e-6)


def test_gather_and_scatter_behaviour():
    """
    Unit tests for gather and scatter_add primitives used by SumLocalMessageFunction.
    """
    coords = jnp.array([[0.1, 0.2], [1.0, -1.0], [2.0, 3.0], [4.0, 5.0]])  # shape (4,2)
    # gather in-bounds
    idx = jnp.array([0, 2, 1])
    g = gather(coordinates=coords, addresses=idx)
    assert g.shape == (3, 2)
    np.testing.assert_allclose(np.array(g), np.array([coords[0], coords[2], coords[1]]))

    # gather out-of-bounds -> should return zeros for out-of-bounds entries (mode="drop", fill_value=0.0)
    idx_oob = jnp.array([0, 10, 2])
    g2 = gather(coordinates=coords, addresses=idx_oob)
    assert g2.shape == (3, 2)
    # row 1 corresponds to index 10 -> zeros
    assert np.allclose(np.array(g2[1]), np.zeros(2))

    # scatter_add: accumulate increments
    accumulator = jnp.zeros((4, 2))
    increments = jnp.array([[1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])
    addresses = jnp.array([0, 0, 2])  # two increments to index 0, one to index 2
    result = scatter_add(accumulator=accumulator, increment=increments, addresses=addresses)
    expected = np.zeros((4, 2))
    expected[0] = np.array([1.0 + 0.5, 0.0 + 0.5])
    expected[2] = np.array([2.0, -1.0])
    np.testing.assert_allclose(np.array(result), expected)


def test_sum_local_message_numeric_identity():
    """
    Small controlled test for SumLocalMessageFunction:
    - build a tiny graph with 3 edge objects and 4 addresses
    - each local MLP (for each port) is patched to identity
    - final_activation is identity, out_size set to input dim so identity is possible
    - expected accumulator computed in numpy and compared exactly
    """
    # Small controlled coordinates
    n_addr = 4
    d = 2  # coordinate dim
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])  # shape (4,2)

    # Build a tiny edge set with n_obj = 3
    # For each edge object i: addresses for port0 and port1
    addr0 = jnp.array([0, 1, 0])  # length 3
    addr1 = jnp.array([1, 2, 3])  # length 3
    n_obj = 3
    # no features (None) to keep input only coordinates concat -> input_dim = 2*d
    edge = JaxEdge(
        address_dict={"0": addr0, "1": addr1},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.ones((n_obj,)),
    )
    # Build context graph with single edge class "edge"
    small_context = JaxGraph(
        edges={"edge": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,  # not used much here
        current_shape=jax_context.current_shape,
    )

    # SumLocalMessageFunction: out_size must equal input_dim = 2*d
    out_size = 2 * d
    fn = SumLocalMessageFunction(out_size=out_size, hidden_size=[], activation=None, final_activation=(lambda x: x))

    rng = jax.random.PRNGKey(222)
    params = fn.init(rngs=rng, context=small_context, coordinates=coords)

    # Patch both port-mlps to identity; module names are like "edge-0-local_message_mlp" and "edge-1-local_message_mlp"
    params = set_dense_layers_to_identity_or_zero(params, "edge-0-local_message_mlp", set_identity=True)
    params = set_dense_layers_to_identity_or_zero(params, "edge-1-local_message_mlp", set_identity=True)

    # Apply function
    out, infos = fn.apply(params, context=small_context, coordinates=coords, get_info=False)
    # out shape should be (n_addr, out_size)
    assert out.shape == (n_addr, out_size)
    out_np = np.array(out)

    # Build expected result:
    # For each object i build input_i = concat(coords[addr0[i]], coords[addr1[i]])
    coords_np = np.array(coords)
    addr0_np = np.array(addr0).astype(int)
    addr1_np = np.array(addr1).astype(int)
    inputs = np.concatenate([coords_np[addr0_np], coords_np[addr1_np]], axis=1)  # (n_obj, 2*d)

    # For port "0": add inputs[i] to accumulator at index addr0[i]
    # For port "1": add inputs[i] to accumulator at index addr1[i]
    expected = np.zeros((n_addr, out_size), dtype=np.float32)
    for i in range(n_obj):
        expected[int(addr0_np[i])] += inputs[i]
    for i in range(n_obj):
        expected[int(addr1_np[i])] += inputs[i]

    np.testing.assert_allclose(out_np, expected, rtol=0.0, atol=1e-6)


def test_sum_local_message_handles_missing_feature_array_and_masking():
    """
    Ensure SumLocalMessageFunction tolerates edge.feature_array == None
    and respects edge.non_fictitious masking (zeroing messages for fictitious edges).
    """
    # reuse small example above but set one non_fictitious to 0
    n_addr = 4
    d = 2
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])
    addr0 = jnp.array([0, 1, 0])
    addr1 = jnp.array([1, 2, 3])
    n_obj = 3
    non_fict = jnp.array([1.0, 0.0, 1.0])  # middle object fictitious

    edge = JaxEdge(
        address_dict={"0": addr0, "1": addr1},
        feature_array=None,
        feature_names=None,
        non_fictitious=non_fict,
    )
    small_context = JaxGraph(
        edges={"edge": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    fn = SumLocalMessageFunction(out_size=2 * d, hidden_size=[], activation=None, final_activation=(lambda x: x))
    rng = jax.random.PRNGKey(333)
    params = fn.init(rngs=rng, context=small_context, coordinates=coords)
    params = set_dense_layers_to_identity_or_zero(params, "edge-0-local_message_mlp", set_identity=True)
    params = set_dense_layers_to_identity_or_zero(params, "edge-1-local_message_mlp", set_identity=True)

    out, _ = fn.apply(params, context=small_context, coordinates=coords, get_info=False)
    out_np = np.array(out)

    # compute expected ignoring contributions from i==1 (non_fictitious==0)
    coords_np = np.array(coords)
    addr0_np = np.array(addr0).astype(int)
    addr1_np = np.array(addr1).astype(int)
    inputs = np.concatenate([coords_np[addr0_np], coords_np[addr1_np]], axis=1)
    expected = np.zeros((n_addr, 2 * d), dtype=np.float32)
    for i in [0, 2]:  # only non_fictitious==1 entries contribute
        expected[int(addr0_np[i])] += inputs[i]
    for i in [0, 2]:
        expected[int(addr1_np[i])] += inputs[i]

    np.testing.assert_allclose(out_np, expected, rtol=0.0, atol=1e-6)
