#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from energnn.gnn.coupler.coupling_function import (
    AttentionLocalMessageFunction,
    EmptyLocalMessageFunction,
    EmptyRemoteMessageFunction,
    EmptySelfMessageFunction,
    IdentityLocalMessageFunction,
    IdentityRemoteMessageFunction,
    IdentitySelfMessageFunction,
    LinearAttentionRemoteMessageFunction,
    LocalMessageFunction,
    MLPSelfMessageFunction,
    RemoteMessageFunction,
    SelfMessageFunction,
    SumLocalMessageFunction,
)

# from tests.utils import build_context, build_context_batch, build_coordinates, build_coordinates_batch
from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph
from tests.utils import TestProblemLoader

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
out_structure = {"node": {"e": jnp.array(0)}, "edge": {"f": jnp.array(0)}}


def assert_single(
    *,
    function: SelfMessageFunction | LocalMessageFunction | RemoteMessageFunction,
    out_structure: dict,
    seed: int,
    context: JaxGraph,
    coordinates: jax.Array,
):
    rngs = jax.random.PRNGKey(seed)
    params_1 = function.init(context=context, coordinates=coordinates, rngs=rngs)
    output_3, infos_3 = function.apply(params_1, context=context, coordinates=coordinates)
    output_4, infos_4 = function.apply(params_1, context=context, coordinates=coordinates, get_info=True)

    chex.assert_trees_all_equal(output_3, output_4)
    assert not infos_3

    return params_1, output_4, infos_4


def assert_batch(
    *,
    params: dict,
    function: SelfMessageFunction | LocalMessageFunction | RemoteMessageFunction,
    context: JaxGraph,
    coordinates: jax.Array,
):

    def apply(params, context, coordinates, get_info):
        return function.apply(params, context=context, coordinates=coordinates, get_info=get_info)

    apply_vmap = jax.vmap(apply, in_axes=[None, 0, 0, None], out_axes=0)
    output_batch_1, infos_1 = apply_vmap(params, context, coordinates, False)
    output_batch_2, infos_2 = apply_vmap(params, context, coordinates, True)

    apply_vmap_jit = jax.jit(apply_vmap)
    output_batch_3, infos_3 = apply_vmap_jit(params, context, coordinates, False)
    output_batch_4, infos_4 = apply_vmap_jit(params, context, coordinates, True)

    chex.assert_trees_all_equal(output_batch_1, output_batch_2, output_batch_3, output_batch_4)
    chex.assert_trees_all_equal(infos_2, infos_4)
    assert not infos_1
    assert not infos_3
    return output_batch_1, infos_1


def test_empty_self_message_function():
    function = EmptySelfMessageFunction()
    params, output, infos = assert_single(
        function=function, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    output, infos = assert_batch(params=params, function=function, context=jax_context_batch, coordinates=coordinates_batch)


def test_identity_self_message_function():
    function = IdentitySelfMessageFunction()
    params, output, infos = assert_single(
        function=function, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    chex.assert_trees_all_equal(coordinates, output)
    output, infos = assert_batch(params=params, function=function, context=jax_context_batch, coordinates=coordinates_batch)
    chex.assert_trees_all_equal(coordinates_batch, output)


def test_mlp_self_message_function():
    function = MLPSelfMessageFunction(hidden_size=[8], activation=nn.relu, final_layer_activation=nn.tanh, out_size=16)
    params, output, infos = assert_single(
        function=function, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    output, infos = assert_batch(params=params, function=function, context=jax_context_batch, coordinates=coordinates_batch)


def test_empty_local_message_function():
    function = EmptyLocalMessageFunction()
    params, output, infos = assert_single(
        function=function, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    output, infos = assert_batch(params=params, function=function, context=jax_context_batch, coordinates=coordinates_batch)


def test_identity_local_message_function():
    function = IdentityLocalMessageFunction()
    params, output, infos = assert_single(
        function=function, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    chex.assert_trees_all_equal(coordinates, output)
    output, infos = assert_batch(params=params, function=function, context=jax_context_batch, coordinates=coordinates_batch)
    chex.assert_trees_all_equal(coordinates_batch, output)


def test_sum_local_message_function():
    function = SumLocalMessageFunction(hidden_size=[16], activation=nn.relu, final_activation=nn.tanh, out_size=16)
    params, output, infos = assert_single(
        function=function, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    output, infos = assert_batch(params=params, function=function, context=jax_context_batch, coordinates=coordinates_batch)


def test_attention_local_message_function():
    function = AttentionLocalMessageFunction(
        out_size=16,
        n_heads=2,
        value_hidden_size=[16],
        value_activation=nn.relu,
        value_out_size=8,
        score_hidden_size=[16],
        score_activation=nn.relu,
        psi_hidden_size=[16],
        psi_activation=nn.relu,
    )
    params, output, infos = assert_single(
        function=function, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    output, infos = assert_batch(params=params, function=function, context=jax_context_batch, coordinates=coordinates_batch)


def test_empty_remote_message_function():
    function = EmptyRemoteMessageFunction()
    params, output, infos = assert_single(
        function=function, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    output, infos = assert_batch(params=params, function=function, context=jax_context_batch, coordinates=coordinates_batch)


def test_identity_remote_message_function():
    function = IdentityRemoteMessageFunction()
    params, output, infos = assert_single(
        function=function, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    output, infos = assert_batch(params=params, function=function, context=jax_context_batch, coordinates=coordinates_batch)


def test_linear_attention_remote_message_function():
    function = LinearAttentionRemoteMessageFunction(
        out_size=16,
        n_heads=4,
        qk_size=4,
        q_hidden_size=[16],
        q_activation=nn.relu,
        k_hidden_size=[16],
        k_activation=nn.relu,
        v_hidden_size=[16],
        v_activation=nn.relu,
        v_out_size=8,
        psi_hidden_size=[16],
        psi_activation=nn.relu,
        kernel_name="elu",
    )
    params, output, infos = assert_single(
        function=function, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    output, infos = assert_batch(params=params, function=function, context=jax_context_batch, coordinates=coordinates_batch)
