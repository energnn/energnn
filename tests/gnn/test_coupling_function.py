#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# TODO
import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from energnn.gnn.coupler.coupling_function import CouplingFunction
from energnn.gnn.coupler.coupling_function import (
    IdentityLocalMessageFunction,
    IdentityRemoteMessageFunction,
    IdentitySelfMessageFunction,
)
from energnn.gnn.utils import MLP
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


def assert_single(*, function: CouplingFunction, seed: int, context: JaxGraph, coordinates: jax.Array):
    # TODO assert infos is in single dimension.
    rngs = jax.random.PRNGKey(seed)
    params_1 = function.init(context=context, coordinates=coordinates, rngs=rngs)
    output_3, infos_3 = function.apply(params_1, context=context, coordinates=coordinates)
    output_4, infos_4 = function.apply(params_1, context=context, coordinates=coordinates, get_info=True)

    chex.assert_trees_all_equal(output_3, output_4)

    return params_1, output_4, infos_4


def assert_batch(*, params: dict, function: CouplingFunction, context: JaxGraph, coordinates: jax.Array):

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
    return output_batch_1, infos_1


def test_coupling_function():
    coupling_function = CouplingFunction(
        phi=MLP(hidden_size=[8], activation=nn.relu, out_size=1),
        self_message_function=IdentitySelfMessageFunction(),
        local_message_function=IdentityLocalMessageFunction(),
        remote_message_function=IdentityRemoteMessageFunction(),
    )

    params, output, infos = assert_single(function=coupling_function, seed=0, context=jax_context, coordinates=coordinates)

    output, infos = assert_batch(
        params=params, function=coupling_function, context=jax_context_batch, coordinates=coordinates_batch
    )
