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

from energnn.gnn import Encoder, IdentityEncoder, MLPEncoder
from energnn.graph import Graph, separate_graphs
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


def assert_single(*, encoder: Encoder, seed: int, context: Graph):
    # TODO assert infos is in single dimension.
    jax_context = JaxGraph.from_numpy_graph(context)
    rngs = jax.random.PRNGKey(seed)
    params_1 = encoder.init(context=jax_context, rngs=rngs)
    (output_2, infos_2), params_2 = encoder.init_with_output(context=jax_context, rngs=rngs)
    output_3, infos_3 = encoder.apply(params_1, context=jax_context)
    output_4, infos_4 = encoder.apply(params_1, context=jax_context, get_info=True)

    chex.assert_trees_all_equal(params_1, params_2)
    chex.assert_trees_all_equal(output_2, output_3, output_4)
    assert not infos_2
    assert not infos_3

    return params_1, output_4, infos_4


def assert_batch(*, params: dict, encoder: Encoder, context_batch: Graph):

    jax_context_batch = JaxGraph.from_numpy_graph(context_batch)

    def apply(params, context, get_info):
        return encoder.apply(params, context=context, get_info=get_info)

    apply_vmap = jax.vmap(apply, in_axes=[None, 0, None], out_axes=0)
    output_batch_1, infos_1 = apply_vmap(params, jax_context_batch, False)
    output_batch_2, infos_2 = apply_vmap(params, jax_context_batch, True)

    apply_vmap_jit = jax.jit(apply_vmap)
    output_batch_3, infos_3 = apply_vmap_jit(params, jax_context_batch, False)
    output_batch_4, infos_4 = apply_vmap_jit(params, jax_context_batch, True)

    chex.assert_trees_all_equal(output_batch_1, output_batch_2, output_batch_3, output_batch_4)
    chex.assert_trees_all_equal(infos_2, infos_4)
    assert not infos_1
    assert not infos_3
    return output_batch_1, infos_1


def test_identity_encoder():
    encoder = IdentityEncoder()
    params, output, infos = assert_single(encoder=encoder, seed=0, context=jax_context)
    chex.assert_trees_all_equal(jax_context, output)
    output, infos = assert_batch(params=params, encoder=encoder, context_batch=jax_context_batch)
    chex.assert_trees_all_equal(jax_context_batch, output)


def test_mlp_encoder():
    encoder = MLPEncoder(hidden_size=[8], out_size=4, activation=nn.relu)
    params, output, infos = assert_single(encoder=encoder, seed=0, context=jax_context)
    output, infos = assert_batch(params=params, encoder=encoder, context_batch=jax_context_batch)
