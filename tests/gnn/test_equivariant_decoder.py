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

from energnn.gnn.decoder import EquivariantDecoder, MLPEquivariantDecoder, ZeroEquivariantDecoder
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


def assert_single(*, decoder: EquivariantDecoder, out_structure: dict, seed: int, context: JaxGraph, coordinates: jax.Array):
    # TODO assert infos is in single dimension.
    rngs = jax.random.PRNGKey(seed)
    params_1 = decoder.init_with_structure(out_structure=out_structure, context=context, coordinates=coordinates, rngs=rngs)
    output_3, infos_3 = decoder.apply(params_1, context=context, coordinates=coordinates)
    output_4, infos_4 = decoder.apply(params_1, context=context, coordinates=coordinates, get_info=True)

    chex.assert_trees_all_equal(output_3, output_4)
    assert not infos_3

    return params_1, output_4, infos_4


def assert_batch(*, params: dict, decoder: EquivariantDecoder, context: JaxGraph, coordinates: jax.Array):

    def apply(params, context, coordinates, get_info):
        return decoder.apply(params, context=context, coordinates=coordinates, get_info=get_info)

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


def test_zero_invariant_decoder():
    decoder = ZeroEquivariantDecoder()
    params, output, infos = assert_single(
        decoder=decoder, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    output, infos = assert_batch(params=params, decoder=decoder, context=jax_context_batch, coordinates=coordinates_batch)


def test_mlp_invariant_decoder():
    decoder = MLPEquivariantDecoder(activation=nn.relu, hidden_size=[16], final_kernel_zero_init=False)
    params, output, infos = assert_single(
        decoder=decoder, out_structure=out_structure, seed=0, context=jax_context, coordinates=coordinates
    )
    output, infos = assert_batch(params=params, decoder=decoder, context=jax_context_batch, coordinates=coordinates_batch)
