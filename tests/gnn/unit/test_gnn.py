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
import pytest

from energnn.gnn import EquivariantGNN, InvariantGNN
from energnn.gnn.encoder import IdentityEncoder
from energnn.gnn.decoder import ZeroEquivariantDecoder, ZeroInvariantDecoder
from energnn.gnn.coupler.coupler import Coupler
from energnn.gnn.coupler.coupling_function import CouplingFunction
from energnn.gnn.coupler.solving_method import ZeroSolvingMethod, NeuralODESolvingMethod
from energnn.gnn.utils import MLP
from energnn.gnn.coupler.coupling_function import (
    IdentityLocalMessageFunction,
    IdentityRemoteMessageFunction,
    IdentitySelfMessageFunction,
)
from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph
from tests.utils import TestProblemLoader

# make deterministic
np.random.seed(0)

# small problem loader used across tests
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

out_structure = {"node": {"e": jnp.array(0)}, "edge": {"f": jnp.array(0)}}


def build_simple_coupler(latent_dim: int):
    """
    Build a Coupler with an MLP phi and identity message functions and a ZeroSolvingMethod.
    phi.out_size will be set by Coupler.init based on solving_method.initialize_coordinates.
    """
    phi = MLP(hidden_size=[8], out_size=latent_dim, activation=nn.relu)
    coupling_function = CouplingFunction(
        phi=phi,
        self_message_function=IdentitySelfMessageFunction(),
        local_message_function=IdentityLocalMessageFunction(),
        remote_message_function=IdentityRemoteMessageFunction(),
    )
    solving_method = ZeroSolvingMethod(latent_dimension=latent_dim)

    return Coupler(coupling_function=coupling_function, solving_method=solving_method)


def test_init_returns_expected_keys_equivariant():
    """init() should return a params dict with encoder, coupler and decoder keys."""
    latent_dim = 7
    coupler = build_simple_coupler(latent_dim=latent_dim)
    decoder = ZeroEquivariantDecoder()
    gnn = EquivariantGNN(encoder=IdentityEncoder(), coupler=coupler, decoder=decoder)

    rngs = jax.random.PRNGKey(0)
    params = gnn.init(rngs=rngs, context=jax_context, out_structure=out_structure)

    assert isinstance(params, dict)
    assert set(params.keys()) == {"encoder", "coupler", "decoder"}
    # each entry should be a pytree/dict-like
    for k in params:
        assert params[k] is not None


def test_apply_types_and_info_equivariant():
    """apply should return (JaxGraph, info_dict) and info contains encoder/coupler/decoder keys."""
    latent_dim = 6
    coupler = build_simple_coupler(latent_dim=latent_dim)
    decoder = ZeroEquivariantDecoder()
    gnn = EquivariantGNN(encoder=IdentityEncoder(), coupler=coupler, decoder=decoder)

    rngs = jax.random.PRNGKey(1)
    params = gnn.init(rngs=rngs, context=jax_context, out_structure=out_structure)

    out_graph, info = gnn.apply(params, context=jax_context, get_info=False)
    assert isinstance(out_graph, JaxGraph)
    assert isinstance(info, dict)
    out_graph2, info2 = gnn.apply(params, context=jax_context, get_info=True)
    assert set(info2.keys()) == {"encoder", "coupler", "decoder"}
    for k in info2:
        # info entries are dicts
        assert isinstance(info2[k], dict)


def test_init_returns_expected_keys_invariant():
    """InvariantGNN.init must return params dict with encoder, coupler and decoder keys."""
    latent_dim = 5
    coupler = build_simple_coupler(latent_dim=latent_dim)
    decoder = ZeroInvariantDecoder()
    gnn = InvariantGNN(encoder=IdentityEncoder(), coupler=coupler, decoder=decoder)

    rngs = jax.random.PRNGKey(2)
    params = gnn.init(rngs=rngs, context=jax_context, out_size=4)

    assert isinstance(params, dict)
    assert set(params.keys()) == {"encoder", "coupler", "decoder"}


def test_apply_types_and_info_invariant():
    """InvariantGNN.apply returns (jax.Array, info_dict) and nested info keys are present."""
    latent_dim = 4
    coupler = build_simple_coupler(latent_dim=latent_dim)
    decoder = ZeroInvariantDecoder()
    gnn = InvariantGNN(encoder=IdentityEncoder(), coupler=coupler, decoder=decoder)

    rngs = jax.random.PRNGKey(3)
    params = gnn.init(rngs=rngs, context=jax_context, out_size=8)

    out_vec, info = gnn.apply(params, context=jax_context, get_info=False)
    assert isinstance(info, dict)
    out_vec2, info2 = gnn.apply(params, context=jax_context, get_info=True)
    assert isinstance(out_vec2, jnp.ndarray)
    assert set(info2.keys()) == {"encoder", "coupler", "decoder"}
    for k in info2:
        assert isinstance(info2[k], dict)


def test_init_deterministic_given_same_rng():
    """Calling init twice with the same RNG must yield identical parameter pytrees."""
    latent_dim = 3
    coupler = build_simple_coupler(latent_dim=latent_dim)
    decoder = ZeroEquivariantDecoder()
    gnn = EquivariantGNN(encoder=IdentityEncoder(), coupler=coupler, decoder=decoder)

    rngs = jax.random.PRNGKey(5)
    params_a = gnn.init(rngs=rngs, context=jax_context, out_structure=out_structure)
    # re-init with same RNG
    params_b = gnn.init(rngs=rngs, context=jax_context, out_structure=out_structure)

    chex.assert_trees_all_equal(params_a, params_b)


def test_apply_raises_on_missing_params_key():
    """If params dict misses required key, apply should raise a KeyError (or similar)."""
    latent_dim = 4
    coupler = build_simple_coupler(latent_dim=latent_dim)
    decoder = ZeroEquivariantDecoder()
    gnn = EquivariantGNN(encoder=IdentityEncoder(), coupler=coupler, decoder=decoder)

    rngs = jax.random.PRNGKey(7)
    params = gnn.init(rngs=rngs, context=jax_context, out_structure=out_structure)

    # drop decoder params intentionally
    broken_params = dict(params)
    broken_params.pop("decoder", None)

    with pytest.raises(Exception):
        # Accept any exception type raised due to missing key
        gnn.apply(broken_params, context=jax_context, get_info=False)


def test_vmap_jit_consistency_equivariant():
    """The EquivariantGNN.apply must be vmappable and jittable across batch axis with consistent outputs."""
    latent_dim = 5
    coupler = build_simple_coupler(latent_dim=latent_dim)
    decoder = ZeroEquivariantDecoder()
    gnn = EquivariantGNN(encoder=IdentityEncoder(), coupler=coupler, decoder=decoder)

    rngs = jax.random.PRNGKey(9)
    params = gnn.init(rngs=rngs, context=jax_context, out_structure=out_structure)

    def apply_fn(params, ctx, get_info):
        return gnn.apply(params, context=ctx, get_info=get_info)

    apply_vmap = jax.vmap(apply_fn, in_axes=[None, 0, None], out_axes=0)
    out1, info1 = apply_vmap(params, jax_context_batch, False)
    out2, info2 = apply_vmap(params, jax_context_batch, True)

    apply_vmap_jit = jax.jit(apply_vmap)
    out3, info3 = apply_vmap_jit(params, jax_context_batch, False)
    out4, info4 = apply_vmap_jit(params, jax_context_batch, True)

    # Use allclose with small tolerance for numerical comparisons when necessary
    chex.assert_trees_all_close(out1, out2, out3, out4, rtol=1e-6, atol=1e-6)
    chex.assert_trees_all_close(info2, info4, rtol=1e-6, atol=1e-6)


def test_vmap_jit_consistency_invariant():
    """The InvariantGNN.apply must be vmappable and jittable across batch axis with consistent outputs."""
    latent_dim = 4
    coupler = build_simple_coupler(latent_dim=latent_dim)
    decoder = ZeroInvariantDecoder()
    gnn = InvariantGNN(encoder=IdentityEncoder(), coupler=coupler, decoder=decoder)

    rngs = jax.random.PRNGKey(11)
    params = gnn.init(rngs=rngs, context=jax_context, out_size=3)

    def apply_fn(params, ctx, get_info):
        return gnn.apply(params, context=ctx, get_info=get_info)

    apply_vmap = jax.vmap(apply_fn, in_axes=[None, 0, None], out_axes=0)
    out1, info1 = apply_vmap(params, jax_context_batch, False)
    out2, info2 = apply_vmap(params, jax_context_batch, True)

    apply_vmap_jit = jax.jit(apply_vmap)
    out3, info3 = apply_vmap_jit(params, jax_context_batch, False)
    out4, info4 = apply_vmap_jit(params, jax_context_batch, True)

    chex.assert_trees_all_close(out1, out2, out3, out4, rtol=1e-6, atol=1e-6)
    chex.assert_trees_all_equal(info2, info4)


def test_end_to_end_zero_pipeline_equivariant():
    """
    End-to-end check with IdentityEncoder -> simple Coupler -> ZeroEquivariantDecoder
    The ZeroEquivariantDecoder outputs should be zeros for requested out_structure.
    """
    latent_dim = 6
    coupler = build_simple_coupler(latent_dim=latent_dim)
    decoder = ZeroEquivariantDecoder()
    gnn = EquivariantGNN(encoder=IdentityEncoder(), coupler=coupler, decoder=decoder)

    rngs = jax.random.PRNGKey(13)
    params = gnn.init(rngs=rngs, context=jax_context, out_structure=out_structure)

    output_graph, info = gnn.apply(params, context=jax_context, get_info=True)
    # Output graph should contain only keys from out_structure
    assert set(output_graph.edges.keys()) == set(out_structure.keys())
    # All feature arrays should be zeros of expected shape
    for key, edge in output_graph.edges.items():
        assert edge.feature_array is not None
        # number of objects equals number in the input context (jax_context)
        n_obj = int(edge.feature_array.shape[0])
        expected_n = int(jax_context.edges[key].feature_array.shape[0])
        assert n_obj == expected_n
        # all zeros
        assert jnp.allclose(edge.feature_array, jnp.zeros_like(edge.feature_array))


def test_end_to_end_zero_pipeline_invariant():
    """
    End-to-end check for InvariantGNN with ZeroInvariantDecoder produces a zero vector of requested size.
    """
    latent_dim = 5
    coupler = build_simple_coupler(latent_dim=latent_dim)
    decoder = ZeroInvariantDecoder()
    gnn = InvariantGNN(encoder=IdentityEncoder(), coupler=coupler, decoder=decoder)

    rngs = jax.random.PRNGKey(17)
    params = gnn.init(rngs=rngs, context=jax_context, out_size=4)

    output_vec, info = gnn.apply(params, context=jax_context, get_info=True)
    assert isinstance(output_vec, jnp.ndarray)
    assert output_vec.shape == (4,)
    assert jnp.allclose(output_vec, jnp.zeros_like(output_vec))
