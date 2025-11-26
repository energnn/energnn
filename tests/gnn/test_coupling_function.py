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
from flax.core.frozen_dict import freeze, unfreeze

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

# deterministic randomness for tests
np.random.seed(0)

# Build small test problem loader and sample contexts
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


def _set_phi_kernel_to_average(params: dict, coord_dim: int) -> dict:
    """
    Modify params['phi'] so that the phi MLP (single dense) computes the average
    of the three concatenated coordinate blocks:
        out_j = (in_j + in_{d+j} + in_{2d+j}) / 3
    This helper locates the Dense layer kernel/bias dict and replaces them.
    Returns a NEW params dict (frozen).
    """
    p = unfreeze(params)
    if "phi" not in p:
        raise KeyError("'phi' key not present in params")
    phi_obj = unfreeze(p["phi"])
    # typical Flax params structure: {'params': {'Dense_0': {'kernel': ..., 'bias': ...}}}
    phi_params = phi_obj.get("params", phi_obj)
    # find a layer dict that contains 'kernel'
    found = False
    for layer_name, layer_val in list(phi_params.items()):
        if isinstance(layer_val, dict) and "kernel" in layer_val:
            kernel = np.array(layer_val["kernel"])
            bias = np.array(layer_val.get("bias", np.zeros(kernel.shape[1], dtype=kernel.dtype)))
            in_dim, out_dim = kernel.shape
            d = coord_dim
            # Expect in_dim == 3 * d and out_dim == d (but we guard)
            if in_dim < 3 * d or out_dim < d:
                # still try to adapt to minimal shared dims
                raise ValueError(f"Unexpected kernel shape {kernel.shape} for coord_dim {coord_dim}")
            new_k = np.zeros_like(kernel, dtype=np.float32)
            # set average: positions (j, j), (d+j, j), (2d+j, j) to 1/3
            for j in range(d):
                new_k[j, j] = 1.0 / 3.0
                new_k[d + j, j] = 1.0 / 3.0
                new_k[2 * d + j, j] = 1.0 / 3.0
            new_b = np.zeros_like(bias, dtype=np.float32)
            phi_params[layer_name]["kernel"] = new_k
            phi_params[layer_name]["bias"] = new_b
            found = True
            break
    if not found:
        raise KeyError("No Dense layer with 'kernel' found inside params['phi']")
    # reassign
    if "params" in phi_obj:
        phi_obj["params"] = phi_params
    else:
        phi_obj = phi_params
    p["phi"] = freeze(phi_obj)
    return freeze(p)


def assert_single(*, function: CouplingFunction, seed: int, context: JaxGraph, coordinates: jax.Array):
    rngs = jax.random.PRNGKey(seed)
    params_1 = function.init(context=context, coordinates=coordinates, rngs=rngs)
    output_3, infos_3 = function.apply(params_1, context=context, coordinates=coordinates)
    output_4, infos_4 = function.apply(params_1, context=context, coordinates=coordinates, get_info=True)

    chex.assert_trees_all_equal(output_3, output_4)
    # When get_info=True, infos should contain 'self','local','remote' (possibly empty dicts)
    assert set(infos_4.keys()) == {"self", "local", "remote"}
    return params_1, output_4, infos_4


def assert_batch(*, params: dict, function: CouplingFunction, context: JaxGraph, coordinates: jax.Array):
    def apply_fn(params, ctx, coords, get_info):
        return function.apply(params, context=ctx, coordinates=coords, get_info=get_info)

    apply_vmap = jax.vmap(apply_fn, in_axes=[None, 0, 0, None], out_axes=0)
    output_batch_1, infos_1 = apply_vmap(params, context, coordinates, False)
    output_batch_2, infos_2 = apply_vmap(params, context, coordinates, True)

    apply_vmap_jit = jax.jit(apply_vmap)
    output_batch_3, infos_3 = apply_vmap_jit(params, context, coordinates, False)
    output_batch_4, infos_4 = apply_vmap_jit(params, context, coordinates, True)

    chex.assert_trees_all_equal(output_batch_1, output_batch_2, output_batch_3, output_batch_4)
    chex.assert_trees_all_equal(infos_2, infos_4)
    return output_batch_1, infos_1


# -------------------------
# Tests
# -------------------------
def test_phi_out_size_and_param_keys_set_on_init():
    """phi.out_size should be set to coordinates width and init returns expected keys"""
    d = coordinates.shape[1]
    coupling = CouplingFunction(
        phi=MLP(hidden_size=[8], activation=nn.relu, out_size=1),
        self_message_function=IdentitySelfMessageFunction(),
        local_message_function=IdentityLocalMessageFunction(),
        remote_message_function=IdentityRemoteMessageFunction(),
    )
    rng = jax.random.PRNGKey(0)
    params = coupling.init(rngs=rng, context=jax_context, coordinates=coordinates)
    # coupling.phi should have been updated
    assert coupling.phi.out_size == d
    # params should contain required keys
    assert set(params.keys()) == {"self", "local", "remote", "phi"}


def test_init_with_output_masks_fictitious_entries():
    """init_with_output should return an output masked by non_fictitious_addresses"""
    # create a context where some addresses are fictitious
    mask = np.array(jax_context.non_fictitious_addresses)
    # flip a few entries to zero
    mask2 = mask.copy()
    mask2[2] = 0.0
    mask2[5] = 0.0
    custom_context = JaxGraph(
        edges=jax_context.edges,
        non_fictitious_addresses=jnp.array(mask2),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )
    coupling = CouplingFunction(
        phi=MLP(hidden_size=[4], activation=nn.relu, out_size=1),
        self_message_function=IdentitySelfMessageFunction(),
        local_message_function=IdentityLocalMessageFunction(),
        remote_message_function=IdentityRemoteMessageFunction(),
    )
    rng = jax.random.PRNGKey(1)
    out, params = coupling.init_with_output(rngs=rng, context=custom_context, coordinates=coordinates)
    out_np = np.array(out)
    # lines 2 and 5 must be zeros because mask2 are 0
    assert np.allclose(out_np[2], 0.0)
    assert np.allclose(out_np[5], 0.0)


def test_apply_returns_infos_keys_and_masks():
    """apply should return infos with 'self','local','remote' and apply mask to output"""
    coupling = CouplingFunction(
        phi=MLP(hidden_size=[8], activation=nn.relu, out_size=1),
        self_message_function=IdentitySelfMessageFunction(),
        local_message_function=IdentityLocalMessageFunction(),
        remote_message_function=IdentityRemoteMessageFunction(),
    )
    rng = jax.random.PRNGKey(2)
    params = coupling.init(rngs=rng, context=jax_context, coordinates=coordinates)
    out, infos = coupling.apply(params, context=jax_context, coordinates=coordinates, get_info=True)
    assert set(infos.keys()) == {"self", "local", "remote"}
    # check mask effect: choose an index and set mask to zero then expect zero output
    mask3 = np.array(jax_context.non_fictitious_addresses)
    mask3[0] = 0.0
    ctx3 = JaxGraph(edges=jax_context.edges, non_fictitious_addresses=jnp.array(mask3), true_shape=jax_context.true_shape, current_shape=jax_context.current_shape)
    out3, _ = coupling.apply(params, context=ctx3, coordinates=coordinates, get_info=False)
    assert np.allclose(np.array(out3[0]), 0.0)


def test_init_is_deterministic_given_same_rng():
    coupling = CouplingFunction(
        phi=MLP(hidden_size=[4], activation=nn.relu, out_size=1),
        self_message_function=IdentitySelfMessageFunction(),
        local_message_function=IdentityLocalMessageFunction(),
        remote_message_function=IdentityRemoteMessageFunction(),
    )
    rng = jax.random.PRNGKey(7)
    p1 = coupling.init(rngs=rng, context=jax_context, coordinates=coordinates)
    p2 = coupling.init(rngs=rng, context=jax_context, coordinates=coordinates)
    chex.assert_trees_all_equal(p1, p2)


def test_apply_numeric_with_identity_messages_and_average_phi():
    """
    Build a coupling with identity message functions. Patch phi so that
    phi(concat(coords,coords,coords)) == coords (average of the three blocks).
    Then apply and compare to expected coords masked.
    """
    d = coordinates.shape[1]
    coupling = CouplingFunction(
        phi=MLP(hidden_size=[], activation=None, out_size=1),  # out_size will be overwritten by init
        self_message_function=IdentitySelfMessageFunction(),
        local_message_function=IdentityLocalMessageFunction(),
        remote_message_function=IdentityRemoteMessageFunction(),
    )
    rng = jax.random.PRNGKey(11)
    params = coupling.init(rngs=rng, context=jax_context, coordinates=coordinates)
    # patch phi params to average the three coord blocks
    params_patched = _set_phi_kernel_to_average(params, coord_dim=d)
    out, infos = coupling.apply(params_patched, context=jax_context, coordinates=coordinates, get_info=True)
    # expected is equal to coordinates multiplied by mask
    mask = np.array(jax_context.non_fictitious_addresses)
    expected = np.array(coordinates) * mask[:, None]
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-6, atol=1e-6)


def test_apply_vmap_and_jit_compatibility():
    """Check that apply can be vmapped/jitted over batch dimension and outputs remain consistent"""
    coupling = CouplingFunction(
        phi=MLP(hidden_size=[4], activation=nn.relu, out_size=1),
        self_message_function=IdentitySelfMessageFunction(),
        local_message_function=IdentityLocalMessageFunction(),
        remote_message_function=IdentityRemoteMessageFunction(),
    )
    rng = jax.random.PRNGKey(13)
    params = coupling.init(rngs=rng, context=jax_context, coordinates=coordinates)

    def apply_fn(params, ctx, coords, get_info):
        return coupling.apply(params, context=ctx, coordinates=coords, get_info=get_info)

    apply_vmap = jax.vmap(apply_fn, in_axes=[None, 0, 0, None], out_axes=0)
    out1, info1 = apply_vmap(params, jax_context_batch, coordinates_batch, False)
    out2, info2 = apply_vmap(params, jax_context_batch, coordinates_batch, True)

    apply_vmap_jit = jax.jit(apply_vmap)
    out3, info3 = apply_vmap_jit(params, jax_context_batch, coordinates_batch, False)
    out4, info4 = apply_vmap_jit(params, jax_context_batch, coordinates_batch, True)

    chex.assert_trees_all_close(out1, out2, out3, out4, rtol=2e-3, atol=1e-5)
    chex.assert_trees_all_close(info2, info4, rtol=2e-3, atol=1e-6)
