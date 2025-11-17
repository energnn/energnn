#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import chex
import flax.linen as nn
from flax.core.frozen_dict import unfreeze, freeze
import jax
import jax.numpy as jnp
import pytest

from energnn.gnn.decoder import EquivariantDecoder, MLPEquivariantDecoder, ZeroEquivariantDecoder
from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph, JaxEdge
from tests.utils import TestProblemLoader, compare_batched_graphs

# Prepare deterministic data and loader
np.random.seed(0)
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

# out_structure must be a FrozenDict (MLP decoder calls .unfreeze())
default_out_structure = freeze({"node": {"e": jnp.array(0)}, "edge": {"f": jnp.array(0)}})


def assert_decoder_vmap_jit_output(*, params: dict, decoder: EquivariantDecoder, context: JaxGraph, coordinates: jax.Array):
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
    assert infos_1 == {}
    assert infos_3 == {}
    assert infos_2 == infos_4


def _set_dense_layers_to_identity_or_zero(params, module_name, set_identity=True):
    """
    Patch params (Flax FrozenDict) such that Dense layers under `module_name` become:
      - identity kernel and zero bias if set_identity=True (square case),
      - zero kernel and zero bias if set_identity=False.

    Returns a new frozen params dict.
    """
    p = unfreeze(params)
    if "params" not in p:
        raise KeyError("'params' key not found in params dict")
    top = p["params"]
    if module_name not in top:
        # If module is nested because of naming, try to find a key that endswith module_name
        candidates = [k for k in top.keys() if k.endswith(module_name)]
        if candidates:
            module_key = candidates[0]
        else:
            raise KeyError(f"Module '{module_name}' not found in params structure: {list(top.keys())}")
    else:
        module_key = module_name
    mod = top[module_key]
    # iterate sublayers
    for layer_name, layer in list(mod.items()):
        if isinstance(layer, dict) and "kernel" in layer:
            k = np.array(layer["kernel"])
            b = np.array(layer.get("bias", np.zeros(k.shape[1], dtype=k.dtype)))
            in_dim, out_dim = k.shape
            if set_identity:
                new_k = np.zeros_like(k)
                for i in range(min(in_dim, out_dim)):
                    new_k[i, i] = 1.0
                new_b = np.zeros_like(b)
            else:
                new_k = np.zeros_like(k)
                new_b = np.zeros_like(b)
            mod[layer_name]["kernel"] = new_k.astype(np.float32)
            mod[layer_name]["bias"] = new_b.astype(np.float32)
    top[module_key] = mod
    p["params"] = top
    return freeze(p)


# ------------------------
# ZeroEquivariantDecoder tests
# ------------------------
def test_zero_equivariant_decoder_single_basic():
    decoder = ZeroEquivariantDecoder()
    rng = jax.random.PRNGKey(0)
    params = decoder.init_with_structure(rngs=rng, context=jax_context, coordinates=coordinates, out_structure=default_out_structure)
    out, info = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=True)

    # Basic checks: edges present only for keys in out_structure
    assert set(out.edges.keys()) == set(default_out_structure.keys())
    for key, feature_names in default_out_structure.items():
        e = out.edges[key]
        n_obj = jax_context.edges[key].feature_array.shape[0]
        # shape and zeros
        assert e.feature_array.shape == (n_obj, len(feature_names))
        np.testing.assert_allclose(np.array(e.feature_array), np.zeros_like(np.array(e.feature_array)))
        # feature_names preserved
        # In our default structure keys are simple; test only presence and length
        assert len(e.feature_names) == len(feature_names)
        assert e.address_dict is None
        # non_fictitious preserved
        np.testing.assert_allclose(np.array(e.non_fictitious), np.array(jax_context.edges[key].non_fictitious))
    # shapes
    assert set(out.true_shape.edges.keys()) == set(default_out_structure.keys())
    assert set(out.current_shape.edges.keys()) == set(default_out_structure.keys())
    # non_fictitious_addresses empty array
    assert np.array(out.non_fictitious_addresses).size == 0
    assert info == {}


def test_zero_equivariant_decoder_batch_vmap_jit():
    decoder = ZeroEquivariantDecoder()
    rng = jax.random.PRNGKey(1)
    params = decoder.init_with_structure(rngs=rng, context=jax_context, coordinates=coordinates, out_structure=default_out_structure)
    assert_decoder_vmap_jit_output(params=params, decoder=decoder, context=jax_context_batch, coordinates=coordinates_batch)


def test_zero_equivariant_decoder_requires_feature_array():
    # If an input edge has feature_array=None, ZeroEquivariantDecoder will try to access .shape -> error
    decoder = ZeroEquivariantDecoder()
    # Build graph with node edge having None features
    node = jax_context.edges["node"]
    edge = jax_context.edges["edge"]
    edge_with_none = JaxEdge(
        address_dict=node.address_dict,
        feature_array=None,
        feature_names=None,
        non_fictitious=node.non_fictitious,
    )
    custom_graph = JaxGraph(
        edges={"node": edge_with_none, "edge": edge},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    rng = jax.random.PRNGKey(2)
    # should raise AttributeError because code accesses feature_array.shape
    with pytest.raises(AttributeError):
        _ = decoder.init_with_structure(rngs=rng, context=custom_graph, coordinates=coordinates, out_structure=default_out_structure)


# ------------------------
# MLPEquivariantDecoder tests
# ------------------------
def test_mlp_equivariant_decoder_init_deterministic():
    decoder = MLPEquivariantDecoder(activation=nn.relu, hidden_size=[8], final_kernel_zero_init=False, out_structure=default_out_structure)
    rng = jax.random.PRNGKey(3)
    p1 = decoder.init_with_structure(rngs=rng, context=jax_context, coordinates=coordinates, out_structure=default_out_structure)
    p2 = decoder.init_with_structure(rngs=rng, context=jax_context, coordinates=coordinates, out_structure=default_out_structure)
    chex.assert_trees_all_equal(p1, p2)


def test_mlp_equivariant_decoder_single_shapes_and_masking():
    # Construct custom graph where some objects are fictitious (mask 0)
    node_edge = jax_context.edges["node"]
    edge_edge = jax_context.edges["edge"]

    # infer n_obj from current jax_context
    def n_obj_from(e):
        if e.feature_array is not None:
            return int(e.feature_array.shape[0])
        return int(np.array(e.non_fictitious).shape[0])

    n_node = n_obj_from(node_edge)
    n_edge = n_obj_from(edge_edge)

    # set first element fictitious for node edge to test masking
    node_nf = jnp.array(np.array(node_edge.non_fictitious))
    node_nf = node_nf.at[0].set(0)
    e1 = JaxEdge(address_dict=node_edge.address_dict, feature_array=jnp.ones((n_node, 2)), feature_names={"a": jnp.array(0), "b": jnp.array(1)}, non_fictitious=node_nf)
    e2 = JaxEdge(address_dict=edge_edge.address_dict, feature_array=jnp.ones((n_edge, 3)), feature_names={"c": jnp.array(0), "d": jnp.array(1), "e": jnp.array(2)}, non_fictitious=edge_edge.non_fictitious)

    custom_graph = JaxGraph(edges={"node": e1, "edge": e2}, non_fictitious_addresses=jax_context.non_fictitious_addresses, true_shape=jax_context.true_shape, current_shape=jax_context.current_shape)

    decoder = MLPEquivariantDecoder(activation=nn.relu, hidden_size=[4], final_kernel_zero_init=False, out_structure=default_out_structure)
    rng = jax.random.PRNGKey(4)
    params = decoder.init_with_structure(rngs=rng, context=custom_graph, coordinates=coordinates, out_structure=default_out_structure)
    out, info = decoder.apply(params, context=custom_graph, coordinates=coordinates, get_info=True)

    # shapes
    assert set(out.edges.keys()) == set(default_out_structure.keys())
    assert out.edges["node"].feature_array.shape == (n_node, len(default_out_structure["node"]))
    assert out.edges["edge"].feature_array.shape == (n_edge, len(default_out_structure["edge"]))

    # Masking: first row for node must be all zeros (we set non_fictitious[0]=0)
    node_out_np = np.array(out.edges["node"].feature_array)
    assert np.allclose(node_out_np[0], 0.0)
    # and at least one non-zero exists for other (unmasked) rows
    assert np.any(np.abs(node_out_np[1:]) > 1e-8)

    assert info == {}


def test_mlp_equivariant_decoder_plain_dict_out_structure_accepts_dict():
    # Passing a plain dict for out_structure should be accepted (no AttributeError).
    # We assert that the decoder runs and returns outputs with expected shapes.
    decoder = MLPEquivariantDecoder(activation=nn.relu, hidden_size=[4], final_kernel_zero_init=False)
    out_struct_plain = {"node": {"e": jnp.array(0)}}
    rng = jax.random.PRNGKey(5)

    # init_with_structure should not raise
    params = decoder.init_with_structure(rngs=rng, context=jax_context, coordinates=coordinates, out_structure=out_struct_plain)

    # apply should also run without raising and produce expected outputs
    out, info = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=True)

    # Verify output keys and shapes match the out_structure
    assert "node" in out.edges
    n_node = int(jax_context.edges["node"].feature_array.shape[0])
    assert out.edges["node"].feature_array.shape == (n_node, len(out_struct_plain["node"]))
    # info is empty dict per implementation
    assert info == {}


def test_mlp_equivariant_decoder_batch_vmap_jit():
    decoder = MLPEquivariantDecoder(activation=nn.relu, hidden_size=[8], final_kernel_zero_init=False, out_structure=default_out_structure)
    rng = jax.random.PRNGKey(6)
    params = decoder.init_with_structure(rngs=rng, context=jax_context, coordinates=coordinates, out_structure=default_out_structure)
    assert_decoder_vmap_jit_output(params=params, decoder=decoder, context=jax_context_batch, coordinates=coordinates_batch)


# -----------------------
# Numeric precise tests
# -----------------------
def test_mlp_equivariant_decoder_numeric_identity_node():
    """
    Make the node-MLP an identity mapping on the gathered coordinates.
    Expected output for each object e: coords[address_e] * non_fictitious_e
    """
    d = coordinates.shape[1]
    # build out_structure such that output size == coordinate dim (identity feasible)
    out_struct_node = freeze({"node": {f"o{i}": jnp.array(i) for i in range(d)}})
    decoder = MLPEquivariantDecoder(activation=None, hidden_size=[], final_kernel_zero_init=False)
    rng = jax.random.PRNGKey(100)
    params = decoder.init_with_structure(rngs=rng, context=jax_context, coordinates=coordinates, out_structure=out_struct_node)

    # Patch node module to act like identity
    params = _set_dense_layers_to_identity_or_zero(params, "node", set_identity=True)

    out_graph, _ = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=False)
    node_out = out_graph.edges["node"].feature_array  # shape (n_obj, d)
    node_edge = jax_context.edges["node"]
    addr = np.array(node_edge.address_dict["0"]).astype(int)
    coords = np.array(coordinates)
    nf = np.array(node_edge.non_fictitious).astype(float)
    expected = coords[addr] * nf[:, None]

    np.testing.assert_allclose(np.array(node_out), expected, rtol=0.0, atol=1e-6)


def test_mlp_equivariant_decoder_numeric_identity_edge():
    """
    Make the edge-MLP an identity mapping on [coords(addr0), coords(addr1), features].
    Expected output for each edge object: concat(coords[addr0], coords[addr1], feature_array) * non_fictitious
    """
    d = coordinates.shape[1]
    edge_feature_dim = int(jax_context.edges["edge"].feature_array.shape[1])
    input_dim = 2 * d + edge_feature_dim
    out_struct_edge = freeze({"edge": {f"o{i}": jnp.array(i) for i in range(input_dim)}})

    decoder = MLPEquivariantDecoder(activation=None, hidden_size=[], final_kernel_zero_init=False)
    rng = jax.random.PRNGKey(101)
    params = decoder.init_with_structure(rngs=rng, context=jax_context, coordinates=coordinates, out_structure=out_struct_edge)

    # Patch edge module to identity
    params = _set_dense_layers_to_identity_or_zero(params, "edge", set_identity=True)

    out_graph, _ = decoder.apply(params, context=jax_context, coordinates=coordinates, get_info=False)
    edge_out = out_graph.edges["edge"].feature_array  # shape (n_obj, input_dim)

    edge = jax_context.edges["edge"]
    addr0 = np.array(edge.address_dict["0"]).astype(int)
    addr1 = np.array(edge.address_dict["1"]).astype(int)
    coords = np.array(coordinates)
    feats = np.array(edge.feature_array)
    nf = np.array(edge.non_fictitious).astype(float)

    expected = np.concatenate([coords[addr0], coords[addr1], feats], axis=1) * nf[:, None]

    np.testing.assert_allclose(np.array(edge_out), expected, rtol=0.0, atol=1e-6)