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
from flax.core.frozen_dict import freeze, unfreeze

from energnn.gnn import Encoder, IdentityEncoder, MLPEncoder
from energnn.graph import Graph, separate_graphs, collate_graphs
from energnn.graph.jax import JaxGraph, JaxEdge
from tests.utils import TestProblemLoader, compare_batched_graphs
from tests.gnn.utils import set_dense_layers_to_identity_or_zero

# make deterministic
np.random.seed(0)
jax_key = jax.random.PRNGKey(0)

# Prepare a small TestProblemLoader and example graphs
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


def _maybe_flax_init_with_output(encoder, *, rngs, context):
    """
    Handle both kinds of init_with_output:
    - For Flax Module: returns (output, params)
    - For IdentityEncoder (plain class): returns ((output, infos), params)
    This helper normalizes both cases to (output, params, infos)
    """
    res = encoder.init_with_output(rngs=rngs, context=context)
    # Flax style: (output, params)
    if (
        isinstance(res, tuple)
        and len(res) == 2
        and not (isinstance(res[0], tuple) and len(res[0]) == 2 and isinstance(res[1], dict))
    ):
        output, params = res
        infos = {}
    # IdentityEncoder style: ((output, infos), params)
    elif isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], tuple) and len(res[0]) == 2:
        (output, infos), params = res
    else:
        # Fallback: try to interpret
        try:
            output, params = res
            infos = {}
        except Exception:
            raise AssertionError("Unexpected init_with_output return structure: %r" % (res,))
    return output, params, infos


def _maybe_apply(encoder, params, context, get_info=False):
    """
    Call encoder.apply and normalize the return value to (output, infos).
    Works for both IdentityEncoder and Flax MLPEncoder.
    """
    res = encoder.apply(params, context=context, get_info=get_info)
    # If res is a tuple (output, infos)
    if isinstance(res, tuple) and len(res) == 2:
        output, infos = res
    else:
        # Maybe Flax returns output only (unlikely here) or nested
        output = res
        infos = {}
    return output, infos


def assert_encoder_vmap_jit_output(*, params: dict, encoder: Encoder, context: JaxGraph):
    def apply(params, context, get_info):
        return encoder.apply(params, context=context, get_info=get_info)

    apply_vmap = jax.vmap(apply, in_axes=[None, 0, None], out_axes=0)
    output_batch_1, infos_1 = apply_vmap(params, context, False)
    output_batch_2, infos_2 = apply_vmap(params, context, True)

    apply_vmap_jit = jax.jit(apply_vmap)
    output_batch_3, infos_3 = apply_vmap_jit(params, context, False)
    output_batch_4, infos_4 = apply_vmap_jit(params, context, True)

    chex.assert_trees_all_equal(output_batch_1, output_batch_2, output_batch_3, output_batch_4)
    chex.assert_trees_all_equal(infos_2, infos_4)
    assert infos_1 == {}
    assert infos_3 == {}
    assert infos_2 == infos_4


# Tests for IdentityEncoder
def test_identity_encoder_single_init_apply_roundtrip():
    encoder = IdentityEncoder()
    rngs = jax.random.PRNGKey(42)
    # init
    params_1 = encoder.init(rngs=rngs, context=jax_context)
    # init_with_output normalized
    output2, params2, infos2 = _maybe_flax_init_with_output(encoder, rngs=rngs, context=jax_context)
    # apply
    output3, infos3 = _maybe_apply(encoder, params_1, jax_context, get_info=False)
    output4, infos4 = _maybe_apply(encoder, params_1, jax_context, get_info=True)

    # params equality
    chex.assert_trees_all_equal(params_1, params2)
    # outputs must match the original context
    chex.assert_trees_all_equal(jax_context, output2, output3, output4)

    # infos are empty dicts
    assert infos2 == {}
    assert infos3 == {}
    assert infos4 == {}


def test_identity_encoder_batch_vmap_jit_consistency():
    encoder = IdentityEncoder()
    rngs = jax.random.PRNGKey(7)
    params = encoder.init(rngs=rngs, context=jax_context)
    assert_encoder_vmap_jit_output(params=params, encoder=encoder, context=jax_context_batch)


# Tests for MLPEncoder
@pytest.fixture(scope="module")
def mlp_encoder():
    return MLPEncoder(hidden_size=[8], out_size=4, activation=nn.relu)


def test_mlp_encoder_init_is_deterministic_and_returns_expected_params(mlp_encoder):
    encoder = mlp_encoder
    rng = jax.random.PRNGKey(1)
    # init twice with same RNG should produce identical params
    params_a = encoder.init(rng, context=jax_context)
    params_b = encoder.init(rng, context=jax_context)
    chex.assert_trees_all_equal(params_a, params_b)

    # init_with_output normalized
    out, params_c, infos = _maybe_flax_init_with_output(encoder, rngs=rng, context=jax_context)
    # params equal to init result
    chex.assert_trees_all_equal(params_a, params_c)
    # output should be a JaxGraph and infos empty dict
    assert isinstance(out, JaxGraph)
    assert infos == {}


def test_mlp_encoder_single_shapes_and_feature_names(mlp_encoder):
    encoder = mlp_encoder
    rng = jax.random.PRNGKey(2)
    params = encoder.init(rng, context=jax_context)
    out, infos = encoder.apply(params, context=jax_context, get_info=True)
    # Basic shape checks per edge
    for key, edge in out.edges.items():
        if edge.feature_array is not None:
            # encoded feature last dim should equal out_size
            assert edge.feature_array.shape[-1] == encoder.out_size
            # feature_names keys 'lat_0'...'lat_{out_size-1}' present
            expected_keys = {f"lat_{i}" for i in range(encoder.out_size)}
            assert set(edge.feature_names.keys()) == expected_keys
        else:
            # when input had None features, output should also have None
            assert edge.feature_names is None

    # infos should be dict (empty per implementation)
    assert isinstance(infos, dict)
    assert infos == {}


def test_mlp_encoder_handles_none_feature_array_gracefully():
    # Build a JaxGraph with one edge having feature_array=None
    edge_with_none = JaxEdge(
        address_dict=jax_context.edges["node"].address_dict,
        feature_array=None,
        feature_names=None,
        non_fictitious=jax_context.edges["node"].non_fictitious,
    )
    custom_graph = JaxGraph(
        edges={"node": edge_with_none, "edge": jax_context.edges["edge"]},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    encoder = MLPEncoder(hidden_size=[4], out_size=3, activation=nn.relu)
    rng = jax.random.PRNGKey(3)
    params = encoder.init(rng, context=custom_graph)
    out, infos = encoder.apply(params, context=custom_graph, get_info=False)

    # node edge had no features, must remain none
    assert out.edges["node"].feature_array is None
    assert out.edges["node"].feature_names is None
    # other edge processed normally
    assert out.edges["edge"].feature_array.shape[-1] == 3


def test_mlp_encoder_jit_and_vmap_compatibility(mlp_encoder):
    encoder = mlp_encoder
    rng = jax.random.PRNGKey(4)
    params = encoder.init(rng, context=jax_context)

    # Define a wrapper callable used by vmap/jit
    def apply_fn(params, ctx, get_info):
        return encoder.apply(params, context=ctx, get_info=get_info)

    # Vectorize across leading batch axis
    apply_vmap = jax.vmap(apply_fn, in_axes=[None, 0, None], out_axes=0)
    # JIT compile the vmapped function
    apply_vmap_jit = jax.jit(apply_vmap)

    out1, info1 = apply_vmap(params, jax_context_batch, False)
    out2, info2 = apply_vmap(params, jax_context_batch, True)
    out3, info3 = apply_vmap_jit(params, jax_context_batch, False)
    out4, info4 = apply_vmap_jit(params, jax_context_batch, True)

    # Compare batched JaxGraph outputs component-wise with numeric tolerance.
    # compare outputs (batched graphs)
    compare_batched_graphs(out1, out2, out3, out4, rtol=2e-3, atol=1e-6)

    # Infos: compare info dicts (should be identical or numerically close)
    # Here infos are empty dicts per implementation; assert equality or closeness
    assert info1 == {}
    assert info3 == {}
    # info2 and info4 should match
    assert info2 == info4


def test_mlp_encoder_multiple_edge_types_independent_processing():
    # Determine object counts from existing jax_context edges (feature_array shape)
    node_edge = jax_context.edges["node"]
    edge_edge = jax_context.edges["edge"]

    # If feature_array is None, fall back to non_fictitious length
    def _n_obj_from_jaxedge(e):
        if e.feature_array is not None:
            return int(e.feature_array.shape[0])
        if e.non_fictitious is not None:
            return int(jnp.array(e.non_fictitious).shape[0])
        raise ValueError("Cannot infer n_obj for JaxEdge")

    n_obj_node = _n_obj_from_jaxedge(node_edge)
    n_obj_edge = _n_obj_from_jaxedge(edge_edge)

    # Create JaxEdges with explicit feature_array shapes (2 and 3 features respectively)
    e1 = JaxEdge(
        address_dict=node_edge.address_dict,
        feature_array=jnp.ones((n_obj_node, 2), dtype=jnp.float32),
        feature_names={"a": jnp.array(0), "b": jnp.array(1)},
        non_fictitious=node_edge.non_fictitious,
    )
    e2 = JaxEdge(
        address_dict=edge_edge.address_dict,
        feature_array=jnp.ones((n_obj_edge, 3), dtype=jnp.float32),
        feature_names={"c": jnp.array(0), "d": jnp.array(1), "e": jnp.array(2)},
        non_fictitious=edge_edge.non_fictitious,
    )

    custom_graph = JaxGraph(
        edges={"A": e1, "B": e2},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    encoder = MLPEncoder(hidden_size=[6], out_size=5, activation=nn.relu)
    rng = jax.random.PRNGKey(5)
    params = encoder.init(rng, context=custom_graph)
    out, infos = encoder.apply(params, context=custom_graph, get_info=False)

    # Each edge has output features of length out_size
    assert out.edges["A"].feature_array.shape[-1] == 5
    assert out.edges["B"].feature_array.shape[-1] == 5
    # Feature names for both edges are the expected lat_* set
    expected_keys = {f"lat_{i}" for i in range(5)}
    assert set(out.edges["A"].feature_names.keys()) == expected_keys
    assert set(out.edges["B"].feature_names.keys()) == expected_keys


def test_mlp_encoder_numeric_identity_single_edge():
    """
    Build a graph whose 'node' features dimension equals the encoder out_size.
    Patch the MLP corresponding to 'node' so it becomes an identity mapping.
    Expect output feature_array for 'node' == input feature_array.
    """
    # Build a simple graph where node features dimension equals out_size=4
    node_edge = jax_context.edges["node"]
    n_obj_node = int(node_edge.feature_array.shape[0])
    d = 4  # choose d == out_size
    e_node = JaxEdge(
        address_dict=node_edge.address_dict,
        feature_array=jnp.linspace(0.0, 1.0, num=n_obj_node * d, dtype=jnp.float32).reshape((n_obj_node, d)),
        feature_names={f"f{i}": jnp.array(i) for i in range(d)},
        non_fictitious=node_edge.non_fictitious,
    )
    custom_graph = JaxGraph(
        edges={"node": e_node, "edge": jax_context.edges["edge"]},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    encoder = MLPEncoder(hidden_size=[], out_size=d, activation=None)
    rng = jax.random.PRNGKey(123)
    params = encoder.init(rng, context=custom_graph)

    # Patch node module to identity
    params = set_dense_layers_to_identity_or_zero(params, "node", set_identity=True)

    out, _ = encoder.apply(params, context=custom_graph, get_info=False)
    np.testing.assert_allclose(np.array(out.edges["node"].feature_array), np.array(e_node.feature_array), rtol=0.0, atol=1e-6)


def test_mlp_encoder_numeric_identity_multiple_edges():
    """
    Build graph with two edges 'A' and 'B', each feature dim == out_size.
    Patch both MLPs to identity and check outputs equal inputs.
    """
    node_edge = jax_context.edges["node"]
    edge_edge = jax_context.edges["edge"]

    n_obj_node = int(node_edge.feature_array.shape[0])
    n_obj_edge = int(edge_edge.feature_array.shape[0])
    d = 3  # out_size

    eA = JaxEdge(
        address_dict=node_edge.address_dict,
        feature_array=jnp.arange(n_obj_node * d, dtype=jnp.float32).reshape((n_obj_node, d)) * 0.1,
        feature_names={f"fa{i}": jnp.array(i) for i in range(d)},
        non_fictitious=node_edge.non_fictitious,
    )
    eB = JaxEdge(
        address_dict=edge_edge.address_dict,
        feature_array=jnp.arange(n_obj_edge * d, dtype=jnp.float32).reshape((n_obj_edge, d)) * 0.2,
        feature_names={f"fb{i}": jnp.array(i) for i in range(d)},
        non_fictitious=edge_edge.non_fictitious,
    )

    custom_graph = JaxGraph(
        edges={"A": eA, "B": eB},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    encoder = MLPEncoder(hidden_size=[], out_size=d, activation=None)
    rng = jax.random.PRNGKey(124)
    params = encoder.init(rng, context=custom_graph)

    # Patch modules to identity
    params = set_dense_layers_to_identity_or_zero(params, "A", set_identity=True)
    params = set_dense_layers_to_identity_or_zero(params, "B", set_identity=True)

    out, _ = encoder.apply(params, context=custom_graph, get_info=False)
    np.testing.assert_allclose(np.array(out.edges["A"].feature_array), np.array(eA.feature_array), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.array(out.edges["B"].feature_array), np.array(eB.feature_array), rtol=0.0, atol=1e-6)
