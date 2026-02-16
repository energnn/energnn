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
import pytest

from energnn.model.encoder.encoder import IdentityEncoder
from energnn.model.encoder.mlp_encoder import MLPEncoder
from energnn.graph import JaxGraph, JaxEdge, GraphStructure, EdgeStructure
from tests.utils import TestProblemLoader, compare_batched_graphs

# make deterministic
np.random.seed(0)

# Prepare a small TestProblemLoader and example graphs
pb_loader = TestProblemLoader(seed=0)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)


# -------------------------
# IdentityEncoder tests
# -------------------------
def test_identity_encoder_single_roundtrip():
    enc = IdentityEncoder()
    out, info = enc(graph=jax_context, get_info=True)
    # should return same graph and empty info
    chex.assert_trees_all_equal(out, jax_context)
    assert info == {}


def test_identity_encoder_batch_vmap_jit_consistency():
    enc = IdentityEncoder()

    def apply_fn(graphs, get_info):
        return enc(graph=graphs, get_info=get_info)

    apply_vmap = jax.vmap(lambda g, gi: enc(graph=g, get_info=gi), in_axes=(0, None), out_axes=0)
    out1, info1 = apply_vmap(jax_context_batch, False)
    out2, info2 = apply_vmap(jax_context_batch, True)
    out3, info3 = jax.jit(apply_vmap)(jax_context_batch, False)
    out4, info4 = jax.jit(apply_vmap)(jax_context_batch, True)

    # compare shapes and structural equality
    chex.assert_trees_all_equal(out1, out2, out3, out4)
    assert info1 == {}
    assert info3 == {}
    assert info2 == info4


# -------------------------
# MLPEncoder tests
# -------------------------
@pytest.fixture(scope="module")
def mlp_encoder():
    # give an explicit seed for deterministic behavior in tests
    return MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[8], out_size=4, activation=None, seed=0)


def test_mlp_encoder_init_is_deterministic_and_returns_graph():
    # Two encoders instantiated with the same seed should produce same outputs
    enc1 = MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[8], out_size=4, activation=None, seed=1)
    enc2 = MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[8], out_size=4, activation=None, seed=1)

    out1, info1 = enc1(graph=jax_context, get_info=False)
    out2, info2 = enc2(graph=jax_context, get_info=False)

    chex.assert_trees_all_equal(out1, out2)
    assert info1 == {}
    assert info2 == {}


def test_mlp_encoder_single_shapes_and_feature_names():
    enc = MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[8], out_size=4, activation=None, seed=2)

    out, infos = enc(graph=jax_context, get_info=True)

    # Basic shape checks per edge
    for key, edge in out.edges.items():
        if edge.feature_array is not None:
            assert edge.feature_array.shape[-1] == enc.out_size
            # feature_names should contain lat_0 ... lat_{out_size-1}
            expected_keys = {f"lat_{i}" for i in range(enc.out_size)}
            assert set(edge.feature_names.keys()) == expected_keys
        else:
            assert edge.feature_names is None

    assert infos == {}


def test_mlp_encoder_handles_none_feature_array_gracefully():
    # Build a JaxGraph with one edge having feature_array=None
    edge_with_none = JaxEdge(
        address_dict=jax_context.edges["arrow"].address_dict,
        feature_array=None,
        feature_names=None,
        non_fictitious=jax_context.edges["arrow"].non_fictitious,
    )
    custom_graph = JaxGraph(
        edges={"arrow": edge_with_none, "source": jax_context.edges["source"]},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    enc = MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[4], out_size=3, activation=None, seed=3)
    out, infos = enc(graph=custom_graph, get_info=False)

    assert out.edges["arrow"].feature_array is None
    assert out.edges["arrow"].feature_names is None
    assert out.edges["source"].feature_array.shape[-1] == 3


def test_mlp_encoder_jit_and_vmap_compatibility(mlp_encoder):
    enc = mlp_encoder
    # One call to ensure everything is fine (also consumes rngs safely)
    _ = enc(graph=jax_context, get_info=False)

    # Vectorize across leading batch axis: vmapping the callable that accepts a single graph
    apply_vmap = jax.vmap(lambda g, gi: enc(graph=g, get_info=gi), in_axes=(0, None), out_axes=0)

    out1, info1 = apply_vmap(jax_context_batch, False)
    out2, info2 = apply_vmap(jax_context_batch, True)
    out3, info3 = jax.jit(apply_vmap)(jax_context_batch, False)
    out4, info4 = jax.jit(apply_vmap)(jax_context_batch, True)

    # compare batched outputs numerically / structurally (helper from tests.utils)
    compare_batched_graphs(out1, out2, out3, out4, rtol=2e-3, atol=1e-6)

    assert info1 == {}
    assert info3 == {}
    assert info2 == info4


def test_mlp_encoder_multiple_edge_types_independent_processing():
    # create two different JaxEdges with specific feature sizes
    arrow_edge = jax_context.edges["arrow"]
    source_edge = jax_context.edges["source"]

    def _n_obj(e):
        if e.feature_array is not None:
            return int(e.feature_array.shape[0])
        return int(jnp.array(e.non_fictitious).shape[0])

    n_obj_arrow = _n_obj(arrow_edge)
    n_obj_source = _n_obj(source_edge)

    e1 = JaxEdge(
        address_dict=arrow_edge.address_dict,
        feature_array=jnp.ones((n_obj_arrow, 2), dtype=jnp.float32),
        feature_names={"a": jnp.array(0), "b": jnp.array(1)},
        non_fictitious=arrow_edge.non_fictitious,
    )
    e2 = JaxEdge(
        address_dict=source_edge.address_dict,
        feature_array=jnp.ones((n_obj_source, 3), dtype=jnp.float32),
        feature_names={"c": jnp.array(0), "d": jnp.array(1), "e": jnp.array(2)},
        non_fictitious=source_edge.non_fictitious,
    )

    custom_graph = JaxGraph(
        edges={"A": e1, "B": e2},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    # create a custom structure for this test
    custom_structure = GraphStructure(
        edges={
            "A": EdgeStructure(address_list=["from", "to"], feature_list=["a", "b"]),
            "B": EdgeStructure(address_list=["id"], feature_list=["c", "d", "e"]),
        }
    )

    enc = MLPEncoder(in_structure=custom_structure, hidden_sizes=[6], out_size=5, activation=None, seed=5)
    out, infos = enc(graph=custom_graph, get_info=False)

    assert out.edges["A"].feature_array.shape[-1] == 5
    assert out.edges["B"].feature_array.shape[-1] == 5
    expected_keys = {f"lat_{i}" for i in range(5)}
    assert set(out.edges["A"].feature_names.keys()) == expected_keys
    assert set(out.edges["B"].feature_names.keys()) == expected_keys


def test_mlp_encoder_numeric_identity():
    """
    Build a graph and replace the MLPs by identity functions to expect exact equality
    (modulo fictitious masking).
    """
    arrow_edge = jax_context.edges["arrow"]
    source_edge = jax_context.edges["source"]

    n_obj_arrow = int(arrow_edge.feature_array.shape[0])
    n_obj_source = int(source_edge.feature_array.shape[0])
    d = 4

    # Create edges with linear values to verify identity mapping
    e_arrow = JaxEdge(
        address_dict=arrow_edge.address_dict,
        feature_array=jnp.linspace(0.0, 1.0, num=n_obj_arrow * d, dtype=jnp.float32).reshape((n_obj_arrow, d)),
        feature_names={f"fa{i}": jnp.array(i) for i in range(d)},
        non_fictitious=arrow_edge.non_fictitious,
    )
    e_source = JaxEdge(
        address_dict=source_edge.address_dict,
        feature_array=jnp.linspace(0.0, 1.0, num=n_obj_source * d, dtype=jnp.float32).reshape((n_obj_source, d)),
        feature_names={f"fs{i}": jnp.array(i) for i in range(d)},
        non_fictitious=source_edge.non_fictitious,
    )

    custom_graph = JaxGraph(
        edges={"arrow": e_arrow, "source": e_source},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    enc = MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[], out_size=d, activation=None, seed=123)
    # Replace both MLPs by identity
    enc.mlp_dict["arrow"] = lambda x: x
    enc.mlp_dict["source"] = lambda x: x

    out, _ = enc(graph=custom_graph, get_info=False)

    expected_arrow = e_arrow.feature_array * jnp.expand_dims(e_arrow.non_fictitious, -1)
    expected_source = e_source.feature_array * jnp.expand_dims(e_source.non_fictitious, -1)

    np.testing.assert_allclose(np.array(out.edges["arrow"].feature_array), np.array(expected_arrow), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.array(out.edges["source"].feature_array), np.array(expected_source), rtol=0.0, atol=1e-6)
