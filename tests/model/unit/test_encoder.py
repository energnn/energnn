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

from energnn.graph import Graph, GraphStructure, HyperEdgeSet, HyperEdgeSetStructure, JaxBackend
from energnn.model.encoder.encoder import IdentityEncoder
from energnn.model.encoder.mlp_encoder import MLPEncoder
from energnn.problem.example import LinearSystemProblemLoader
from tests.utils import compare_batched_graphs

# make deterministic
np.random.seed(0)

# Prepare a small LinearSystemProblemLoader and example graphs
pb_loader = LinearSystemProblemLoader(seed=0)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)


# IdentityEncoder tests
def test_identity_encoder_single_roundtrip():
    enc = IdentityEncoder()
    out, info = enc(graph=jax_context, step_with_metrics=True)
    # should return same graph and empty info
    chex.assert_trees_all_equal(out, jax_context)
    assert info == {}


def test_identity_encoder_batch_vmap_jit_consistency():
    enc = IdentityEncoder()

    def apply_fn(graphs, step_with_metrics):
        return enc(graph=graphs, step_with_metrics=step_with_metrics)

    apply_vmap = jax.vmap(lambda g, gi: enc(graph=g, step_with_metrics=gi), in_axes=(0, None), out_axes=0)
    out1, info1 = apply_vmap(jax_context_batch, False)
    out2, info2 = apply_vmap(jax_context_batch, True)
    out3, info3 = jax.jit(apply_vmap)(jax_context_batch, False)
    out4, info4 = jax.jit(apply_vmap)(jax_context_batch, True)

    # compare shapes and structural equality
    chex.assert_trees_all_equal(out1, out2, out3, out4)
    assert info1 == {}
    assert info3 == {}
    assert info2 == info4


# MLPEncoder tests
@pytest.fixture(scope="module")
def mlp_encoder():
    # give an explicit seed for deterministic behavior in tests
    return MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[8], out_size=4, activation=None, seed=0)


def test_mlp_encoder_init_is_deterministic_and_returns_graph():
    # Two encoders instantiated with the same seed should produce same outputs
    enc1 = MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[8], out_size=4, activation=None, seed=1)
    enc2 = MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[8], out_size=4, activation=None, seed=1)

    out1, info1 = enc1(graph=jax_context, step_with_metrics=False)
    out2, info2 = enc2(graph=jax_context, step_with_metrics=False)

    chex.assert_trees_all_equal(out1, out2)
    assert info1 == {}
    assert info2 == {}


def test_mlp_encoder_single_shapes_and_feature_names():
    enc = MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[8], out_size=4, activation=None, seed=2)

    out, infos = enc(graph=jax_context, step_with_metrics=True)

    # Basic shape checks per edge
    for key, edge in out.hyper_edge_sets.items():
        if edge.feature_array is not None:
            assert edge.feature_array.shape[-1] == enc.out_size
            # feature_names should contain lat_0 ... lat_{out_size-1}
            expected_keys = {f"lat_{i}" for i in range(enc.out_size)}
            assert set(edge.feature_names.keys()) == expected_keys
        else:
            assert edge.feature_names is None

    assert infos == {}


def test_mlp_encoder_handles_none_feature_array_gracefully():
    # Build a Graph with one edge having feature_array=None
    edge_with_none = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict=jax_context.hyper_edge_sets["line"].port_dict,
        feature_array=None,
        feature_names=None,
        non_fictitious=jax_context.hyper_edge_sets["line"].non_fictitious,
    )
    custom_graph = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"line": edge_with_none, "bus": jax_context.hyper_edge_sets["bus"]},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    enc = MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[4], out_size=3, activation=None, seed=3)
    out, infos = enc(graph=custom_graph, step_with_metrics=False)

    assert out.hyper_edge_sets["line"].feature_array is None
    assert out.hyper_edge_sets["line"].feature_names is None
    assert out.hyper_edge_sets["bus"].feature_array.shape[-1] == 3


def test_mlp_encoder_jit_and_vmap_compatibility(mlp_encoder):
    enc = mlp_encoder
    # One call to ensure everything is fine (also consumes rngs safely)
    _ = enc(graph=jax_context, step_with_metrics=False)

    # Vectorize across leading batch axis: vmapping the callable that accepts a single graph
    apply_vmap = jax.vmap(lambda g, gi: enc(graph=g, step_with_metrics=gi), in_axes=(0, None), out_axes=0)

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
    line_edge = jax_context.hyper_edge_sets["line"]
    bus_edge = jax_context.hyper_edge_sets["bus"]

    def _n_obj(e):
        if e.feature_array is not None:
            return int(e.feature_array.shape[0])
        return int(jnp.array(e.non_fictitious).shape[0])

    n_obj_line = _n_obj(line_edge)
    n_obj_bus = _n_obj(bus_edge)

    e1 = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict=line_edge.port_dict,
        feature_array=jnp.ones((n_obj_line, 2), dtype=jnp.float32),
        feature_names={"a": jnp.array(0), "b": jnp.array(1)},
        non_fictitious=line_edge.non_fictitious,
    )
    e2 = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict=bus_edge.port_dict,
        feature_array=jnp.ones((n_obj_bus, 3), dtype=jnp.float32),
        feature_names={"c": jnp.array(0), "d": jnp.array(1), "e": jnp.array(2)},
        non_fictitious=bus_edge.non_fictitious,
    )

    custom_graph = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"A": e1, "B": e2},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    # create a custom structure for this test
    custom_structure = GraphStructure(
        hyper_edge_sets={
            "A": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=["a", "b"]),
            "B": HyperEdgeSetStructure(port_list=["id"], feature_list=["c", "d", "e"]),
        }
    )

    enc = MLPEncoder(in_structure=custom_structure, hidden_sizes=[6], out_size=5, activation=None, seed=5)
    out, infos = enc(graph=custom_graph, step_with_metrics=False)

    assert out.hyper_edge_sets["A"].feature_array.shape[-1] == 5
    assert out.hyper_edge_sets["B"].feature_array.shape[-1] == 5
    expected_keys = {f"lat_{i}" for i in range(5)}
    assert set(out.hyper_edge_sets["A"].feature_names.keys()) == expected_keys
    assert set(out.hyper_edge_sets["B"].feature_names.keys()) == expected_keys


def test_mlp_encoder_numeric_identity():
    """
    Build a graph and replace the MLPs by identity functions to expect exact equality
    (modulo fictitious masking).
    """
    line_edge = jax_context.hyper_edge_sets["line"]
    bus_edge = jax_context.hyper_edge_sets["bus"]

    n_obj_line = int(line_edge.feature_array.shape[0])
    n_obj_bus = int(bus_edge.feature_array.shape[0])
    d = 4

    # Create edges with linear values to verify identity mapping
    e_line = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict=line_edge.port_dict,
        feature_array=jnp.linspace(0.0, 1.0, num=n_obj_line * d, dtype=jnp.float32).reshape((n_obj_line, d)),
        feature_names={f"fa{i}": jnp.array(i) for i in range(d)},
        non_fictitious=line_edge.non_fictitious,
    )
    e_bus = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict=bus_edge.port_dict,
        feature_array=jnp.linspace(0.0, 1.0, num=n_obj_bus * d, dtype=jnp.float32).reshape((n_obj_bus, d)),
        feature_names={f"fs{i}": jnp.array(i) for i in range(d)},
        non_fictitious=bus_edge.non_fictitious,
    )

    custom_graph = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"line": e_line, "bus": e_bus},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    enc = MLPEncoder(in_structure=pb_loader.context_structure, hidden_sizes=[], out_size=d, activation=None, seed=123)
    # Replace both MLPs by identity
    enc.mlp_dict["line"] = lambda x: x
    enc.mlp_dict["bus"] = lambda x: x

    out, _ = enc(graph=custom_graph, step_with_metrics=False)

    expected_line = e_line.feature_array * jnp.expand_dims(e_line.non_fictitious, -1)
    expected_bus = e_bus.feature_array * jnp.expand_dims(e_bus.non_fictitious, -1)

    np.testing.assert_allclose(
        np.array(out.hyper_edge_sets["line"].feature_array), np.array(expected_line), rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(np.array(out.hyper_edge_sets["bus"].feature_array), np.array(expected_bus), rtol=0.0, atol=1e-6)


def test_encoder_apply_preserves_none_feature_edges(monkeypatch):
    # Build graph with one edge having None features and another with features
    node_edge_with_none = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict=jax_context.hyper_edge_sets["bus"].port_dict,
        feature_array=None,
        feature_names=None,
        non_fictitious=jax_context.hyper_edge_sets["bus"].non_fictitious,
    )
    edge_with_feat = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict=jax_context.hyper_edge_sets["line"].port_dict,
        feature_array=jnp.ones((jax_context.hyper_edge_sets["line"].feature_array.shape[0], 1), dtype=jnp.float32),
        feature_names={"susceptance": jnp.array(0)},
        non_fictitious=jax_context.hyper_edge_sets["line"].non_fictitious,
    )
    g = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"bus": node_edge_with_none, "line": edge_with_feat},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    in_structure = GraphStructure(
        hyper_edge_sets={
            "bus": HyperEdgeSetStructure(port_list=["id"], feature_list=None),
            "line": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=["susceptance"]),
        }
    )

    d = 4
    enc = MLPEncoder(in_structure=in_structure, hidden_sizes=[], out_size=d, activation=None, seed=123)
    out_graph, _ = enc(graph=g, step_with_metrics=False)

    # bus edge had None -> must remain None
    assert out_graph.hyper_edge_sets["bus"].feature_array is None


# Tests for mixed precision (dtype) support
def test_mlp_encoder_bf16_output_dtype_and_closeness():
    from flax import nnx

    kwargs = dict(in_structure=pb_loader.context_structure, hidden_sizes=[8], out_size=4, activation=None)
    enc_bf16 = MLPEncoder(**kwargs, dtype=jnp.bfloat16, seed=7)
    enc_fp32 = MLPEncoder(**kwargs, seed=7)

    out_bf16, _ = enc_bf16(graph=jax_context, step_with_metrics=False)
    out_fp32, _ = enc_fp32(graph=jax_context, step_with_metrics=False)

    for key, hyper_edge_set in out_bf16.hyper_edge_sets.items():
        if hyper_edge_set.feature_array is not None:
            # output is cast back to the input features dtype
            assert hyper_edge_set.feature_array.dtype == jnp.float32
            np.testing.assert_allclose(
                np.array(hyper_edge_set.feature_array),
                np.array(out_fp32.hyper_edge_sets[key].feature_array),
                rtol=0.1,
                atol=0.1,
            )
    # parameters remain stored in float32
    for leaf in jax.tree.leaves(nnx.state(enc_bf16, nnx.Param)):
        assert leaf.dtype == jnp.float32
