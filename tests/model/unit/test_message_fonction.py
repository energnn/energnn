#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from energnn.graph import GraphStructure, HyperEdgeSetStructure
from energnn.graph import Graph, HyperEdgeSet, JaxBackend
from energnn.model.coupler.message_passing.message_passing_function import (
    IdentityMessagePassingFunction,
    LocalSumMessagePassingFunction,
)
from energnn.model.utils import gather, scatter_add
from energnn.problem.example import LinearSystemProblemLoader

# deterministic
np.random.seed(0)

# Small fixture graphs from LinearSystemProblemLoader
pb_loader = LinearSystemProblemLoader(seed=0, batch_size=4, n_max=10)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)
coordinates = jnp.array(np.random.uniform(size=(10, 7)))
coordinates_batch = jnp.array(np.random.uniform(size=(4, 10, 7)))


def _unbatch_graph(batched_graph: Graph, coordinates_batch: jax.Array, idx: int = 0) -> Graph:
    """
    Extract a single graph from a batched Graph by taking index `idx` on leading batch axis
    for arrays that have a leading batch dimension.
    """
    batch_size = int(coordinates_batch.shape[0])
    edges = {}
    for k, e in batched_graph.hyper_edge_sets.items():
        # feature_array
        fa = e.feature_array
        if fa is None:
            fa_s = None
        else:
            if hasattr(fa, "shape") and fa.shape[0] == batch_size:
                fa_s = fa[idx]
            else:
                fa_s = fa

        # non_fictitious
        nf = e.non_fictitious
        if hasattr(nf, "shape") and nf.shape[0] == batch_size:
            nf_s = nf[idx]
        else:
            nf_s = nf

        # address_dict
        addr_s = None
        if e.port_dict is not None:
            addr_s = {}
            for aname, aarr in e.port_dict.items():
                if hasattr(aarr, "shape") and aarr.shape[0] == batch_size:
                    addr_s[aname] = aarr[idx]
                else:
                    addr_s[aname] = aarr

        edges[k] = HyperEdgeSet(
            backend=JaxBackend(),
            port_dict=addr_s,
            feature_array=fa_s,
            feature_names=e.feature_names,
            non_fictitious=nf_s,
        )

    return Graph(
        backend=JaxBackend(),
        hyper_edge_sets=edges,
        non_fictitious_addresses=batched_graph.non_fictitious_addresses,
        true_shape=batched_graph.true_shape,
        current_shape=batched_graph.current_shape,
    )


class IdentityMLP:
    def __call__(self, x):
        # returns input as float32 jax array
        return jnp.asarray(x, dtype=jnp.float32)


class ConstantMLP:
    def __init__(self, out_vec):
        self.out_vec = jnp.asarray(out_vec, dtype=jnp.float32)

    def __call__(self, x):
        # tile to batch size
        n = x.shape[0]
        return jnp.tile(self.out_vec[None, :], (n, 1))


def patch_all_mlps_to_identity(mf: LocalSumMessagePassingFunction):
    for ek in list(mf.mlp_tree.keys()):
        for pk in list(mf.mlp_tree[ek].keys()):
            mf.mlp_tree[ek][pk] = IdentityMLP()


def compute_expected_local_sum(
    graph: Graph, coords: jnp.ndarray, mlp_tree_funcs: dict | None, final_activation, out_size: int | None = None
) -> jnp.ndarray:
    """
    Reproduce the LocalSumMessageFunction internal ops to compute expected accumulator.
    mlp_tree_funcs: mapping edge_key -> port_key -> callable(x) -> (n_obj, out_size)
    If mlp_tree_funcs is None, uses identity for each port.
    """
    acc = None
    if out_size is not None:
        acc = jnp.zeros((coords.shape[0], out_size), dtype=jnp.float32)

    for edge_key, edge in graph.hyper_edge_sets.items():
        # build input_array
        parts = []
        if edge.feature_names is not None and edge.feature_array is not None:
            parts.append(edge.feature_array)
        for port_name, port_addr in edge.port_dict.items():
            parts.append(gather(coordinates=coords, addresses=port_addr))
        input_array = jnp.concatenate(parts, axis=-1)
        non_fict = jnp.expand_dims(edge.non_fictitious, -1)

        for port_name, port_addr in edge.port_dict.items():
            if mlp_tree_funcs is None:
                mlp = nnx.identity  # identity
            else:
                mlp = mlp_tree_funcs.get(edge_key, {}).get(port_name, nnx.identity)
            inc = mlp(input_array) * non_fict
            if acc is None:
                # initialize accumulator with correct out_size and n_addresses
                acc = jnp.zeros((coords.shape[0], int(inc.shape[-1])), dtype=jnp.float32)
            acc = scatter_add(accumulator=acc, increment=inc, addresses=port_addr)
    if acc is None:
        # no edges -> zeros
        acc = jnp.zeros((coords.shape[0], 0), dtype=jnp.float32)
    return final_activation(acc)


def _assert_vmap_jit_consistent(mf, ctx_batch: Graph, coords_batch: jnp.ndarray, rtol=1e-6, atol=1e-6):
    """
    Ensure vmapped and vmapped+jit versions produce consistent outputs.
    Precondition: mf._build_missing_mlps must already have been called on a non-batched sample.
    """
    apply_vmap = jax.vmap(lambda g, c, gi: mf(graph=g, coordinates=c, step_with_metrics=gi), in_axes=(0, 0, None), out_axes=0)
    out1, info1 = apply_vmap(ctx_batch, coords_batch, False)
    out2, info2 = apply_vmap(ctx_batch, coords_batch, True)
    out3, info3 = jax.jit(apply_vmap)(ctx_batch, coords_batch, False)
    out4, info4 = jax.jit(apply_vmap)(ctx_batch, coords_batch, True)

    chex.assert_trees_all_close(out1, out2, atol=1e-6)
    chex.assert_trees_all_close(info2, info4, atol=1e-6)
    assert info1 == {}
    assert info3 == {}
    return out1, info1


# Tests for IdentityMessageFunction
def test_identity_returns_coordinates():
    imf = IdentityMessagePassingFunction()
    out, info = imf(graph=jax_context, coordinates=coordinates, step_with_metrics=True)
    np.testing.assert_allclose(np.array(out), np.array(coordinates))
    assert info == {}


def test_identity_vmapped_and_jitted():
    imf = IdentityMessagePassingFunction()
    # batch vmapped
    out_b, _ = jax.vmap(lambda g, c, gi: imf(graph=g, coordinates=c, step_with_metrics=gi), in_axes=(0, 0, None))(
        jax_context_batch, coordinates_batch, False
    )
    np.testing.assert_allclose(np.array(out_b), np.array(coordinates_batch))
    # jit+vmap after simple call (no RNG) -> same
    out_b_jit, _ = jax.jit(jax.vmap(lambda g, c, gi: imf(graph=g, coordinates=c, step_with_metrics=gi), in_axes=(0, 0, None)))(
        jax_context_batch, coordinates_batch, False
    )
    np.testing.assert_allclose(np.array(out_b_jit), np.array(coordinates_batch))


def test_identity_dtype_and_shape():
    imf = IdentityMessagePassingFunction()
    out, _ = imf(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == coordinates.shape


# Tests for LocalSumMessageFunction
def test_mlp_tree_initialization_from_structure():
    mf = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=0,
    )
    # Check that mlp_tree is correctly populated based on in_graph_structure
    expected_keys = set(pb_loader.context_structure.hyper_edge_sets.keys())
    assert set(mf.mlp_tree.keys()) == expected_keys
    for ek in expected_keys:
        edge_struct = pb_loader.context_structure.hyper_edge_sets[ek]
        assert set(mf.mlp_tree[ek].keys()) == set(edge_struct.port_list)
        for pk in mf.mlp_tree[ek].keys():
            assert callable(mf.mlp_tree[ek][pk])


def test_mlp_tree_input_sizes_with_and_without_features():
    # create structure with one edge having features and one without
    struct = GraphStructure(
        hyper_edge_sets={
            "A": HyperEdgeSetStructure(port_list=["id"], feature_list=["v1", "v2"]),
            "B": HyperEdgeSetStructure(port_list=["id"], feature_list=None),
        }
    )
    in_array_size = 5
    mf = LocalSumMessagePassingFunction(
        in_graph_structure=struct,
        in_array_size=in_array_size,
        hidden_sizes=[2],
        out_size=4,
        seed=1,
    )
    assert set(mf.mlp_tree.keys()) == {"A", "B"}
    # A: in_array_size (5) * n_ports (1) + n_features (2) = 7
    assert mf.mlp_tree["A"]["id"].sequential.layers[0].in_features == 7
    # B: in_array_size (5) * n_ports (1) + n_features (0) = 5
    assert mf.mlp_tree["B"]["id"].sequential.layers[0].in_features == 5


def test_output_shape_and_dtype():
    out_size = 5
    mf = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=out_size,
        outer_activation=nnx.identity,
        seed=3,
    )
    out, info = mf(graph=jax_context, coordinates=coordinates, step_with_metrics=True)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (coordinates.shape[0], out_size)
    assert info == {}


def test_non_fictitious_masking():
    # build a small graph with one edge with 3 objects and 4 addresses
    n_addr = 4
    d = 2
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])
    addr0 = jnp.array([0, 1, 0])
    addr1 = jnp.array([1, 2, 3])
    n_obj = 3
    non_fict = jnp.array([1.0, 0.0, 1.0])  # middle object fictitious

    edge = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict={"from": addr0, "to": addr1},
        feature_array=None,
        feature_names=None,
        non_fictitious=non_fict,
    )
    small_context = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"line": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    mf = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=d,
        hidden_sizes=[],
        activation=None,
        out_size=2 * d,
        outer_activation=nnx.identity,
        seed=10,
    )
    # patch mlps to constant ones so we can detect zeroing
    const = jnp.array([1.0] * (2 * d))
    for ek in list(mf.mlp_tree.keys()):
        for pk in list(mf.mlp_tree[ek].keys()):
            mf.mlp_tree[ek][pk] = ConstantMLP(const)

    out, _ = mf(graph=small_context, coordinates=coords, step_with_metrics=False)
    out_np = np.array(out)
    # contributions from object with non_fict==0 (index 1) must be zero
    # compute expected manually using compute_expected_local_sum
    expected = np.array(compute_expected_local_sum(small_context, coords, mf.mlp_tree, nnx.identity))
    # since we set mlps to constant, compare
    np.testing.assert_allclose(out_np, expected, rtol=0.0, atol=1e-6)


def test_final_activation_applied():
    # test with tanh activation
    mf = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[],
        activation=None,
        out_size=2,
        outer_activation=jnp.tanh,
        seed=11,
    )
    # patch to constant 1.0 vectors
    for ek in list(mf.mlp_tree.keys()):
        for pk in list(mf.mlp_tree[ek].keys()):
            mf.mlp_tree[ek][pk] = ConstantMLP(jnp.array([1.0, -1.0]))
    out, _ = mf(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    # expected: tanh(accumulator)
    expected = np.array(jnp.tanh(compute_expected_local_sum(jax_context, coordinates, mf.mlp_tree, nnx.identity)))
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-6, atol=1e-6)


def test_local_sum_numeric_identity_basic():
    # This is the small case we attempted earlier; we reproduce expected using compute_expected_local_sum
    n_addr = 4
    d = 2
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])
    addr0 = jnp.array([0, 1, 0])
    addr1 = jnp.array([1, 2, 3])
    n_obj = 3
    edge = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict={"from": addr0, "to": addr1},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.ones((n_obj,)),
    )
    small_context = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"line": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    mf = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=d,
        hidden_sizes=[],
        activation=None,
        out_size=2 * d,
        outer_activation=nnx.identity,
        seed=222,
    )
    patch_all_mlps_to_identity(mf)

    out, _ = mf(graph=small_context, coordinates=coords, step_with_metrics=False)
    expected = compute_expected_local_sum(small_context, coords, mf.mlp_tree, nnx.identity)
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-6, atol=1e-6)


def test_local_sum_with_features_included():
    # Create edge with features; ensure features are included before gathered coords
    n_addr = 3
    coords = jnp.array([[1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])
    addr0 = jnp.array([0, 1])
    addr1 = jnp.array([1, 2])
    n_obj = 2
    feat = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    edge = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict={"from": addr0, "to": addr1},
        feature_array=feat,
        feature_names={"a": jnp.array(0), "b": jnp.array(1)},
        non_fictitious=jnp.ones((n_obj,)),
    )
    g = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"line": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    mf = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coords.shape[1],
        hidden_sizes=[],
        activation=None,
        out_size=feat.shape[1] + coords.shape[1] * 2,
        outer_activation=nnx.identity,
        seed=99,
    )
    patch_all_mlps_to_identity(mf)

    out, _ = mf(graph=g, coordinates=coords, step_with_metrics=False)
    expected = compute_expected_local_sum(g, coords, mf.mlp_tree, nnx.identity)
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-6, atol=1e-6)


def test_multiple_edges_and_ports_independent_processing():
    # Create graph with two edges "line" and "bus" with distinct constant mlp outputs; verify sum is correct
    n_addr = 4
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])
    addr_a0 = jnp.array([0, 1, 2])
    addr_a1 = jnp.array([1, 2, 3])
    addr_b = jnp.array([0, 1, 3])
    edge_a = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict={"from": addr_a0, "to": addr_a1},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.ones((3,)),
    )
    edge_b = HyperEdgeSet(
        backend=JaxBackend(), port_dict={"id": addr_b}, feature_array=None, feature_names=None, non_fictitious=jnp.ones((3,))
    )
    g = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"line": edge_a, "bus": edge_b},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    mf = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coords.shape[1],
        hidden_sizes=[],
        activation=None,
        out_size=2,
        outer_activation=nnx.identity,
        seed=44,
    )
    # patch line ports to constant [1,0], bus port to constant [0,2]
    for pk in mf.mlp_tree["line"].keys():
        mf.mlp_tree["line"][pk] = ConstantMLP(jnp.array([1.0, 0.0]))
    for pk in mf.mlp_tree["bus"].keys():
        mf.mlp_tree["bus"][pk] = ConstantMLP(jnp.array([0.0, 2.0]))

    out, _ = mf(graph=g, coordinates=coords, step_with_metrics=False)
    # compute expected via compute_expected_local_sum with a custom mlp mapping
    mlp_map = {
        "line": {
            p: (lambda x, v=jnp.array([1.0, 0.0]): jnp.tile(v[None, :], (x.shape[0], 1))) for p in mf.mlp_tree["line"].keys()
        },
        "bus": {
            p: (lambda x, v=jnp.array([0.0, 2.0]): jnp.tile(v[None, :], (x.shape[0], 1))) for p in mf.mlp_tree["bus"].keys()
        },
    }
    expected = compute_expected_local_sum(g, coords, mlp_map, nnx.identity)
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-6, atol=1e-6)


def test_deterministic_with_seed():
    mf1 = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=7,
    )
    mf2 = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=7,
    )
    out1, _ = mf1(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    out2, _ = mf2(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    chex.assert_trees_all_close(out1, out2, atol=1e-6)


def test_vmap_jit_safety_after_build():
    mf = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[2],
        activation=None,
        out_size=4,
        outer_activation=nnx.identity,
        seed=8,
    )
    out_b, _ = _assert_vmap_jit_consistent(mf, jax_context_batch, coordinates_batch)
    # just check shapes
    assert np.array(out_b).shape[0] == coordinates_batch.shape[0]


def test_empty_graph_returns_zeros():
    # graph with no edges
    g = Graph(
        backend=JaxBackend(), hyper_edge_sets={}, non_fictitious_addresses=jnp.ones((5,)), true_shape=None, current_shape=None
    )
    mf = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=3,
        hidden_sizes=[2],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=9,
    )
    out, _ = mf(graph=g, coordinates=jnp.zeros((5, 3)), step_with_metrics=False)
    # Expect zeros with shape (n_addr, out_size)
    assert out.shape == (5, 3)


def test_addresses_out_of_bounds_handling():
    # Create edge with addresses containing out-of-bounds index
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    addr_from = jnp.array([0, 10])  # 10 is out of bounds
    addr_to = jnp.array([0, 1])
    edge = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict={"from": addr_from, "to": addr_to},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.ones((2,)),
    )
    g = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"line": edge},
        non_fictitious_addresses=jnp.ones((2,)),
        true_shape=None,
        current_shape=None,
    )
    mf = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coords.shape[1],
        hidden_sizes=[],
        activation=None,
        out_size=4,
        outer_activation=nnx.identity,
        seed=15,
    )
    patch_all_mlps_to_identity(mf)
    out, _ = mf(graph=g, coordinates=coords, step_with_metrics=False)
    # ensure it runs and shape is correct
    assert out.shape == (coords.shape[0], 4)


# Tests for the fused mode of LocalSumMessagePassingFunction
FusedLocalSumMessagePassingFunction = partial(LocalSumMessagePassingFunction, fuse_ports=True)


def compute_expected_fused_local_sum(
    graph: Graph, coords: jnp.ndarray, mlp_dict_funcs: dict, active_ports: dict, out_size: int, final_activation
) -> jnp.ndarray:
    """Reproduce the FusedLocalSumMessagePassingFunction internal ops to compute expected accumulator."""
    acc = jnp.zeros((coords.shape[0], out_size), dtype=jnp.float32)
    for edge_key, edge in graph.hyper_edge_sets.items():
        if edge_key not in mlp_dict_funcs:
            continue
        parts = []
        if edge.feature_names is not None and edge.feature_array is not None:
            parts.append(edge.feature_array)
        for port_name, port_addr in edge.port_dict.items():
            parts.append(gather(coordinates=coords, addresses=port_addr))
        input_array = jnp.concatenate(parts, axis=-1)
        non_fict = jnp.expand_dims(edge.non_fictitious, -1)
        out = mlp_dict_funcs[edge_key](input_array * non_fict) * non_fict
        for i, port_name in enumerate(active_ports[edge_key]):
            inc = out[..., i * out_size : (i + 1) * out_size]
            acc = scatter_add(accumulator=acc, increment=inc, addresses=edge.port_dict[port_name])
    return final_activation(acc)


def test_fused_mlp_dict_initialization_from_structure():
    out_size = 3
    mf = FusedLocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=out_size,
        outer_activation=nnx.identity,
        seed=0,
    )
    expected_keys = set(pb_loader.context_structure.hyper_edge_sets.keys())
    assert set(mf.mlp_tree.keys()) == expected_keys
    for ek in expected_keys:
        edge_struct = pb_loader.context_structure.hyper_edge_sets[ek]
        n_ports = len(edge_struct.port_list)
        # one single MLP per class, predicting out_size values per port
        assert mf.mlp_tree[ek].sequential.layers[-1].out_features == out_size * n_ports
        assert mf.active_ports[ek] == list(edge_struct.port_list)


def test_fused_mlp_input_sizes_with_and_without_features():
    struct = GraphStructure(
        hyper_edge_sets={
            "A": HyperEdgeSetStructure(port_list=["p1", "p2"], feature_list=["v1", "v2"]),
            "B": HyperEdgeSetStructure(port_list=["id"], feature_list=None),
        }
    )
    in_array_size = 5
    out_size = 4
    mf = FusedLocalSumMessagePassingFunction(
        in_graph_structure=struct,
        in_array_size=in_array_size,
        hidden_sizes=[2],
        out_size=out_size,
        seed=1,
    )
    # A: in_array_size (5) * n_ports (2) + n_features (2) = 12, out: 4 * 2 ports = 8
    assert mf.mlp_tree["A"].sequential.layers[0].in_features == 12
    assert mf.mlp_tree["A"].sequential.layers[-1].out_features == 8
    # B: in_array_size (5) * n_ports (1) = 5, out: 4 * 1 port = 4
    assert mf.mlp_tree["B"].sequential.layers[0].in_features == 5
    assert mf.mlp_tree["B"].sequential.layers[-1].out_features == 4


def test_fused_output_shape_and_dtype():
    out_size = 5
    mf = FusedLocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=out_size,
        outer_activation=nnx.identity,
        seed=3,
    )
    out, info = mf(graph=jax_context, coordinates=coordinates, step_with_metrics=True)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (coordinates.shape[0], out_size)
    assert info == {}


def test_fused_numeric_correctness_with_constant_mlp():
    n_addr = 4
    d = 2
    out_size = 2
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])
    addr0 = jnp.array([0, 1, 0])
    addr1 = jnp.array([1, 2, 3])
    non_fict = jnp.array([1.0, 0.0, 1.0])  # middle object fictitious
    edge = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict={"from": addr0, "to": addr1},
        feature_array=None,
        feature_names=None,
        non_fictitious=non_fict,
    )
    small_context = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"line": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    mf = FusedLocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=d,
        hidden_sizes=[],
        activation=None,
        out_size=out_size,
        outer_activation=nnx.identity,
        seed=10,
    )
    # constant output: chunk [1, 2] for port "from", chunk [3, 4] for port "to"
    const = jnp.array([1.0, 2.0, 3.0, 4.0])
    mf.mlp_tree["line"] = ConstantMLP(const)

    out, _ = mf(graph=small_context, coordinates=coords, step_with_metrics=False)
    expected = compute_expected_fused_local_sum(
        small_context, coords, {"line": ConstantMLP(const)}, mf.active_ports, out_size, nnx.identity
    )
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=0.0, atol=1e-6)
    # fictitious object (index 1) must not contribute anywhere
    manual = np.zeros((n_addr, out_size))
    manual[0] += [1.0, 2.0]  # from of edge 0
    manual[1] += [3.0, 4.0]  # to of edge 0
    manual[0] += [1.0, 2.0]  # from of edge 2
    manual[3] += [3.0, 4.0]  # to of edge 2
    np.testing.assert_allclose(np.array(out), manual, rtol=0.0, atol=1e-6)


def test_fused_blacklist_excluded_from_output_and_scatter():
    n_addr = 4
    d = 2
    out_size = 2
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])
    addr0 = jnp.array([0, 1, 0])
    addr1 = jnp.array([1, 2, 3])
    edge = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict={"from": addr0, "to": addr1},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.ones((3,)),
    )
    small_context = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"line": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    struct = GraphStructure(hyper_edge_sets={"line": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=None)})
    mf = FusedLocalSumMessagePassingFunction(
        in_graph_structure=struct,
        in_array_size=d,
        hidden_sizes=[],
        activation=None,
        out_size=out_size,
        outer_activation=nnx.identity,
        port_scatter_blacklist={"line": ["from"]},
        seed=10,
    )
    # blacklisted port must not appear in active ports, and MLP output must only cover "to"
    assert mf.active_ports["line"] == ["to"]
    assert mf.mlp_tree["line"].sequential.layers[-1].out_features == out_size

    const = jnp.array([3.0, 4.0])
    mf.mlp_tree["line"] = ConstantMLP(const)
    out, _ = mf(graph=small_context, coordinates=coords, step_with_metrics=False)
    # only "to" addresses (1, 2, 3) receive messages
    manual = np.zeros((n_addr, out_size))
    manual[1] += [3.0, 4.0]
    manual[2] += [3.0, 4.0]
    manual[3] += [3.0, 4.0]
    np.testing.assert_allclose(np.array(out), manual, rtol=0.0, atol=1e-6)


def test_fused_fully_blacklisted_class_is_skipped():
    struct = GraphStructure(
        hyper_edge_sets={
            "A": HyperEdgeSetStructure(port_list=["p1", "p2"], feature_list=None),
            "B": HyperEdgeSetStructure(port_list=["id"], feature_list=None),
        }
    )
    mf = FusedLocalSumMessagePassingFunction(
        in_graph_structure=struct,
        in_array_size=2,
        hidden_sizes=[],
        activation=None,
        out_size=2,
        outer_activation=nnx.identity,
        port_scatter_blacklist={"A": ["p1", "p2"]},
        seed=0,
    )
    assert "A" not in mf.mlp_tree
    assert "B" in mf.mlp_tree

    # a graph containing class A must still be processed without error
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    edge_a = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict={"p1": jnp.array([0, 1]), "p2": jnp.array([1, 2])},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.ones((2,)),
    )
    edge_b = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict={"id": jnp.array([0, 2])},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.ones((2,)),
    )
    g = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"A": edge_a, "B": edge_b},
        non_fictitious_addresses=jnp.ones((3,)),
        true_shape=None,
        current_shape=None,
    )
    mf.mlp_tree["B"] = ConstantMLP(jnp.array([1.0, 0.0]))
    out, _ = mf(graph=g, coordinates=coords, step_with_metrics=False)
    manual = np.zeros((3, 2))
    manual[0] += [1.0, 0.0]
    manual[2] += [1.0, 0.0]
    np.testing.assert_allclose(np.array(out), manual, rtol=0.0, atol=1e-6)


def test_fused_deterministic_with_seed():
    mf1 = FusedLocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=7,
    )
    mf2 = FusedLocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=7,
    )
    out1, _ = mf1(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    out2, _ = mf2(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    chex.assert_trees_all_close(out1, out2, atol=1e-6)


def test_fused_vmap_jit_safety():
    mf = FusedLocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[2],
        activation=None,
        out_size=4,
        outer_activation=nnx.identity,
        seed=8,
    )
    out_b, _ = _assert_vmap_jit_consistent(mf, jax_context_batch, coordinates_batch)
    assert np.array(out_b).shape[0] == coordinates_batch.shape[0]


def test_fused_empty_graph_returns_zeros():
    g = Graph(
        backend=JaxBackend(), hyper_edge_sets={}, non_fictitious_addresses=jnp.ones((5,)), true_shape=None, current_shape=None
    )
    mf = FusedLocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=3,
        hidden_sizes=[2],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=9,
    )
    out, _ = mf(graph=g, coordinates=jnp.zeros((5, 3)), step_with_metrics=False)
    assert out.shape == (5, 3)


def test_local_sum_blacklist_excluded_from_scatter():
    # blacklist behavior of the existing per-port variant, for parity with the fused one
    n_addr = 4
    d = 2
    out_size = 2
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])
    addr0 = jnp.array([0, 1, 0])
    addr1 = jnp.array([1, 2, 3])
    edge = HyperEdgeSet(
        backend=JaxBackend(),
        port_dict={"from": addr0, "to": addr1},
        feature_array=None,
        feature_names=None,
        non_fictitious=jnp.ones((3,)),
    )
    small_context = Graph(
        backend=JaxBackend(),
        hyper_edge_sets={"line": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    struct = GraphStructure(hyper_edge_sets={"line": HyperEdgeSetStructure(port_list=["from", "to"], feature_list=None)})
    mf = LocalSumMessagePassingFunction(
        in_graph_structure=struct,
        in_array_size=d,
        hidden_sizes=[],
        activation=None,
        out_size=out_size,
        outer_activation=nnx.identity,
        port_scatter_blacklist={"line": ["from"]},
        seed=10,
    )
    assert set(mf.mlp_tree["line"].keys()) == {"to"}
    mf.mlp_tree["line"]["to"] = ConstantMLP(jnp.array([3.0, 4.0]))
    out, _ = mf(graph=small_context, coordinates=coords, step_with_metrics=False)
    manual = np.zeros((n_addr, out_size))
    manual[1] += [3.0, 4.0]
    manual[2] += [3.0, 4.0]
    manual[3] += [3.0, 4.0]
    np.testing.assert_allclose(np.array(out), manual, rtol=0.0, atol=1e-6)


# Tests for mixed precision (dtype) support
def test_mlp_dtype_computation_and_param_storage():
    from energnn.model.utils import MLP as _MLP

    mlp = _MLP(in_size=4, hidden_sizes=[8], out_size=3, dtype=jnp.bfloat16, seed=0)
    out = mlp(jnp.ones((5, 4), dtype=jnp.float32))
    assert out.dtype == jnp.bfloat16
    # parameters remain stored in float32
    for leaf in jax.tree.leaves(nnx.state(mlp, nnx.Param)):
        assert leaf.dtype == jnp.float32


def test_local_sum_bf16_output_dtype_and_closeness():
    mf_fp32 = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=5,
    )
    mf_bf16 = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        dtype=jnp.bfloat16,
        seed=5,
    )
    out_fp32, _ = mf_fp32(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    out_bf16, info = mf_bf16(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    # output is cast back to the coordinates dtype
    assert out_bf16.dtype == coordinates.dtype
    assert out_bf16.shape == out_fp32.shape
    assert info == {}
    # bf16 has ~2-3 significant decimal digits: results must be close but not identical
    np.testing.assert_allclose(np.array(out_bf16), np.array(out_fp32), rtol=0.05, atol=0.05)


def test_local_sum_dtype_none_is_unchanged():
    kwargs = dict(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
    )
    mf_default = LocalSumMessagePassingFunction(**kwargs, seed=6)
    mf_explicit = LocalSumMessagePassingFunction(**kwargs, dtype=None, seed=6)
    out1, _ = mf_default(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    out2, _ = mf_explicit(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    assert out1.dtype == jnp.float32
    np.testing.assert_array_equal(np.array(out1), np.array(out2))


def test_local_sum_bf16_vmap_jit_safety():
    mf = LocalSumMessagePassingFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[2],
        activation=None,
        out_size=4,
        outer_activation=nnx.identity,
        dtype=jnp.bfloat16,
        seed=8,
    )
    apply_vmap = jax.vmap(lambda g, c: mf(graph=g, coordinates=c, step_with_metrics=False), in_axes=(0, 0), out_axes=0)
    out_vmap, _ = apply_vmap(jax_context_batch, coordinates_batch)
    out_jit, _ = jax.jit(apply_vmap)(jax_context_batch, coordinates_batch)
    assert out_vmap.dtype == coordinates_batch.dtype
    assert out_vmap.shape[0] == coordinates_batch.shape[0]
    np.testing.assert_allclose(np.array(out_vmap), np.array(out_jit), rtol=2e-2, atol=2e-2)


def test_local_sum_bf16_fp32_scatter_output_dtype_and_closeness():
    kwargs = dict(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
    )
    mf_fp32 = LocalSumMessagePassingFunction(**kwargs, seed=5)
    mf_hybrid = LocalSumMessagePassingFunction(**kwargs, dtype=jnp.bfloat16, scatter_dtype=jnp.float32, seed=5)
    out_fp32, _ = mf_fp32(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    out_hybrid, _ = mf_hybrid(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    # output is cast back to the coordinates dtype
    assert out_hybrid.dtype == coordinates.dtype
    assert out_hybrid.shape == out_fp32.shape
    np.testing.assert_allclose(np.array(out_hybrid), np.array(out_fp32), rtol=0.05, atol=0.05)
    # parameters remain stored in float32
    for leaf in jax.tree.leaves(nnx.state(mf_hybrid, nnx.Param)):
        assert leaf.dtype == jnp.float32


def test_local_sum_scatter_dtype_none_is_unchanged():
    kwargs = dict(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        dtype=jnp.bfloat16,
    )
    mf_default = LocalSumMessagePassingFunction(**kwargs, seed=6)
    mf_explicit = LocalSumMessagePassingFunction(**kwargs, scatter_dtype=None, seed=6)
    out1, _ = mf_default(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    out2, _ = mf_explicit(graph=jax_context, coordinates=coordinates, step_with_metrics=False)
    np.testing.assert_array_equal(np.array(out1), np.array(out2))
