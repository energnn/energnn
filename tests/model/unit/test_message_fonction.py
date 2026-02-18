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
from flax import nnx

from energnn.model.coupler.neural_ode.message_function import LocalSumMessageFunction, IdentityMessageFunction
from energnn.model.utils import gather, scatter_add
from energnn.graph import GraphStructure, EdgeStructure
from energnn.graph.jax import JaxGraph, JaxEdge
from tests.utils import TestProblemLoader

# deterministic
np.random.seed(0)

# Small fixture graphs from TestProblemLoader
pb_loader = TestProblemLoader(seed=0, batch_size=4, n_max=10)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)
coordinates = jnp.array(np.random.uniform(size=(10, 7)))
coordinates_batch = jnp.array(np.random.uniform(size=(4, 10, 7)))


def _unbatch_graph(batched_graph: JaxGraph, coordinates_batch: jax.Array, idx: int = 0) -> JaxGraph:
    """
    Extract a single graph from a batched JaxGraph by taking index `idx` on leading batch axis
    for arrays that have a leading batch dimension.
    """
    batch_size = int(coordinates_batch.shape[0])
    edges = {}
    for k, e in batched_graph.edges.items():
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
        if e.address_dict is not None:
            addr_s = {}
            for aname, aarr in e.address_dict.items():
                if hasattr(aarr, "shape") and aarr.shape[0] == batch_size:
                    addr_s[aname] = aarr[idx]
                else:
                    addr_s[aname] = aarr

        edges[k] = JaxEdge(
            address_dict=addr_s,
            feature_array=fa_s,
            feature_names=e.feature_names,
            non_fictitious=nf_s,
        )

    return JaxGraph(
        edges=edges,
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


def patch_all_mlps_to_identity(mf: LocalSumMessageFunction):
    for ek in list(mf.mlp_tree.keys()):
        for pk in list(mf.mlp_tree[ek].keys()):
            mf.mlp_tree[ek][pk] = IdentityMLP()


def compute_expected_local_sum(
    graph: JaxGraph, coords: jnp.ndarray, mlp_tree_funcs: dict | None, final_activation, out_size: int | None = None
) -> jnp.ndarray:
    """
    Reproduce the LocalSumMessageFunction internal ops to compute expected accumulator.
    mlp_tree_funcs: mapping edge_key -> port_key -> callable(x) -> (n_obj, out_size)
    If mlp_tree_funcs is None, uses identity for each port.
    """
    acc = None
    if out_size is not None:
        acc = jnp.zeros((coords.shape[0], out_size), dtype=jnp.float32)

    for edge_key, edge in graph.edges.items():
        # build input_array
        parts = []
        if edge.feature_names is not None and edge.feature_array is not None:
            parts.append(edge.feature_array)
        for port_name, port_addr in edge.address_dict.items():
            parts.append(gather(coordinates=coords, addresses=port_addr))
        input_array = jnp.concatenate(parts, axis=-1)
        non_fict = jnp.expand_dims(edge.non_fictitious, -1)

        for port_name, port_addr in edge.address_dict.items():
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


def _assert_vmap_jit_consistent(mf, ctx_batch: JaxGraph, coords_batch: jnp.ndarray, rtol=1e-6, atol=1e-6):
    """
    Ensure vmapped and vmapped+jit versions produce consistent outputs.
    Precondition: mf._build_missing_mlps must already have been called on a non-batched sample.
    """
    apply_vmap = jax.vmap(lambda g, c, gi: mf(graph=g, coordinates=c, get_info=gi), in_axes=(0, 0, None), out_axes=0)
    out1, info1 = apply_vmap(ctx_batch, coords_batch, False)
    out2, info2 = apply_vmap(ctx_batch, coords_batch, True)
    out3, info3 = jax.jit(apply_vmap)(ctx_batch, coords_batch, False)
    out4, info4 = jax.jit(apply_vmap)(ctx_batch, coords_batch, True)

    chex.assert_trees_all_equal(out1, out2, out3, out4)
    chex.assert_trees_all_equal(info2, info4)
    assert info1 == {}
    assert info3 == {}
    return out1, info1


# Tests for IdentityMessageFunction
def test_identity_returns_coordinates():
    imf = IdentityMessageFunction()
    out, info = imf(graph=jax_context, coordinates=coordinates, get_info=True)
    np.testing.assert_allclose(np.array(out), np.array(coordinates))
    assert info == {}


def test_identity_vmapped_and_jitted():
    imf = IdentityMessageFunction()
    # batch vmapped
    out_b, _ = jax.vmap(lambda g, c, gi: imf(graph=g, coordinates=c, get_info=gi), in_axes=(0, 0, None))(
        jax_context_batch, coordinates_batch, False
    )
    np.testing.assert_allclose(np.array(out_b), np.array(coordinates_batch))
    # jit+vmap after simple call (no RNG) -> same
    out_b_jit, _ = jax.jit(jax.vmap(lambda g, c, gi: imf(graph=g, coordinates=c, get_info=gi), in_axes=(0, 0, None)))(
        jax_context_batch, coordinates_batch, False
    )
    np.testing.assert_allclose(np.array(out_b_jit), np.array(coordinates_batch))


def test_identity_dtype_and_shape():
    imf = IdentityMessageFunction()
    out, _ = imf(graph=jax_context, coordinates=coordinates, get_info=False)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == coordinates.shape


# Tests for LocalSumMessageFunction
def test_mlp_tree_initialization_from_structure():
    mf = LocalSumMessageFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=0,
    )
    # Check that mlp_tree is correctly populated based on in_graph_structure
    expected_keys = set(pb_loader.context_structure.edges.keys())
    assert set(mf.mlp_tree.keys()) == expected_keys
    for ek in expected_keys:
        edge_struct = pb_loader.context_structure.edges[ek]
        assert set(mf.mlp_tree[ek].keys()) == set(edge_struct.address_list)
        for pk in mf.mlp_tree[ek].keys():
            assert callable(mf.mlp_tree[ek][pk])


def test_mlp_tree_input_sizes_with_and_without_features():
    # create structure with one edge having features and one without
    struct = GraphStructure(
        edges={
            "A": EdgeStructure(address_list=["id"], feature_list=["v1", "v2"]),
            "B": EdgeStructure(address_list=["id"], feature_list=None),
        }
    )
    in_array_size = 5
    mf = LocalSumMessageFunction(
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
    mf = LocalSumMessageFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=out_size,
        outer_activation=nnx.identity,
        seed=3,
    )
    out, info = mf(graph=jax_context, coordinates=coordinates, get_info=True)
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

    edge = JaxEdge(address_dict={"from": addr0, "to": addr1}, feature_array=None, feature_names=None, non_fictitious=non_fict)
    small_context = JaxGraph(
        edges={"arrow": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    mf = LocalSumMessageFunction(
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

    out, _ = mf(graph=small_context, coordinates=coords, get_info=False)
    out_np = np.array(out)
    # contributions from object with non_fict==0 (index 1) must be zero
    # compute expected manually using compute_expected_local_sum
    expected = np.array(compute_expected_local_sum(small_context, coords, mf.mlp_tree, nnx.identity))
    # since we set mlps to constant, compare
    np.testing.assert_allclose(out_np, expected, rtol=0.0, atol=1e-6)


def test_final_activation_applied():
    # test with tanh activation
    mf = LocalSumMessageFunction(
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
    out, _ = mf(graph=jax_context, coordinates=coordinates, get_info=False)
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
    edge = JaxEdge(
        address_dict={"from": addr0, "to": addr1}, feature_array=None, feature_names=None, non_fictitious=jnp.ones((n_obj,))
    )
    small_context = JaxGraph(
        edges={"arrow": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    mf = LocalSumMessageFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=d,
        hidden_sizes=[],
        activation=None,
        out_size=2 * d,
        outer_activation=nnx.identity,
        seed=222,
    )
    patch_all_mlps_to_identity(mf)

    out, _ = mf(graph=small_context, coordinates=coords, get_info=False)
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
    edge = JaxEdge(
        address_dict={"from": addr0, "to": addr1},
        feature_array=feat,
        feature_names={"a": jnp.array(0), "b": jnp.array(1)},
        non_fictitious=jnp.ones((n_obj,)),
    )
    g = JaxGraph(
        edges={"arrow": edge},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    mf = LocalSumMessageFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coords.shape[1],
        hidden_sizes=[],
        activation=None,
        out_size=feat.shape[1] + coords.shape[1] * 2,
        outer_activation=nnx.identity,
        seed=99,
    )
    patch_all_mlps_to_identity(mf)

    out, _ = mf(graph=g, coordinates=coords, get_info=False)
    expected = compute_expected_local_sum(g, coords, mf.mlp_tree, nnx.identity)
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-6, atol=1e-6)


def test_multiple_edges_and_ports_independent_processing():
    # Create graph with two edges "arrow" and "source" with distinct constant mlp outputs; verify sum is correct
    n_addr = 4
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [2.0, -1.0]])
    addr_a0 = jnp.array([0, 1, 2])
    addr_a1 = jnp.array([1, 2, 3])
    addr_b = jnp.array([0, 1, 3])
    edge_a = JaxEdge(
        address_dict={"from": addr_a0, "to": addr_a1}, feature_array=None, feature_names=None, non_fictitious=jnp.ones((3,))
    )
    edge_b = JaxEdge(address_dict={"id": addr_b}, feature_array=None, feature_names=None, non_fictitious=jnp.ones((3,)))
    g = JaxGraph(
        edges={"arrow": edge_a, "source": edge_b},
        non_fictitious_addresses=jnp.ones((n_addr,)),
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    mf = LocalSumMessageFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coords.shape[1],
        hidden_sizes=[],
        activation=None,
        out_size=2,
        outer_activation=nnx.identity,
        seed=44,
    )
    # patch arrow ports to constant [1,0], source port to constant [0,2]
    for pk in mf.mlp_tree["arrow"].keys():
        mf.mlp_tree["arrow"][pk] = ConstantMLP(jnp.array([1.0, 0.0]))
    for pk in mf.mlp_tree["source"].keys():
        mf.mlp_tree["source"][pk] = ConstantMLP(jnp.array([0.0, 2.0]))

    out, _ = mf(graph=g, coordinates=coords, get_info=False)
    # compute expected via compute_expected_local_sum with a custom mlp mapping
    mlp_map = {
        "arrow": {
            p: (lambda x, v=jnp.array([1.0, 0.0]): jnp.tile(v[None, :], (x.shape[0], 1))) for p in mf.mlp_tree["arrow"].keys()
        },
        "source": {
            p: (lambda x, v=jnp.array([0.0, 2.0]): jnp.tile(v[None, :], (x.shape[0], 1))) for p in mf.mlp_tree["source"].keys()
        },
    }
    expected = compute_expected_local_sum(g, coords, mlp_map, nnx.identity)
    np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-6, atol=1e-6)


def test_deterministic_with_seed():
    mf1 = LocalSumMessageFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=7,
    )
    mf2 = LocalSumMessageFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coordinates.shape[1],
        hidden_sizes=[4],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=7,
    )
    out1, _ = mf1(graph=jax_context, coordinates=coordinates, get_info=False)
    out2, _ = mf2(graph=jax_context, coordinates=coordinates, get_info=False)
    chex.assert_trees_all_equal(out1, out2)


def test_vmap_jit_safety_after_build():
    mf = LocalSumMessageFunction(
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
    g = JaxGraph(edges={}, non_fictitious_addresses=jnp.ones((5,)), true_shape=None, current_shape=None)
    mf = LocalSumMessageFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=3,
        hidden_sizes=[2],
        activation=None,
        out_size=3,
        outer_activation=nnx.identity,
        seed=9,
    )
    out, _ = mf(graph=g, coordinates=jnp.zeros((5, 3)), get_info=False)
    # Expect zeros with shape (n_addr, out_size)
    assert out.shape == (5, 3)


def test_addresses_out_of_bounds_handling():
    # Create edge with addresses containing out-of-bounds index
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    addr_from = jnp.array([0, 10])  # 10 is out of bounds
    addr_to = jnp.array([0, 1])
    edge = JaxEdge(
        address_dict={"from": addr_from, "to": addr_to}, feature_array=None, feature_names=None, non_fictitious=jnp.ones((2,))
    )
    g = JaxGraph(edges={"arrow": edge}, non_fictitious_addresses=jnp.ones((2,)), true_shape=None, current_shape=None)
    mf = LocalSumMessageFunction(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=coords.shape[1],
        hidden_sizes=[],
        activation=None,
        out_size=4,
        outer_activation=nnx.identity,
        seed=15,
    )
    patch_all_mlps_to_identity(mf)
    out, _ = mf(graph=g, coordinates=coords, get_info=False)
    # ensure it runs and shape is correct
    assert out.shape == (coords.shape[0], 4)
