#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np
import chex
import jax
import jax.numpy as jnp

from energnn.model.decoder.equivariant_decoder import MLPEquivariantDecoder
from energnn.graph import GraphStructure, EdgeStructure
from energnn.graph.jax import JaxGraph, JaxEdge, JaxGraphShape
from tests.utils import TestProblemLoader

# Prepare deterministic data and loader
np.random.seed(0)
pb_loader = TestProblemLoader(seed=0, batch_size=4, n_max=10)
pb_batch = next(iter(pb_loader))
jax_context_batch, _ = pb_batch.get_context()
jax_context = jax.tree.map(lambda x: x[0], jax_context_batch)
coordinates = jnp.array(np.random.uniform(size=(10, 7)))
coordinates_batch = jnp.array(np.random.uniform(size=(4, 10, 7)))

# out_structure must be a GraphStructure
default_out_structure = GraphStructure(
    edges={
        "source": EdgeStructure(address_list=["id"], feature_list=["e"]),
        "arrow": EdgeStructure(address_list=["from", "to"], feature_list=["f"]),
    }
)


def assert_decoder_vmap_jit_output(*, decoder: MLPEquivariantDecoder, context: JaxGraph, coordinates: jax.Array):
    def apply(graph, coords, get_info):
        return decoder(graph=graph, coordinates=coords, get_info=get_info)

    # map over batch axis (graph batch and coords batch)
    apply_vmap = jax.vmap(apply, in_axes=(0, 0, None), out_axes=0)

    output_batch_1, infos_1 = apply_vmap(context, coordinates, False)
    output_batch_2, infos_2 = apply_vmap(context, coordinates, True)

    apply_vmap_jit = jax.jit(apply_vmap)
    output_batch_3, infos_3 = apply_vmap_jit(context, coordinates, False)
    output_batch_4, infos_4 = apply_vmap_jit(context, coordinates, True)

    chex.assert_trees_all_equal(output_batch_1, output_batch_2, output_batch_3, output_batch_4)
    chex.assert_trees_all_equal(infos_2, infos_4)
    assert infos_1 == {}
    assert infos_3 == {}


# -------------------------
# MLPEquivariantDecoder tests
# -------------------------


def test_mlp_equivariant_decoder_init_deterministic():
    """
    Two decoders created with the same seed must produce the same outputs on the same input.
    """
    dec1 = MLPEquivariantDecoder(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=7,
        out_structure=default_out_structure,
        activation=jax.nn.relu,
        hidden_sizes=[8],
        seed=3,
    )
    dec2 = MLPEquivariantDecoder(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=7,
        out_structure=default_out_structure,
        activation=jax.nn.relu,
        hidden_sizes=[8],
        seed=3,
    )

    out1, info1 = dec1(graph=jax_context, coordinates=coordinates, get_info=False)
    out2, info2 = dec2(graph=jax_context, coordinates=coordinates, get_info=False)

    chex.assert_trees_all_equal(out1, out2)
    assert info1 == {}
    assert info2 == {}


def test_mlp_equivariant_decoder_single_shapes_and_masking():
    # Construct custom graph where some objects are fictitious (mask 0)
    node_edge = jax_context.edges["source"]
    edge_edge = jax_context.edges["arrow"]

    def n_obj_from(e):
        if e.feature_array is not None:
            return int(e.feature_array.shape[0])
        return int(np.array(e.non_fictitious).shape[0])

    n_node = n_obj_from(node_edge)
    n_edge = n_obj_from(edge_edge)

    # set first element fictitious for source edge to test masking
    node_nf = jnp.array(np.array(node_edge.non_fictitious))
    node_nf = node_nf.at[0].set(0)
    e1 = JaxEdge(
        address_dict=node_edge.address_dict,
        feature_array=jnp.ones((n_node, 2)),
        feature_names={"a": jnp.array(0), "b": jnp.array(1)},
        non_fictitious=node_nf,
    )
    e2 = JaxEdge(
        address_dict=edge_edge.address_dict,
        feature_array=jnp.ones((n_edge, 3)),
        feature_names={"c": jnp.array(0), "d": jnp.array(1), "e": jnp.array(2)},
        non_fictitious=edge_edge.non_fictitious,
    )

    custom_graph = JaxGraph(
        edges={"source": e1, "arrow": e2},
        non_fictitious_addresses=jax_context.non_fictitious_addresses,
        true_shape=jax_context.true_shape,
        current_shape=jax_context.current_shape,
    )

    custom_in_structure = GraphStructure(
        edges={
            "source": EdgeStructure(address_list=["id"], feature_list=["a", "b"]),
            "arrow": EdgeStructure(address_list=["from", "to"], feature_list=["c", "d", "e"]),
        }
    )

    decoder = MLPEquivariantDecoder(
        in_graph_structure=custom_in_structure,
        in_array_size=7,
        out_structure=default_out_structure,
        activation=jax.nn.relu,
        hidden_sizes=[4],
        seed=4,
    )

    out, info = decoder(graph=custom_graph, coordinates=coordinates, get_info=True)

    # shapes
    assert set(out.edges.keys()) == set(default_out_structure.edges.keys())
    assert out.edges["source"].feature_array.shape == (n_node, len(default_out_structure.edges["source"].feature_list))
    assert out.edges["arrow"].feature_array.shape == (n_edge, len(default_out_structure.edges["arrow"].feature_list))

    # Masking: first row for source must be all zeros (we set non_fictitious[0]=0)
    node_out_np = np.array(out.edges["source"].feature_array)
    assert np.allclose(node_out_np[0], 0.0)
    # and at least one non-zero exists for other (unmasked) rows
    assert np.any(np.abs(node_out_np[1:]) > 1e-8)

    assert info == {}


def test_mlp_equivariant_decoder_batch_vmap_jit():
    decoder = MLPEquivariantDecoder(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=7,
        out_structure=default_out_structure,
        activation=jax.nn.relu,
        hidden_sizes=[8],
        seed=6,
    )
    assert_decoder_vmap_jit_output(decoder=decoder, context=jax_context_batch, coordinates=coordinates_batch)


def test_mlp_equivariant_decoder_mlp_dict_initialization():
    """
    Check that MLPs are correctly initialized in decoder.mlp_dict based on out_structure.
    """
    out_structure = GraphStructure(edges={"source": EdgeStructure(address_list=["id"], feature_list=["y0", "y1"])})
    decoder = MLPEquivariantDecoder(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=7,
        out_structure=out_structure,
        activation=jax.nn.relu,
        hidden_sizes=[4],
        seed=42,
    )

    assert isinstance(decoder.mlp_dict, dict)
    assert "source" in decoder.mlp_dict
    assert callable(decoder.mlp_dict["source"])


def test_mlp_equivariant_decoder_numeric_identity_node():
    """
    Make the source-MLP act like identity on gathered coordinates.
    Expected: coords[address] * non_fictitious
    """
    d = coordinates.shape[1]
    out_struct_node = GraphStructure(
        edges={"source": EdgeStructure(address_list=["id"], feature_list=[f"o{i}" for i in range(d)])}
    )
    decoder = MLPEquivariantDecoder(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=d,
        out_structure=out_struct_node,
        activation=None,
        hidden_sizes=[],
        seed=100,
    )

    # function that returns only the first d columns (coords)
    def select_coords(x):
        return x[..., :d]

    decoder.mlp_dict["source"] = select_coords

    out_graph, _ = decoder(graph=jax_context, coordinates=coordinates, get_info=False)
    node_out = out_graph.edges["source"].feature_array  # shape (n_obj, d)
    node_edge = jax_context.edges["source"]
    addr = np.array(node_edge.address_dict["id"]).astype(int)
    coords = np.array(coordinates)
    nf = np.array(node_edge.non_fictitious).astype(float)
    expected = coords[addr] * nf[:, None]

    np.testing.assert_allclose(np.array(node_out), expected, rtol=0.0, atol=1e-6)


def test_mlp_equivariant_decoder_numeric_identity_edge():
    """
    Make the arrow-MLP act like identity on [coords(addr0), coords(addr1), features].
    Expected: concat(coords[addr0], coords[addr1], feature_array) * non_fictitious
    """
    d = coordinates.shape[1]
    edge_feature_dim = int(jax_context.edges["arrow"].feature_array.shape[1])
    input_dim = 2 * d + edge_feature_dim
    out_struct_edge = GraphStructure(
        edges={"arrow": EdgeStructure(address_list=["from", "to"], feature_list=[f"o{i}" for i in range(input_dim)])}
    )

    decoder = MLPEquivariantDecoder(
        in_graph_structure=pb_loader.context_structure,
        in_array_size=d,
        out_structure=out_struct_edge,
        activation=None,
        hidden_sizes=[],
        seed=101,
    )

    def identity(x):
        return x

    decoder.mlp_dict["arrow"] = identity

    out_graph, _ = decoder(graph=jax_context, coordinates=coordinates, get_info=False)
    edge_out = out_graph.edges["arrow"].feature_array  # shape (n_obj, input_dim)

    edge = jax_context.edges["arrow"]
    addr0 = np.array(edge.address_dict["from"]).astype(int)
    addr1 = np.array(edge.address_dict["to"]).astype(int)
    coords = np.array(coordinates)
    feats = np.array(edge.feature_array)
    nf = np.array(edge.non_fictitious).astype(float)

    expected = np.concatenate([coords[addr0], coords[addr1], feats], axis=1) * nf[:, None]

    np.testing.assert_allclose(np.array(edge_out), expected, rtol=0.0, atol=1e-6)
