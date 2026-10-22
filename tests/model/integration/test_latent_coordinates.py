#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
"""End-to-end checks of GNN.compute_latent_coordinates(_batch) on a real model."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from energnn.graph import separate_graphs
from energnn.model.decoder.decoder import Decoder
from energnn.model.ready_to_use import TinyRecurrentEquivariantGNN
from energnn.problem.example import LinearSystemProblemLoader

BATCH_SIZE = 4
N_MAX = 10

pb_loader = LinearSystemProblemLoader(seed=0, batch_size=BATCH_SIZE, n_max=N_MAX)
pb_batch = next(iter(pb_loader))
context_batch, _ = pb_batch.get_context()
context_list = separate_graphs(context_batch)


@pytest.fixture
def model():
    return TinyRecurrentEquivariantGNN(
        in_structure=pb_loader.context_structure, out_structure=pb_loader.decision_structure, seed=0
    )


def _state_as_numpy(model):
    return jax.tree.map(np.array, nnx.state(model))


def _assert_states_equal(state_a, state_b):
    leaves_a, treedef_a = jax.tree.flatten(state_a)
    leaves_b, treedef_b = jax.tree.flatten(state_b)
    assert treedef_a == treedef_b
    for a, b in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(a, b)


def test_batch_shape_dtype_and_finiteness(model):
    latent_dim = model.coupler.phi.out_size
    latents = model.compute_latent_coordinates_batch(graph=context_batch)
    assert latents.shape == (BATCH_SIZE, N_MAX, latent_dim)
    assert latents.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(latents)))


def test_single_shape_dtype_and_finiteness(model):
    latent_dim = model.coupler.phi.out_size
    graph = context_list[0]
    latents = model.compute_latent_coordinates(graph=graph)
    assert latents.shape == (graph.non_fictitious_addresses.shape[0], latent_dim)
    assert latents.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(latents)))


def test_batch_matches_separate_graphs_in_eval_mode(model):
    """Batched and per-graph computations agree on non-fictitious addresses once the normalizer is frozen."""
    model.eval()
    latents_batch = model.compute_latent_coordinates_batch(graph=context_batch)
    for i, graph in enumerate(context_list):
        latents_single = model.compute_latent_coordinates(graph=graph)
        mask = np.array(graph.non_fictitious_addresses, dtype=bool)
        np.testing.assert_allclose(np.array(latents_single)[mask], np.array(latents_batch[i])[mask], rtol=1e-5, atol=1e-6)


def test_single_matches_coordinates_fed_to_decoder(model):
    """The returned latents are exactly those the decoder consumes in a full forward pass."""

    class CapturingDecoder(Decoder):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.coordinates = nnx.data(None)

        def __call__(self, coordinates, graph, step_with_metrics: bool = False):
            self.coordinates = nnx.data(coordinates)
            return self.wrapped(coordinates=coordinates, graph=graph, step_with_metrics=step_with_metrics)

    model.eval()
    model.decoder = CapturingDecoder(model.decoder)
    graph = context_list[0]

    model(graph=graph, step_with_metrics=False)
    latents = model.compute_latent_coordinates(graph=graph)

    np.testing.assert_allclose(np.array(latents), np.array(model.decoder.coordinates), rtol=1e-6, atol=1e-7)


def test_eval_mode_does_not_mutate_state(model):
    """In eval mode the computation is pure: no parameter nor normalizer statistic is touched."""
    model.eval()
    before = _state_as_numpy(model)
    model.compute_latent_coordinates_batch(graph=context_batch)
    model.compute_latent_coordinates(graph=context_list[0])
    _assert_states_equal(before, _state_as_numpy(model))


def test_train_mode_updates_normalizer_like_forward(model):
    """In train mode the normalizer statistics are updated, exactly as a forward_batch call would do."""
    model_ref = TinyRecurrentEquivariantGNN(
        in_structure=pb_loader.context_structure, out_structure=pb_loader.decision_structure, seed=0
    )
    before = _state_as_numpy(model)

    model.compute_latent_coordinates_batch(graph=context_batch)
    model_ref.forward_batch(graph=context_batch, step_with_metrics=False)

    after = _state_as_numpy(model)
    with pytest.raises(AssertionError):
        _assert_states_equal(before, after)
    _assert_states_equal(nnx.state(model, nnx.Param), nnx.state(model_ref, nnx.Param))
    _assert_states_equal(after, _state_as_numpy(model_ref))


def test_batch_is_jittable(model):
    model.eval()
    eager = model.compute_latent_coordinates_batch(graph=context_batch)

    @nnx.jit
    def jitted(model_, graph):
        return model_.compute_latent_coordinates_batch(graph=graph)

    np.testing.assert_allclose(np.array(jitted(model, context_batch)), np.array(eager), rtol=1e-6, atol=1e-7)


def test_is_differentiable_wrt_params(model):
    """Latent coordinates can be used as a training target: gradients w.r.t. parameters are finite and non-zero."""
    model.eval()

    def loss(model_):
        return jnp.sum(model_.compute_latent_coordinates_batch(graph=context_batch) ** 2)

    grads = nnx.grad(loss)(model)
    leaves = jax.tree.leaves(grads)
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(bool(jnp.any(g != 0)) for g in leaves)
