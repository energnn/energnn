#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from energnn.model.ready_to_use import TinyRecurrentEquivariantGNN
from energnn.model.utils import MLP
from energnn.problem.example import LinearSystemProblemLoader

pb_loader = LinearSystemProblemLoader(seed=0, batch_size=4, n_max=10)
pb_batch = next(iter(pb_loader))
context_batch, _ = pb_batch.get_context()


def test_ready_defaults_are_fused_fp32():
    model = TinyRecurrentEquivariantGNN(
        in_structure=pb_loader.context_structure, out_structure=pb_loader.decision_structure, seed=0
    )
    mf = model.coupler.message_functions[0]
    assert mf.fuse_ports is True
    # fused mode: one single MLP per hyper-edge class
    for key in mf.mlp_tree:
        assert isinstance(mf.mlp_tree[key], MLP)
    # full float32 computation by default
    assert mf.dtype is None
    assert mf.scatter_dtype is None
    assert model.coupler.phi.dtype is None
    assert model.encoder.dtype is None
    assert model.decoder.dtype is None
    for leaf in jax.tree.leaves(nnx.state(model, nnx.Param)):
        assert leaf.dtype == jnp.float32


def test_ready_bf16_opt_in_propagates_to_all_components():
    model = TinyRecurrentEquivariantGNN(
        in_structure=pb_loader.context_structure,
        out_structure=pb_loader.decision_structure,
        dtype=jnp.bfloat16,
        scatter_dtype=jnp.float32,
        seed=0,
    )
    mf = model.coupler.message_functions[0]
    assert mf.dtype == jnp.bfloat16
    assert mf.scatter_dtype == jnp.float32
    assert model.coupler.phi.dtype == jnp.bfloat16
    assert model.encoder.dtype == jnp.bfloat16
    assert model.decoder.dtype == jnp.bfloat16
    # parameters remain stored in float32
    for leaf in jax.tree.leaves(nnx.state(model, nnx.Param)):
        assert leaf.dtype == jnp.float32
    decision, _ = model.forward_batch(graph=context_batch, step_with_metrics=False)
    assert decision.feature_flat_array.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(decision.feature_flat_array)))


def test_ready_defaults_forward_batch_is_finite_float32():
    model = TinyRecurrentEquivariantGNN(
        in_structure=pb_loader.context_structure, out_structure=pb_loader.decision_structure, seed=0
    )
    decision, _ = model.forward_batch(graph=context_batch, step_with_metrics=False)
    flat = decision.feature_flat_array
    assert flat.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(flat)))


def test_ready_opt_out_recovers_per_port():
    model = TinyRecurrentEquivariantGNN(
        in_structure=pb_loader.context_structure,
        out_structure=pb_loader.decision_structure,
        fuse_ports=False,
        seed=0,
    )
    mf = model.coupler.message_functions[0]
    assert mf.fuse_ports is False
    # per-port mode: one dict of MLPs per hyper-edge class
    for key in mf.mlp_tree:
        assert isinstance(mf.mlp_tree[key], dict)
    decision, _ = model.forward_batch(graph=context_batch, step_with_metrics=False)
    assert bool(jnp.all(jnp.isfinite(decision.feature_flat_array)))


def test_ready_bf16_close_to_fp32():
    kwargs = dict(in_structure=pb_loader.context_structure, out_structure=pb_loader.decision_structure, seed=0)
    model_bf16 = TinyRecurrentEquivariantGNN(**kwargs, dtype=jnp.bfloat16)
    model_fp32 = TinyRecurrentEquivariantGNN(**kwargs)
    out_bf16, _ = model_bf16.forward_batch(graph=context_batch, step_with_metrics=False)
    out_fp32, _ = model_fp32.forward_batch(graph=context_batch, step_with_metrics=False)
    np.testing.assert_allclose(
        np.array(out_bf16.feature_flat_array), np.array(out_fp32.feature_flat_array), rtol=0.1, atol=0.1
    )
