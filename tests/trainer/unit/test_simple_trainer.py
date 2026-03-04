#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from unittest.mock import MagicMock

from energnn.trainer import SimpleTrainer
from energnn.trainer.simple_trainer import _cast_cotangent_to_primal_dtype
from energnn.model import SimpleGNN, IdentityEncoder
from energnn.graph import JaxGraph, JaxEdge
from tests.utils import TestProblemLoader


class IdentityNormalizer(nnx.Module):
    def __call__(self, graph, get_info=False):
        return graph, {}


def create_tiny_model(context_structure):
    class SimpleDecoder(nnx.Module):
        def __call__(self, coordinates, graph, get_info=False):
            # No params here, just pass through
            decision = JaxGraph(
                edges={
                    "source": JaxEdge(
                        address_dict=None,
                        feature_array=coordinates,
                        feature_names={"value": jnp.array(0)},
                        non_fictitious=jnp.ones(coordinates.shape[0]),
                    )
                },
                non_fictitious_addresses=jnp.ones(coordinates.shape[0]),
                true_shape=graph.true_shape,
                current_shape=graph.current_shape,
            )
            return decision, {}

    class SimpleCoupler(nnx.Module):
        def __init__(self):
            # One param to update
            self.linear = nnx.Linear(1, 1, rngs=nnx.Rngs(1))

        def __call__(self, graph, get_info=False):
            x = graph.edges["source"].feature_array
            return self.linear(x), {}

    return SimpleGNN(
        normalizer=IdentityNormalizer(),
        encoder=IdentityEncoder(),
        coupler=SimpleCoupler(),
        decoder=SimpleDecoder(),
    )


def test_cast_cotangent_to_primal_dtype():
    primal = {"a": jnp.array([1.0], dtype=jnp.float32), "b": jnp.array([1], dtype=jnp.int32), "c": "not-an-array"}
    cotangent = {"a": jnp.array([2.0], dtype=jnp.float64), "b": jnp.array([2.0], dtype=jnp.float32), "c": "not-an-array"}

    casted = _cast_cotangent_to_primal_dtype(cotangent, primal)

    assert casted["a"].dtype == jnp.float32
    assert casted["b"].dtype == jnp.int32
    assert casted["c"] == "not-an-array"


def test_trainer_init():
    loader = TestProblemLoader()
    model = create_tiny_model(loader.context_structure)
    optimizer = optax.sgd(1e-3)
    trainer = SimpleTrainer(model=model, gradient_transformation=optimizer)

    assert trainer.model is model
    assert isinstance(trainer.optimizer, nnx.Optimizer)
    assert trainer.train_step == 0
    assert trainer.best_metrics == float("inf")


def test_training_step_basic():
    loader = TestProblemLoader(dataset_size=4, batch_size=4)
    model = create_tiny_model(loader.context_structure)
    # Using SGD with huge learning rate to be absolutely sure we see a change
    optimizer = optax.sgd(100.0)
    trainer = SimpleTrainer(model=model, gradient_transformation=optimizer)

    batch = next(iter(loader))

    # Get initial parameter values
    params = nnx.state(model, nnx.Param)
    leaves_before = jax.tree.leaves(params)

    # Perform one training step
    infos = trainer.training_step(batch, get_info=True)

    assert isinstance(infos, dict)
    assert any(k.startswith("1_context") for k in infos.keys())
    assert any(k.startswith("3_gradient") for k in infos.keys())
    assert any(k.startswith("4_update") for k in infos.keys())

    # Get updated parameter values
    params_after = nnx.state(model, nnx.Param)
    leaves_after = jax.tree.leaves(params_after)

    # Check if any parameter has changed
    changed = False
    for b, a in zip(leaves_before, leaves_after):
        if not jnp.allclose(b, a, atol=1e-7):
            changed = True
            break

    assert changed, "Parameters did not change after training step"


def test_eval_step():
    loader = TestProblemLoader(dataset_size=4, batch_size=4)
    model = create_tiny_model(loader.context_structure)
    trainer = SimpleTrainer(model=model, gradient_transformation=optax.sgd(1e-3))

    batch = next(iter(loader))
    metrics, infos = trainer.eval_step(0, batch)

    assert isinstance(metrics, list)
    assert len(metrics) == 4
    assert isinstance(infos, dict)
    assert any(k.startswith("1_context") for k in infos.keys())
    assert any(k.startswith("2_forward") for k in infos.keys())
    assert any(k.startswith("3_metrics") for k in infos.keys())


def test_eval():
    loader = TestProblemLoader(dataset_size=8, batch_size=4)
    model = create_tiny_model(loader.context_structure)
    trainer = SimpleTrainer(model=model, gradient_transformation=optax.sgd(1e-3))

    metrics, infos = trainer.eval(loader)

    assert isinstance(metrics, float)
    assert isinstance(infos, dict)
    assert "metrics" in infos
    assert infos["metrics"] == metrics


def test_run_evaluation_updates_best_metrics():
    loader = TestProblemLoader(dataset_size=4, batch_size=4)
    model = create_tiny_model(loader.context_structure)
    trainer = SimpleTrainer(model=model, gradient_transformation=optax.sgd(1e-3))

    trainer.eval = MagicMock(return_value=(0.5, {"some": "info"}))

    res = trainer.run_evaluation(val_loader=loader)
    assert res == 0.5
    assert trainer.best_metrics == 0.5

    # Second call with worse metrics
    trainer.eval = MagicMock(return_value=(0.8, {"some": "info"}))
    res = trainer.run_evaluation(val_loader=loader)
    assert res == 0.8
    assert trainer.best_metrics == 0.5  # Kept previous best


def test_save_load_checkpoint(tmp_path):
    from orbax.checkpoint import CheckpointManager

    loader = TestProblemLoader()
    model = create_tiny_model(loader.context_structure)
    trainer = SimpleTrainer(model=model, gradient_transformation=optax.sgd(1e-3))
    trainer.train_step = 42

    m_cp = MagicMock(spec=CheckpointManager)
    m_cp.directory = tmp_path
    m_cp.save.return_value = True

    path = trainer.save_checkpoint(checkpoint_manager=m_cp, metrics=0.123)
    assert path == str(tmp_path / "42")
    m_cp.save.assert_called_once()

    # Load checkpoint
    m_cp.latest_step.return_value = 42
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(trainer.optimizer)
    restored_data = {"default": {"model": model_state, "optimizer": opt_state, "step": 42, "metrics": 0.123}}
    m_cp.restore.return_value = restored_data

    trainer.train_step = 0  # reset
    trainer.load_checkpoint(checkpoint_manager=m_cp)
    assert trainer.train_step == 42


def test_train_loop_basic():
    # Small loaders
    train_loader = TestProblemLoader(dataset_size=4, batch_size=2)
    val_loader = TestProblemLoader(dataset_size=2, batch_size=2)

    model = create_tiny_model(train_loader.context_structure)
    trainer = SimpleTrainer(model=model, gradient_transformation=optax.sgd(1e-3))

    # Mock run_evaluation to avoid real eval overhead and just track calls
    trainer.run_evaluation = MagicMock(return_value=0.1)

    n_epochs = 2
    res = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        eval_period=1,  # Eval every step
        log_period=1,
        progress_bar=False,
        eval_before_training=True,
    )

    # 2 epochs, 2 batches per epoch -> 4 training steps
    assert trainer.train_step == 4
    # Expected calls to run_evaluation: 1 (before) + 4 (during each step) = 5
    assert trainer.run_evaluation.call_count == 5


def test_train_with_tracker_and_storage():
    train_loader = TestProblemLoader(dataset_size=2, batch_size=2)
    val_loader = TestProblemLoader(dataset_size=2, batch_size=2)
    model = create_tiny_model(train_loader.context_structure)
    trainer = SimpleTrainer(model=model, gradient_transformation=optax.sgd(1e-3))

    from energnn.tracker import Tracker
    from energnn.storage import Storage
    from orbax.checkpoint import CheckpointManager

    m_tracker = MagicMock(spec=Tracker)
    m_storage = MagicMock(spec=Storage)
    m_cp = MagicMock(spec=CheckpointManager)
    m_cp.save.return_value = True
    m_cp.directory = MagicMock()
    m_cp.directory.__truediv__.return_value = "path/to/ckpt"

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=1,
        tracker=m_tracker,
        storage=m_storage,
        checkpoint_manager=m_cp,
        log_period=1,
        progress_bar=False,
    )

    # run_evaluation called (at least after epoch)
    # run_evaluation calls save_checkpoint which calls m_cp.save
    assert m_cp.save.called
    # run_evaluation calls storage.upload if local_ckpt_path is not None
    assert m_storage.upload.called
    # m_cp.wait_until_finished should be called at the end of train
    assert m_cp.wait_until_finished.called
