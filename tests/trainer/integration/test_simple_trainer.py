# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import diffrax
import optax
import orbax.checkpoint as ocp
from flax import nnx

from energnn.model import (
    CenterReduceNormalizer,
    LocalSumMessageFunction,
    MLP,
    MLPEncoder,
    MLPEquivariantDecoder,
    NeuralODECoupler,
    SimpleGNN,
)
from energnn.trainer import SimpleTrainer
from tests.utils import TestProblemLoader


def test_simple_trainer(tmp_path):

    train_loader = TestProblemLoader(seed=0)
    val_loader = TestProblemLoader(seed=1)

    normalizer = CenterReduceNormalizer(
        update_limit=1000,
        in_structure=train_loader.context_structure,
    )
    encoder = MLPEncoder(
        in_structure=train_loader.context_structure,
        hidden_sizes=[],
        final_activation=nnx.leaky_relu,
        out_size=4,
        seed=64,
    )
    coupler = NeuralODECoupler(
        phi=MLP(in_size=4, hidden_sizes=[], out_size=4, seed=64, final_activation=nnx.tanh),
        message_functions=[
            LocalSumMessageFunction(
                in_graph_structure=train_loader.context_structure,
                in_array_size=4,
                out_size=4,
                hidden_sizes=[4],
                activation=nnx.leaky_relu,
                final_activation=nnx.tanh,
                outer_activation=nnx.tanh,
                encoded_feature_size=4,
                seed=64,
            )
        ],
        dt=0.25,
        stepsize_controller=diffrax.ConstantStepSize(),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        solver=diffrax.Euler(),
        max_steps=10,
    )

    decoder = MLPEquivariantDecoder(
        in_array_size=4,
        in_graph_structure=train_loader.context_structure,
        encoded_feature_size=4,
        out_structure=train_loader.decision_structure,
        hidden_sizes=[4],
        activation=nnx.leaky_relu,
        seed=64,
    )
    model = SimpleGNN(normalizer=normalizer, encoder=encoder, coupler=coupler, decoder=decoder)

    ckpt_manager = ocp.CheckpointManager(
        directory=tmp_path / "test_trainer", options=ocp.CheckpointManagerOptions(max_to_keep=10)
    )

    trainer = SimpleTrainer(
        model=model,
        gradient_transformation=optax.adam(1e-3),
    )

    _ = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=ckpt_manager,
        n_epochs=1,
        log_period=None,
        eval_period=None,
        progress_bar=True,
        eval_before_training=False,
        eval_after_epoch=True,
    )
    metrics_1 = trainer.run_evaluation(val_loader=val_loader, progress_bar=True)

    _ = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=ckpt_manager,
        n_epochs=1,
        log_period=None,
        eval_period=None,
        progress_bar=True,
        eval_before_training=False,
        eval_after_epoch=False,
    )
    metrics_2 = trainer.run_evaluation(val_loader=val_loader, progress_bar=True)

    new_ckpt_manager = ocp.CheckpointManager(directory=tmp_path / "test_trainer")

    trainer.load_checkpoint(new_ckpt_manager, step=4)
    metrics_3 = trainer.run_evaluation(val_loader=val_loader, progress_bar=True)

    _ = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=new_ckpt_manager,
        n_epochs=1,
        log_period=None,
        eval_period=None,
        progress_bar=True,
        eval_before_training=False,
        eval_after_epoch=False,
    )
    metrics_4 = trainer.run_evaluation(val_loader=val_loader, progress_bar=True)

    assert metrics_1 == metrics_3 != metrics_2 == metrics_4
