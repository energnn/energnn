#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

import diffrax
import flax.linen as nn
import jax.numpy as jnp
import optax
from flax import nnx

from energnn.model import (
    CenterReduceNormalizer,
    LocalSumMessageFunction,
    MLPEncoder,
    MLPEquivariantDecoder,
    NeuralODECoupler,
    SimpleGNN,
)
from energnn.storage import DummyStorage
from energnn.tracker import DummyTracker
from energnn.trainer import SimpleTrainer
from tests.utils import TestProblemLoader

train_loader = TestProblemLoader(
    dataset_size=8,
    n_batch=4,
    context_edge_params={
        "node": {"n_obj": 10, "feature_list": ["a", "b"], "address_list": ["0"]},
        "edge": {"n_obj": 10, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
    },
    oracle_edge_params={
        "node": {"n_obj": 10, "feature_list": ["e"]},
        "edge": {"n_obj": 10, "feature_list": ["f"]},
    },
    n_addr=10,
    shuffle=True,
)

val_loader = TestProblemLoader(
    dataset_size=8,
    n_batch=4,
    context_edge_params={
        "node": {"n_obj": 10, "feature_list": ["a", "b"], "address_list": ["0"]},
        "edge": {"n_obj": 10, "feature_list": ["c", "d"], "address_list": ["0", "1"]},
    },
    oracle_edge_params={
        "node": {"n_obj": 10, "feature_list": ["e"]},
        "edge": {"n_obj": 10, "feature_list": ["f"]},
    },
    n_addr=10,
    shuffle=True,
)

storage = DummyStorage()
tracker = DummyTracker()


def test_simple_trainer(tmp_path):
    normalizer = CenterReduceNormalizer(update_limit=1000, use_running_average=False)
    encoder = MLPEncoder(hidden_size=[16], activation=nnx.relu, out_size=7, seed=64)
    coupler = NeuralODECoupler(
        phi_hidden_size=[16],
        phi_activation=nn.relu,
        phi_final_activation=nn.tanh,
        message_functions=[
            LocalSumMessageFunction(
                out_size=4, hidden_size=[8], activation=nn.relu, final_activation=nn.tanh, seed=64
            ),
            LocalSumMessageFunction(
                out_size=4, hidden_size=[8], activation=nn.relu, final_activation=nn.tanh, seed=64
            ),
        ],
        latent_dimension=8,
        dt=0.1,
        stepsize_controller=diffrax.ConstantStepSize(),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        solver=diffrax.Euler(),
        max_steps=1000,
        seed=64,
    )

    decoder = MLPEquivariantDecoder(
        out_structure={"node": {"e": jnp.array([0])}, "edge": {"f": jnp.array([0])}},
        hidden_size=[16],
        activation=nnx.relu,
        seed=64,
    )
    model = SimpleGNN(normalizer=normalizer, encoder=encoder, coupler=coupler, decoder=decoder)
    trainer = SimpleTrainer(model=model, optimizer=optax.adam(1e-3))

    _ = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=1,
        log_period=10,
        eval_period=10,
        out_dir=tmp_path,
        last_id="last",
        best_id="best",
        storage=storage,
        tracker=tracker,
    )

    path_amortizer = tmp_path / "last"
    new_amortizer = SimpleTrainer.load(path_amortizer)

    _ = new_amortizer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=1,
        log_period=10,
        eval_period=10,
        out_dir=tmp_path,
        last_id="last",
        best_id="best",
        storage=storage,
        tracker=tracker,
    )
