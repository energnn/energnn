#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import diffrax
import flax
import optax
from energnn.model_registry import DummyRegistry
from jax.random import PRNGKey
from omegaconf import DictConfig

from energnn.amortizer import SimpleAmortizer
from energnn.gnn import Coupler, EquivariantGNN, IdentityEncoder
from energnn.gnn.coupler.coupling_function import CouplingFunction
from energnn.gnn.coupler.coupling_function import IdentityRemoteMessageFunction
from energnn.gnn.coupler.coupling_function import MLPSelfMessageFunction
from energnn.gnn.coupler.coupling_function import SumLocalMessageFunction
from energnn.gnn.coupler.solving_method import NeuralODESolvingMethod
from energnn.gnn.decoder import MLPEquivariantDecoder
from energnn.gnn.utils import MLP
from energnn.normalizer import Postprocessor, Preprocessor
from energnn.normalizer.normalization_function import CenterReduceFunction
from energnn.tracker import DummyTracker
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

#TODO : change to LocalRegistry
registry = DummyRegistry()
tracker = DummyTracker()


def test_create(tmp_path):
    preprocessor = Preprocessor(f=CenterReduceFunction())
    postprocessor = Postprocessor(f=CenterReduceFunction())
    gnn = EquivariantGNN(
        encoder=IdentityEncoder(),
        coupler=Coupler(
            solving_method=NeuralODESolvingMethod(
                latent_dimension=4,
                dt=0.25,
                stepsize_controller=diffrax._step_size_controller.ConstantStepSize(),
                solver=diffrax._solver.Euler(),
                adjoint=diffrax._adjoint.RecursiveCheckpointAdjoint(),
                max_steps=1000,
            ),
            coupling_function=CouplingFunction(
                phi=MLP(hidden_size=[8], activation=flax.linen.relu, out_size=4),
                self_message_function=MLPSelfMessageFunction(
                    hidden_size=[8], out_size=4, activation=flax.linen.relu, final_layer_activation=flax.linen.relu
                ),
                local_message_function=SumLocalMessageFunction(
                    hidden_size=[8], out_size=4, activation=flax.linen.relu, final_activation=flax.linen.relu
                ),
                remote_message_function=IdentityRemoteMessageFunction(),
            ),
        ),
        decoder=MLPEquivariantDecoder(hidden_size=[8], activation=flax.linen.relu),
    )
    optimizer = optax.adam(1e-3)
    amortizer = SimpleAmortizer(preprocessor=preprocessor, postprocessor=postprocessor, gnn=gnn, optimizer=optimizer,
                                project_name="test-amortizer", run_id="test-run")

    problem_cfg = DictConfig({})
    amortizer.init(rngs=PRNGKey(0), loader=train_loader, problem_cfg=problem_cfg)

    _ = amortizer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        problem_cfg=problem_cfg,
        n_epochs=1,
        registry=registry,
        tracker=tracker,
    )

    #TODO : retry with LocalRegistry
    # path_amortizer = tmp_path / "last"
    # new_amortizer = SimpleAmortizer.load(path_amortizer)
    #
    # _ = new_amortizer.train(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     problem_cfg=DictConfig({}),
    #     n_epochs=1,
    #     registry=registry,
    #     tracker=tracker,
    # )
