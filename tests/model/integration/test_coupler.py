# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import diffrax
from flax import nnx

from energnn.model import (
    GATv2MessagePassingFunction,
    GlobalAggregationMessagePassingFunction,
    LocalSumMessagePassingFunction,
    MLP,
    MultiHeadQKVMessagePassingFunction,
    PerformerMessagePassingFunction,
    NODECoupler,
    RecurrentCoupler,
)
from energnn.model.coupler.message_passing.recurrent_coupler import (
    VirtualAddressRecurrentCoupler,
)
from energnn.problem.example import LinearSystemProblemLoader


def test_neural_ode_coupler():
    loader = LinearSystemProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    coupler = NODECoupler(
        phi=MLP(in_size=4, hidden_sizes=[], out_size=4, seed=64, final_activation=nnx.tanh),
        message_functions=[
            LocalSumMessagePassingFunction(
                in_graph_structure=loader.context_structure,
                in_array_size=4,
                out_size=4,
                hidden_sizes=[4],
                activation=nnx.leaky_relu,
                final_activation=nnx.tanh,
                outer_activation=nnx.tanh,
                seed=64,
            )
        ],
        dt=0.25,
        stepsize_controller=diffrax.ConstantStepSize(),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        solver=diffrax.Euler(),
        max_steps=10,
    )

    def f(coupler, graph):
        return coupler(graph=graph, get_info=False)

    coupler_vmap = nnx.jit(nnx.vmap(f, in_axes=(None, 0), out_axes=0))

    coordinates_batch, _ = coupler_vmap(coupler=coupler, graph=context_batch)


def test_recurrent_coupler_with_gatv2_message_function():
    loader = LinearSystemProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    coupler = RecurrentCoupler(
        phi=MLP(in_size=4, hidden_sizes=[], out_size=4, seed=64, final_activation=nnx.tanh),
        message_functions=[
            GATv2MessagePassingFunction(
                in_graph_structure=loader.context_structure,
                in_array_size=4,
                out_size=4,
                hidden_sizes=[4],
                activation=nnx.leaky_relu,
                final_activation=None,
                outer_activation=nnx.tanh,
                seed=64,
            )
        ],
        n_steps=4,
    )

    def f(coupler, graph):
        return coupler(graph=graph, get_info=False)

    coupler_vmap = nnx.jit(nnx.vmap(f, in_axes=(None, 0), out_axes=0))

    coordinates_batch, _ = coupler_vmap(coupler=coupler, graph=context_batch)


def test_recurrent_coupler_with_gatv2_score_uses_receiver():
    loader = LinearSystemProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    coupler = RecurrentCoupler(
        phi=MLP(in_size=4, hidden_sizes=[], out_size=4, seed=64, final_activation=nnx.tanh),
        message_functions=[
            GATv2MessagePassingFunction(
                in_graph_structure=loader.context_structure,
                in_array_size=4,
                out_size=4,
                hidden_sizes=[4],
                activation=nnx.leaky_relu,
                final_activation=None,
                outer_activation=nnx.tanh,
                score_uses_receiver=True,
                seed=64,
            )
        ],
        n_steps=4,
    )

    def f(coupler, graph):
        return coupler(graph=graph, get_info=False)

    coupler_vmap = nnx.jit(nnx.vmap(f, in_axes=(None, 0), out_axes=0))

    coordinates_batch, _ = coupler_vmap(coupler=coupler, graph=context_batch)




def test_recurrent_coupler_with_global_aggregation_message_function():
    """RecurrentCoupler wraps GlobalAggregationMessagePassingFunction (Item 2 of
    attention-backlog sec 3.2): single per-address value MLP, mean-reduced with
    corrected denominator (sum non_fictitious + eps), broadcast to every real
    receiver. Verifies the message function plugs into the coupler interface
    and survives vmap + jit across a batched input."""
    loader = LinearSystemProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    coupler = RecurrentCoupler(
        phi=MLP(in_size=4, hidden_sizes=[], out_size=4, seed=64, final_activation=nnx.tanh),
        message_functions=[
            GlobalAggregationMessagePassingFunction(
                in_graph_structure=loader.context_structure,
                in_array_size=4,
                out_size=4,
                hidden_sizes=[4],
                activation=nnx.leaky_relu,
                final_activation=None,
                outer_activation=nnx.tanh,
                seed=64,
            )
        ],
        n_steps=4,
    )

    def f(coupler, graph):
        return coupler(graph=graph, get_info=False)

    coupler_vmap = nnx.jit(nnx.vmap(f, in_axes=(None, 0), out_axes=0))

    coordinates_batch, _ = coupler_vmap(coupler=coupler, graph=context_batch)



def test_recurrent_coupler_with_multi_head_qkv_message_function():
    """RecurrentCoupler wraps MultiHeadQKVMessagePassingFunction (Item 3 of
    attention-backlog sec 3.3): single per-address Q MLP, per-(class, port)
    K and V MLP trees, bilinear K^T Q score with sqrt(d_qk) scaling,
    scatter_add weighted-value aggregation. Verifies the message function
    plugs into the coupler interface and survives vmap + jit across a
    batched input."""
    loader = LinearSystemProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    coupler = RecurrentCoupler(
        phi=MLP(in_size=4, hidden_sizes=[], out_size=4, seed=64, final_activation=nnx.tanh),
        message_functions=[
            MultiHeadQKVMessagePassingFunction(
                in_graph_structure=loader.context_structure,
                in_array_size=4,
                hidden_sizes=[4],
                d_qk=8,
                out_size=4,
                activation=nnx.leaky_relu,
                final_activation=None,
                outer_activation=nnx.tanh,
                seed=64,
            )
        ],
        n_steps=4,
    )

    def f(coupler, graph):
        return coupler(graph=graph, get_info=False)

    coupler_vmap = nnx.jit(nnx.vmap(f, in_axes=(None, 0), out_axes=0))

    coordinates_batch, _ = coupler_vmap(coupler=coupler, graph=context_batch)



def test_recurrent_coupler_with_performer_message_function():
    """RecurrentCoupler wraps PerformerMessagePassingFunction (Item 4 of
    attention-backlog sec 3.4): three per-address MLPs (Q, K, V),
    all-to-all linear attention with kernel-trick aggregation
    (sum V K^T) Q and sqrt(d_qk) scaling, no softmax, no random-feature
    kernel. Verifies the message function plugs into the coupler interface
    and survives vmap + jit across a batched input."""
    loader = LinearSystemProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    coupler = RecurrentCoupler(
        phi=MLP(in_size=4, hidden_sizes=[], out_size=4, seed=64, final_activation=nnx.tanh),
        message_functions=[
            PerformerMessagePassingFunction(
                in_graph_structure=loader.context_structure,
                in_array_size=4,
                hidden_sizes=[4],
                d_qk=8,
                out_size=4,
                activation=nnx.leaky_relu,
                final_activation=None,
                outer_activation=nnx.tanh,
                seed=64,
            )
        ],
        n_steps=4,
    )

    def f(coupler, graph):
        return coupler(graph=graph, get_info=False)

    coupler_vmap = nnx.jit(nnx.vmap(f, in_axes=(None, 0), out_axes=0))

    coordinates_batch, _ = coupler_vmap(coupler=coupler, graph=context_batch)

def test_recurrent_coupler():
    loader = LinearSystemProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    coupler = RecurrentCoupler(
        phi=MLP(in_size=4, hidden_sizes=[], out_size=4, seed=64, final_activation=nnx.tanh),
        message_functions=[
            LocalSumMessagePassingFunction(
                in_graph_structure=loader.context_structure,
                in_array_size=4,
                out_size=4,
                hidden_sizes=[4],
                activation=nnx.leaky_relu,
                final_activation=nnx.tanh,
                outer_activation=nnx.tanh,
                seed=64,
            )
        ],
        n_steps=4,
    )

    def f(coupler, graph):
        return coupler(graph=graph, get_info=False)

    coupler_vmap = nnx.jit(nnx.vmap(f, in_axes=(None, 0), out_axes=0))

    coordinates_batch, _ = coupler_vmap(coupler=coupler, graph=context_batch)


def test_recurrent_coupler_with_virtual_address():
    """RecurrentCoupler extended with a shared virtual state (Item 5 of
    attention-backlog sec 3.5): two parallel Euler updates on the per-address
    state h and on h_virtual, with h_virtual concatenated into phi's input
    (design (c)) and F_virtual = phi_virtual(masked_mean(h) || h_virtual_old)
    (design (alpha)). Verifies the coupler plugs into the GNN pipeline pattern
    and survives vmap + jit across a batched input."""
    loader = LinearSystemProblemLoader(seed=0).__iter__()
    problem_batch = next(loader)
    context_batch, _ = problem_batch.get_context()

    virtual_size = 4
    coupler = VirtualAddressRecurrentCoupler(
        phi=MLP(
            in_size=4 + virtual_size,
            hidden_sizes=[],
            out_size=4,
            seed=64,
            final_activation=nnx.tanh,
        ),
        phi_virtual=MLP(
            in_size=4 + virtual_size,
            hidden_sizes=[],
            out_size=virtual_size,
            seed=64,
            final_activation=nnx.tanh,
        ),
        message_functions=[
            PerformerMessagePassingFunction(
                in_graph_structure=loader.context_structure,
                in_array_size=4,
                hidden_sizes=[4],
                d_qk=8,
                out_size=4,
                activation=nnx.leaky_relu,
                final_activation=None,
                outer_activation=nnx.tanh,
                seed=64,
            )
        ],
        n_steps=4,
        virtual_address_size=virtual_size,
    )

    def f(coupler, graph):
        return coupler(graph=graph, get_info=False)

    coupler_vmap = nnx.jit(nnx.vmap(f, in_axes=(None, 0), out_axes=0))

    coordinates_batch, _ = coupler_vmap(coupler=coupler, graph=context_batch)
