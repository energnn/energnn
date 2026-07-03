# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from energnn.graph import JaxGraph
from energnn.model.utils import MLP
from .message_passing_function import MessagePassingFunction
from ..coupler import Coupler

logger = logging.getLogger(__name__)


class RecurrentCoupler(Coupler):
    r"""
    Simplified version of the Neural Ordinary Differential Equation solver.

    The following recurrent system is used.:

    .. math::
        \forall a \in \mathcal{A}_x, h_a(t+\delta t) = h_a(t+\delta t) +
        \delta t \times \phi_\theta(\psi^1_\theta(h;x)_a, \dots, \psi^n_\theta(h;x)_a),

    with the following initial condition:

    .. math::
        \forall a \in \mathcal{A}_x, h_a(t=0) = [0, \dots, 0].

    :param phi: Outer MLP :math:`\phi_\theta`.
    :param message_functions: List of message functions :math:`(\psi^i_\theta)_i`.
    :param n_steps: Number of message passing steps.
    """

    def __init__(
        self,
        phi: MLP,
        message_functions: list[MessagePassingFunction],
        n_steps: int,
    ):
        super().__init__()
        self.phi = phi
        self.message_functions = nnx.List(message_functions)
        self.n_steps = n_steps

        self.dt = 1 / self.n_steps

    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[jax.Array, dict]:

        def F(t, coordinates, graph):
            """Residual function."""
            messages = []
            for m in self.message_functions:
                message, info = m(graph=graph, coordinates=coordinates)
                messages.append(message)
            messages = jnp.concatenate(messages, axis=-1)
            return self.phi(messages)

        h = jnp.zeros([jnp.shape(graph.non_fictitious_addresses)[0], self.phi.out_size])

        dt = 1 / self.n_steps
        for _ in range(self.n_steps):
            h = h + dt * F(0, h, graph)

        return h, {}

    @staticmethod
    def log_solved():
        """Log a message indicating successful ODE solve."""
        logger.info("ODE solved")


class VirtualAddressRecurrentCoupler(Coupler):
    r"""Recurrent coupler with a shared virtual state propagated alongside
    per-address coordinates (Item 5 of the attention-backlog specification, section 3.5).

    Extends :class:`RecurrentCoupler` with a single shared vector
    :math:`h_{\mathrm{virtual}} \in \mathbb{R}^{d_{v}}` evolved in parallel
    with the per-address state :math:`h \in \mathbb{R}^{n_{\mathrm{addr}} \times d_h}`.
    The virtual state acts as a global memory channel that propagates
    information beyond local topology in a single Euler step.

    Two forward-Euler updates are run jointly over ``n_steps``:

    .. math::
        h(t + \delta t)
        = h(t) + \delta t \cdot \phi\!\big( \mathrm{concat}\big(
            \psi_1(h(t)), \dots, \psi_n(h(t)),
            h_{\mathrm{virtual}}(t) \otimes \mathbf{1}_{n_{\mathrm{addr}}}
        \big) \big),

    .. math::
        h_{\mathrm{virtual}}(t + \delta t)
        = h_{\mathrm{virtual}}(t) + \delta t \cdot \phi_{\mathrm{virtual}}\!\big(
            \mathrm{concat}\big(
                \mathrm{mean}_{a \in \mathcal{A}_x^{\mathrm{real}}}\, h_a(t),
                h_{\mathrm{virtual}}(t)
            \big)
        \big),

    where :math:`\psi_i` are the message functions, :math:`\phi` is the
    per-address residual MLP, :math:`\phi_{\mathrm{virtual}}` is the
    virtual-state residual MLP, and the mean is taken over non-fictitious
    addresses only (denominator
    :math:`\sum_a \mathbb{1}[a \in \mathcal{A}_x^{\mathrm{real}}] + \varepsilon`).

    Design choices:

    - **Injection**. The virtual state is concatenated to the message
      vector before :math:`\phi`. No change is required to the
      :class:`MessagePassingFunction` ABC. Message functions from Items 1-4
      are reused unchanged.
    - **F_virtual**. Mean pool of :math:`h` over real addresses,
      concatenated with :math:`h_{\mathrm{virtual}}` and passed through
      :math:`\phi_{\mathrm{virtual}}`. Reuses the corrected denominator
      pattern of :class:`GlobalAggregationMessagePassingFunction` (Item 2).
    - **Virtual address count**. Single vector for v1; the
      ``virtual_address_size`` parameter is the only size knob.

    **Equivariance**. The output ``h`` permutes with the address permutation
    of the input; ``h_virtual`` is invariant. The masked mean over real
    addresses is permutation-invariant, the broadcast to all addresses
    preserves the equivariance of ``h``.

    **Degenerate cases**. With ``virtual_address_size = 0`` the virtual state
    is empty and the coupler reduces to :class:`RecurrentCoupler` semantics
    (modulo floating-point error from the concatenation of zero-size tensors).
    With ``n_steps = 1`` a single Euler step is applied.

    :param phi: Outer per-address MLP. Must satisfy
        ``phi.in_size == sum(message.out_size for message in message_functions) + virtual_address_size``
        and ``phi.out_size`` equals the latent dimension of the per-address state.
    :param phi_virtual: Virtual-state MLP. Must satisfy
        ``phi_virtual.in_size == phi.out_size + virtual_address_size`` and
        ``phi_virtual.out_size == virtual_address_size``.
    :param message_functions: List of message functions to wrap.
    :param n_steps: Number of forward-Euler steps.
    :param virtual_address_size: Dimension of the shared virtual state.
    :param eps: Small constant added to the denominator of the masked mean
        to avoid division by zero on graphs with no real addresses. Defaults
        to ``1e-9`` to match :class:`GlobalAggregationMessagePassingFunction`.

    **Compatibility with the H2MG architecture**. The virtual state propagates
    in parallel with the per-address state but is never consumed by the
    message functions directly (design (c)). Per-(class, port) message
    functions (``LocalSum``, ``GATv2``, ``MultiHeadQKV``) consume only the
    per-address coordinates ``h`` and the encoded hyper-edge features; their
    ABC contract is unchanged. The virtual state enters the per-address
    update only through :math:`\phi`, which means the virtual contribution
    is a residual on top of the standard message aggregation. With
    ``virtual_address_size = 0`` the coupler reduces to
    :class:`RecurrentCoupler` semantics exactly.

    **Why this design vs naive concatenation**. Earlier composition experiments
    showed that placing :class:`GlobalAggregationMessagePassingFunction`
    alongside :class:`LocalSumMessagePassingFunction` in
    :attr:`RecurrentCoupler.message_functions` degrades eval on AC LF Small
    in 8/8 configurations tested, and that the same flat concatenation pattern
    with :class:`PerformerMessagePassingFunction` does not improve over the
    LocalSum baseline. The shared virtual state evolved by
    :math:`F_{\mathrm{virtual}}` combines signals through a separate channel
    rather than through a single flat concatenation.

    References:
        The attention-backlog specification, section 3.5 (Item 5).
    """

    def __init__(
        self,
        phi: MLP,
        phi_virtual: MLP,
        message_functions: list[MessagePassingFunction],
        n_steps: int,
        virtual_address_size: int,
        eps: float = 1e-9,
    ):
        super().__init__()
        # Validate the size contracts so misconfiguration fails at construction
        # rather than inside the first jit-compiled forward.
        expected_phi_in = sum(mf.out_size for mf in message_functions) + virtual_address_size
        if phi.in_size != expected_phi_in:
            raise ValueError(
                f"phi.in_size={phi.in_size} does not match "
                f"sum(mf.out_size) + virtual_address_size = {expected_phi_in}. "
                "phi consumes the concatenation of all message outputs plus the broadcast "
                "virtual state (design (c))."
            )
        if virtual_address_size > 0:
            expected_phi_v_in = phi.out_size + virtual_address_size
            if phi_virtual.in_size != expected_phi_v_in:
                raise ValueError(
                    f"phi_virtual.in_size={phi_virtual.in_size} does not match "
                    f"phi.out_size + virtual_address_size = {expected_phi_v_in}. "
                    "phi_virtual consumes the concatenation of masked_mean(h) and "
                    "h_virtual_old (design (alpha))."
                )
            if phi_virtual.out_size != virtual_address_size:
                raise ValueError(
                    f"phi_virtual.out_size={phi_virtual.out_size} does not match "
                    f"virtual_address_size={virtual_address_size}. The virtual state "
                    "residual must match the virtual state dimension."
                )
        self.phi = phi
        self.phi_virtual = phi_virtual
        self.message_functions = nnx.List(message_functions)
        self.n_steps = n_steps
        self.virtual_address_size = virtual_address_size
        self.eps = eps
        self.dt = 1 / self.n_steps

    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[jax.Array, dict]:
        """Run the joint per-address and virtual-state ODE solver.

        :param graph: Input H2MG graph (single instance shape; the trainer vmaps the batch).
        :param get_info: Unused; kept for the :class:`Coupler` ABC.
        :return: ``(h, info)`` where ``h`` has shape
            ``(n_addr, phi.out_size)`` and ``info`` is an empty dict.
        """
        n_addr = jnp.shape(graph.non_fictitious_addresses)[0]
        h = jnp.zeros([n_addr, self.phi.out_size])
        h_virtual = jnp.zeros(self.virtual_address_size)
        dt = self.dt

        for _ in range(self.n_steps):
            h_old = h

            # F: per-address residual update.
            messages = []
            for m in self.message_functions:
                msg, _ = m(graph=graph, coordinates=h)
                messages.append(msg)
            msg_cat = jnp.concatenate(messages, axis=-1)
            # Inject the virtual state into phi's input (design (c)).
            if self.virtual_address_size > 0:
                msg_v_broadcast = jnp.broadcast_to(
                    h_virtual[None, :],
                    (n_addr, self.virtual_address_size),
                )
                phi_input = jnp.concatenate([msg_cat, msg_v_broadcast], axis=-1)
            else:
                phi_input = msg_cat
            h = h + dt * self.phi(phi_input)

            # F_virtual: virtual state residual update (design (alpha)).
            if self.virtual_address_size > 0:
                mask = jnp.expand_dims(graph.non_fictitious_addresses, -1)
                h_masked = h_old * mask
                denom = jnp.sum(graph.non_fictitious_addresses) + self.eps
                h_mean = jnp.sum(h_masked, axis=0) / denom
                virt_input = jnp.concatenate([h_mean, h_virtual], axis=-1)
                h_virtual = h_virtual + dt * self.phi_virtual(virt_input)

        info = {"h_virtual_final": h_virtual} if get_info else {}
        return h, info

    @staticmethod
    def log_solved():
        """Log a message indicating successful ODE solve."""
        logger.info("ODE solved")
