# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable

from flax import nnx
from flax.typing import Array

from energnn.graph.jax import JaxGraph

Activation = Callable[[Array], Array]


class Encoder(nnx.Module, ABC):

    def __init__(self, *, seed: int = 0):
        self.rngs = nnx.Rngs(seed)

    @abstractmethod
    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Should encode the input graph into a graph with the same edges classes and features.

        :param graph: Input graph to encode.
        :param get_info: If True, returns additional info for tracking purpose.

        :raises NotImplementedError: If the subclass does not override this constructor.
        """
        raise NotImplementedError


class IdentityEncoder(Encoder):
    r"""
    Identity encoder that returns the input graph unchanged.

    .. math::
        \tilde{x} = x
    """

    def __init__(self):
        super().__init__()

    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """Apply the identity encoder and return the input graph without changes.

        :param context: Input graph to encode.
        :param get_info: If True, returns additional info for tracking purpose.
        """
        return graph, {}
