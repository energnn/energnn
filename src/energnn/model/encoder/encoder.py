# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from __future__ import annotations

from abc import ABC, abstractmethod

from flax import nnx

from energnn.graph import JaxGraph


class Encoder(nnx.Module, ABC):

    @abstractmethod
    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """Encode the input graph into a graph with the same edge classes and features.

        :param graph: Input graph to encode.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: A tuple containing:
            - Encoded graph with transformed features
            - A dictionary with additional information if get_info=True, empty dict otherwise
        :raises NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError


class IdentityEncoder(Encoder):
    r"""Identity encoder that returns the input graph unchanged.

    .. math::
        \tilde{x} = x
    """

    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """Apply the identity encoder and return the input graph without changes.

        :param context: Input graph to encode.
        :param get_info: If True, returns additional info for tracking purpose.
        """
        return graph, {}
