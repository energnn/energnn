# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from abc import ABC, abstractmethod

from flax import nnx

from energnn.graph import Graph


class Encoder(nnx.Module, ABC):

    @abstractmethod
    def __call__(self, graph: Graph, step_with_metrics: bool = False) -> tuple[Graph, dict]:
        """Encode the input graph into a graph with the same hyper-edge set classes and features.

        :param graph: Input graph to encode.
        :param step_with_metrics: Whether this step collects metrics. Implementations that produce metrics should
            expose a `return_metrics` constructor flag and return them only when both are True.
        :return: A tuple containing:
            - Encoded graph with transformed features
            - A dictionary of metrics for tracking purpose, empty dict when not collected
        :raises NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError


class IdentityEncoder(Encoder):
    r"""Identity encoder that returns the input graph unchanged.

    .. math::
        \tilde{x} = x
    """

    def __call__(self, graph: Graph, step_with_metrics: bool = False) -> tuple[Graph, dict]:
        """Apply the identity encoder and return the input graph without changes.

        :param context: Input graph to encode.
        :param step_with_metrics: Whether this step collects metrics. Implementations that produce metrics should
            expose a `return_metrics` constructor flag and return them only when both are True.
        """
        return graph, {}
