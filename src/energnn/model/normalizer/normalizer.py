# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

from flax import nnx

from energnn.graph import Graph


class Normalizer(nnx.Module, ABC):
    """Interface for a normalizer.

    A normalizer transforms the input graph features into a distribution
    more suitable for neural network training (e.g., standardization, normalization).
    """

    @abstractmethod
    def __call__(self, graph: Graph, step_with_metrics: bool = False) -> tuple[Graph, dict]:
        """Normalize the input graph features.

        :param graph: Input graph to normalize.
        :param step_with_metrics: Whether this step collects metrics. Implementations that produce metrics should
            expose a `return_metrics` constructor flag and return them only when both are True.
        :return: A tuple containing:
            - Normalized graph with transformed features
            - A dictionary of metrics for tracking purpose, empty dict when not collected
        :raises NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError
