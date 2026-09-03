# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import jax
from flax import nnx

from energnn.graph import Graph


class Coupler(nnx.Module, ABC):
    """Interface for a coupler.

    A coupler takes as input a graph and returns latent coordinates for each address.
    Graph information should be injected into the latent coordinates in a permutation-equivariant manner.
    """

    @abstractmethod
    def __call__(self, graph: Graph, step_with_metrics: bool = False) -> tuple[jax.Array, dict]:
        """Compute latent coordinates from the input graph.

        :param graph: Input graph to process.
        :param step_with_metrics: Whether this step collects metrics. Implementations that produce metrics should
            expose a `return_metrics` constructor flag and return them only when both are True.
        :return: A tuple containing:
            - Latent coordinates array with shape (num_addresses, latent_dim)
            - A dictionary of metrics for tracking purpose, empty dict when not collected
        :raises NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError
