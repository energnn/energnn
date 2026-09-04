# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import jax
from flax import nnx

from energnn.graph import Graph


class Decoder(ABC, nnx.Module):
    """Interface for all decoders.

    A decoder takes as input latent coordinates and an encoded graph context,
    and produces either a new graph with predictions or a global output vector.
    """

    @abstractmethod
    def __call__(
        self, *, graph: Graph, coordinates: jax.Array, step_with_metrics: bool = False
    ) -> tuple[Graph | jax.Array, dict]:
        """Decode latent coordinates into predictions.

        :param graph: Encoded graph providing context for decoding.
        :param coordinates: Latent coordinates array with shape (num_addresses, latent_dim).
        :param step_with_metrics: Whether this step collects metrics. Implementations that produce metrics should
            expose a `return_metrics` constructor flag and return them only when both are True.
        :return: A tuple containing:
            - Either a new Graph with prediction features or a global output array
            - A dictionary of metrics for tracking purpose, empty dict when not collected
        :raises NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError
