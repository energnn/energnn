# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod

import jax
from flax import nnx

from energnn.graph import JaxGraph


class Coupler(nnx.Module, ABC):
    """Interface for a coupler.

    A coupler takes as input a graph and returns latent coordinates for each address.
    Graph information should be injected into the latent coordinates in a permutation-equivariant manner.
    """

    @abstractmethod
    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[jax.Array, dict]:
        """Compute latent coordinates from the input graph.

        :param graph: Input graph to process.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: A tuple containing:
            - Latent coordinates array with shape (num_addresses, latent_dim)
            - A dictionary with additional information if get_info=True, empty dict otherwise
        :raises NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError
