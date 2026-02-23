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


class Decoder(ABC, nnx.Module):
    """Interface for all decoders.

    A decoder takes as input latent coordinates and an encoded graph context,
    and produces either a new graph with predictions or a global output vector.
    """

    @abstractmethod
    def __call__(
        self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False
    ) -> tuple[JaxGraph | jax.Array, dict]:
        """Decode latent coordinates into predictions.

        :param graph: Encoded graph providing context for decoding.
        :param coordinates: Latent coordinates array with shape (num_addresses, latent_dim).
        :param get_info: If True, returns additional info for tracking purpose.
        :return: A tuple containing:
            - Either a new JaxGraph with prediction features or a global output array
            - A dictionary with additional information if get_info=True, empty dict otherwise
        :raises NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError
