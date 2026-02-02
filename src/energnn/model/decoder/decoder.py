# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod

import jax
from flax import nnx

from energnn.graph.jax.graph import JaxGraph


class Decoder(ABC, nnx.Module):
    """Interface for all decoders.

    :param seed: Random seed for weight initialization.
    """

    def __init__(self, *, seed: int = 0):
        self.rngs = nnx.Rngs(seed)

    @abstractmethod
    def __call__(
        self, *, graph: JaxGraph, coordinates: jax.Array, get_info: bool = False
    ) -> tuple[JaxGraph | jax.Array, dict]:
        raise NotImplementedError
