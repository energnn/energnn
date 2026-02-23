# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod

from flax import nnx

from energnn.graph import JaxGraph


class Normalizer(nnx.Module, ABC):
    """Interface for a normalizer.

    A normalizer transforms the input graph features into a distribution
    more suitable for neural network training (e.g., standardization, normalization).
    """

    @abstractmethod
    def __call__(self, graph: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """Normalize the input graph features.

        :param graph: Input graph to normalize.
        :param get_info: If True, returns additional info for tracking purpose.
        :return: A tuple containing:
            - Normalized graph with transformed features
            - A dictionary with additional information if get_info=True, empty dict otherwise
        :raises NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError
