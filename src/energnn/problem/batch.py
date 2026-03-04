# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod

from energnn.graph import GraphStructure, JaxGraph


class ProblemBatch(ABC):
    """
    Abstract base class for handling batches of problem instances.

    Subclasses should implement methods to retrieve batch of context,
    compute gradients and metrics for batches of decision graphs,
    and provide an initial zero decision batch.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the batch handler.

        Implementations may accept parameters like batch size.

        :raises NotImplementedError: If not overridden in subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def get_context(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Retrieve the batch of context graphs :math:`x`.

        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :returns: A tuple of:
            - **Graph**: A batched context object.
            - **dict**: A dictionary of additional information (empty if get_info=False).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict]:
        r"""
        Compute gradients :math:`\nabla_y f` for a batched of decision graphs :math:`y`.

        :param decision: Batched decision graph at which to evaluate gradient.
        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :param cfg: An optional configuration dict.
        :returns: A tuple of:
            - **Graph**: A batched context object.
            - **dict**: A dictionary of additional information (empty if get_info=False).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[list[float], dict]:
        """
        Evaluate scalar metrics for each decision graph in the batch.

        :param decision: Batched decision graph to evaluate.
        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :param cfg: An optional configuration dict.
        :returns: A tuple of:
            - **list[float]**: list of metric values.
            - **dict**: A dictionary of additional information (empty if get_info=False).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def context_structure(self) -> GraphStructure:
        """Should define the structure of all context graphs."""
        raise NotImplementedError

    @property
    @abstractmethod
    def decision_structure(self) -> GraphStructure:
        """Should define the structure of all decision graphs."""
        raise NotImplementedError
