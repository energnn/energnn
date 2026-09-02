# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

from energnn.graph import Graph, GraphStructure


class ProblemBatch(ABC):
    """
    Abstract base class for handling batches of problem instances.

    Subclasses should implement methods to retrieve batch of context,
    compute gradients and scores for batches of decision graphs,
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
    def get_context(self, step_with_metrics: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        """
        Retrieve the batch of context graphs :math:`x`.

        :param step_with_metrics: Whether this step collects metrics. Return metrics only when True (and, by
            convention, only if the problem was built to produce them).
        :param step: Training step number passed by the trainer. Useful for scheduling.
        :returns: A tuple of:
            - **Graph**: A batched context object.
            - **dict**: A dictionary of metrics for tracking purpose (empty if `step_with_metrics=False`).

        :raises NotImplementedError: If the subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, *, decision: Graph, step_with_metrics: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        r"""
        Compute gradients :math:`\nabla_y f` for a batched of decision graphs :math:`y`.

        :param decision: Batched decision graph at which to evaluate gradient.
        :param step_with_metrics: Whether this step collects metrics. Return metrics only when True (and, by
            convention, only if the problem was built to produce them).
        :param step: Training step number passed by the trainer. Useful for scheduling.
        :returns: A tuple of:
            - **Graph**: A batched context object.
            - **dict**: A dictionary of metrics for tracking purpose (empty if `step_with_metrics=False`).

        :raises NotImplementedError: If the subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_score(
        self, *, decision: Graph, step_with_metrics: bool = False, step: int | None = None
    ) -> tuple[list[float], dict]:
        """
        Evaluate a scalar `score` for each decision graph in the batch.

        :param decision: Batched decision graph to evaluate.
        :param step_with_metrics: Whether this step collects metrics. Return metrics only when True (and, by
            convention, only if the problem was built to produce them).
        :param step: Training step number passed by the trainer. Useful for scheduling.
        :returns: A tuple of:
            - **list[float]**: list of score values.
            - **dict**: A dictionary of metrics for tracking purpose (empty if `step_with_metrics=False`).

        :raises NotImplementedError: If the subclass does not override this constructor.
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
