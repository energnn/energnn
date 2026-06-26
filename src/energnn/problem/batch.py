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
    evaluate scores for batches of decision graphs,
    and provide metadata about the graph structures.
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
    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        """
        Retrieve the batch of context graphs :math:`x`.

        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :param step: Training step number passed by the trainer. Useful for scheduling.
        :returns: A tuple of:
            - **Graph**: A batched context object.
            - **dict**: A dictionary of additional information (empty if `get_info=False`).

        :raises NotImplementedError: If the subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_score(self, *, decision: Graph, get_info: bool = False, step: int | None = None) -> tuple[list[float], dict]:
        """
        Evaluate a scalar `score` for each decision graph in the batch.

        :param decision: Batched decision graph to evaluate.
        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :param step: Training step number passed by the trainer. Useful for scheduling.
        :returns: A tuple of:
            - **list[float]**: list of score values.
            - **dict**: A dictionary of additional information (empty if `get_info=False`).

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


class UnsupervisedProblemBatch(ABC, ProblemBatch):
    """
    Base class for unsupervised learning or optimization problems on batches.

    This class focuses on problems where the objective is defined by a gradient
    of a cost function with respect to the decision variables.
    """

    @abstractmethod
    def get_gradient(self, *, decision: JaxGraph, get_info: bool = False, step: int | None = None) -> tuple[JaxGraph, dict]:
        r"""
        Compute gradients :math:`\nabla_y f` for a batch of decision graphs :math:`y`.

        The gradient guides optimization algorithms by indicating the direction of
        steepest increase of the objective function.

        :param decision: Batched decision graph to evaluate.
        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :param step: Training step number passed by the trainer. Useful for scheduling.
        :returns: A tuple of:
            - **Graph**: A batched context object.
            - **dict**: A dictionary of additional information (empty if `get_info=False`).

        :raises NotImplementedError: If the subclass does not override this constructor.
        """
        raise NotImplementedError


class SupervisedProblemBatch(ABC, ProblemBatch):
    """
    Base class for supervised learning problems on batches.

    This class focuses on problems where a ground truth target (oracle) is available
    for each problem instance in the batch.
    """

    @abstractmethod
    def get_target(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        """
        Retrieve the target graphs :math:`y^*` of the problem batch.

        The target graphs contain the ground truth labels or optimal decisions
        associated with the context graphs.

        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :return: A tuple containing:
            - **Graph**: The target graph object.
            - **dict**: A dictionary of additional information (empty if `get_info=False`).

        :raises NotImplementedError: If the subclass does not override this constructor.
        """
        raise NotImplementedError
