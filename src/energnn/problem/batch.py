# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from typing import Any

from jax.tree_util import register_pytree_node_class

from energnn.graph import Graph, GraphStructure


class ProblemBatch(ABC):
    """
    Abstract base class for handling batches of problem instances.

    Subclasses should implement methods to retrieve batch of context,
    evaluate scores for batches of decision graphs,
    and provide metadata about the graph structures.
    """

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses as JAX PyTrees."""
        super().__init_subclass__(**kwargs)
        register_pytree_node_class(cls)

    @abstractmethod
    def __init__(self):
        """
        Initialize the batch handler.

        Implementations may accept parameters like batch size.

        :raises NotImplementedError: If not overridden in subclass.
        """
        raise NotImplementedError

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """
        Flatten the ProblemBatch into a list of children and auxiliary data for JAX.
        By default, all items in __dict__ are considered children.
        """
        # We filter out attributes that might not be JAX-compatible if necessary,
        # but usually Graph objects are fine.
        children = tuple(self.__dict__.values())
        aux_data = tuple(self.__dict__.keys())
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[str, ...], children: tuple[Any, ...]) -> "ProblemBatch":
        """Reconstruct the ProblemBatch from children and auxiliary data."""
        instance = cls.__new__(cls)
        instance.__dict__.update(zip(aux_data, children))
        return instance

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


class SelfSupervisedProblemBatch(ProblemBatch):
    """
    Base class for self-supervised learning or optimization problems on batches.

    This class focuses on problems where the objective is defined by a gradient
    of a cost function with respect to the decision variables.
    """

    @abstractmethod
    def get_gradient(self, *, decision: Graph, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
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


class SupervisedProblemBatch(ProblemBatch):
    """
    Base class for supervised learning problems on batches.

    This class focuses on problems where a ground truth target (oracle) is available
    for each problem instance in the batch.
    """

    @abstractmethod
    def get_loss(self, *, decision: Graph, get_info: bool = False, step: int | None = None) -> tuple[float, dict]:
        r"""
        Compute the loss value for a given decision :math:`y`.

        The loss guides optimization algorithms such as gradient descent.

        :param decision: A decision graph at which to evaluate the loss.
        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :param step: Training step number passed by the trainer. Useful for scheduling.
        :return: A tuple containing:
            - **float**: The loss value.
            - **dict**: A dictionary of additional information (empty if `get_info=False`).

        :raises NotImplementedError: If the subclass does not override this constructor.
        """
        raise NotImplementedError
