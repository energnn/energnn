# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from typing import Any

from jax.tree_util import register_pytree_node_class

from energnn.graph import Graph, GraphStructure


class Problem(ABC):
    """
    Base abstract class for graph-based optimization or learning problems.

    Subclasses must implement methods to retrieve the problem context graph,
    evaluate scores, and provide problem metadata.

    Notes:
        - All returned Graph objects must adhere to the energnn.graph.Graph API.
        - Methods returning tuples will return additional information in the dict when
          `get_info=True` for tracking purpose.
    """

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses as JAX PyTrees."""
        super().__init_subclass__(**kwargs)
        register_pytree_node_class(cls)

    @abstractmethod
    def __init__(self):
        """
        Initialize the problem instance.

        This constructor may accept parameters specific to the problem definition,
        such as hyperparameters, or graph dimensions.

        :raises NotImplementedError: If the subclass does not override this constructor.
        """
        raise NotImplementedError

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        """
        Flatten the Problem into a list of children and auxiliary data for JAX.
        """
        children = tuple(self.__dict__.values())
        aux_data = tuple(self.__dict__.keys())
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[str, ...], children: tuple[Any, ...]) -> "Problem":
        """Reconstruct the Problem from children and auxiliary data."""
        instance = cls.__new__(cls)
        instance.__dict__.update(zip(aux_data, children))
        return instance

    @abstractmethod
    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        """
        Retrieve the context graph math:`x` of the problem instance.

        The context graph encompasses all fixed inputs required to define
        the instance, such as node features, hyper-edge indices, and any static attributes.

        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :param step: Training step number passed by the trainer. Useful for scheduling.
        :return: A tuple containing:
            - **Graph**: The context graph object.
            - **dict**: A dictionary of additional information (empty if `get_info=False`).

        :raises NotImplementedError: If the subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_score(self, *, decision: Graph, get_info: bool = False, step: int | None = None) -> tuple[float, dict]:
        """Should return a scalar `score` that evaluates the decision graph :math:`y`.

        :param decision: The decision graph to evaluate.
        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :param step: Training step number passed by the trainer. Useful for scheduling.
        :return: A tuple containing:
            - **float**: A float as score value.
            - **dict**: A dictionary of additional information (empty if `get_info=False`).

        :raises NotImplementedError: If the subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, *, path: str) -> None:
        """
        Serialize the problem instance to disk.

        This method should make all necessary states persist to reconstruct
        the problem later.

        :param path: Filesystem path or directory to save problem data.

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


class SelfSupervisedProblem(Problem):
    """
    Base class for self-supervised learning or optimization problems.

    This class focuses on problems where the objective is defined by a gradient
    of a cost function with respect to the decision variables.
    """

    @abstractmethod
    def get_gradient(self, *, decision: Graph, get_info: bool = False, step: int | None = None) -> tuple[Graph, dict]:
        r"""
        Compute the gradient graph :math:`\nabla_y f` for a given decision :math:`y`.

        The gradient guides optimization algorithms such as gradient descent.

        :param decision: A decision graph at which to evaluate the gradient.
        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :param step: Training step number passed by the trainer. Useful for scheduling.
        :return: A tuple containing:
            - **Graph**: The gradient graph with the same structure as decision.
            - **dict**: A dictionary of additional information (empty if `get_info=False`).

        :raises NotImplementedError: If the subclass does not override this constructor.
        """
        raise NotImplementedError


class SupervisedProblem(Problem):
    """
    Base class for supervised learning problems.

    This class focuses on problems where a ground truth target (oracle) is available.
    """

    @abstractmethod
    def get_loss(self, *, decision: Graph, get_info: bool = False, step: int | None = None) -> tuple[float, dict]:
        r"""
        Compute the loss graph :math:`\mathcal{L} = f(y, y^{\star})` for a given decision :math:`y`.

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
