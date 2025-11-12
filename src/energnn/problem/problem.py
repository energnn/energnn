# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod

from energnn.graph import Graph
from energnn.problem.metadata import ProblemMetadata
from omegaconf import DictConfig


class Problem(ABC):
    """
    Base abstract class for graph-based optimization or learning problems.

    Subclasses must implement methods to retrieve the problem context graph,
    an initial zero decision graph, compute gradients, evaluate metrics,
    and provide problem metadata.

    Notes:
        - All returned Graph objects must adhere to the energnn.graph.Graph API.
        - Methods returning tuples will return additional information in the dict when
          `get_info=True` for tracking purpose.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the problem instance.

        This constructor may accept parameters specific to the problem definition,
        such as hyperparameters, or graph dimensions.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_context(self, get_info: bool = False) -> tuple[Graph, dict]:
        """
        Retrieve the context graph math:`x` of the problem instance.

        The context graph encompasses all fixed inputs required to define
        the instance, such as node features, edge indices, and any static attributes.

        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :return: A tuple containing:
            - **Graph**: The context graph object.
            - **dict**: A dictionary of additional information (empty if get_info=False).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_zero_decision(self, get_info: bool = False) -> tuple[Graph, dict]:
        """
        Construct a decision graph math:`y` filled with zeros.

        This method provides a baseline or starting point for optimization routines.

        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :return: A tuple containing:
            - **Graph**: The zero-initialized decision graph.
            - **dict**: A dictionary of additional information (empty if get_info=False).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, *, decision: Graph, get_info: bool = False, cfg: DictConfig | None) -> tuple[Graph, dict]:
        r"""
        Compute the gradient graph :math:`\nabla_y f` for a given decision :math:`y`.

        The gradient guides optimization algorithms such as gradient descent.

        :param decision: A decision graph at which to evaluate the gradient.
        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :param cfg: An optional configuration dict.
        :return: A tuple containing:
            - **Graph**: The gradient graph with same structure as decision.
            - **dict**: A dictionary of additional information (empty if get_info=False).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self, *, decision: Graph, get_info: bool = False, cfg: DictConfig | None) -> tuple[float, dict]:
        """Should return a scalar metrics that evaluates the decision graph :math:`y`.

        :param decision: The decision graph to evaluate.
        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :param cfg: An optional configuration dict.
        :return: A tuple containing:
            - **float**: A float as metric value.
            - **dict**: A dictionary of additional information (empty if get_info=False).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    def get_decision_structure(self) -> dict:
        """
        Returns a dictionary containing the structure of decision graphs compatible with the problem instance."

        This helper uses `get_zero_decision` to extract feature dimensions
        and returns a mapping from edge keys to integer feature shapes.

        :return: A dict mapping edge identifiers to feature dimension dicts.
        """
        zero_decision_graph, _ = self.get_zero_decision()

        def to_int(feature_names):
            return {key: int(value) for key, value in feature_names.items()}

        return {edge_key: to_int(edge.feature_names) for edge_key, edge in zero_decision_graph.edges.items()}

    @abstractmethod
    def get_metadata(self) -> ProblemMetadata:
        """
        Retrieve metadata describing problem characteristics.

        Metadata include problem name, configuration ID, version, context shape, decision shape.

        :return: A ProblemMetadata instance encapsulating metadata.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, *, path: str) -> None:
        """
        Serialize the problem instance to disk.

        This method should persist all necessary state to reconstruct
        the problem later.

        :param path: Filesystem path or directory to save problem data.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError
