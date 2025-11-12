# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod

from omegaconf import DictConfig

from energnn.graph import Graph, separate_graphs


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
    def get_context(self, get_info: bool = False) -> tuple[Graph, dict]:
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
    def get_gradient(self, *, decision: Graph, get_info: bool = False, cfg: DictConfig | None) -> tuple[Graph, dict]:
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
    def get_metrics(self, *, decision: Graph, get_info: bool = False, cfg: DictConfig | None) -> tuple[list[float], dict]:
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

    @abstractmethod
    def get_zero_decision(self, get_info: bool = False) -> tuple[Graph, dict]:
        """
        Construct a batched decision graph math:`y` filled with zeros.

        :param get_info: Flag indicating if additional information should be returned for tracking purpose.
        :returns: A tuple of:
            - **Graph**: A batched context object.
            - **dict**: A dictionary of additional information (empty if get_info=False).

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    def get_decision_structure(self) -> dict:
        """
        Obtain the structure template of decision graphs in the batch.

        Internally calls `get_zero_decision` and separates the first graph
        to extract feature dimensions.

        :returns: A dict mapping edge identifiers to feature dimension dicts.
        """
        zero_decision_graph_batch, _ = self.get_zero_decision()
        zero_decision_graph = separate_graphs(zero_decision_graph_batch)[0]

        def to_int(feature_names):
            return {key: int(value) for key, value in feature_names.items()}

        return {edge_key: to_int(edge.feature_names) for edge_key, edge in zero_decision_graph.edges.items()}
