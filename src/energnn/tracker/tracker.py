# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

from omegaconf import DictConfig


class Tracker(ABC):
    """
    Abstract base class defining the interface for experiment tracking.

    Concrete implementations should manage the lifecycle of training runs,
    including initialization, logging of configurations and metrics, and association
    of artifacts such as datasets and models.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the tracker client.

        Concrete implementations may set up connections to tracking backends (e.g., Neptune),
        authenticate, or configure default project names.
        """
        raise NotImplementedError

    @abstractmethod
    def init_run(self, *, name: str, tags: dict[str, str], cfg: DictConfig):
        """Should initialize a training run, associate it with tags, and log its config.

        :param name: Name for the run.
        :param tags: List of tags to categorize the run.
        :param cfg: Configuration object containing experiment parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def stop_run(self):
        """
        Stop the currently active training run.

        This method should flush any pending logs and finalize the run record,
        ensuring that all metrics and artifacts are properly saved in the backend.
        """
        raise NotImplementedError

    @abstractmethod
    def run_append(self, *, infos: dict, step: int) -> None:
        """
        Should track the `infos` dictionary.

        :param infos: Information dictionary
        :param step: Training or evaluation step associated with these infos.
        """
        raise NotImplementedError
