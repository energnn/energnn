# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
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
    def init_run(self, *, name: str, tags: list[str], cfg: DictConfig):
        """Should initialize a training run, associate it with tags and log its config.

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
    def get_amortizer_path(self, *, tag: str) -> str:
        """
        Should fetch the unique ID of a gnn, based on its tag.

        :param tag: Tag or key associated with the saved amortizer.
        """
        raise NotImplementedError

    @abstractmethod
    def run_track_dataset(self, *, infos: dict, target_path: str) -> None:
        """
        Should associate the current run with its dataset.

        :param infos: Dictionary of dataset metadata to log (e.g., name, version, split).
        :param target_path: Path where the dataset is stored.
        """
        raise NotImplementedError

    @abstractmethod
    def run_track_amortizer(self, *, id: str, target_path: str) -> None:
        """
        Should associate the current run with its gnn.

        :param id: Unique identifier of the amortizer.
        :param target_path: Path where the amortizer is stored.
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
