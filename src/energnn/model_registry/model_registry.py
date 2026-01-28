# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod

from energnn.amortizer import SimpleAmortizer
from energnn.amortizer.metadata import AmortizerMetadata


class ModelRegistry(ABC):
    """Abstract base class for model registries.

    A model registry implementation is necessary for the train method of SimpleAmortizer to work.
    It provides methods for registering, retrieving, and deleting models, as well as for listing all registered models.
    """

    @abstractmethod
    def register_trainer(self, trainer: SimpleAmortizer, best: bool = False, last: bool = False) -> bool:
        """
        Registers a SimpleAmortizer class into the registry.

        Uploads the file to remote storage and stores its hash and metadata in the feature store database.

        :param trainer: The SimpleAmortizer class to register.
        :param best: Whether the trainer is the best one so far.
        :param last: Whether the trainer is the last one so far.
        :return: True if the instance is registered successfully, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trainer_metadata(self, project_name: str, run_id: str, step: int) -> AmortizerMetadata | None:
        """
        Retrieves the metadata corresponding to the specified trainer export.

        :param project_name: The project in which the training run was done.
        :param run_id: The ID of the training run that produced the trainer to search.
        :param step: The training step in the run at which the trainer was stored.
        :return: The AmortizerMetadata corresponding to the specified trainer export, or None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def download_trainer(self, project_name: str, run_id: str, step: int) -> SimpleAmortizer:
        """
        Downloads and load the SimpleAmortizer class corresponding to the specified trainer export.

        :param project_name: The project in which the training run was done.
        :param run_id: The ID of the training run that produced the trainer to search.
        :param step: The training step in the run at which the trainer was stored.
        :return: The SimpleAmortizer class corresponding to the specified trainer export.
        """
        raise NotImplementedError
