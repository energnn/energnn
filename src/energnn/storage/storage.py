# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from abc import ABC, abstractmethod


class Storage(ABC):
    """
    Interface for dataset and GNN storage.

    This abstract base class defines the required methods for interacting with external storage systems (e.g., local disk,
    S3 bucket, etc.) used to persist datasets and graph neural network (GNN) models or files.

    Subclasses must implement the following:
    - A method to **upload** a file (e.g., dataset, model checkpoint) or directory to the storage backend.
    - A method to **download** a file from the storage backend to the local environment.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the storage instance

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def upload(self, source_path: str, target_path: str) -> None:
        """
        Should upload a file or directory to the storage backend.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def download(self, source_path: str, target_path: str, overwrite: bool = False, unzip: bool = True) -> None:
        """
        Should download a file or directory from the storage backend.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, target_path: str) -> None:
        """
        Should delete a file from the storage backend.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError
