# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import json
import os
import pickle
from datetime import datetime

from .metadata import ProblemMetadata


class ProblemDataset(dict):
    """
    Dictionary-like container for datasets of problem instances.

    Stores dataset-level metadata and a list of ProblemMetadata instances.

    :param name: Identifier for the dataset.
    :param split: Dataset split name (e.g., "train", "val", "test").
    :param version: Version number of the dataset.
    :param instances: List of ProblemMetadata objects describing each instance.
    :param size: Total number of instances in the dataset.
    :param context_max_shape: Maximum dimensions of context graphs across instances.
    :param decision_max_shape: Maximum dimensions of decision graphs across instances.
    :param generation_date: Timestamp when the dataset was generated.
    :param selection_criteria: A dictionnary that contains some criteria
    :param tags: Key-value tags associated to the dataset for grouping or filtering.
    """

    def __init__(
        self,
        name: str,
        split: str,
        version: int,
        instances: list[ProblemMetadata],
        size: int,
        context_max_shape: dict,
        decision_max_shape: dict,
        generation_date: datetime,
        selection_criteria: dict,
        tags=None,
    ) -> None:
        super().__init__()
        if tags is None:
            tags = {}
        self["name"] = name
        self["split"] = split
        self["version"] = version
        self["size"] = size
        self["context_max_shape"] = context_max_shape
        self["decision_max_shape"] = decision_max_shape
        self["generation_date"] = generation_date
        self["selection_criteria"] = selection_criteria
        self["instances"] = instances
        self["tags"] = tags

    def get_infos_for_feature_store(self) -> dict:
        """
        Retrieve the dataset's information to send to the feature store.

        Excludes the full list of instances.

        :returns: A dict containing all dataset fields except "instances", with
                  "generation_date" converted to ISO string.
        """
        res = self.copy()
        res.pop("instances")
        res["generation_date"] = str(self.generation_date)
        return res

    def get_locally_missing_instances(self, path: str) -> list[ProblemMetadata]:
        """
        Identify instances whose files are missing in a local directory.

        :param path: Base directory where instance files should be stored.
        :returns: List of metadata of instances not present under `path`.
        """
        return [
            instance
            for instance in self.instances
            if not os.path.exists(os.path.join(path, instance.storage_path))
        ]

    def get_instance_paths(self) -> list[str]:
        """
        List the storage paths for all instances in the dataset.

        :returns: List of instance file paths as stored in metadata.
        """
        return [instance.storage_path for instance in self.instances]

    def to_json(self, file_path: str):
        """
        Serialize the dataset to a JSON file for human-readable archives.

        Note: JSON output will not preserve Python types on load.

        :param file_path: Target JSON file path.
        :raises IOError: If writing to the file system fails.
        """
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(self, handle, indent=4, ensure_ascii=False, default=str)

    def to_pickle(self, file_path: str):
        """
        Serialize the dataset to a pickle file for efficient reload.

        :param file_path: Target pickle file path.
        :raises IOError: If writing to the file system fails.
        """
        with open(file_path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, file_path: str) -> "ProblemDataset":
        """
        Load a dataset from a pickle file produced by `to_pickle`.

        :param file_path: Source pickle file path.
        :returns: Restored `ProblemDataset` instance.
        """
        with open(file_path, "rb") as handle:
            dataset = pickle.load(handle)
        return dataset

    @property
    def name(self) -> str:
        return self.get("name")

    @property
    def split(self) -> str:
        return self.get("split")

    @property
    def version(self) -> int:
        return self.get("version")

    @property
    def size(self) -> int:
        return self.get("size")

    @property
    def context_max_shape(self) -> dict:
        return self.get("context_max_shape")

    @property
    def decision_max_shape(self) -> dict:
        return self.get("decision_max_shape")

    @property
    def generation_date(self) -> str:
        return self.get("generation_date")

    @property
    def selection_criteria(self) -> dict:
        return self.get("selection_criteria")

    @property
    def instances(self) -> list[ProblemMetadata]:
        return self.get("instances")

    @property
    def tags(self) -> dict:
        return self.get("tags")
