# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import hashlib
import io
import json
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import requests

from energnn.feature_store.config_info import ProblemGenerationConfigInfo
from energnn.problem import Problem
from energnn.problem import ProblemDataset
from energnn.problem.metadata import ProblemMetadata

logger = logging.getLogger(__name__)


class FeatureStoreClient:
    r"""

    Client interface for interacting with an EnerGNN Feature Store server.

    This client allows registering, retrieving, and downloading configuration files,
    problem instances, and datasets related to an EnerGNN project. It manages metadata
    and data storage through HTTP requests to a remote feature store

    :param project_name: Identifies the EnerGNN project, for HTTP requests to the feature store and storage location.
    :param feature_store_url: Where to send the requests to.
    """

    project_name: str
    feature_store_url: str
    config_url: str
    instance_url: str
    dataset_url: str

    def __init__(self, *, project_name: str, feature_store_url: str):
        self.project_name = project_name
        self.feature_store_url = feature_store_url
        self.config_url = self.feature_store_url + "/config"
        self.instance_url = self.feature_store_url + "/instance"
        self.dataset_url = self.feature_store_url + "/dataset"

    def register_config(self, config_path: str, config_id: str) -> bool:
        """
        Registers a configuration file into the feature store.

        Uploads the file to remote storage and stores its hash and metadata in the feature store database.

        :param config_path: Path to the configuration file.
        :param config_id: Identifier to reference the configuration.
        :return: True if the configuration is registered successfully, False otherwise.
        """

        with open(config_path, "rb") as file:
            hash_func = hashlib.new("md5")
            hash_func.update(file.read())
            local_hash: str = hash_func.hexdigest()
        storage_uuid = uuid.uuid4()
        register_info = {"config_id": config_id, "hash": local_hash, "tags": {}, "storage_path": str(storage_uuid)}

        zip_files_to_send(config_path)
        with open(config_path + ".zip", "rb") as file:
            response = requests.post(url=self.config_url,
                                     params={"project_name": self.project_name},
                                     files={"config_file": file,
                                            "config": (None, json.dumps(register_info), "application/json")})
        if response.status_code != 200:
            logger.error(response.json())
            return False
        os.remove(config_path + ".zip")
        return True

    def get_configs_metadata(self) -> list[ProblemGenerationConfigInfo]:
        """
        Retrieves metadata of all registered configurations for this project.

        :return: A list of configuration metadata.
        """
        response = requests.get(url=self.config_url + "s", params={"project_name": self.project_name})
        return response.json()

    def get_config_metadata(self, config_id: str) -> ProblemGenerationConfigInfo | None:
        """
        Retrieves metadata of a specific configuration.

        :param config_id: Configuration identifier.
        :return: Configuration metadata if found, None otherwise.
        """
        response = requests.get(url=self.config_url, params={"project_name": self.project_name, "config_id": config_id})
        if response.status_code == 400:
            return None
        return response.json()

    def remove_config(self, config_id: str) -> bool:
        """
        Removes an instance generation configuration from the feature store.

        :param config_id: Configuration identifier.
        :return: True if the configuration has been removed, False instead.
        """
        response = requests.delete(url=self.config_url, params={"project_name": self.project_name, "config_id": config_id})
        if response.status_code != 200:
            logger.error(response.json())
            return False
        return True

    def download_config(self, config_id: str, output_dir: Path, unzip: bool = True) -> Path:
        response = requests.get(url=self.config_url + "/download",
                                params={"project_name": self.project_name, "config_id": config_id})
        if response.status_code != 200:
            raise Exception(f"Error while trying to download configuration {config_id} for project {self.project_name}"
                            f" : {response.json()['message']}.")
        return write_zip_from_response(response, output_dir, unzip)

    def register_instance(self, instance: Problem) -> bool:
        """
        Registers a problem instance into the feature store and uploads it to remote storage.

        :param instance: A Problem object to register.
        :return: True if successfully registered and uploaded, False otherwise.
        """
        distant_storage_name = str(uuid.uuid4())
        instance_infos: ProblemMetadata = instance.get_metadata()
        instance_infos["storage_path"] = distant_storage_name
        with TemporaryDirectory() as tmp_dir_name:
            tmp_dir_path = Path(tmp_dir_name)
            instance_path = tmp_dir_path / instance_infos.name
            instance.save(path=instance_path)
            zip_files_to_send(str(instance_path))
            with open(str(instance_path) + ".zip", "rb") as file:
                response = requests.post(url=self.instance_url,
                                         params={"project_name": self.project_name},
                                         files={"instance_file": file,
                                                "instance": (None, json.dumps(instance_infos), "application/json")})
            if response.status_code != 200:
                logger.error(response.json())
                return False
        return True

    def get_instances_metadata(
        self,
        min_version: int,
        config_id: str | None = None,
        date_filters: dict[str, tuple[datetime, datetime]] | None = None,
        tag_filters: dict | None = None,
    ) -> list[ProblemMetadata] | None:
        """
        Retrieve from the feature store the list of ProblemMetadata corresponding to the chosen filter parameters.

        :param min_version: (optional) Minimal code version of the problem to retrieve.
        :param config_id: (optional) Identifier of the configuration file used to generate the instances.
        :param date_filters: (optional) For any potential date in the problems metadata, defines the range to select from.
        :param tag_filters: (optional) Any specific key/value tags to filter the instances with.
        :return: List of problem metadata.
        """
        params: dict = {"min_version": min_version, "project_name": self.project_name}
        body: dict = {}
        if config_id is not None:
            params["config_id"] = config_id
        if date_filters is not None:
            body["date_filters"] = date_filters
        if tag_filters is not None:
            body["tag_filters"] = tag_filters
        response = requests.get(url=self.instance_url + "s", params=params, json=body)
        if response.status_code != 200:
            logger.error(response.json())
            return None
        return response.json()

    def get_instance_metadata(self, name: str, config_id: str, code_version: int) -> ProblemMetadata | None:
        """
        Retrieve from the feature store the ProblemMetadata of an instance by name, config ID and version.

        :param name: Name of the problem instance.
        :param config_id: Configuration identifier.
        :param code_version: Code version of the problem to retrieve.
        :return: Metadata if found, else None.
        """
        instance_key = {"project_name": self.project_name, "name": name, "config_id": config_id, "code_version": code_version}
        response = requests.get(url=self.instance_url, params=instance_key)
        if response.status_code != 200:
            logger.error(response.json())
            return None
        return response.json()

    def download_instance(self, name: str, config_id: str, code_version: int, output_dir: Path, unzip: bool = True) -> Path:
        """
        Downloads a registered problem instance if not already available locally.

        :param name: Instance name.
        :param config_id: Configuration identifier.
        :param code_version: Code version.
        :param output_dir: Directory where to save the instance.
        :param unzip: If True, unzip the downloaded file.
        :return: Local path of the downloaded instance.
        :raises Exception: If the instance does not exist in the feature store.
        """
        metadata = self.get_instance_metadata(name, config_id, code_version)
        if metadata is None:
            raise Exception(f"Instance with name '{name}', config ID '{config_id}' and version {code_version} does not exist.")
        storage_path = metadata["storage_path"]
        local_path = output_dir / storage_path
        if not local_path.exists():
            instance_key = {"project_name": self.project_name, "name": name,
                            "config_id": config_id, "code_version": code_version}
            response = requests.get(url=self.instance_url + "/download", params=instance_key)
            if response.status_code != 200:
                raise Exception(f"Error while trying to download instance : {response.json()['message']}.")
            return write_zip_from_response(response, output_dir, unzip)
        else:
            logger.info(f"Instance with name '{name}', config ID '{config_id}' and version {code_version} already downloaded")
        return local_path

    def remove_instance(self, name: str, config_id: str, code_version: int) -> bool:
        """
        Remove from the feature store an instance by name, config ID and version (also remove it from the associated storage).

        :param name: Name of the problem instance.
        :param config_id: Configuration identifier.
        :param code_version: Code version of the problem to retrieve.
        :return: True if the instance was cleanly removed, False otherwise.
        """
        instance_key = {"project_name": self.project_name, "name": name, "config_id": config_id, "code_version": code_version}
        response = requests.delete(url=self.instance_url, params=instance_key)
        if response.status_code != 200:
            logger.error(response.json())
            return False
        logger.info(f"Successfully removed instance {name}/{config_id}/{code_version} from feature store.")
        return True

    def register_dataset(self, dataset: ProblemDataset) -> bool:
        """
        Registers a dataset in the feature store and uploads it to remote storage.

        :param dataset: ProblemDataset object to register.
        :return: True if successfully registered and uploaded, False otherwise.
        """
        dataset_file_name = f"{dataset.name}_{dataset.split}_{dataset.version}"
        logger.info(f"Registering {dataset_file_name} in the feature store")

        storage_path = str(uuid.uuid4())
        dataset_infos = dataset.get_infos_for_feature_store()
        dataset_infos["storage_path"] = storage_path
        with TemporaryDirectory() as tmp_dir_name:
            tmp_dir_path = Path(tmp_dir_name)
            dataset_file_path = tmp_dir_path / dataset_file_name
            dataset.to_pickle(str(dataset_file_path))
            zip_files_to_send(str(dataset_file_path))
            with open(str(dataset_file_path) + ".zip", "rb") as file:
                response = requests.post(url=self.dataset_url,
                                         params={"project_name": self.project_name},
                                         files={"dataset_file": file,
                                                "dataset": (None, json.dumps(dataset_infos), "application/json")})
            if response.status_code != 200:
                logger.error(response.json())
                return False
        return True

    def get_datasets_metadata(self):
        """
        Retrieve metadata of all datasets registered in the feature store.

        :return: List of dataset metadata.
        """
        response = requests.get(url=self.instance_url + "s", params={"project_name": self.project_name})
        if response.status_code != 200:
            logger.error(response.json())
            return None
        return response.json()

    def get_dataset_metadata(self, name: str, split: str, version: int):
        """
        Retrieves metadata of a specific dataset from the feature store by name, split, version.

        :param name: Dataset name.
        :param split: Dataset split (e.g., train, test).
        :param version: Dataset version number.
        :return: Metadata if found, None otherwise.
        """
        dataset_key = {"project_name": self.project_name, "name": name, "split": split, "version": version}
        response = requests.get(url=self.dataset_url, params=dataset_key)
        if response.status_code != 200:
            logger.error(response.json())
            return None
        return response.json()

    def download_dataset(
        self, name: str, split: str, version: int, output_dir: Path, download_instances: bool = True
    ) -> ProblemDataset:
        """
        Downloads a dataset from the feature store, using its unique identifier (name, split, version).
        All Problem instances of the dataset are downloaded locally (if they are not already available) if the
         download_instances parameter is set to True.

        :param name: Dataset name.
        :param split: Dataset split.
        :param version: Dataset version
        :param output_dir: Local directory to store the dataset and its instances.
        :param download_instances: If True, downloads instances of the dataset not already available locally.
        :return: A ProblemDataset object, containing the metadata of the downloaded dataset and its instances' ProblemMetadata.
        :raises MissingDatasetError: If the dataset does not exist in the feature store.
        """
        key = f"{name}_{split}_{version}"
        metadata = self.get_dataset_metadata(name, split, version)
        if metadata is None:
            raise MissingDatasetError(f"Dataset {key} does not exist")
        storage_path = metadata["storage_path"]
        local_path = output_dir / storage_path
        if not local_path.exists():
            dataset_key = {"project_name": self.project_name, "name": name, "split": split, "version": version}
            response = requests.get(url=self.dataset_url + "/download", params=dataset_key)
            if response.status_code != 200:
                raise Exception(f"Error while trying to download dataset : {response.json()['message']}.")
            write_zip_from_response(response, output_dir, unzip=False)
        else:
            logger.info(f"Dataset file for {key} already downloaded")
        dataset: ProblemDataset = ProblemDataset.from_pickle(local_path / key)

        # Download missing instances of the list contained in the dataset
        if download_instances:
            to_download = dataset.get_locally_missing_instances(str(output_dir))
            if len(to_download) == 0:
                logger.info(f"All of {key} instances already downloaded")
            else:
                logger.info(f"Downloading problem instances of {key} missing locally ({len(to_download)} instances).")
            for instance in to_download:
                try:
                    self.download_instance(name=instance.name, config_id=instance.config_id,
                                           code_version=instance.code_version, output_dir=output_dir,
                                           unzip=False)
                except OSError as exc:
                    logger.error(f"Error while downloading instances : {exc} "
                                 f"(still {len(to_download) - to_download.index(instance)} to download).")
                    break

        return dataset

    def remove_dataset(self, name: str, split: str, version: int) -> bool:
        str_key = f"{name}_{split}_{version}"
        dataset_key = {"project_name": self.project_name, "name": name, "split": split, "version": version}
        response = requests.delete(url=self.dataset_url, params=dataset_key)
        if response.status_code != 200:
            logger.error(response.json())
            return False
        logger.info(f"Successfully removed dataset {str_key} from feature store.")
        return True


def write_zip_from_response(response: requests.Response, output_dir: Path, unzip: bool) -> Path:
    filename = response.headers["Content-Disposition"].split("filename=")[1][1:-1]
    local_path = output_dir / filename
    if not unzip:
        with open(local_path, "wb") as file:
            file.write(response.content)
    else:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            os.mkdir(local_path)
            zip_file.extractall(local_path)
    return local_path


def zip_files_to_send(source_path: str):
    if os.path.isdir(source_path):
        shutil.make_archive(source_path, "zip", source_path)
    else:
        zipfile.ZipFile(source_path + ".zip", mode="w").write(source_path, os.path.basename(source_path))


class MissingDatasetError(Exception):
    """
    Raised when a requested dataset is not found in the feature store.
    """
    pass
