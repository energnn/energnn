#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import logging
import os
import shutil
from pathlib import Path

from .storage import Storage

logger = logging.getLogger(__name__)


class LocalStorage(Storage):
    """
    Local storage in filesystem.

    :param root_directory: Root directory for local storage.
    """

    root_directory: Path

    def __init__(self, root_directory: str):
        self.root_directory = Path(root_directory)

    def upload(self, source_path: str, target_path: str) -> None:
        """
        Upload a file or directory from `source_path` to `target_path`, relative to the root directory.

        :param source_path: Absolute or relative path to the file or directory to be uploaded.
        :param target_path: Destination path relative to the `root_directory` where the resource should be uploaded.
        """
        abs_target_path = self.root_directory / target_path
        abs_target_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Uploading '{source_path}' to '{target_path}'")
        if os.path.isdir(source_path):
            shutil.copytree(source_path, abs_target_path, dirs_exist_ok=True)
        else:
            shutil.copy(source_path, abs_target_path)

    def download(self, source_path: str, target_path: str, overwrite: bool = False, unzip: bool = True) -> None:
        """
        Retrieve a file or directory from `source_path` (relative to `root_directory`) to a local `target_path`.

        :param source_path: Path relative to the internal `root_directory` pointing to the file or directory to download.
        :param target_path: Absolute or relative path on the local file system where the content should be copied.
        :param overwrite: If False and the target already exists, the download is skipped. If True, the target is overwritten.
        :param unzip: Currently unused, but can be extended to handle automatic extraction of zip archives after download.
        """
        if os.path.exists(target_path) and not overwrite:
            logger.info("'{}' already exists.".format(target_path))
        else:
            logger.info(f"Downloading '{source_path}' to '{target_path}'")
            abs_source_path = self.root_directory / source_path
            Path(target_path).mkdir(parents=True, exist_ok=True)
            if abs_source_path.is_dir():
                shutil.copytree(abs_source_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy(abs_source_path, target_path)

    def delete(self, target_path: str) -> None:
        """
        Delete a file or directory from `target_path`, relative to the root directory.

        :param target_path: File path relative to the `root_directory` of the resource to delete.
        """
        logger.info(f"Deleting '{target_path}'")
        abs_source_path = self.root_directory / target_path
        if abs_source_path.is_dir():
            shutil.rmtree(abs_source_path)
        else:
            Path(abs_source_path).unlink(True)
