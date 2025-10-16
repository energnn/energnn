#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import os
import shutil
import zipfile
from pathlib import Path

import boto3

from .storage import Storage


class S3Storage(Storage):
    """
    S3 remote storage implementation using client-side AES256 encryption.

    This class handles uploading and downloading files and directories to S3,
    transparently compressing data as ZIP archives and performing encryption with a
    customer-provided key.

    Encryption is performed at upload and download time using the AES256 algorithm and
    The encryption key must be made available via a file

    :param bucket: Name of the S3 bucket where data will be stored.
    :param key_bin_file_path: Path to the binary file containing the AES256 encryption key.
    """

    bucket: str
    key: bytes

    def __init__(self, bucket: str, key_bin_file_path: str):
        """
        Initialize the S3Storage with bucket name and key file path.

        Reads the binary encryption key from the given file location.

        :param bucket: Target S3 bucket name.
        :param key_bin_file_path: Filesystem path to the AES256 key file.
        """
        self.bucket = bucket
        self.key = Path(key_bin_file_path).read_bytes()

    def upload(self, source_path: str, target_path: str) -> None:
        """
        Compress and upload a file or directory to S3 with AES256 encryption.

        If `source_path` is a directory, it will be archived into a ZIP file
        named `<source_path>.zip`. If it's a file, only that single file will be
        zipped into `<source_path>.zip`.

        After zipping, the archive is uploaded to the configured S3 bucket under
        the specified `target_path` key with AES256 encryption. The local ZIP archive
        is removed after a successful upload.

        :param source_path: Local filesystem path to a file or directory to upload.
        :param target_path: S3 object key under which to store the ZIP archive.
        """
        if os.path.isdir(source_path):
            shutil.make_archive(source_path, "zip", source_path)
        else:
            zipfile.ZipFile(source_path + ".zip", mode="w").write(source_path, os.path.basename(source_path))
        client = boto3.client("s3")
        args = {"SSECustomerAlgorithm": "AES256", "SSECustomerKey": self.key}
        client.upload_file(Filename=source_path + ".zip", Bucket=self.bucket, Key=target_path, ExtraArgs=args)
        client.close()
        os.remove(source_path + ".zip")

    def download(self, source_path: str, target_path: str, overwrite: bool = False, unzip: bool = True) -> None:
        """
        Download and optionally extract a ZIP archive from S3 with decryption.

        Retrieves the object stored at `source_path` key in the configured S3 bucket.
        The downloaded file is decrypted using the same AES256 key and
        then unzipped into the `target_path` directory or file path. The local ZIP
        archive is deleted after extraction if `unzip` is True.

        :param source_path: S3 object key of the ZIP archive to download.
        :param target_path: Local filesystem path where data will be written.
        :param overwrite: Whether to overwrite existing local data at `target_path`, defaults to False.
        :param unzip: If True, the downloaded data will be unzipped into `target_path`.
                      If False, the ZIP file is saved directly to `target_path` with
                      no extraction. Defaults to True.
        """
        if os.path.exists(target_path) and not overwrite:
            print("{} already exists.".format(target_path))
            return None
        else:
            print("Writing {}.".format(target_path))
            client = boto3.client("s3")
            args = {
                "SSECustomerAlgorithm": "AES256",
                "SSECustomerKey": self.key,
            }
            filename = target_path + ".zip" if unzip else target_path
            client.download_file(Bucket=self.bucket, Key=source_path, Filename=filename, ExtraArgs=args)
            client.close()
            if unzip:
                shutil.unpack_archive(filename, target_path, "zip")
                os.remove(target_path + ".zip")

    def delete(self, target_path: str) -> None:
        """
        Delete and object from S3 bucket.

        :param target_path: S3 object key of the object to delete.
        """
        print("Deleting {} from bucket {}.".format(target_path, self.bucket))
        client = boto3.client("s3")
        client.delete_object(Bucket=self.bucket, Key=target_path)
        client.close()
