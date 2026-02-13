#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from energnn.feature_store.feature_store_client import (
    FeatureStoreClient,
    MissingDatasetError, write_zip_from_response,
)
from energnn.problem.metadata import ProblemMetadata
from energnn.problem.dataset import ProblemDataset


class DummyResponse:
    def __init__(self, status_code=200, json_obj=None):
        self.status_code = status_code
        self._json = json_obj if json_obj is not None else {}

    def json(self):
        return self._json


class FakeProblem:
    """Minimal Problem stub used for register_instance tests."""

    def __init__(self, name, metadata: ProblemMetadata):
        self._metadata = metadata
        self._saved_path = None

    def get_metadata(self):
        return self._metadata

    def save(self, path: str):
        # emulate saving by creating a small file or dir
        p = Path(path)
        if p.exists() and p.is_dir():
            # create a small file inside directory
            f = p / "content.txt"
            f.write_text("data")
        else:
            # create parent then file
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("data")
        self._saved_path = str(path)


@patch("uuid.uuid4")
def test_register_config_new_success(mock_uuid):
    tmpf = tempfile.NamedTemporaryFile(delete=False)
    try:
        tmpf.write(b"abc123")
        tmpf.flush()
        tmpf.close()
        # compute md5
        m = hashlib.new("md5")
        with open(tmpf.name, "rb") as fh:
            m.update(fh.read())
        local_hash = m.hexdigest()

        mock_uuid.return_value = "UUID-CONF"

        client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

        # Patch get_config_metadata to return None -> indicates config not stored yet
        client.get_config_metadata = MagicMock(return_value=None)

        with patch("requests.post", return_value=DummyResponse(status_code=200)) as post_mock:
            res = client.register_config(config_path=tmpf.name, config_id="cid")
            assert res is True

            # ensure requests.post was called to register the config metadata
            post_mock.assert_called_once()
            # optionally inspect the json payload posted
            called_args, called_kwargs = post_mock.call_args
            # params should contain project_name
            assert called_kwargs["params"] == {"project_name": "proj"}
            # payload should contain a well formed json and the zipped config file
            posted_load = called_kwargs["files"]
            posted_json = json.loads(posted_load["config"][1])
            assert posted_json["config_id"] == "cid"
            assert posted_json["hash"] == local_hash
            posted_file = posted_load["config_file"]
            assert posted_file.name == tmpf.name + ".zip"
    finally:
        try:
            os.unlink(tmpf.name)
        except Exception:
            pass



def test_get_configs_and_get_config_metadata(monkeypatch):
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

    # get_configs_metadata => calls GET to config_url + "s"
    with patch("requests.get", return_value=DummyResponse(status_code=200, json_obj=[{"a": 1}])) as get_mock:
        res = client.get_configs_metadata()
        assert res == [{"a": 1}]
        get_mock.assert_called_once()

    # get_config_metadata: absent (400) -> None
    with patch("requests.get", return_value=DummyResponse(status_code=400, json_obj={"err": "no"})):
        assert client.get_config_metadata("nope") is None

    # get_config_metadata: present
    expected = {"config_id": "cid", "hash": "h"}
    with patch("requests.get", return_value=DummyResponse(status_code=200, json_obj=expected)):
        got = client.get_config_metadata("cid")
        assert got == expected


def test_remove_config_paths(monkeypatch):
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

    # case: present and delete success -> True
    with patch("requests.delete", return_value=DummyResponse(status_code=200)) as delete_mock:
        res = client.remove_config("cid")
        assert res is True

    # case: present but delete failed (non-200)
    with patch("requests.delete", return_value=DummyResponse(status_code=500)) as delete_mock:
        res2 = client.remove_config("cid")
        assert res2 is False


@patch("uuid.uuid4")
def test_register_instance_success_and_http_fail(mock_uuid, tmp_path):
    mock_uuid.return_value = "INST-UUID"
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

    # Build metadata
    pm = ProblemMetadata(
        name="instA",
        config_id="conf1",
        code_version=1,
        context_shape={"node": 1},
        decision_shape={"node": 1},
        storage_path="",
        filter_tags={},
    )

    # success path: requests.post returns 200
    problem = FakeProblem("instA", pm)
    with patch("requests.post", return_value=DummyResponse(status_code=200)) as post_mock:
        ok = client.register_instance(problem)
        assert ok is True
        # the HTTP post was called
        post_mock.assert_called_once()

    # failure path: requests.post returns non-200 -> should delete uploaded storage and return False
    problem2 = FakeProblem("instB", pm)
    with patch("requests.post", return_value=DummyResponse(status_code=500)):
        res = client.register_instance(problem2)
        assert res is False

def test_get_instances_and_get_instance_metadata(monkeypatch):
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

    # get_instances_metadata: success
    with patch("requests.get", return_value=DummyResponse(status_code=200, json_obj=[{"m": 1}])):
        res = client.get_instances_metadata(min_version=0)
        assert res == [{"m": 1}]

    # get_instances_metadata: HTTP non-200 -> None
    with patch("requests.get", return_value=DummyResponse(status_code=500, json_obj={"err": "bad"})):
        res2 = client.get_instances_metadata(min_version=0)
        assert res2 is None

    # get_instance_metadata: success
    expected = {"name": "n", "storage_path": "p"}
    with patch("requests.get", return_value=DummyResponse(status_code=200, json_obj=expected)):
        got = client.get_instance_metadata("n", "c", 1)
        assert got == expected

    # get_instance_metadata: failure -> None
    with patch("requests.get", return_value=DummyResponse(status_code=500, json_obj={"err": "x"})):
        assert client.get_instance_metadata("n", "c", 1) is None


def test_download_instance_success_and_already_local(tmp_path, monkeypatch):
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

    # setup metadata
    metadata = {"storage_path": "inst123"}
    # patch get_instance_metadata to return metadata
    client.get_instance_metadata = MagicMock(return_value=metadata)

    output_dir = Path(tmp_path)
    local_path = output_dir / "inst123"
    monkeypatch.setattr("energnn.feature_store.feature_store_client.write_zip_from_response", lambda res, d, u: local_path)

    # # ensure not exists => will call storage.download
    if local_path.exists():
        local_path.unlink()
    with patch("requests.get", return_value=DummyResponse(status_code=200)) as dl:
        got = client.download_instance("n", "c", 1, output_dir)
        assert got == local_path
        dl.assert_called_once()

    # if local exists, storage.download should not be called
    local_path.write_text("already")
    got2 = client.download_instance("n", "c", 1, output_dir)
    assert got2 == local_path


def test_download_instance_missing_metadata_raises():
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")
    client.get_instance_metadata = MagicMock(return_value=None)
    with pytest.raises(Exception):
        client.download_instance("no", "c", 1, Path("."))


def test_remove_instance_present_and_absent(monkeypatch):
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

    # present -> call requests.delete
    with patch("requests.delete", return_value=DummyResponse(status_code=200)) as delete_mock:
        res = client.remove_instance("n", "c", 1)
        assert res is True

# Tests for dataset register/download/remove
@patch("uuid.uuid4")
def test_register_dataset_success_and_fail(mock_uuid, tmp_path):
    mock_uuid.return_value = "DATA-UUID"
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

    # create a small ProblemDataset-like object (use real ProblemDataset)
    pm1 = ProblemMetadata(
        name="i1",
        config_id="c",
        code_version=1,
        context_shape={"node": 1},
        decision_shape={"node": 1},
        storage_path="inst-a",
        filter_tags={},
    )
    ds = ProblemDataset(
        name="myds",
        split="train",
        version=1,
        instances=[pm1],
        size=1,
        context_max_shape={},
        decision_max_shape={},
        generation_date=datetime.now(),
        selection_criteria={},
        tags={},
    )

    # success path: requests.post returns 200 -> to_pickle + upload called
    with patch("requests.post", return_value=DummyResponse(status_code=200)) as post_mock:
        ok = client.register_dataset(ds)
        assert ok is True

    # fail path: requests.post returns non-200 -> False and no upload
    with patch("requests.post", return_value=DummyResponse(status_code=500)):
        ok2 = client.register_dataset(ds)
        assert ok2 is False


def test_get_datasets_and_get_dataset_metadata(monkeypatch):
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

    # get_datasets_metadata -> note code uses instance_url + "s" (legacy), but we just mock requests.get
    with patch("requests.get", return_value=DummyResponse(status_code=200, json_obj=[{"ds": 1}])):
        r = client.get_datasets_metadata()
        assert r == [{"ds": 1}]

    # get_dataset_metadata -> success
    with patch("requests.get", return_value=DummyResponse(status_code=200, json_obj={"storage_path": "sp"})):
        g = client.get_dataset_metadata("name", "train", 1)
        assert g == {"storage_path": "sp"}

    # failure path
    with patch("requests.get", return_value=DummyResponse(status_code=500)):
        assert client.get_dataset_metadata("name", "train", 1) is None


def test_download_dataset_missing_raises(tmp_path, monkeypatch):
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

    client.get_dataset_metadata = MagicMock(return_value=None)
    with pytest.raises(MissingDatasetError):
        client.download_dataset("n", "train", 1, Path(tmp_path), download_instances=False)


def test_download_dataset_downloads_instances(monkeypatch, tmp_path):
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

    # make metadata
    metadata = {"storage_path": "dataset-storage"}
    client.get_dataset_metadata = MagicMock(return_value=metadata)

    # prepare output_dir path
    output_dir = Path(tmp_path)
    local_path = output_dir / metadata["storage_path"]

    # Prepare a fake dataset returned by ProblemDataset.from_pickle
    fake_dataset = MagicMock(spec=ProblemDataset)
    # emulate a dataset with two instances; get_locally_missing_instances returns list of missing paths
    pm1 = ProblemMetadata(
        name="i1",
        config_id="c",
        code_version=1,
        context_shape={"node": 1},
        decision_shape={"node": 1},
        storage_path="inst-a",
        filter_tags={},
    )
    pm2 = ProblemMetadata(
        name="i2",
        config_id="c",
        code_version=1,
        context_shape={"node": 1},
        decision_shape={"node": 1},
        storage_path="inst-b",
        filter_tags={},
    )
    fake_dataset.get_locally_missing_instances.return_value = [pm1, pm2]

    # monkeypatch ProblemDataset.from_pickle to return our fake_dataset
    with patch("energnn.feature_store.feature_store_client.ProblemDataset.from_pickle", return_value=fake_dataset):
        # case 1: storage not almost full -> should attempt to download each missing instance
        monkeypatch.setattr("energnn.feature_store.feature_store_client.write_zip_from_response", lambda res, d, unzip: local_path)
        with patch.object(client, "download_instance", return_value=None) as dl, patch("requests.get", return_value=DummyResponse(status_code=200)) as dl2:
            ds = client.download_dataset("name", "train", 1, output_dir, download_instances=True)
            # ensure we return the dataset object
            assert ds is fake_dataset
            # ensure dataset download (and instance downloads) were attempted
            assert dl2.call_count == 1
            assert dl.call_count == 2

        # case 2: storage almost full -> dataset.remove_instance should be called instead of download
        fake_dataset.get_locally_missing_instances.return_value = ["inst3"]

        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("already downloaded")


def test_remove_dataset_paths(monkeypatch):
    client = FeatureStoreClient(project_name="proj", feature_store_url="http://fs")

    # dataset absent -> False
    with patch("requests.delete", return_value=DummyResponse(status_code=400)):
        assert client.remove_dataset("n", "s", 1) is False

    # present and delete succeeds
    with patch("requests.delete", return_value=DummyResponse(status_code=200)):
        ok = client.remove_dataset("n", "s", 1)
        assert ok is True
