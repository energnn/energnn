#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from datetime import datetime
import json
from pathlib import Path

import pytest

from energnn.problem.metadata import ProblemMetadata
from energnn.problem.dataset import ProblemDataset


def make_metadata(name: str, storage_path: str) -> ProblemMetadata:
    """Helper to create a ProblemMetadata with minimal required fields."""
    return ProblemMetadata(
        name=name,
        config_id="cfg-1",
        code_version=1,
        context_shape={"node": 3},
        decision_shape={"node": 2},
        storage_path=storage_path,
        filter_tags={"tag": "val"},
    )


def test_dataset_properties_reflect_input():
    m1 = make_metadata("inst1", "a.pkl")
    m2 = make_metadata("inst2", "b.pkl")
    gen_date = datetime(2023, 1, 2, 3, 4, 5)

    ds = ProblemDataset(
        name="mydataset",
        split="train",
        version=7,
        instances=[m1, m2],
        size=2,
        context_max_shape={"node": 10},
        decision_max_shape={"node": 5},
        generation_date=gen_date,
        selection_criteria={"min_size": 0},
        tags={"owner": "me"},
    )

    assert ds.name == "mydataset"
    assert ds.split == "train"
    assert ds.version == 7
    assert ds.size == 2
    assert ds.context_max_shape == {"node": 10}
    assert ds.decision_max_shape == {"node": 5}
    assert ds.selection_criteria == {"min_size": 0}
    assert ds.tags == {"owner": "me"}
    # instances present and in order
    assert len(ds.instances) == 2
    assert ds.instances[0].name == "inst1"
    assert ds.instances[1].storage_path == "b.pkl"


def test_get_infos_for_feature_store_removes_instances_and_datestring():
    m = make_metadata("inst", "some/path.pkl")
    gen_date = datetime(2024, 6, 1, 12, 0, 0)
    ds = ProblemDataset(
        name="ds",
        split="val",
        version=1,
        instances=[m],
        size=1,
        context_max_shape={},
        decision_max_shape={},
        generation_date=gen_date,
        selection_criteria={},
        tags={},
    )

    info = ds.get_infos_for_feature_store()
    # instances key must be removed
    assert "instances" not in info
    # generation_date must be stringified
    assert isinstance(info["generation_date"], str)
    assert info["generation_date"] == str(gen_date)
    # other fields stay present
    assert info["name"] == "ds"
    assert info["size"] == 1


def test_get_instance_paths_list():
    m1 = make_metadata("i1", "p/a.pkl")
    m2 = make_metadata("i2", "p/b.pkl")
    ds = ProblemDataset(
        name="ds",
        split="test",
        version=2,
        instances=[m1, m2],
        size=2,
        context_max_shape={},
        decision_max_shape={},
        generation_date=datetime.now(),
        selection_criteria={},
        tags={},
    )
    assert ds.get_instance_paths() == ["p/a.pkl", "p/b.pkl"]


def test_get_locally_missing_instances_detects_missing_files(tmp_path: Path):
    # create two metadata entries: create file for first, not for second
    m1 = make_metadata("i1", "exists.pkl")
    m2 = make_metadata("i2", "missing.pkl")

    # create the existing file under tmp_path
    (tmp_path / "exists.pkl").write_text("ok")

    ds = ProblemDataset(
        name="ds",
        split="train",
        version=1,
        instances=[m1, m2],
        size=2,
        context_max_shape={},
        decision_max_shape={},
        generation_date=datetime.now(),
        selection_criteria={},
        tags={},
    )

    missing = ds.get_locally_missing_instances(str(tmp_path))
    assert missing == [m2]


def test_to_json_writes_file_and_content(tmp_path: Path):
    m = make_metadata("inst", "x.pkl")
    ds = ProblemDataset(
        name="nm",
        split="split",
        version=3,
        instances=[m],
        size=1,
        context_max_shape={"node": 1},
        decision_max_shape={"node": 1},
        generation_date=datetime(2022, 2, 2, 2, 2, 2),
        selection_criteria={"c": "v"},
        tags={"k": "v"},
    )
    out = tmp_path / "ds.json"
    ds.to_json(str(out))

    loaded = json.loads(out.read_text(encoding="utf-8"))
    # Instances removed? to_json writes full dict (it writes 'instances' too), but we check keys exist
    assert "name" in loaded and loaded["name"] == "nm"
    assert "generation_date" in loaded
    assert "instances" in loaded


def test_to_pickle_from_pickle_roundtrip_preserves_data(tmp_path: Path):
    m1 = make_metadata("i1", "a.pkl")
    m2 = make_metadata("i2", "b.pkl")
    gen_date = datetime(2021, 12, 31, 23, 59, 59)
    ds = ProblemDataset(
        name="my",
        split="s",
        version=10,
        instances=[m1, m2],
        size=2,
        context_max_shape={"node": 3},
        decision_max_shape={"node": 2},
        generation_date=gen_date,
        selection_criteria={"a": 1},
        tags={"t": 1},
    )

    pfile = tmp_path / "ds.pkl"
    ds.to_pickle(str(pfile))
    ds2 = ProblemDataset.from_pickle(str(pfile))
    assert isinstance(ds2, ProblemDataset)
    # fields preserved
    assert ds2.name == "my"
    assert ds2.split == "s"
    assert ds2.version == 10
    assert ds2.size == 2
    # generation_date preserved as datetime via pickle
    assert ds2.generation_date == gen_date
    # instances list preserved length and storage paths
    assert len(ds2.instances) == 2
    assert ds2.instances[0].storage_path == "a.pkl"


def test_get_locally_missing_instances_with_subpaths(tmp_path: Path):
    # Instances have nested paths
    m1 = make_metadata("i1", "sub/f1.pkl")
    m2 = make_metadata("i2", "sub/f2.pkl")
    # create only f1 under tmp_path/sub
    (tmp_path / "sub").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sub" / "f1.pkl").write_text("ok")

    ds = ProblemDataset(
        name="ds",
        split="s",
        version=1,
        instances=[m1, m2],
        size=2,
        context_max_shape={},
        decision_max_shape={},
        generation_date=datetime.now(),
        selection_criteria={},
        tags={},
    )

    missing = ds.get_locally_missing_instances(str(tmp_path))
    assert missing == [m2]


def test_behaviour_with_empty_instances_list(tmp_path: Path):
    ds = ProblemDataset(
        name="empty",
        split="s",
        version=1,
        instances=[],
        size=0,
        context_max_shape={},
        decision_max_shape={},
        generation_date=datetime.now(),
        selection_criteria={},
        tags={},
    )
    assert ds.get_instance_paths() == []
    assert ds.get_locally_missing_instances(str(tmp_path)) == []
    infos = ds.get_infos_for_feature_store()
    assert "instances" not in infos
    assert "generation_date" in infos
