# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from energnn.amortizer import SimpleAmortizer
from energnn.trainer_registry import LocalRegistry
from unittest.mock import MagicMock

def test_local_registry(tmp_path):
    registry = LocalRegistry("test_amortizer", tmp_path)

    amortizer: SimpleAmortizer = MagicMock()
    amortizer.project_name = "test_amortizer"
    amortizer.run_id = "test_run"
    amortizer.train_step = 1000
    metadata = {"name": "test_amortizer",
                "run_id": "test_run",
                "train_step": 1000}
    amortizer.get_metadata = MagicMock(return_value=metadata)
    amortizer.save = MagicMock()
    key = f"{amortizer.project_name}_{amortizer.run_id}_{amortizer.train_step}"
    registry.register_trainer(amortizer)
    assert (tmp_path / f"{key}_False_False").exists()
    amortizer.save.assert_called_with(name="trainer.pkl", directory=str(tmp_path / f"{key}_False_False"))

    stored_metadata = registry.get_trainer_metadata(amortizer.run_id, amortizer.train_step)
    metadata["last"] = False
    metadata["best"] = False
    assert stored_metadata == metadata

    SimpleAmortizer.load = MagicMock(return_value=amortizer)
    stored_amortizer = registry.download_trainer(amortizer.run_id, amortizer.train_step)
    assert stored_amortizer == amortizer
    SimpleAmortizer.load.assert_called_with(str(tmp_path / f"{key}_False_False" / "trainer.pkl"))
