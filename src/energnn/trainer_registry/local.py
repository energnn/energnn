# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import json
import logging
import os
from pathlib import Path

from energnn.amortizer import SimpleAmortizer
from energnn.amortizer.metadata import TrainerMetadata
from energnn.trainer_registry import TrainerRegistry


class LocalRegistry(TrainerRegistry):
    """
    Local implementation model registry for storing and retrieving trained models locally.

    :param local_directory: Path to the local directory where models are stored.
    """
    local_directory: Path

    def __init__(self, local_directory: Path):
        self.local_directory = local_directory

    def register_trainer(self, trainer: SimpleAmortizer, best: bool = False, last: bool = False) -> bool:
        save_name = f"{trainer.project_name}_{trainer.run_id}_{trainer.train_step}_{best}_{last}"
        save_dir = self.local_directory / save_name
        metadata: TrainerMetadata = trainer.get_metadata()
        metadata["best"] = best
        metadata["last"] = last
        os.makedirs(save_dir, exist_ok=True)
        trainer.save(name="trainer.pkl", directory=str(save_dir))
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        return True

    def get_trainer_metadata(self, project_name: str, run_id: str, step: int) -> TrainerMetadata | None:
        key = f"{project_name}_{run_id}_{step}"
        stored_trainers = os.listdir(self.local_directory)
        for trainer_name in stored_trainers:
            if trainer_name.startswith(key):
                metadata_path = self.local_directory / trainer_name / "metadata.json"
                with open(metadata_path, "r") as f:
                    return TrainerMetadata(**json.load(f))
        logging.error(f"Trainer with key {key} not found in local directory.")
        return None

    def download_trainer(self, project_name: str, run_id: str, step: int) -> SimpleAmortizer | None:
        key = f"{project_name}_{run_id}_{step}"
        stored_trainers = os.listdir(self.local_directory)
        for trainer_name in stored_trainers:
            if trainer_name.startswith(key):
                trainer_path = self.local_directory / trainer_name / "trainer.pkl"
                return SimpleAmortizer.load(str(trainer_path))
        logging.error(f"Trainer with key {key} not found in local directory.")
        return None
