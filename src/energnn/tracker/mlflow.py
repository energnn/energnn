# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
import json
from tempfile import TemporaryDirectory

import flatdict
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.energnn.tracker import Tracker

class MlflowTracker(Tracker):

    def __init__(self, project_name: str, tracking_uri: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(project_name)

    def init_run(self, *, name: str, tags: dict[str, str], cfg: DictConfig):
        mlflow.start_run(run_name=name,
                         tags=tags)
        cfg_dict = stringify_unsupported(OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_params(cfg_dict)

    def stop_run(self):
        mlflow.end_run()

    def get_amortizer_path(self, *, tag: str) -> str:
        pass

    def run_track_dataset(self, *, infos: dict, target_path: str) -> None:
        with TemporaryDirectory() as tmp_dir:
            with open(f"{tmp_dir}/infos.json", "w") as f:
                json.dump(infos, f, indent=2)
            mlflow.log_artifact(f"{tmp_dir}/infos.json", artifact_path=f"datasets/{target_path}")

    def run_track_amortizer(self, *, id: str, target_path: str) -> None:
        mlflow.log_param(f"amortizers/{target_path}", id)

    def run_append(self, *, infos: dict, step: int) -> None:
        flat_infos = flatdict.FlatDict(infos, delimiter="/")
        for k, val in flat_infos.items():
            if (isinstance(val, dict)) or (np.size(val) == 0) or (np.all(np.isnan(val))):
                flat_infos.pop(k)
        metrics = {k: np.nanmean(v) for k, v in flat_infos.items()}
        mlflow.log_metrics(metrics, step=step)

def stringify_unsupported(d, parent_key="", sep="/") -> dict:
    """
    Flatten nested containers and stringify unsupported datatypes for Neptune Scale logging.

    Recursively traverses dicts, lists, tuples, and sets, flattening keys with a separator.
    Converts values not in supported types (int, float, str, datetime, bool, list, set)
    to strings.

    :param d: Input data structure to flatten.
    :param parent_key: Prefix for nested keys during recursion.
    :param sep: Separator used between nested key levels.
    :returns: Flattened dictionary with primitive or "stringified" values.
    """

    supported_datatypes = [int, float, str, datetime, bool, list, set]

    items = {}
    if not isinstance(d, (dict, list, tuple, set)):
        return d if type(d) in supported_datatypes else str(d)
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, (dict, list, tuple, set)):
                items |= stringify_unsupported(v, new_key, sep=sep)
            else:
                items[new_key] = v if type(v) in supported_datatypes else str(v)
    elif isinstance(d, (list, tuple, set)):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, (dict, list, tuple, set)):
                items.update(stringify_unsupported(v, new_key, sep=sep))
            else:
                items[new_key] = v if type(v) in supported_datatypes else str(v)
    return items
