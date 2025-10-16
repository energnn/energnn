#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from datetime import datetime

import flatdict
import numpy as np
from neptune_scale import Run
from neptune_scale.projects import create_project
from omegaconf import DictConfig, OmegaConf

from .tracker import Tracker


class NeptuneScaleTracker(Tracker):
    """
    Scalable Neptune AI tracker using `neptune-scale` client.

    Supports efficient logging of configurations, metrics, datasets, and model artifacts
    for high-throughput experiment tracking.

    :param project_name: Neptune name of the project.
    """

    run: Run
    project_name: str

    def __init__(self, project_name: str) -> None:
        """
        Initialize the NeptuneScaleTracker by creating or connecting to a Neptune project.

        :param project_name: Neptune name of the project.
        """
        self.project_name = project_name
        self.project = create_project(name=project_name, workspace="argo")

    def init_run(self, *, name: str, tags: list[str], cfg: DictConfig) -> None:
        """
        Start a new run and log experiment configuration to Neptune Scale.

        :param name: Run name for identification.
        :param tags: Tags to categorize and filter runs.
        :param cfg: Experiment parameters as an OmegaConf DictConfig.
        """
        self.run = Run(project=self.project_name, name=name, tags=tags)
        cfg_dict = stringify_unsupported(OmegaConf.to_container(cfg, resolve=True))
        self.run.log_configs(cfg_dict)

    def stop_run(self) -> None:
        """
        Close the current run, ensuring all logs are persisted.
        """
        self.run.close()

    def get_amortizer_path(self, *, tag: str) -> str:
        """Gets the path to the best amortizer for the run that bears the provided tag."""
        # TODO : validate use of this method
        # runs_table_df = self.project.fetch_runs_table(tag=tag, columns=[]).to_pandas()
        # if len(runs_table_df) != 1:
        #     raise ValueError(f"Found too many runs for tag {tag}")
        # run_id = runs_table_df["sys/id"].values[0]
        # with neptune.init_run(project=self.project_name, with_id=run_id, mode="read-only") as run:
        #     path = run["amortizers/best"].fetch()
        # return path
        return ""

    def run_track_dataset(self, *, infos: dict, target_path: str) -> None:
        """
        Log dataset metadata for the current run in Neptune Scale.

        :param infos: Dictionary of dataset metadata (e.g., name, version, split).
        :param target_path: Storage path for the dataset.
        """
        prefix: str = f"datasets/{target_path}/"
        flat_infos = flatdict.FlatDict(infos, delimiter="/")
        prefixed_flat_infos = {prefix + k: v for k, v in flat_infos.items()}
        self.run.log_configs(prefixed_flat_infos)

    def run_track_amortizer(self, *, id: str, target_path: str) -> None:
        """
        Log amortizer model identifier for the current run in Neptune Scale.

        :param id: Unique amortizer identifier.
        :param target_path: Storage path for the amortizer artifact.
        """
        self.run.log_configs({f"amortizers/{target_path}": id})

    def run_append(self, *, infos: dict, step: int) -> None:
        """
        Append scalar metrics to Neptune Scale run at a specified step.

        Filters out entries that are empty, nested dicts, or all-NaN arrays,
        then logs the mean of each metric.

        :param infos: Nested dictionary of metrics or arrays.
        :param step: Numeric step index for the metrics.
        """
        flat_infos = flatdict.FlatDict(infos, delimiter="/")
        for k, val in flat_infos.items():
            if (isinstance(val, dict)) or (np.size(val) == 0) or (np.all(np.isnan(val))):
                flat_infos.pop(k)
        metrics = {k: np.nanmean(v) for k, v in flat_infos.items()}
        self.run.log_metrics(metrics, step=step)


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
