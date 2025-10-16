#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import flatdict
import neptune
import numpy as np
from neptune.utils import stringify_unsupported
from neptune import Run
from omegaconf import DictConfig, OmegaConf

from .tracker import Tracker


class NeptuneTracker(Tracker):
    """
    Tracker implementation using Neptune AI for experiment logging.

    This tracker logs experiment runs, parameters, metrics, datasets, and model artifacts
    to Neptune. It initializes a Neptune project connection and handles run lifecycle.

    :param project_name: Neptune project identifier.
    """

    run: Run
    project_name: str

    def __init__(self, project_name: str) -> None:
        """
        Create a NeptuneTracker and initialize Neptune project.

        :param project_name: Name of the Neptune project.
        """
        self.project_name = project_name
        self.project = neptune.init_project(project=project_name)

    def init_run(self, *, name: str, tags: list[str], cfg: DictConfig) -> None:
        """
        Start a new Neptune run and log configuration parameters.

        :param name: Descriptive run name for display in Neptune.
        :param tags: List of tags for organizing runs in the Neptune UI.
        :param cfg: OmegaConf configuration object containing experiment parameters.
        """
        self.run = neptune.init_run(project=self.project_name, name=name, tags=tags)
        cfg_dict = stringify_unsupported(OmegaConf.to_container(cfg, resolve=True))
        self.run["parameters"] = cfg_dict

    def stop_run(self) -> None:
        """
        Stop the current Neptune run, ensuring all pending logs are flushed.
        """
        self.run.stop()

    def get_amortizer_path(self, *, tag: str) -> str:
        """
        Retrieve the best amortizer model path logged under a specific run tag.

        :param tag: Tag used to identify the run containing the saved amortizer.
        :return: Remote path or identifier of the best amortizer artifact.
        :raises ValueError: If multiple or no runs match the provided tag.
        """
        # Demander
        runs_table_df = self.project.fetch_runs_table(tag=tag, columns=[]).to_pandas()
        if len(runs_table_df) != 1:
            raise ValueError(f"Found too many runs for tag {tag}")
        run_id = runs_table_df["sys/id"].values[0]
        with neptune.init_run(project=self.project_name, with_id=run_id, mode="read-only") as run:
            path = run["amortizers/best"].fetch()
        return path

    def run_track_dataset(self, *, infos: dict, target_path: str) -> None:
        """
        Log dataset metadata under the current run.

        :param infos: Dictionary containing dataset information (e.g., name, version, split).
        :param target_path: Storage path or identifier for the dataset.
        """
        self.run["datasets/" + target_path] = infos

    def run_track_amortizer(self, *, id: str, target_path: str) -> None:
        """
        Log amortizer model identifier under the current run.

        :param id: Identifier of the amortizer
        :param target_path: Storage path for the amortizer artifact.
        """
        self.run["amortizers/" + target_path] = id

    def run_append(self, *, infos: dict, step: int) -> None:
        """
        Append scalar metrics to the Neptune run at a given step.

        Flattens nested info dictionaries and logs each scalar metric.

        :param infos: Nested dictionary of metrics or information.
        :param step: Training or evaluation step number.
        """
        flat_infos = flatdict.FlatDict(infos, delimiter="/")
        for k, val in flat_infos.items():
            if (not isinstance(val, dict)) and (np.size(val) > 0) and (not np.all(np.isnan(val))):
                self.run[k].append(np.nanmean(val), step=step)
