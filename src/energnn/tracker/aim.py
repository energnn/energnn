# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import subprocess
import sys
from datetime import datetime

import flatdict
import numpy as np
from importlib.metadata import distributions
from aim import Run, Repo
from omegaconf import DictConfig, OmegaConf

from .tracker import Tracker


class AimTracker(Tracker):

    def __init__(self, project_name: str, tracking_uri: str) -> None:
        self.aim_run = None
        self.repo = Repo(tracking_uri)
        self.project_name = project_name

    def init_run(self, *, name: str, tags: dict[str, str], cfg: DictConfig):
        self.aim_run = Run(repo=self.repo, experiment=self.project_name)
        self.aim_run['__system_params'] = {
            'packages': {i.metadata["Name"].lower(): i.version for i in distributions()},
            'git_info': get_git_info(),
            'executable': sys.executable,
            'arguments': sys.argv,
        }
        self.aim_run.name = name
        for tag, value in tags.items():
            self.aim_run.add_tag(f"{tag}:{value}")
        cfg_dict = stringify_unsupported(OmegaConf.to_container(cfg, resolve=True))
        self.aim_run['config'] = cfg_dict

    def stop_run(self):
        self.aim_run.close()

    def run_append(self, *, infos: dict, step: int) -> None:
        flat_infos = flatdict.FlatDict(infos, delimiter="/")
        for k, val in flat_infos.items():
            if (isinstance(val, dict)) or (np.size(val) == 0) or (np.all(np.isnan(val))):
                continue
            self.aim_run.track(np.nanmean(val), name=k, step=step)


def stringify_unsupported(d, parent_key="", sep="/") -> dict:
    """
    Flatten nested containers and stringify unsupported datatypes for logging.

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

def get_git_info():
    git_info = {}
    try:
        r = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # not a git repo
        return git_info
    else:
        output = r.stdout.decode('utf-8').strip().lower()
        if output != 'true':
            # malformed result
            return git_info

    cmds = {
        'branch': ('git', 'rev-parse', '--abbrev-ref', 'HEAD'),
        'remote_origin_url': ('git', 'config', '--get', 'remote.origin.url'),
        'commit': ('git', 'log', '--pretty=format:%h/%ad/%an', '--date=iso-strict', '-1'),
    }
    results = {}
    for key, cmd in cmds.items():
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            continue
        else:
            output = r.stdout.decode('utf-8').strip()
            results[key] = output

    try:
        commit_hash, commit_timestamp, commit_author = results.get('commit').split('/')
    except (ValueError, AttributeError):
        commit_hash = commit_timestamp = commit_author = None

    git_info.update(
        {
            'branch': results.get('branch'),
            'remote_origin_url': results.get('remote_origin_url'),
            'commit': {'hash': commit_hash, 'timestamp': commit_timestamp, 'author': commit_author},
        }
    )

    return git_info