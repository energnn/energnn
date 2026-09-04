# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import time
import warnings

import jax
import numpy as np


def numpify_metrics_dict(metrics: dict) -> dict:
    """
    Convert all numeric entries in an information dictionary to numpy scalar arrays.

    This function iterates over the provided `metrics` dictionary and ensures that each
    value is converted to a numpy scalar. 1-dimensional arrays (jax.Array, numpy.ndarray,
    or list) are averaged along their only axis. Scalar values are wrapped with `np.array`.
    Nested dictionaries are not supported, and a warning will be emitted if encountered.

    :param metrics: A mapping from string keys to values that are either:
        - jax.Array or numpy.ndarray of dimension < 2
        - Python list of numbers (dimension < 2)
        - Python int or float
        - dict (unsupported, emits a warning)
    :raises ValueError:
        If a value in `metrics` is not one of the supported types.
    :returns:
        A new dictionary with the same keys as `metrics` and numpy scalar values.
        All numeric arrays or lists are reduced via their mean, and scalars are
        converted to zero-dimensional numpy arrays.
    """
    np_metrics_dict = {}
    for k, numerical_metric in metrics.items():
        if isinstance(numerical_metric, jax.Array) or isinstance(numerical_metric, np.ndarray):
            assert numerical_metric.ndim < 2
            np_metrics_dict[k] = np.mean(np.asarray(numerical_metric))
        elif isinstance(numerical_metric, list):
            metric_array = np.asarray(numerical_metric)
            assert metric_array.ndim < 2
            np_metrics_dict[k] = np.mean(metric_array)
        elif isinstance(numerical_metric, (float, int)):
            np_metrics_dict[k] = np.array(numerical_metric)
        elif isinstance(numerical_metric, dict):
            warnings.warn("Nested information dict are not supported.")
        else:
            raise ValueError(f"Unsupported metric : {numerical_metric}, of type {type(numerical_metric)} with key: {k}")
    return np_metrics_dict


def append_scores_and_metrics(
    scores_acc: np.ndarray,
    metrics_acc: dict[str, np.ndarray],
    scores_batch: np.ndarray,
    metrics_batch: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Append batched metrics and information arrays to existing accumulators.

    This function concatenates the provided `scores_batch` to the existing
    1-D `scores_acc` array. Similarly, it appends each entry in `metrics_batch`
    to the corresponding array in `metrics_acc`. Nested dicts in `metrics_batch`
    are not supported and will trigger a warning.

    :param scores_acc: A 1-dimensional numpy array acting as the accumulator for metrics.
    :param metrics_acc: A dictionary mapping string keys to 1-dimensional numpy arrays
                      that act as accumulators for auxiliary information.
    :param scores_batch: A 1-dimensional numpy array of new metric values to append.
    :param metrics_batch: A dictionary mapping string keys to values that are either:
                        - 1-dimensional numpy.ndarray or jax.Array
                        - Python list
                        - Python int or float (will emit a warning but not be appended)

    :returns: A tuple containing:
            - Updated metrics accumulator (numpy.ndarray)
            - Updated metrics accumulator (dict[str, numpy.ndarray])

    :raises ValueError:
        If a value in `metrics_batch` is neither a nested dict, array, list, nor scalar.
    """

    scores_acc = np.append(scores_acc, scores_batch)
    for k, v in metrics_batch.items():
        if isinstance(v, dict):
            warnings.warn("Does not support nested dict")
        elif isinstance(v, (list, np.ndarray, jax.Array)):
            metrics_acc[k] = np.append(metrics_acc[k], v)
        elif isinstance(v, (int, float)):
            warnings.warn("Batched metrics should be arrays")
        else:
            raise ValueError(f"Unsupported metrics type : {type(v)}")
    return scores_acc, metrics_acc


class TaskLogger:
    """
    Context manager for logging timing information of a code block.

    Use this class in a `with` statement to automatically log the start,
    end, and duration of a given task.

    :param logger: An object with `info` and `error` methods (e.g., a Python `logging.Logger`).
    :param task_name: A descriptive name for the task being logged.
    """

    def __init__(self, logger, task_name: str):
        """
        Initialize the TaskLogger context manager.

        :param logger: A logging-like object with `metrics` and `error` methods.
        :param task_name: Human-readable name for the task.
        """
        self.logger = logger
        self.task_name = task_name

    def __enter__(self):
        """
        Enter the runtime context and record the start time.

        :returns: This TaskLogger instance for use within the context.
        """
        self.logger.info(f"{self.task_name}...")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context and log completion or error with elapsed time.

        :param exc_type: Exception type if raised in the context, else None.
        :param exc_value: Exception instance if raised, else None.
        :param traceback: Traceback object if an exception was raised, else None.

        :returns: False to propagate exceptions, True to suppress them.
        """
        end_time = time.perf_counter()
        self.elapsed_time = (end_time - self.start_time) * 1000  # in ms
        if exc_type is None:
            self.logger.info(f"{self.task_name} completed in {self.elapsed_time:.2f} ms")
        else:
            self.logger.error(f"{self.task_name} failed after {self.elapsed_time:.2f} ms due to: {exc_value}")
