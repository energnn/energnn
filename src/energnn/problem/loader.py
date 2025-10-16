#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from abc import ABC, abstractmethod
from typing import Sized, Iterator

from .dataset import ProblemDataset
from .batch import ProblemBatch


class ProblemLoader(ABC, Sized, Iterator[ProblemBatch]):
    """
    Abstract base class for problem loaders that yield batches of problem instances.

    Iterates over a provided `ProblemDataset` in consecutive batches.

    :param dataset: Dataset containing problem instance metadata.
    :param batch_size: Number of instances per batch returned by the iterator.
    :param shuffle: If true, randomly shuffle the dataset.
    """

    @abstractmethod
    def __init__(self, dataset: ProblemDataset, batch_size: int, shuffle: bool = False):
        """
        Initialize the problem loader.

        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[ProblemBatch]:
        """
        Return the loader iterator.

        Should optionally reshuffle data if `shuffle=True`.

        :returns: Iterator over batches.
        """
        raise NotImplementedError

    @abstractmethod
    def __next__(self) -> ProblemBatch:
        """
        Retrieve the next batch of problems.

        :returns: A `ProblemBatch` containing up to `batch_size` problem instances.


        :raises StopIteration: if there are no further items.
        :raises NotImplementedError: if subclass does not override this constructor.
        """

        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Number of batches per epoch.


        :raises NotImplementedError: if subclass does not override this constructor.
        """
        raise NotImplementedError
