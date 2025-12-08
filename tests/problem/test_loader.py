#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from math import ceil
import random

import pytest

from datetime import datetime
from energnn.problem.dataset import ProblemDataset
from energnn.problem.metadata import ProblemMetadata
from energnn.problem.batch import ProblemBatch


class FakeProblemBatch(ProblemBatch):
    """
    Minimal ProblemBatch implementation for tests.
    Stores a small list of ProblemMetadata instances.
    """

    def __init__(self, instances):
        # instances is a list of ProblemMetadata
        self._instances = instances

    def get_context(self, get_info: bool = False):
        # For testing we simply return the list as "context" and an empty info dict
        return self._instances, {}

    def get_gradient(self, *, decision, get_info: bool = False, cfg=None):
        raise NotImplementedError

    def get_metrics(self, *, decision, get_info: bool = False, cfg=None):
        raise NotImplementedError

    def get_zero_decision(self, get_info: bool = False):
        raise NotImplementedError


class SimpleProblemLoader:
    """
    Minimal concrete ProblemLoader used for unit tests.
    Implements __iter__, __next__, __len__ and supports optional shuffling
    with a reproducible seed.
    """

    def __init__(self, dataset: ProblemDataset, batch_size: int, shuffle: bool = False, seed: int | None = None):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        # store seed for deterministic re-shuffling across __iter__ calls
        self._seed = seed
        self._indices = list(range(len(self.dataset.instances)))
        self._pos = 0
        # prepare initial order (but do not call reset here to keep semantics explicit)
        self.reset(self._seed)

    def reset(self, seed: int | None = None):
        # Build index list and optionally shuffle deterministically with provided seed
        self._indices = list(range(len(self.dataset.instances)))
        if self.shuffle:
            rnd = random.Random(seed)
            rnd.shuffle(self._indices)
        self._pos = 0

    def __iter__(self):
        # Reset with stored seed so repeated iter(...) yield reproducible order when seed provided.
        self.reset(self._seed)
        return self

    def __next__(self):
        if self._pos >= len(self._indices):
            raise StopIteration
        i0 = self._pos
        i1 = min(self._pos + self.batch_size, len(self._indices))
        batch_idx = self._indices[i0:i1]
        instances = [self.dataset.instances[i] for i in batch_idx]
        self._pos = i1
        return FakeProblemBatch(instances)

    def __len__(self):
        return (len(self.dataset.instances) + self.batch_size - 1) // self.batch_size


def make_problem_metadata(i: int, storage_base: str = "inst") -> ProblemMetadata:
    """Create a small ProblemMetadata with deterministic fields for testing."""
    name = f"name_{i}"
    config_id = f"cfg_{i}"
    code_version = i
    context_shape = {"node": 10}
    decision_shape = {"node": 2}
    storage_path = f"{storage_base}_{i}.pkl"
    filter_tags = {"tag": i % 2}
    return ProblemMetadata(
        name=name,
        config_id=config_id,
        code_version=code_version,
        context_shape=context_shape,
        decision_shape=decision_shape,
        storage_path=storage_path,
        filter_tags=filter_tags,
    )


def make_dataset(n_instances: int) -> ProblemDataset:
    instances = [make_problem_metadata(i) for i in range(n_instances)]
    now = datetime.utcnow()
    ds = ProblemDataset(
        name="ds_test",
        split="train",
        version=1,
        instances=instances,
        size=len(instances),
        context_max_shape={"node": 10},
        decision_max_shape={"node": 2},
        generation_date=now,
        selection_criteria={},
        tags={"unit_test": True},
    )
    return ds


# -------------------------
# Tests
# -------------------------
def test_len_matches_ceil_of_dataset_over_batchsize():
    N = 10
    B = 3
    ds = make_dataset(N)
    loader = SimpleProblemLoader(dataset=ds, batch_size=B, shuffle=False)
    assert len(loader) == ceil(N / B)


def test_iteration_returns_all_instances_once_in_order_when_no_shuffle():
    N = 7
    B = 3
    ds = make_dataset(N)
    loader = SimpleProblemLoader(dataset=ds, batch_size=B, shuffle=False)
    collected = []
    count_batches = 0
    for batch in loader:
        count_batches += 1
        batch_instances, _ = batch.get_context()
        for meta in batch_instances:
            collected.append(meta.storage_path)
    # number of batches matches length
    assert count_batches == len(loader)
    expected = [m.storage_path for m in ds.instances]
    assert collected == expected


def test_stopiteration_after_len_calls():
    N = 5
    B = 2
    ds = make_dataset(N)
    loader = SimpleProblemLoader(dataset=ds, batch_size=B, shuffle=False)
    it = iter(loader)
    # consume exactly len batches
    for _ in range(len(loader)):
        _ = next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_iter_resets_and_can_be_reused():
    N = 8
    B = 3
    ds = make_dataset(N)
    loader = SimpleProblemLoader(dataset=ds, batch_size=B, shuffle=False)
    # first pass
    first_pass = []
    for batch in loader:
        inst, _ = batch.get_context()
        first_pass.extend([m.storage_path for m in inst])
    # second pass - iter resets
    second_pass = []
    for batch in loader:
        inst, _ = batch.get_context()
        second_pass.extend([m.storage_path for m in inst])
    assert first_pass == second_pass


def test_batch_size_greater_than_dataset_returns_single_batch():
    N = 4
    B = 10
    ds = make_dataset(N)
    loader = SimpleProblemLoader(dataset=ds, batch_size=B, shuffle=False)
    assert len(loader) == 1
    batches = list(loader)
    assert len(batches) == 1
    batch_instances, _ = batches[0].get_context()
    assert len(batch_instances) == N


def test_shuffle_changes_order_when_true_and_different_seeds():
    N = 12
    B = 4
    ds = make_dataset(N)
    # Two loaders with different seeds should (very likely) produce different orders
    loader1 = SimpleProblemLoader(dataset=ds, batch_size=B, shuffle=True, seed=1)
    loader2 = SimpleProblemLoader(dataset=ds, batch_size=B, shuffle=True, seed=2)

    order1 = []
    for batch in loader1:
        inst, _ = batch.get_context()
        order1.extend([m.storage_path for m in inst])

    order2 = []
    for batch in loader2:
        inst, _ = batch.get_context()
        order2.extend([m.storage_path for m in inst])

    # They should be permutations of the original set
    assert set(order1) == set(order2) == set(m.storage_path for m in ds.instances)
    # Very likely the orders are not identical (non-deterministic but probability of equality is minuscule)
    assert order1 != order2


def test_shuffle_false_preserves_order():
    N = 6
    B = 2
    ds = make_dataset(N)
    loader = SimpleProblemLoader(dataset=ds, batch_size=B, shuffle=False)
    order = []
    for batch in loader:
        inst, _ = batch.get_context()
        order.extend([m.storage_path for m in inst])
    expected = [m.storage_path for m in ds.instances]
    assert order == expected


def test_invalid_batch_size_raises():
    ds = make_dataset(3)
    with pytest.raises(ValueError):
        _ = SimpleProblemLoader(dataset=ds, batch_size=0, shuffle=False)
    with pytest.raises(ValueError):
        _ = SimpleProblemLoader(dataset=ds, batch_size=-1, shuffle=False)


def test_batch_size_one_behavior():
    N = 5
    ds = make_dataset(N)
    loader = SimpleProblemLoader(dataset=ds, batch_size=1, shuffle=False)
    assert len(loader) == N
    collected = []
    for batch in loader:
        inst, _ = batch.get_context()
        assert len(inst) == 1
        collected.append(inst[0].storage_path)
    assert collected == [m.storage_path for m in ds.instances]
