#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np

from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph
from energnn.model.normalizer import CenterReduceNormalizer, TDigestNormalizer
from tests.utils import TestProblemLoader

np.random.seed(0)

n = 6
pb_loader = TestProblemLoader(
    dataset_size=4,
    n_batch=7,
    context_edge_params={
        "node": {"n_obj": n, "feature_list": ["a", "b"], "address_list": ["0"]},
        "edge": {"n_obj": n, "feature_list": ["c", "d", "g"], "address_list": ["0", "1"]},
    },
    oracle_edge_params={
        "node": {"n_obj": n, "feature_list": ["e"]},
        "edge": {"n_obj": n, "feature_list": ["f"]},
    },
    n_addr=n,
    shuffle=False,
)
pb_batch = next(iter(pb_loader))
context_batch, _ = pb_batch.get_context()
jax_context_batch = JaxGraph.from_numpy_graph(context_batch)
context = separate_graphs(context_batch)[0]
jax_context = JaxGraph.from_numpy_graph(context)


def test_center_reduce_normalizer():
    normalizer = CenterReduceNormalizer(update_limit=1000, use_running_average=False)
    output, infos = normalizer(graph=jax_context)
    output_batch, infos_batch = normalizer(graph=jax_context_batch)


def test_tdigest_normalizer():
    normalizer = TDigestNormalizer(update_limit=1000, use_running_average=True)
    output, infos = normalizer(graph=jax_context)
    output_batch, infos_batch = normalizer(graph=jax_context_batch)
