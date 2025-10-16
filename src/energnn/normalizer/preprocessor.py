#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
from __future__ import annotations
import logging
import pickle
from itertools import islice

import jax
import jax.numpy as jnp
from tqdm import tqdm

from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph
from energnn.normalizer.normalization_function.normalization_function import NormalizationFunction
from energnn.normalizer.utils import apply_inverse_tree, apply_tree, out_tree_to_graph
from energnn.problem import ProblemLoader

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocessor for bijective, permutation-equivariant normalization of graph features.

    This class fits a `NormalizationFunction` over batches of graphs, computes
    necessary normalization parameters, and applies forward/inverse transformations
    to graph data.

    :param f: NormalizationFunction instance to apply to graph features.
    :param max_loaded_batch_count: Maximum number of batches to sample for fitting.
                                   If None, uses default attribute value.

    :ivar params: Normalization parameters computed after fitting.
    :ivar _fitted: Boolean indicating whether parameters have been fit.
    """

    f: NormalizationFunction
    params: dict = {}
    max_loaded_batch_count: int = 10

    def __init__(self, f: NormalizationFunction, max_loaded_batch_count: int | None = None):
        """
        Construct a Preprocessor with a given normalization function.

        :param f: NormalizationFunction to fit and apply.
        :param max_loaded_batch_count: Number of batches to sample for fitting.
        """
        self.f = f
        if max_loaded_batch_count is not None:
            self.max_loaded_batch_count = max_loaded_batch_count
        self._fitted = False

    def to_pickle(self, *, file_path: str) -> None:
        """
        Serialize and save the Preprocessor to disk.

        :param file_path: Path to output pickle file.
        """
        with open(file_path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, *, file_path: str) -> Preprocessor:
        """
        Load a Preprocessor instance from a pickle file.

        :param file_path: Path to input pickle file.
        :return: Deserialized Preprocessor.
        """
        with open(file_path, "rb") as handle:
            normalizer = pickle.load(handle)
        return normalizer

    def preprocess(self, in_graph: JaxGraph, get_info=False) -> tuple[JaxGraph, dict]:
        """
        Apply fitted normalization to a single graph instance.

        :param in_graph: Input JaxGraph with raw features.
        :param get_info: Whether to return quantile statistics before/after.
        :return: Tuple of (normalized_graph, info_dict).
                 If `get_info`, info contains 'input_graph' and 'output_graph' quantiles.
        :raises RuntimeError: If Preprocessor has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor parameters not yet fitted. Call fit_problem_loader first.")

        in_tree = {k: e.feature_array for k, e in in_graph.edges.items()}
        non_fictitious_tree = {k: jnp.expand_dims(e.non_fictitious, -1) for k, e in in_graph.edges.items()}
        out_tree = apply_tree(self.f, self.params, in_tree, non_fictitious_tree)
        out_graph = out_tree_to_graph(out_tree, in_graph)
        if get_info:
            infos = {"input_graph": in_graph.quantiles(), "output_graph": out_graph.quantiles()}
        else:
            infos = {}
        return out_graph, infos

    def preprocess_batch(self, in_graph: JaxGraph, get_info=False) -> tuple[JaxGraph, dict]:
        """
        Apply normalization to a batch of graph instances.

        :param in_graph: A batched JaxGraph with raw features.
        :param get_info: Whether to return quantile statistics before/after.
        :return: Tuple (normalized_graph, info_dict).
                 If `get_info`, info contains 'input_graph' and 'output_graph' quantiles.
        :raises RuntimeError: If Preprocessor has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor parameters not yet fitted. Call fit_problem_loader first.")

        in_tree = {k: e.feature_array for k, e in in_graph.edges.items()}
        non_fictitious_tree = {k: jnp.expand_dims(e.non_fictitious, -1) for k, e in in_graph.edges.items()}
        out_tree = jax.vmap(apply_tree, in_axes=[None, None, 0, 0])(self.f, self.params, in_tree, non_fictitious_tree)
        out_graph = out_tree_to_graph(out_tree, in_graph)
        if get_info:
            infos = {"input_graph": in_graph.quantiles(), "output_graph": out_graph.quantiles()}
        else:
            infos = {}
        return out_graph, infos

    def preprocess_inverse(self, in_graph: JaxGraph, get_info=False) -> tuple[JaxGraph, dict]:
        """
        Revert normalization on a single graph instance.

        :param in_graph: Normalized JaxGraph to invert.
        :param get_info: Whether to return quantile statistics before/after.
        :return: Tuple (denormalized_graph, info_dict).
                 If `get_info`, info contains 'input_graph' and 'output_graph' quantiles.
        :raises RuntimeError: If Preprocessor has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor parameters not yet fitted. Call fit_problem_loader first.")

        in_tree = {k: e.feature_array for k, e in in_graph.edges.items()}
        non_fictitious_tree = {k: jnp.expand_dims(e.non_fictitious, -1) for k, e in in_graph.edges.items()}
        out_tree = apply_inverse_tree(self.f, self.params, in_tree, non_fictitious_tree)
        out_graph = out_tree_to_graph(out_tree, in_graph)
        if get_info:
            infos = {"input_graph": in_graph.quantiles(), "output_graph": out_graph.quantiles()}
        else:
            infos = {}
        return out_graph, infos

    def preprocess_inverse_batch(self, in_graph: JaxGraph, get_info=False) -> tuple[JaxGraph, dict]:
        """
        Revert normalization on a batch of graph instances.

        :param in_graph: A normalized batched JaxGraph.
        :param get_info: Whether to return quantile statistics before/after.
        :return: Tuple (denormalized_graph, info_dict).
                 If `get_info`, info contains 'input_graph' and 'output_graph' quantiles.
        :raises RuntimeError: If Preprocessor has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor parameters not yet fitted. Call fit_problem_loader first.")

        in_tree = {k: e.feature_array for k, e in in_graph.edges.items()}
        non_fictitious_tree = {k: jnp.expand_dims(e.non_fictitious, -1) for k, e in in_graph.edges.items()}
        out_tree = jax.vmap(apply_inverse_tree, in_axes=[None, None, 0, 0])(self.f, self.params, in_tree, non_fictitious_tree)
        out_graph = out_tree_to_graph(out_tree, in_graph)
        if get_info:
            infos = {"input_graph": in_graph.quantiles(), "output_graph": out_graph.quantiles()}
        else:
            infos = {}
        return out_graph, infos

    def fit_problem_loader(
        self, problem_loader: ProblemLoader, refit: bool = False, progress_bar: bool = True, device: jax.Device | None = None
    ) -> None:
        """
        Fit normalization parameters using a ProblemLoader data source.

        Samples up to `max_loaded_batch_count` batches from the loader, computes normalization parameters,
        and sets the fitted flag.

        :param problem_loader: A problem loader to iter over problem instances.
        :param refit: whether to refit when it has already been fitted, defaults to False.
        :param progress_bar: Display a progress bar during fitting, defaults to True.
        :param device: JAX device for graph conversion (e.g., GPU), defaults to None.
        :return: None
        """
        if self._fitted and not refit:
            return

        logger.info("Fit preprocessor parameters")

        problem_batch = next(iter(problem_loader))
        context_batch, _ = problem_batch.get_context()
        contexts = separate_graphs(context_batch)
        jax_context = JaxGraph.from_numpy_graph(contexts[0], device=device)
        in_tree = {k: e.feature_array for k, e in jax_context.edges.items()}
        aux = jax.tree.map(self.f.init_aux, in_tree)

        for pb_batch in tqdm(
            islice(problem_loader, self.max_loaded_batch_count),
            desc="Fitting preprocessor",
            unit="batch",
            disable=not progress_bar,
        ):
            context_batch, _ = pb_batch.get_context()
            context_list = separate_graphs(graph_batch=context_batch)
            for context in context_list:
                context.unpad()
                jax_context = JaxGraph.from_numpy_graph(context, device=device)
                in_tree = {k: e.feature_array for k, e in jax_context.edges.items()}
                aux = jax.tree.map(self.f.update_aux, in_tree, aux)

        self.params = jax.tree.map(self.f.compute_params, in_tree, aux)
        self._fitted = True
