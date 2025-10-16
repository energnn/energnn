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
from omegaconf import DictConfig
from tqdm import tqdm

from energnn.graph import separate_graphs
from energnn.graph.jax import JaxGraph
from energnn.normalizer.normalization_function import NormalizationFunction
from energnn.normalizer.utils import apply_inverse_tree, gradient_inverse_tree, out_tree_to_graph
from energnn.problem import ProblemLoader

logger = logging.getLogger(__name__)


class Postprocessor:
    """
    Postprocessor for bijective, permutation-equivariant transformation of model outputs and gradient preconditioning.

    Applies the inverse of a fitted `NormalizationFunction` to decision graphs,
    and preconditions gradients by multiplying with the inverse Jacobian matrix of the postprocessing.

    :param f: Normalization function instance to apply and invert on model outputs.
    :param max_loaded_batch_count: Maximum number of batches to sample for fitting parameters.

    :ivar params: Parameters computed for inverse normalization.
    :ivar _fitted: Boolean indicating if parameters have been fit.
    """

    f: NormalizationFunction
    params: dict = {}
    max_loaded_batch_count: int = 10

    def __init__(self, f: NormalizationFunction, max_loaded_batch_count: int | None = None):
        """
        Initialize a Postprocessor with a given normalization function.

        :param f: NormalizationFunction to fit and apply inversely.
        :param max_loaded_batch_count: Number of batches used to compute params.
        """
        self.f = f
        if max_loaded_batch_count is not None:
            self.max_loaded_batch_count = max_loaded_batch_count
        self._fitted = False

    def to_pickle(self, *, file_path: str) -> None:
        """
        Serialize and save the Postprocessor to disk.

        :param file_path: Output file path for pickle.
        """
        with open(file_path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, *, file_path: str) -> Postprocessor:
        """
        Load a Postprocessor instance from a pickle file.

        :param file_path: Path to the pickle file.
        :return: Deserialized Postprocessor object.
        """
        with open(file_path, "rb") as handle:
            normalizer = pickle.load(handle)
        return normalizer

    def postprocess(self, in_graph: JaxGraph, get_info=False) -> tuple[JaxGraph, dict]:
        """
        Apply inverse normalization to a single decision graph.

        :param in_graph: JaxGraph containing model output features.
        :param get_info: If True, return quantile statistics before/after.
        :return: Tuple (denormalized_graph, info).
                 If `get_info`, info contains 'input_graph' and 'output_graph' quantiles.
        :raises RuntimeError: If Postprocessor is not yet fitted.
        """
        if not self._fitted:
            raise RuntimeError("Postprocessor parameters not yet fitted. Call fit_problem_loader first.")

        in_tree = {k: e.feature_array for k, e in in_graph.edges.items()}
        non_fictitious_tree = {k: jnp.expand_dims(e.non_fictitious, -1) for k, e in in_graph.edges.items()}
        out_tree = apply_inverse_tree(self.f, self.params, in_tree, non_fictitious_tree)
        out_graph = out_tree_to_graph(out_tree, in_graph)
        if get_info:
            infos = {"input_graph": in_graph.quantiles(), "output_graph": out_graph.quantiles()}
        else:
            infos = {}
        return out_graph, infos

    def postprocess_batch(self, in_graph: JaxGraph, get_info=False) -> tuple[JaxGraph, dict]:
        """
        Apply inverse normalization to a batch of decision graphs.

        :param in_graph: Batched JaxGraph of model outputs.
        :param get_info: Whether to return quantile statistics before/after.
        :return: Tuple (denormalized_graph, info_dict).
                 If `get_info`, info contains 'input_graph' and 'output_graph' quantiles.
        :raises RuntimeError: If Postprocessor has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Postprocessor parameters not yet fitted. Call fit_problem_loader first.")

        in_tree = {k: e.feature_array for k, e in in_graph.edges.items()}
        non_fictitious_tree = {k: jnp.expand_dims(e.non_fictitious, -1) for k, e in in_graph.edges.items()}
        out_tree = jax.vmap(apply_inverse_tree, in_axes=(None, None, 0, 0))(self.f, self.params, in_tree, non_fictitious_tree)
        out_graph = out_tree_to_graph(out_tree, in_graph)
        if get_info:
            infos = {"input_graph": in_graph.quantiles(), "output_graph": out_graph.quantiles()}
        else:
            infos = {}
        return out_graph, infos

    def precondition_gradient(self, out_graph: JaxGraph, grad_graph: JaxGraph, get_info=False) -> tuple[JaxGraph, dict]:
        """
        Precondition a gradient graph based on the provided decision graph.

        :param out_graph: Decision graph after postprocess.
        :param grad_graph: Graph of raw gradients matching `out_graph` structure.
        :param get_info: If True, return quantile statistics for gradients.
        :return: Tuple (preconditioned_grad_graph, info_dict) where info may include 'input_grad' and 'output_grad'.
        :raises RuntimeError: If Postprocessor has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Postprocessor parameters not yet fitted. Call fit_problem_loader first.")

        in_tree = {k: e.feature_array for k, e in out_graph.edges.items()}
        non_fictitious_tree = {k: jnp.expand_dims(e.non_fictitious, -1) for k, e in out_graph.edges.items()}
        prec_grad_tree = gradient_inverse_tree(self.f, self.params, in_tree, non_fictitious_tree)
        grad_tree = {k: e.feature_array for k, e in grad_graph.edges.items()}
        prec_grad_tree = jax.tree.map(lambda a, b: a / b, grad_tree, prec_grad_tree)
        prec_grad_graph = out_tree_to_graph(prec_grad_tree, out_graph)
        if get_info:
            infos = {"input_grad": grad_graph.quantiles(), "output_grad": prec_grad_graph.quantiles()}
        else:
            infos = {}
        return prec_grad_graph, infos

    def precondition_gradient_batch(self, out_graph: JaxGraph, grad_graph: JaxGraph, get_info=False) -> tuple[JaxGraph, dict]:
        """
        Precondition a batch of gradient graphs based on a batch of decision graphs.

        :param out_graph: Batched decision graphs after postprocess.
        :param grad_graph: Batched gradient graphs matching `out_graph` structure.
        :param get_info: If True, return quantile statistics for gradients.
        :return: Tuple (preconditioned_batch_grad_graph, info_dict) where info may include 'input_grad' and 'output_grad'.
        :raises RuntimeError: If Postprocessor has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Postprocessor parameters not yet fitted. Call fit_problem_loader first.")

        in_tree = {k: e.feature_array for k, e in out_graph.edges.items()}
        non_fictitious_tree = {k: 1.0 + 0.0 * jnp.expand_dims(e.non_fictitious, -1) for k, e in out_graph.edges.items()}
        prec_grad_tree = jax.vmap(gradient_inverse_tree, in_axes=(None, None, 0, 0))(
            self.f, self.params, in_tree, non_fictitious_tree
        )
        grad_tree = {k: e.feature_array for k, e in grad_graph.edges.items()}
        prec_grad_tree = jax.tree.map(lambda a, b: a / b, grad_tree, prec_grad_tree)
        prec_grad_graph = out_tree_to_graph(prec_grad_tree, out_graph)
        if get_info:
            infos = {"input_grad": grad_graph.quantiles(), "output_grad": prec_grad_graph.quantiles()}
        else:
            infos = {}
        return prec_grad_graph, infos

    def fit_problem_loader(
        self,
        problem_loader: ProblemLoader,
        problem_cfg: DictConfig | None = None,
        refit: bool = False,
        progress_bar: bool = True,
        device: jax.Device | None = None,
    ) -> None:
        """
        Fits the postprocessor parameters to a problem loader.

        Samples up to `max_loaded_batch_count` batches from the loader, computes normalization parameters,
        and sets the fitted flag.

        :param problem_loader: A problem loader to iter over problem instances.
        :param problem_cfg: Configuration for gradient computation.
        :param refit: Whether to refit when it has already been fitted, defaults to False
        :param progress_bar: Display a progress bar during fitting, defaults to True.
        :param device: JAX device for graph conversion (e.g., GPU), defaults to None.
        :return: None
        """
        if self._fitted and not refit:
            return

        logger.debug("Fit postprocessor parameters")

        problem_batch = next(iter(problem_loader))
        zero_decision, _ = problem_batch.get_zero_decision()
        gradient_batch, _ = problem_batch.get_gradient(decision=zero_decision, cfg=problem_cfg)
        gradients = separate_graphs(gradient_batch)
        jax_gradient = JaxGraph.from_numpy_graph(gradients[0], device=device)
        in_tree = {k: e.feature_array for k, e in jax_gradient.edges.items()}
        aux = jax.tree.map(self.f.init_aux, in_tree)

        for pb_batch in tqdm(
            islice(problem_loader, self.max_loaded_batch_count),
            desc="Fitting postprocessor",
            unit="batch",
            disable=not progress_bar,
        ):
            zero_decision, _ = pb_batch.get_zero_decision()
            gradient_batch, _ = pb_batch.get_gradient(decision=zero_decision, cfg=problem_cfg)
            gradient_list = separate_graphs(graph_batch=gradient_batch)
            for gradient in gradient_list:
                gradient.unpad()
                jax_gradient = JaxGraph.from_numpy_graph(gradient, device=device)
                in_tree = {k: -e.feature_array for k, e in jax_gradient.edges.items()}
                aux = jax.tree.map(self.f.update_aux, in_tree, aux)

        self.params = jax.tree.map(self.f.compute_params, in_tree, aux)
        self._fitted = True
